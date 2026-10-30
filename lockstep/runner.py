"""Run one engine configuration on a sentinel prompt set, with boundary capture on.

python -m lockstep.runner --config cfg.json --prompts prompts.json --out DIR [--max-tokens 1]
cfg.json: {"engine_kwargs": {...vllm.LLM kwargs...}, "env": {...extra env vars...}}
Writes DIR/capture/rank*.pkl (fingerprints), DIR/outputs.json (tokens), DIR/meta.json.
Env vars in cfg["env"] must be applied by the caller *before* launching this process
(see explore.launch); they are recorded here for provenance.
"""
import argparse
import json
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=1)
    ap.add_argument("--order", default="batch", choices=["batch", "serial"])
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--per-request-seed", action="store_true", help="seed_i = i for prompt i (seeded sampling)")
    ap.add_argument("--lora", default=None, help="LoRA adapter directory applied to every request")
    ap.add_argument("--two-pass", action="store_true",
                    help="generate every prompt once (discarded), then again; the recorded pass reuses KV state "
                         "from the first one (e.g. from a CPU offload tier when the GPU cache is too small)")
    ap.add_argument("--score-from", default=None,
                    help="trainer-side recompute: score prompt+completion of a previous run's outputs.json "
                         "in one prefill (prompt_logprobs) instead of generating")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    os.makedirs(a.out, exist_ok=True)
    if os.environ.get("LOCKSTEP_MODE", "boundary") != "off" or os.environ.get("LOCKSTEP_TRACE_ONLY"):
        # LOCKSTEP_TRACE_ONLY with LOCKSTEP_MODE=off: hidden-decision tracing without boundary capture
        os.environ["LOCKSTEP_DIR"] = os.path.join(a.out, "capture")

    load = None
    if os.environ.get("LOCKSTEP_SYNTH_LOAD"):  # controlled co-tenant: matmul loop on the same GPU (E15)
        import subprocess, sys, atexit
        # the loop exits when its parent dies: an engine that fails to start must not leave
        # an orphaned load behind (E15b left 29 such orphans running for three days)
        code = ("import torch,time,os\n"
                "pp=%d\n"
                "a=torch.randn(4096,4096,device='cuda',dtype=torch.bfloat16)\n"
                "while os.getppid()==pp:\n"
                "    for _ in range(50): a=(a@a).clamp_(-1,1)\n"
                "    torch.cuda.synchronize(); time.sleep(float(%r))\n"
                % (os.getpid(), os.environ.get("LOCKSTEP_SYNTH_SLEEP", "0.002")))
        load = subprocess.Popen([sys.executable, "-c", code])
        atexit.register(load.kill)

    from vllm import LLM, SamplingParams
    import vllm

    ek = dict(cfg["engine_kwargs"])
    if isinstance(ek.get("kv_transfer_config"), dict):
        from vllm.config import KVTransferConfig
        ek["kv_transfer_config"] = KVTransferConfig(**ek["kv_transfer_config"])
    llm = LLM(**ek)
    prompts = json.load(open(a.prompts))
    if a.per_request_seed:
        sps = [SamplingParams(temperature=a.temperature, max_tokens=a.max_tokens, logprobs=1, seed=i)
               for i in range(len(prompts))]
    else:
        sps = [SamplingParams(temperature=a.temperature, max_tokens=a.max_tokens, logprobs=1)] * len(prompts)
    lora = None
    if a.lora:
        from vllm.lora.request import LoRARequest
        lora = LoRARequest("adapter", 1, a.lora)
    t0 = time.time()
    if a.score_from:
        from vllm.inputs import TokensPrompt
        prev = json.load(open(a.score_from))
        sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)
        reqs = [TokensPrompt(prompt_token_ids=list(o["prompt_token_ids"]) + list(o["token_ids"])) for o in prev]
        res = llm.generate(reqs, sp, use_tqdm=False) if a.order == "batch" else             [llm.generate([r], sp, use_tqdm=False)[0] for r in reqs]
        outs = []
        for o, r in zip(prev, res):
            n0, ids = len(o["prompt_token_ids"]), list(o["prompt_token_ids"]) + list(o["token_ids"])
            pl = r.prompt_logprobs or []
            sc = [repr(pl[j][ids[j]].logprob) if j < len(pl) and pl[j] and ids[j] in pl[j] else None
                  for j in range(n0, len(ids))]
            outs.append({"token_ids": o["token_ids"], "logprobs": sc, "prompt_len": n0, "pkey": o.get("pkey"),
                         "rollout_logprobs": o["logprobs"]})
        json.dump(outs, open(os.path.join(a.out, "outputs.json"), "w"))
        json.dump({"config": cfg, "vllm": vllm.__version__, "score_from": a.score_from, "gen_wall_s": time.time() - t0},
                  open(os.path.join(a.out, "meta.json"), "w"), indent=1)
        print("RUN_DONE", a.out)
        return
    if a.two_pass:
        for i, p in enumerate(prompts):
            llm.generate([p], sps[i], use_tqdm=False, lora_request=lora)
    if a.order == "serial":
        res = [llm.generate([p], sps[i], use_tqdm=False, lora_request=lora)[0] for i, p in enumerate(prompts)]
    else:
        res = llm.generate(prompts, sps, use_tqdm=False, lora_request=lora)
    dt = time.time() - t0
    import zlib
    import numpy as np
    def lps(o):  # exact float logprob of each chosen token (bitwise comparable via repr)
        return [repr(lp[t].logprob) if lp and t in lp else None for t, lp in zip(o.token_ids, o.logprobs or [])]
    outs = [{"token_ids": list(r.outputs[0].token_ids), "logprobs": lps(r.outputs[0]), "text": r.outputs[0].text,
             "prompt_len": len(r.prompt_token_ids), "prompt_token_ids": list(r.prompt_token_ids),
             "pkey": zlib.crc32(np.asarray(r.prompt_token_ids, dtype=np.int64).tobytes())} for r in res]
    json.dump(outs, open(os.path.join(a.out, "outputs.json"), "w"))
    meta = {"config": cfg, "vllm": vllm.__version__, "gen_wall_s": dt, "order": a.order,
            "max_tokens": a.max_tokens, "n_prompts": len(prompts),
            "env": {k: v for k, v in os.environ.items() if k.startswith(("VLLM_", "LOCKSTEP_", "CUDA_VISIBLE", "TORCHINDUCTOR"))}}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print("RUN_DONE", a.out, f"{dt:.2f}s")
    if load is not None:
        load.kill()


if __name__ == "__main__":
    main()
