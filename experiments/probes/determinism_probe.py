"""Probe R3': does greedy output stay bitwise-identical across serving configurations?

One invocation = one engine configuration; results are dumped to JSON and compared
offline by compare_determinism.py. Run each config in a fresh process.

Knobs:
  --tp N                    tensor-parallel size
  --order batch|serial|shuffled   how requests are submitted (batch composition)
  --prefix-cache            enable prefix caching
  --warm                    run the prompt set once before the measured pass (cache hits)
  --max-batched-tokens K    chunked-prefill chunk budget
  --cpu-offload-gb G        use vLLM OffloadingConnector with G GB of CPU memory (KV tier)
  env VLLM_BATCH_INVARIANT=1 toggles vLLM batch-invariant kernels
"""
import argparse
import json
import os
import random
import time


def build_prompts(path, n, shared_prefix_tokens, seed):
    """Long shared system prompt + distinct user turns from UltraChat."""
    rng = random.Random(seed)
    qs = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            data = d.get("data") or d.get("messages") or []
            q = data[0] if data and isinstance(data[0], str) else (data[0].get("content") if data else None)
            if q and 20 < len(q) < 1500:
                qs.append(q)
            if len(qs) >= n * 5:
                break
    rng.shuffle(qs)
    filler = ("You are a careful assistant deployed in a production agent harness. "
              "Follow the tool-use policy, cite sources, and reason step by step. ") * max(1, shared_prefix_tokens // 30)
    return [filler + "\n\nUser: " + q + "\nAssistant:" for q in qs[:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="local UltraChat JSONL dataset")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--shared-prefix-tokens", type=int, default=1024)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--order", default="batch", choices=["batch", "serial", "shuffled"])
    ap.add_argument("--prefix-cache", action="store_true")
    ap.add_argument("--warm", action="store_true")
    ap.add_argument("--max-batched-tokens", type=int, default=8192)
    ap.add_argument("--cpu-offload-gb", type=float, default=0)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--enforce-eager", action="store_true", help="disable CUDA graphs")
    ap.add_argument("--no-compile", action="store_true", help="disable torch.compile (compilation level 0)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from vllm import LLM, SamplingParams

    kw = dict(model=a.model, tensor_parallel_size=a.tp, enable_prefix_caching=a.prefix_cache,
              max_num_batched_tokens=a.max_batched_tokens, max_model_len=a.max_model_len,
              gpu_memory_utilization=a.gpu_mem, seed=a.seed, enable_chunked_prefill=True,
              enforce_eager=a.enforce_eager)
    if a.no_compile:
        kw["compilation_config"] = 0
    if a.cpu_offload_gb:
        from vllm.config import KVTransferConfig
        kw["kv_transfer_config"] = KVTransferConfig(
            kv_connector="OffloadingConnector", kv_role="kv_both",
            kv_connector_extra_config={"cpu_bytes_to_use": int(a.cpu_offload_gb * 2**30)})
    llm = LLM(**kw)
    prompts = build_prompts(a.data, a.n, a.shared_prefix_tokens, a.seed)
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens, logprobs=1, seed=a.seed)

    def generate(order):
        idx = list(range(len(prompts)))
        if order == "shuffled":
            random.Random(a.seed + 1).shuffle(idx)
        outs = {}
        if order == "serial":
            for i in idx:
                outs[i] = llm.generate([prompts[i]], sp, use_tqdm=False)[0]
        else:
            res = llm.generate([prompts[i] for i in idx], sp, use_tqdm=False)
            outs = {i: r for i, r in zip(idx, res)}
        return [outs[i] for i in range(len(prompts))]

    if a.warm:
        generate("batch")
    t0 = time.time()
    res = generate(a.order)
    dt = time.time() - t0
    recs = []
    for r in res:
        o = r.outputs[0]
        lps = [lp[t].logprob if lp and t in lp else None for t, lp in zip(o.token_ids, o.logprobs or [])]
        recs.append({"token_ids": list(o.token_ids), "logprobs": lps,
                     "num_cached_tokens": getattr(r, "num_cached_tokens", None)})
    import vllm
    meta = {k: v for k, v in vars(a).items()}
    meta.update(batch_invariant=os.environ.get("VLLM_BATCH_INVARIANT", "0"), vllm=vllm.__version__, wall_s=dt)
    json.dump({"meta": meta, "records": recs}, open(a.out, "w"))
    print("done", a.out, f"{dt:.1f}s")


if __name__ == "__main__":
    main()
