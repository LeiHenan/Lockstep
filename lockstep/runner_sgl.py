"""SGLang counterpart of lockstep.runner (offline Engine API).

python -m lockstep.runner_sgl --config cfg.json --prompts p.json --out DIR [--max-tokens 16] [--order batch|serial]
cfg.json: {"engine_kwargs": {...sglang.Engine kwargs...}, "env": {...}}
Must be launched with PYTHONPATH including lockstep/sitecustom and LOCKSTEP_ENGINE=sglang
(the explore scheduler does this when the spec sets "engine": "sglang").
"""
import argparse
import json
import os
import time
import zlib

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=1)
    ap.add_argument("--order", default="batch", choices=["batch", "serial"])
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    os.makedirs(a.out, exist_ok=True)
    os.environ["LOCKSTEP_DIR"] = os.path.join(a.out, "capture")
    os.environ["LOCKSTEP_ENGINE"] = "sglang"
    from lockstep.sglang_adapter import register
    register()  # in this process too (scheduler may run in-process for some configs)
    import sglang as sgl

    eng = sgl.Engine(**cfg["engine_kwargs"])
    tok = eng.tokenizer_manager.tokenizer
    prompts = json.load(open(a.prompts))
    sp = {"temperature": 0.0, "max_new_tokens": a.max_tokens}
    t0 = time.time()
    if a.order == "serial":
        res = [eng.generate(prompt=p, sampling_params=sp, return_logprob=True) for p in prompts]
    else:
        res = eng.generate(prompt=prompts, sampling_params=sp, return_logprob=True)
    dt = time.time() - t0
    outs = []
    for p, r in zip(prompts, res):
        ids = tok(p)["input_ids"]
        mi = r.get("meta_info") or {}
        olp = mi.get("output_token_logprobs") or []
        toks = list(r.get("output_ids") or [t[1] for t in olp])
        outs.append({"token_ids": toks, "logprobs": [repr(t[0]) for t in olp], "text": r.get("text"),
                     "prompt_len": len(ids), "pkey": zlib.crc32(np.asarray(ids, dtype=np.int64).tobytes())})
    json.dump(outs, open(os.path.join(a.out, "outputs.json"), "w"))
    import sglang
    meta = {"config": cfg, "engine": "sglang", "sglang": sglang.__version__, "gen_wall_s": dt, "order": a.order,
            "max_tokens": a.max_tokens, "n_prompts": len(prompts)}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    eng.shutdown()
    print("RUN_DONE", a.out, f"{dt:.2f}s")


if __name__ == "__main__":
    main()
