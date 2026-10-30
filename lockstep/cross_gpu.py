"""Do identically configured instances on different GPU generations compute the same prefill? (review round 9)
Offline, from the instances' own generations in existing deployments.

python -m lockstep.cross_gpu NAME=ROOT [NAME=ROOT ...] > cross_gpu.json

For Qwen3-8B on the 32-prompt sentinel set, budget 2048 and 256 sequences, flag on, vLLM 0.31: every instance
with exactly that configuration, grouped by the first generated token's log-probability over all prompts (its
prefill class). Each pair of platforms is compared on their largest classes: prompts whose first-token
log-probabilities agree bit for bit, whose generations agree, and the attention implementation the servers
logged ("FlashAttention version N"; vLLM names the backend FLASH_ATTN either way).
"""
import collections
import glob
import json
import os
import re
import sys

ARGS = "--max-num-batched-tokens 2048 --max-num-seqs 256"


def _args(spec):
    return " ".join(x for x in (spec or {}).get("args", []) if x not in ("--served-model-name", "m"))


def platform(root):
    classes, example, fa = collections.Counter(), {}, collections.Counter()
    for f in glob.glob(os.path.join(root, "**", "local_b.json"), recursive=True):
        d = os.path.dirname(f)
        if "failed_attempts" in d:
            continue
        try:
            meta = json.load(open(os.path.join(d, "meta.json")))
            model = json.load(open(os.path.join(d, "inst_b", "cfg.json")))["engine_kwargs"]["model"]
        except Exception:
            continue
        if not model.endswith("Qwen3-8B"):
            continue
        for side in ("a", "b"):
            spec = meta.get(side) or {}
            if _args(spec) != ARGS or spec.get("ibi") is not True:
                continue
            rows = json.load(open(os.path.join(d, f"local_{side}.json")))
            if len(rows) != 32:
                continue
            key = json.dumps([(r["logprobs"] or [None])[0] for r in rows])
            classes[key] += 1
            example[key] = rows
            log = os.path.join(d, f"server_{side}.log")
            if os.path.exists(log):
                m = re.search(r"FlashAttention version (\d)", open(log, errors="replace").read())
                fa[m.group(1) if m else "?"] += 1
    return classes, example, fa


def main(pairs):
    plats = {}
    for p in pairs:
        name, root = p.split("=", 1)
        plats[name] = platform(root)
    out = {"platforms": {n: {"instances": sum(c.values()), "classes": sorted(c.values(), reverse=True),
                             "flash_attention_version": dict(fa)} for n, (c, _, fa) in plats.items()},
           "pairs": []}
    names = list(plats)
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            (cx, ex, _), (cy, ey, _) = plats[x], plats[y]
            if not cx or not cy:
                continue
            a, b = ex[cx.most_common(1)[0][0]], ey[cy.most_common(1)[0][0]]
            out["pairs"].append({"a": x, "b": y, "n": len(a),
                                 "first_token_logprob_equal": sum((u["logprobs"] or [None])[0] == (v["logprobs"] or [None])[0]
                                                                  for u, v in zip(a, b)),
                                 "generations_equal": sum(u["token_ids"] == v["token_ids"] for u, v in zip(a, b))})
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
