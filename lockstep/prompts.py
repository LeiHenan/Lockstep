"""Sentinel prompt sets.

python -m lockstep.prompts --data dataset.jsonl --out prompts.json --n 32 [--shared-prefix-tokens 1024] [--long-frac 0.25]
Mixes a shared system prefix (exercises prefix caching) with distinct UltraChat questions;
a fraction of prompts is lengthened to cross compile-range / chunk boundaries.
"""
import argparse
import json
import random


def build(path, n, shared_prefix_tokens, long_frac, seed):
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
    out = []
    for i, q in enumerate(qs[:n]):
        body = q
        if rng.random() < long_frac:  # lengthen with extra context to cross range boundaries
            body = "\n\n".join(qs[(i + j) % len(qs)] for j in range(1, 8)) + "\n\nQuestion: " + q
        out.append(filler + "\n\nUser: " + body + "\nAssistant:")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="local UltraChat JSONL dataset")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--shared-prefix-tokens", type=int, default=1024)
    ap.add_argument("--long-frac", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    json.dump(build(a.data, a.n, a.shared_prefix_tokens, a.long_frac, a.seed), open(a.out, "w"))


if __name__ == "__main__":
    main()
