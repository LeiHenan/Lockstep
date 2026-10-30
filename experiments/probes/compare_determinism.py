"""Compare determinism-probe outputs against a reference run.

Usage: python compare_determinism.py ref.json other1.json other2.json ...
Reports per run: fraction of sequences bitwise-identical (token ids), first-divergence
position stats, and max |logprob diff| over the common (pre-divergence) prefix.
"""
import json
import statistics
import sys


def load(p):
    return json.load(open(p))


def compare(ref, oth):
    same, firsts, maxdiff = 0, [], 0.0
    for r, o in zip(ref["records"], oth["records"]):
        a, b = r["token_ids"], o["token_ids"]
        k = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
        if k is None and len(a) == len(b):
            same += 1
        else:
            firsts.append(k if k is not None else min(len(a), len(b)))
        upto = k if k is not None else min(len(a), len(b))
        for x, y in zip(r["logprobs"][:upto], o["logprobs"][:upto]):
            if x is not None and y is not None:
                maxdiff = max(maxdiff, abs(x - y))
    n = len(ref["records"])
    return {"n": n, "identical_frac": same / n, "diverged": len(firsts),
            "first_div_median": statistics.median(firsts) if firsts else None,
            "first_div_min": min(firsts) if firsts else None, "max_logprob_diff_prefix": maxdiff}


def label(m):
    keys = ["tp", "order", "prefix_cache", "warm", "max_batched_tokens", "cpu_offload_gb", "batch_invariant"]
    return " ".join(f"{k}={m[k]}" for k in keys)


if __name__ == "__main__":
    ref = load(sys.argv[1])
    print("REF:", label(ref["meta"]))
    for p in sys.argv[2:]:
        o = load(p)
        c = compare(ref, o)
        cached = [r.get("num_cached_tokens") for r in o["records"] if r.get("num_cached_tokens") is not None]
        c["mean_cached_tokens"] = sum(cached) / len(cached) if cached else None
        print(label(o["meta"]), "->", json.dumps(c))
