"""E3: does observation perturb numerics? Compare exact logprobs/tokens between runs.

python -m lockstep.e3_perturb DIR   (DIR = explore output dir of the e3 spec)
Prints a matrix of (token-identical fraction, logprob-bitwise-identical fraction) for all run pairs.
"""
import itertools
import json
import os
import sys


def load(d):
    return {o["pkey"]: o for o in json.load(open(os.path.join(d, "outputs.json")))}


def main():
    root = sys.argv[1]
    runs = sorted(n for n in os.listdir(root) if os.path.exists(os.path.join(root, n, "outputs.json")))
    data = {n: load(os.path.join(root, n)) for n in runs}
    res = {}
    for a, b in itertools.combinations(runs, 2):
        ks = data[a].keys() & data[b].keys()
        tok = sum(data[a][k]["token_ids"] == data[b][k]["token_ids"] for k in ks) / max(1, len(ks))
        lp = sum(data[a][k].get("logprobs") == data[b][k].get("logprobs") for k in ks) / max(1, len(ks))
        res[f"{a}|{b}"] = {"tok": tok, "logprob_bitwise": lp, "n": len(ks)}
        print(f"{a:16s} {b:16s} tok={tok:.3f} logprob_bitwise={lp:.3f} n={len(ks)}")
    json.dump(res, open(os.path.join(root, "e3_pairs.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
