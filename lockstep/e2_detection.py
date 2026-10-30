"""E2: detection efficiency — hidden-state fingerprints vs. generated-token comparison.

For a pair of runs (a known violation), estimate P(detect) when only k prompts are used:
  * fingerprint signal: any differing kappa among the k prompts' prefill positions
  * token signal (T tokens): any of the k prompts' first T generated tokens differ
  * log-prob signal (T tokens, if outputs carry log-probs): any of the first T sampled tokens'
    log-probs differ bitwise; T=1 needs only the prefill, like the fingerprint signal
Probabilities are estimated by sampling k-subsets (seeded) from the prompt set.

python -m lockstep.e2_detection RUN_A RUN_B [--trials 2000] [--ks 1,2,4,8,16,32] [--Ts 1,4,16]
"""
import argparse
import json
import os
import random

from lockstep import align


def per_prompt_signals(run_a, run_b):
    import numpy as np
    keep = align.prompt_keys(run_a)
    ta = align.consumed_only(align.kappa_table(align.load_records(run_a), keep)[0])
    tb = align.consumed_only(align.kappa_table(align.load_records(run_b), keep)[0])
    common, ia, ib = np.intersect1d(ta.keys, tb.keys, assume_unique=True, return_indices=True)
    ok = ta.widths[ia] == tb.widths[ib]
    common, ia, ib = common[ok], ia[ok], ib[ok]
    pk = (common >> np.uint64(31)).astype(np.int64)
    bad = ta.vals[ia] != tb.vals[ib]
    fp = {int(p): False for p in np.unique(pk)}
    for p in np.unique(pk[bad]):
        fp[int(p)] = True
    oa = {o["pkey"]: o["token_ids"] for o in json.load(open(os.path.join(run_a, "outputs.json")))}
    ob = {o["pkey"]: o["token_ids"] for o in json.load(open(os.path.join(run_b, "outputs.json")))}
    la = {o["pkey"]: o.get("logprobs") for o in json.load(open(os.path.join(run_a, "outputs.json")))}
    lb = {o["pkey"]: o.get("logprobs") for o in json.load(open(os.path.join(run_b, "outputs.json")))}
    first_lp_div = {}  # first generated position whose log-prob differs bitwise (index 0 = prefill only)
    for p in la.keys() & lb.keys():
        if la[p] is None or lb[p] is None:
            continue
        first_lp_div[p] = next((i for i, (u, v) in enumerate(zip(la[p], lb[p])) if u != v), None)
    first_tok_div = {}
    for p in oa.keys() & ob.keys():
        x, y = oa[p], ob[p]
        d = next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), None)
        first_tok_div[p] = d  # None = identical within generated length
    prompts = sorted(set(fp) & set(first_tok_div))
    return prompts, fp, first_tok_div, first_lp_div


def estimate(prompts, fp, ftd, ks, Ts, trials, seed=0, fld=None):
    rng = random.Random(seed)
    fld = fld or {}
    use_lp = all(p in fld for p in prompts)
    res = {"fingerprint": {}, "tokens": {T: {} for T in Ts}}
    if use_lp:
        res["logprobs"] = {T: {} for T in Ts}
    for k in ks:
        if k > len(prompts):
            continue
        hit_fp, hit_tok, hit_lp = 0, {T: 0 for T in Ts}, {T: 0 for T in Ts}
        for _ in range(trials):
            s = rng.sample(prompts, k)
            hit_fp += any(fp[p] for p in s)
            for T in Ts:
                hit_tok[T] += any(ftd[p] is not None and ftd[p] < T for p in s)
                if use_lp:
                    hit_lp[T] += any(fld[p] is not None and fld[p] < T for p in s)
        res["fingerprint"][k] = hit_fp / trials
        for T in Ts:
            res["tokens"][T][k] = hit_tok[T] / trials
            if use_lp:
                res["logprobs"][T][k] = hit_lp[T] / trials
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--ks", default="1,2,4,8,16,32")
    ap.add_argument("--Ts", default="1,4,16")
    ap.add_argument("--json")
    a = ap.parse_args()
    prompts, fp, ftd, fld = per_prompt_signals(a.run_a, a.run_b)
    ks = [int(x) for x in a.ks.split(",")]
    Ts = [int(x) for x in a.Ts.split(",")]
    res = estimate(prompts, fp, ftd, ks, Ts, a.trials, fld=fld)
    res["n_prompts"] = len(prompts)
    res["frac_prompts_fp_diverged"] = sum(fp[p] for p in prompts) / max(1, len(prompts))
    res["frac_prompts_tok_diverged_within"] = {T: sum(ftd[p] is not None and ftd[p] < T for p in prompts) / max(1, len(prompts)) for T in Ts}
    if fld:
        res["frac_prompts_lp_diverged_within"] = {T: sum(fld.get(p) is not None and fld[p] < T for p in prompts) / max(1, len(prompts)) for T in Ts}
    print(json.dumps(res, indent=1))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
