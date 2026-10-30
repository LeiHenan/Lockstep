"""P1 summary over runs (results_prereg/P1_KV_POOL_PREREGISTRATION.md).

python -m lockstep.p1_report RUN_DIR [RUN_DIR ...] > p1_report.json

Per run and consumer (after lockstep.pool_demo --compare): identity equal to the producer's, hits, prompts whose
pool-served output differs from the consumer's own (log-probabilities or tokens), and the error of the importance
ratio exp(lp_served - lp_own) - 1 over the generated positions before the two token sequences first differ (as
an RL trainer that recomputes the sampler's log-probabilities would see it). Also the prediction checks P1-1..P1-4.
"""
import json
import math
import os
import sys


def _lps(row):
    return [float(x) for x in row["logprobs"]]


def ratio_errors(pool, own):
    errs = []
    for x, y in zip(pool, own):
        for t, (a, b, la, lb) in enumerate(zip(x["token_ids"], y["token_ids"], _lps(x), _lps(y))):
            errs.append(abs(math.exp(la - lb) - 1))
            if a != b:  # the positions after the first different token condition on different prefixes
                break
    return errs


def _q(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))] if v else None


def main(dirs):
    runs, agg = [], {}
    for d in dirs:
        rep = json.load(open(os.path.join(d, "pool_report.json")))
        for tag, c in rep["consumers"].items():
            pool = json.load(open(os.path.join(d, f"pool_{tag}.json")))
            own = json.load(open(os.path.join(d, f"own_{tag}.json")))
            e = ratio_errors(pool, own)
            row = {"run": os.path.basename(d), "consumer": c["name"], "key": c["key"], "n": rep["n"],
                   "identity_eq_producer": c["identity_eq_producer"], "hits": c["hits"],
                   "served_ne_own": c["served_ne_own"], "served_ne_own_tokens": c["served_ne_own_tokens"],
                   "own_ne_producer": c["own_ne_producer"], "ratio_err_p50": _q(e, 0.5),
                   "ratio_err_p99": _q(e, 0.99), "ratio_err_max": max(e) if e else None, "positions": len(e)}
            runs.append(row)
            fam = "F06" if "f06" in row["run"] else "F8"
            a = agg.setdefault(f"{fam}/{c['name']}/{c['key']}", {"runs": 0, "prompts": 0, "hits": 0,
                                                                   "served_ne_own": 0, "served_ne_own_tokens": 0})
            a["runs"] += 1
            a["prompts"] += rep["n"]
            for k in ("hits", "served_ne_own", "served_ne_own_tokens"):
                a[k] += c[k]
    tok = [r for r in runs if r["key"] == "tokens"]
    ide = [r for r in runs if r["key"] == "identity"]
    checks = {
        "P1-1_plant_tokens_half_of_hits": all(r["served_ne_own"] * 2 >= r["hits"] for r in tok if r["consumer"] == "plant"),
        "P1-1_f06_cold_tokens_some_run": any(r["served_ne_own"] for r in tok if r["consumer"] == "cold" and "f06" in r["run"]),
        "P1-2_identity_never_differs": all(r["served_ne_own"] == 0 for r in ide),
        "P1-3_equal_identity_hits_as_tokens": all(
            r["hits"] == next(t["hits"] for t in tok if t["run"] == r["run"] and t["consumer"] == r["consumer"])
            for r in ide if r["identity_eq_producer"]),
        "P1-3_plant_identity_hits_zero": all(r["hits"] == 0 for r in ide if r["consumer"] == "plant"),
        "P1-4_budget_identity_hits": [r["hits"] for r in ide if r["consumer"] == "budget"],
        "P1-4_budget_tokens_served_ne_own": [r["served_ne_own"] for r in tok if r["consumer"] == "budget"],
    }
    print(json.dumps({"checks": checks, "aggregate": agg, "runs": runs}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
