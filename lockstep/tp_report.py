"""TP14 analysis (results_prereg/TP14_PREREGISTRATION.md).

python -m lockstep.tp_report RUN_DIR > tp14_report.json

Per deployment (prefill TP2 + decode TP2): each decode rank's first guard verdict and whether its identity equalled
the producer's rank (admitted on identity), every rank's coverage, the instances' own generations compared (all
outputs, first token), the disaggregated outputs against the decode instance's own, the decisions in which the two
instances' ranks differ (per rank), and how many decisions differ between the two ranks of one instance. Checks
TP-1..TP-6.
"""
import glob
import json
import os
import sys

from lockstep import ablate_identity, decisions


def _canon(use):
    return {f"{v.get('subgraph')}/{v.get('srcn') or v.get('kernel')}": json.dumps(v.get("chosen"), sort_keys=True)
            for v in use.values()}


def _rank_use(d, side, r):
    return decisions.load_rank(os.path.join(d, f"inst_{side}", "capture"), r)[0]


def row(d):
    la, lb = json.load(open(os.path.join(d, "local_a.json"))), json.load(open(os.path.join(d, "local_b.json")))
    pd = json.load(open(os.path.join(d, "outputs.json")))
    first = lambda x: (x["logprobs"] or [None])[0]  # noqa: E731
    ev = [json.loads(l) for l in open(os.path.join(d, "inst_b", "capture", "live_guard.jsonl"))]
    checks = [e for e in ev if e.get("event") == "check"]
    first_by_rank = {}
    for c in checks:
        first_by_rank.setdefault(c.get("rank", 0), c)
    serves = [json.loads(l) for l in open(os.path.join(d, "inst_a", "capture", "live_guard.jsonl"))
              if '"serve"' in l]
    cov = {f"b_r{r}": c.get("coverage") for r, c in first_by_rank.items()}
    cov.update({f"a_r{s.get('rank', 0)}": s.get("coverage") for s in serves})
    ranks = sorted(first_by_rank)
    diff_across, diff_within = {}, {}
    for r in ranks:
        ua, ub = _canon(_rank_use(d, "a", r)), _canon(_rank_use(d, "b", r))
        diff_across[r] = sorted(k for k in set(ua) | set(ub) if ua.get(k) != ub.get(k))
    for side in ("a", "b"):
        u0, u1 = _canon(_rank_use(d, side, 0)), _canon(_rank_use(d, side, 1))
        diff_within[side] = sum(u0.get(k) != u1.get(k) for k in set(u0) | set(u1))
    try:
        static_equal = ablate_identity.ident(os.path.join(d, "inst_a"), "budget") == \
            ablate_identity.ident(os.path.join(d, "inst_b"), "budget")
    except Exception:
        static_equal = None
    verdicts = {r: first_by_rank[r]["result"] for r in ranks}
    return {"name": os.path.basename(d), "n": len(lb),
            "first_verdicts": verdicts,
            "identity_admitted": bool(ranks) and all(v == "admitted" for v in verdicts.values()),
            "admitted": bool(ranks) and all(str(v).startswith("admitted") for v in verdicts.values()),
            "all_results": sorted({c["result"] for c in checks}),
            "coverage": cov, "covered": all(not v for v in cov.values()),
            "local_equal": sum(a == b for a, b in zip(la, lb)),
            "first_token_equal": sum(first(a) == first(b) for a, b in zip(la, lb)),
            "pd_equal_decode_local": sum(p == b for p, b in zip(pd, lb)),
            "rank_decisions_differ_across": {str(r): v[:8] for r, v in diff_across.items()},
            "n_rank_decisions_differ_across": {str(r): len(v) for r, v in diff_across.items()},
            "rank_decisions_differ_within": diff_within,
            "static_hash_equal": static_equal,
            "identity_ms": [c.get("identity_ms") for c in checks[:2]]}


def main(run):
    rows = [row(d) for d in sorted(glob.glob(os.path.join(run, "*")))
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "local_b.json"))]
    main_rows = [r for r in rows if not r["name"].startswith("warm_")]
    unsafe = [r["name"] for r in rows if r["admitted"] and r["pd_equal_decode_local"] < r["n"]]
    kv_unsafe = [r for r in main_rows if r["first_token_equal"] < r["n"]]
    print(json.dumps({
        "rows": rows,
        "TP-1_all_covered": all(r["covered"] for r in rows),
        "TP-2_identity_unsafe_admits": [r["name"] for r in rows if r["identity_admitted"] and r["first_token_equal"] < r["n"]],
        "TP-3_exact": sum(r["pd_equal_decode_local"] == r["n"] for r in rows), "deployments": len(rows),
        "unsafe_admits": unsafe,
        "TP-4_v032_off_diverged": sum(r["local_equal"] < r["n"] for r in main_rows if r["name"].startswith("v032_off")),
        "TP-5_kv_unsafe_with_identity_differing": [r["name"] for r in kv_unsafe if not r["identity_admitted"]],
        "kv_unsafe": [r["name"] for r in kv_unsafe],
        "TP-6_flag_on_hetero_identical": [(r["name"], r["identity_admitted"]) for r in main_rows
                                          if "_on_hetero" in r["name"] and r["first_token_equal"] == r["n"]],
        "static_hash_unsafe_admits": [r["name"] for r in kv_unsafe if r["static_hash_equal"]],
    }, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
