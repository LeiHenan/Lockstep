"""G7 analysis (results_prereg/G7_HELDOUT_SHAPES_PREREGISTRATION.md).

python -m lockstep.analyze_g7 RUN_DIR PROMPTS_META.json > g7_report.json

Per deployment: the live verdicts (identity alone and with the canary), whether the identities differed,
whether the canaries agreed, and per prompt group (sentinel, short, unrelated, long) whether the two
instances' prefills (first generated token's log-probability) and whole outputs differ. Checks G1-G5.
"""
import glob
import json
import os
import sys


def main(run, meta):
    groups = json.load(open(meta))["groups"]
    rows = []
    for d in sorted(glob.glob(os.path.join(run, "*", "local_b.json"))):
        d = os.path.dirname(d)
        name = os.path.basename(d)
        la, lb = json.load(open(os.path.join(d, "local_a.json"))), json.load(open(os.path.join(d, "local_b.json")))
        pd = json.load(open(os.path.join(d, "outputs.json")))
        ev = [json.loads(l) for l in open(os.path.join(d, "inst_b", "capture", "live_guard.jsonl"))]
        checks = [e for e in ev if e.get("event") == "check"]
        res = sorted({e.get("result") for e in checks})
        ident_equal = "admitted" in res  # vLLM's check passed with both identities folded in
        ca = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "inst_*", "capture", "canary.json")))]
        pre = [(x["logprobs"] or [None])[0] != (y["logprobs"] or [None])[0] for x, y in zip(la, lb)]
        gen = [x != y for x, y in zip(la, lb)]
        per = {g: {"prefill_diff": sum(pre[a:b]), "output_diff": sum(gen[a:b]), "n": b - a} for g, (a, b) in groups.items()}
        rows.append({"name": name, "flag": "on" if "_on_" in name else "off", "prime": name.startswith("prime"),
                     "verdicts": res, "identity_equal": ident_equal,
                     "canary_equal": (ca[0] == ca[1]) if len(ca) == 2 else None,
                     "kv_unsafe": any(pre), "outputs_differ": any(gen), "per_group": per,
                     "nonsentinel_prefill_diff": sum(v["prefill_diff"] for g, v in per.items() if g != "sentinel"),
                     "canary_prompt_differs": pre[0],
                     "pd_eq_local_b": sum(x == y for x, y in zip(pd, lb)), "n": len(lb)})
    main_ = [r for r in rows if not r["prime"]]
    g1_unsafe = [r["name"] for r in rows if r["identity_equal"] and r["kv_unsafe"]]
    g2_unneeded_on = [r["name"] for r in main_ if r["flag"] == "on" and not r["identity_equal"] and not r["outputs_differ"]]
    g3_canary_miss = [r["name"] for r in rows if r["nonsentinel_prefill_diff"] and not r["canary_prompt_differs"]]
    hybrid_admit = lambda r: r["identity_equal"] or "admitted_by_canary" in r["verdicts"]
    g4_unsafe = [r["name"] for r in rows if hybrid_admit(r) and r["kv_unsafe"]]
    g4_inexact = [r["name"] for r in rows if r["pd_eq_local_b"] != r["n"]]
    g5 = sum(1 for r in main_ if r["flag"] == "off" and r["outputs_differ"])
    out = {"deployments": len(rows), "main": len(main_),
           "divergent_outputs": sum(r["outputs_differ"] for r in rows), "kv_unsafe": sum(r["kv_unsafe"] for r in rows),
           "G1_identity_unsafe_admits": g1_unsafe, "G2_flag_on_unneeded_blocks": g2_unneeded_on,
           "G3_canary_misses": g3_canary_miss, "G4_hybrid_unsafe_admits": g4_unsafe, "G4_pd_not_exact": g4_inexact,
           "G5_natural_divergent_flag_off_main": g5,
           "identity_unneeded_blocks": [r["name"] for r in rows if not r["identity_equal"] and not r["kv_unsafe"]],
           "hybrid_unneeded_blocks": [r["name"] for r in rows if not hybrid_admit(r) and not r["kv_unsafe"]],
           "rows": rows}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
