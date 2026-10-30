"""LD analysis (results_prereg/LD_LATE_DECISION_PREREGISTRATION.md).

python -m lockstep.ld_report RUN_DIR > ld_report.json

Per deployment: when the prefill instance changed its decision (test_retuned), the guard events after it
(republished, invalidated, check results), and per prompt whether the disaggregated output equals the decode
instance's own generation, split by whether the prompt's request started before or after the change.
"""
import glob
import json
import os
import sys


def _ev(d, side):
    f = os.path.join(d, f"inst_{side}", "capture", "live_guard.jsonl")
    return [json.loads(l) for l in open(f)] if os.path.exists(f) else []


def main(run):
    rows = []
    for f in sorted(glob.glob(os.path.join(run, "ld_*", "local_b.json"))):
        d = os.path.dirname(f)
        name = os.path.basename(d)
        meta = json.load(open(os.path.join(d, "meta.json")))
        outs = json.load(open(os.path.join(d, "outputs.json")))
        la, lb = json.load(open(os.path.join(d, "local_a.json"))), json.load(open(f))
        ea, eb = _ev(d, "a"), _ev(d, "b")
        rt = next((e for e in ea if e.get("event") == "test_retuned"), None)
        t_rt = rt["t"] if rt else None
        req = meta.get("request_t") or []
        exact = [x == y for x, y in zip(outs, lb)]
        post = [t_rt is not None and t > t_rt for t in req]
        after = lambda ev: [e for e in ev if t_rt is not None and e.get("t", 0) > t_rt]
        rep = next((e for e in after(ea) if e.get("event") == "republished"), None)
        inv = next((e for e in after(eb) if e.get("event") == "invalidated"), None)
        checks_after = [e.get("result") for e in after(eb) if e.get("event") == "check"]
        rows.append({
            "name": name, "scenario": name.split("_")[1].upper(),
            "retuned": rt.get("changed") if rt else None,
            "first_checks": [e.get("result") for e in eb if e.get("event") == "check"][:1],
            "checks_after_retune": sorted(set(checks_after)), "n_checks_after": len(checks_after),
            "republish_s": rep["t"] - t_rt if rep else None, "invalidate_s": inv["t"] - t_rt if inv else None,
            "canary_dropped": sum(e.get("event") == "canary_dropped" for e in ea + eb),
            "prompts_before": sum(not p for p in post), "prompts_after": sum(post),
            "inexact_before": sum((not p) and (not x) for p, x in zip(post, exact)),
            "inexact_after": sum(p and (not x) for p, x in zip(post, exact)),
            "local_a_ne_local_b": sum(x != y for x, y in zip(la, lb)),
            "new_usage_after_first_check": sum(e.get("event") == "republished" for e in ea),
        })
    by = lambda s: [r for r in rows if r["scenario"] == s]
    checks = {
        "LD-0_retune_applied_and_changes_numerics": all(r["retuned"] and r["local_a_ne_local_b"] for s in
                                                        ("LD1", "LD2", "LD3", "LD4") for r in by(s)),
        "LD-1_canary_only_harm_after_only": all(r["inexact_after"] > 0 and r["inexact_before"] == 0 for r in by("LD1")),
        "LD-2_stale_readmitted_and_harm": all("admitted_by_canary" in r["checks_after_retune"] and r["inexact_after"] > 0
                                              for r in by("LD2")),
        "LD-3_blocked_and_exact": all("blocked" in r["checks_after_retune"] and r["inexact_before"] + r["inexact_after"] == 0
                                      for s in ("LD3", "LD4") for r in by(s)),
        "LD-4_control_clean": all(r["invalidate_s"] is None and "blocked" not in r["first_checks"] and
                                  r["inexact_before"] + r["inexact_after"] == 0 and r["new_usage_after_first_check"] == 0
                                  for r in by("LD5")),
    }
    print(json.dumps({"checks": checks, "rows": rows}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
