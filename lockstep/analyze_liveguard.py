"""Experiment A analysis (results_prereg/PLANTED_GUARD_PREREGISTRATION.md): in-connector guard.

python -m lockstep.analyze_liveguard RUNS/liveguard > liveguard_report.json

Per scenario: ground truth (do the two instances compute bit-identically when each serves every prompt
alone?), the live guard's handshake verdicts (decode-side `check` events), whether disaggregated outputs
equal the decode instance's own outputs, and the time to compute an identity. The offline guard's verdict
(lockstep.guard --pd) is joined when present. Overhead: per-request latency of rounds >= 1 (round 0
contains the handshake) with the guard on vs off, and the first request of round 0.
"""
import glob
import json
import os
import statistics as st
import sys


def _load(d, f):
    return json.load(open(os.path.join(d, f)))


def scenario(d):
    o, a, b = _load(d, "outputs.json"), _load(d, "local_a.json"), _load(d, "local_b.json")
    key = lambda r: (r["token_ids"], r["logprobs"])
    ev = [json.loads(l) for f in glob.glob(os.path.join(d, "inst_*", "capture", "live_guard.jsonl")) for l in open(f)]
    checks = [e for e in ev if e.get("event") == "check"]
    serves = [e for e in ev if e.get("event") == "serve"]
    verdicts = sorted({e["result"] for e in checks})
    meta = _load(d, "meta.json")
    lat = [x for x in meta.get("request_latency_s", []) if isinstance(x, list)]
    a_eq_b = sum(key(x) == key(y) for x, y in zip(a, b))
    return {"name": os.path.basename(os.path.normpath(d)),
            "instances_identical": a_eq_b == len(a), "local_a_eq_b": f"{a_eq_b}/{len(a)}",
            "pd_eq_local_b": f"{sum(key(x) == key(y) for x, y in zip(o, b))}/{len(o)}",
            "empty_outputs": sum(1 for r in o if not r["token_ids"]),
            "guard_on": bool(ev), "verdicts": verdicts, "n_checks": len(checks),
            "identity_ms": [round(e["identity_ms"], 3) for e in checks + serves],
            "n_trace_records": sorted({e.get("n_records") for e in checks + serves}),
            "latency_round0_first_s": next((x[1] for x in lat if x[0] == 0), None),
            "latency_rounds1plus_median_s": st.median([x[1] for x in lat if x[0] >= 1]) if any(x[0] >= 1 for x in lat) else None,
            "later_rounds_equal_to_round0": meta.get("later_rounds_equal_to_round0")}


def main(root):
    rows = [scenario(d) for d in sorted(glob.glob(os.path.join(root, "*/"))) if os.path.exists(os.path.join(d, "local_b.json"))]
    off = {}
    p = os.path.join(root, "guard_pd_offline.json")
    if os.path.exists(p):
        try:
            off = {s["scenario"]: s["allow"] for s in json.load(open(p))["scenarios"]}
        except Exception:
            off = {}
    for r in rows:
        r["offline_allow"] = off.get(r["name"])
    guarded = [r for r in rows if r["guard_on"] and r["verdicts"]]
    correct = [r for r in guarded if (r["verdicts"] == ["admitted"]) == r["instances_identical"]]
    unsafe = [r["name"] for r in guarded if "admitted" in r["verdicts"] and not r["instances_identical"]]
    pd_ok = [r for r in guarded if r["pd_eq_local_b"].split("/")[0] == r["pd_eq_local_b"].split("/")[1]]
    live_vs_off = [r["name"] for r in guarded if r["offline_allow"] is not None and r["offline_allow"] != (r["verdicts"] == ["admitted"])]
    ms = [x for r in guarded for x in r["identity_ms"]]
    on = [r["latency_rounds1plus_median_s"] for r in rows if r["name"].startswith("ovh_on") and r["latency_rounds1plus_median_s"]]
    offl = [r["latency_rounds1plus_median_s"] for r in rows if r["name"].startswith("ovh_off") and r["latency_rounds1plus_median_s"]]
    summary = {"guarded_scenarios": len(guarded), "verdict_correct": len(correct), "unsafe_admits": unsafe,
               "blocked": sum(1 for r in guarded if r["verdicts"] == ["blocked"]),
               "pd_equals_decode_local": len(pd_ok), "empty_outputs": sum(r["empty_outputs"] for r in guarded),
               "live_disagrees_with_offline": live_vs_off,
               "identity_ms_median": st.median(ms) if ms else None, "identity_ms_max": max(ms) if ms else None,
               "overhead": {"on_median_s": on, "off_median_s": offl,
                            "ratio_of_means": (st.mean(on) / st.mean(offl)) if on and offl else None}}
    print(json.dumps({"summary": summary, "scenarios": rows}, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1])
