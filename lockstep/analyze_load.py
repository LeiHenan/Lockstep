"""Load-test analysis (results_prereg/LOADTEST_PREREGISTRATION.md): L1-L4 over all rounds.

python -m lockstep.analyze_load RUNS/loadtest [--rounds 1,2,3] > load_report.json

Per (policy, concurrency): mean and range over rounds of throughput, TTFT/TPOT p50/p99; the guard's
handshake verdicts (live_guard.jsonl of the decode instance) and short outputs per run.
L1: pd_guard vs pd, throughput and TPOT p50 within 2% (means over rounds). L3: pd_block complete.
L4: every pd_guard handshake admitted, every pd_block handshake blocked.
"""
import argparse
import glob
import json
import os
import re
import statistics as st

from lockstep.xfer_rate import rate

M = ["throughput_req_s", "throughput_tok_s", "ttft_p50", "ttft_p99", "tpot_p50", "tpot_p99", "e2e_p50"]


def verdicts(d):
    out = []
    for f in glob.glob(os.path.join(d, "inst_*", "capture", "live_guard.jsonl")):
        out += [json.loads(l)["result"] for l in open(f) if '"check"' in l]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--rounds", default=None)
    a = ap.parse_args()
    keep = set(a.rounds.split(",")) if a.rounds else None
    runs = {}
    for d in sorted(glob.glob(os.path.join(a.root, "*_c*_r*/"))):
        m = re.match(r"(.+)_c(\d+)_r(\d+)$", os.path.basename(os.path.normpath(d)))
        if not m or not os.path.exists(os.path.join(d, "load.json")) or (keep and m[3] not in keep):
            continue
        s = json.load(open(os.path.join(d, "load.json")))["summary"]
        s.update(round=int(m[3]), verdicts=sorted(set(verdicts(d))), n_checks=len(verdicts(d)))
        if m[1] != "colo2":
            s["xfer"] = rate(d)
        runs.setdefault((m[1], int(m[2])), []).append(s)
    table = {}
    for (pol, c), rs in sorted(runs.items()):
        row = {"rounds": sorted(r["round"] for r in rs)}
        for k in M:
            xs = [r[k] for r in rs if r.get(k) is not None]
            row[k] = {"mean": st.mean(xs), "min": min(xs), "max": max(xs)} if xs else None
        row["short_outputs"] = sum(r["short_outputs"] for r in rs)
        row["verdicts"] = sorted({v for r in rs for v in r["verdicts"]})
        row["n_checks"] = [r["n_checks"] for r in rs]
        if pol != "colo2":
            row["xfer_gb_s"] = [r["xfer"]["per_transfer_gb_s"] for r in rs]
        table[f"{pol}_c{c}"] = row
    l1 = {}
    for c in sorted({c for _, c in runs}):
        p, g = table.get(f"pd_c{c}"), table.get(f"pd_guard_c{c}")
        if p and g:
            rel = {k: g[k]["mean"] / p[k]["mean"] - 1 for k in ("throughput_req_s", "tpot_p50")}
            l1[c] = {"rel_diff": rel, "within_2pct": all(abs(v) <= 0.02 for v in rel.values()),
                     "pd_round_spread": {k: p[k]["max"] / p[k]["min"] - 1 for k in ("throughput_req_s", "tpot_p50")}}
    l3 = {k: v["short_outputs"] == 0 for k, v in table.items() if k.startswith("pd_block")}
    l4 = {k: v["verdicts"] for k, v in table.items() if k.startswith(("pd_guard", "pd_block"))}
    print(json.dumps({"L1": l1, "L3_complete": l3, "L4_verdicts": l4, "table": table}, indent=1))


if __name__ == "__main__":
    main()
