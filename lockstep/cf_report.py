"""CF analysis (results_prereg/CF_KVE_PREREGISTRATION.md): does a fresh process that replays one side's timed
decisions reproduce that side's outputs?

python -m lockstep.cf_report RUN_DIR MIRROR_RUNS > cf_report.json
"""
import glob
import json
import os
import sys

PAIRS = {"k1_t3_r1": "k1l1/k1_t3_r1", "v7s_ww_r1": "v7s/lgm_qwen8b_homo_off_ww_r1", "v7s_ww_r2": "v7s/lgm_qwen8b_homo_off_ww_r2"}


def main(run, runs):
    rows = []
    for pair, src in PAIRS.items():
        loc = {s: json.load(open(os.path.join(runs, src, f"local_{s}.json"))) for s in "ab"}
        for side in "ab":
            d = os.path.join(run, f"cf_{pair}_{side}")
            if not os.path.exists(os.path.join(d, "outputs.json")):
                rows.append({"run": os.path.basename(d), "missing": True})
                continue
            out = json.load(open(os.path.join(d, "outputs.json")))
            other = "b" if side == "a" else "a"
            dec = {}
            for f in glob.glob(os.path.join(d, "inst_a", "capture", "decisions_*.json")):
                dec.update(json.load(open(f)))
            forced = [v.get("forced") for v in dec.values() if v.get("forced")]
            rows.append({"run": os.path.basename(d), "n": len(out),
                         "eq_replayed_side": sum(x == y for x, y in zip(out, loc[side])),
                         "eq_other_side": sum(x == y for x, y in zip(out, loc[other])),
                         "sides_differ": sum(x != y for x, y in zip(loc["a"], loc["b"])),
                         "forced": len(forced), "forced_applied": sum(bool(f.get("applied")) for f in forced)})
    ok = [r for r in rows if not r.get("missing")]
    checks = {"CF-1": bool(ok) and len(ok) == 6 and all(r["eq_replayed_side"] == r["n"] for r in ok),
              "CF-2": bool(ok) and all(r["forced"] and r["forced_applied"] == r["forced"] for r in ok)}
    print(json.dumps({"checks": checks, "rows": rows}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
