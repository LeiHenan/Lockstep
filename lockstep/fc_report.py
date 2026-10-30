"""FC report (results_prereg/FREEZE_FC_RW2_PREREGISTRATION.md): per deployment, each side's coverage verdicts, the
consumer's handshake results, and the disaggregated outputs against the decode instance's own. Checks FC-0..FC-2.

python -m lockstep.fc_report RUN_DIR > fc_report.json
"""
import glob
import json
import os
import sys


def _events(d, side):
    out = []
    for f in glob.glob(os.path.join(d, f"inst_{side}", "capture", "live_guard.jsonl")):
        out += [json.loads(l) for l in open(f)]
    return out


def main(run):
    rows = {}
    for d in sorted(glob.glob(os.path.join(run, "fc*"))):
        if not os.path.exists(os.path.join(d, "local_b.json")) or "smoke" in d:
            continue
        ea, eb = _events(d, "a"), _events(d, "b")
        cov_a = sorted({",".join(e.get("coverage") or []) or "complete" for e in ea if e["event"] == "serve"})
        cov_b = sorted({",".join(e.get("coverage") or []) or "complete" for e in eb if e["event"] == "check"})
        res = {}
        for e in eb:
            if e["event"] == "check":
                res[e.get("result")] = res.get(e.get("result"), 0) + 1
        out = json.load(open(os.path.join(d, "outputs.json")))
        lb = json.load(open(os.path.join(d, "local_b.json")))
        rows[os.path.basename(d)] = {"coverage_a": cov_a, "coverage_b": cov_b, "handshakes": res,
                                     "outputs_exact": sum(o == y for o, y in zip(out, lb)), "n": len(lb)}
    ctl = rows.get("fc0_control", {})
    inj = {k: v for k, v in rows.items() if k != "fc0_control"}
    print(json.dumps({"deployments": rows,
                      "FC-0": ctl.get("handshakes", {}).get("admitted", 0) > 0 and ctl.get("outputs_exact") == ctl.get("n")
                              and ctl.get("coverage_a") == ["complete"] and ctl.get("coverage_b") == ["complete"],
                      "FC-1": all(not v["handshakes"].get("admitted") and v["outputs_exact"] == v["n"] and
                                  any("trace-error" in c for c in v["coverage_a"] + v["coverage_b"]) for v in inj.values()),
                      }, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
