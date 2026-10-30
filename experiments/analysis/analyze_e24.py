"""E24: evaluate the E7d screen predictions with 12 processes per configuration (pre-registered in
results_prereg/E24_PREREGISTRATION.md). Usage: python analyze_e24.py RUNS_ROOT"""
import json, math, os, sys
from lockstep import decisions as D

R = os.path.join(sys.argv[1], "e24")
PRED = {"qwen8b_mbt1024": "STABLE", "llama8b_mbt2048": "STABLE", "qwen8b_mbt3072": "UNSTABLE",
        "qwen8b_mbt12288": "UNSTABLE", "qwen4b_mbt8192": "UNSTABLE"}


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((c - h) / d, 2), round((c + h) / d, 2)]


out = {}
for spec in sorted(os.listdir(R)):
    sd = os.path.join(R, spec)
    if not os.path.exists(os.path.join(sd, "report.json")):
        continue
    cls = json.load(open(os.path.join(sd, "report.json")))["classes"]
    warm = [[r for r in c if r != "base" and os.path.exists(os.path.join(sd, r, "outputs.json"))] for c in cls]
    warm = [c for c in warm if c]
    n = sum(len(c) for c in warm)
    mc = max(warm, key=len)
    outliers = [o for c in warm if c is not mc for o in c]
    dref = D.load(os.path.join(sd, mc[0]))
    attributed = sum(1 for o in outliers if D.diff(dref, D.load(os.path.join(sd, o))))
    name = spec.replace("e24_", "")
    obs = "UNSTABLE" if len(warm) > 1 else "STABLE"
    out[name] = {"prediction": PRED[name], "observed": obs, "correct": obs == PRED[name], "warm": n,
                 "classes": len(warm), "outliers": len(outliers), "flip_rate_ci95": wilson(len(outliers), n),
                 "outliers_with_decision_diff": attributed, "decisions_per_process": len(dref)}
    print(name, out[name])
json.dump(out, open(os.path.join(R, "e24_report.json"), "w"), indent=1)
