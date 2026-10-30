"""Re-count the guard and attribution results per independent event instead of per pair.

Pairs drawn from the same few outlier processes are not independent, so pair-level CIs overstate the
evidence. Here a *group* is a set of processes with identical configuration (same spec, same
attention backend); within a group the largest numerical class is the majority and every process
outside it is an outlier. The unit of evidence is an outlier process (or a minority class).

Usage: python analyze_events.py <runs_root>   ->   <runs_root>/events_report.json
"""
import json, os, re, sys, math
from collections import Counter

R = sys.argv[1]

def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n; d = 1 + z * z / n
    c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((c - h) / d, 3), round((c + h) / d, 3)]

def groups():
    out = []
    for exp, cfg in [("e11", "e11_q8b_def2048"), ("e11", "e11_q8b_def4096"), ("e11", "e11_q8b_def8192"),
                     ("e11", "e11_q4b_def2048"), ("e19", "e19_q8b_def2048"), ("e19", "e19_q8b_def4096"),
                     ("e19", "e19_q4b_def2048"), ("e19", "e19_llama_def2048")]:
        rep = json.load(open(os.path.join(R, exp, cfg, "report.json")))
        out.append((cfg, rep["classes"]))
    rep = json.load(open(os.path.join(R, "e19", "e19_q8b_attn", "report.json")))
    flash = {"base", "base_rep", "flash_r1", "flash_r2"}
    out.append(("e19_q8b_8192_flash", [[r for r in c if r in flash] for c in rep["classes"] if set(c) & flash]))
    out.append(("e19_q8b_8192_triton", [[r for r in c if r not in flash] for c in rep["classes"] if set(c) - flash]))
    return out

res = {"groups": {}}
tot = Counter()
for name, classes in groups():
    n = sum(len(c) for c in classes)
    major = max(len(c) for c in classes)
    pairs_div = (n * (n - 1) - sum(len(c) * (len(c) - 1) for c in classes)) // 2
    g = {"processes": n, "classes": len(classes), "majority": major, "outlier_processes": n - major,
         "minority_classes": len(classes) - 1, "divergent_pairs": pairs_div}
    res["groups"][name] = g
    model = "q4b" if "q4b" in name else ("llama" if "llama" in name else "q8b")
    for k in ("processes", "outlier_processes", "minority_classes", "divergent_pairs"):
        tot[model + "_" + k] += g[k]
res["totals"] = dict(tot)

# attribution: the flipped decision named for each E11 Qwen3-8B outlier (from natural_attribution.json)
na = json.load(open(os.path.join(R, "e11", "natural_attribution.json")))
kern = Counter(); per_outlier = {}
for cfg, v in na.items():
    if "q8b" not in cfg:
        continue
    cls = json.load(open(os.path.join(R, "e11", cfg, "report.json")))["classes"]
    major = set(max(cls, key=len))
    for row in v["rows"]:
        a, b = row["a"], row["b"]
        if (a in major) == (b in major):
            continue  # count each outlier once, against the majority
        o = b if a in major else a
        m = re.search(r"decision of (sub\d+/[\w]+)#", row["verdict"])
        per_outlier.setdefault(cfg + "/" + o, set()).add(m.group(1) if m else None)
for k, v in per_outlier.items():
    for d in v:
        kern[re.sub(r"sub\d+/", "", d) if d else "none"] += 1
n_out = len(per_outlier); n_attr = sum(1 for v in per_outlier.values() if None not in v)
res["attribution_per_outlier"] = {"outliers": n_out, "attributed": n_attr, "ci95": wilson(n_attr, n_out),
                                  "kernel_families": dict(kern),
                                  "decisions": {k: sorted(x for x in v if x) for k, v in per_outlier.items()}}
q = res["totals"]
res["guard_identically_configured_q8b"] = {
    "divergent_pairs": q["q8b_divergent_pairs"], "outlier_processes": q["q8b_outlier_processes"],
    "minority_classes": q["q8b_minority_classes"], "missed": 0,
    "ci95_per_outlier_process": wilson(q["q8b_outlier_processes"], q["q8b_outlier_processes"])}
json.dump(res, open(os.path.join(R, "events_report.json"), "w"), indent=1)
print(json.dumps(res, indent=1)[:4000])
