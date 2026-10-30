"""Attribution for the divergent processes of E19 (not evaluated before; 2026-10-17).
For each identically configured group (same spec, same attention backend), every process outside the
largest class is compared with a process of the largest class: attributed iff the subgraph that
produced the first divergent value contains the earliest differing traced decision.
Usage: python analyze_attr_e19.py RUNS_ROOT"""
import json, os, sys
from lockstep import decisions as D

R = os.path.join(sys.argv[1], "e19")
FLASH = {"base", "base_rep", "flash_r1", "flash_r2"}
res = {}
for spec in ["e19_q8b_def2048", "e19_q8b_def4096", "e19_q8b_attn"]:
    sd = os.path.join(R, spec)
    rep = json.load(open(os.path.join(sd, "report.json")))
    groups = [rep["classes"]] if spec != "e19_q8b_attn" else [
        [[r for r in c if r in FLASH] for c in rep["classes"] if set(c) & FLASH],
        [[r for r in c if r not in FLASH] for c in rep["classes"] if set(c) - FLASH]]
    for gi, cls in enumerate(groups):
        major = max(cls, key=len); ref = major[0]
        dref = D.load(os.path.join(sd, ref))
        for c in cls:
            if c is major:
                continue
            for o in c:
                pair = rep["pairs"].get(f"{ref}|{o}") or rep["pairs"].get(f"{o}|{ref}")
                site = pair.get("earliest_site") if pair else None
                prod = site[0] - 1 if site and site[1] == "in" else (site[0] if site else None)
                dd = D.diff(dref, D.load(os.path.join(sd, o)))
                subs = sorted({x["subgraph"] for x in dd if x.get("subgraph") is not None})
                ok = bool(subs) and subs[0] == prod
                res[f"{spec}#{gi}/{o}"] = {"first_site": site, "producer": prod, "decision_subgraphs": subs,
                                           "kernels": sorted({x["kernel"] for x in dd}), "attributed": ok}
n = len(res); k = sum(v["attributed"] for v in res.values())
print(json.dumps(res, indent=1)); print("attributed", k, "of", n)
json.dump(res, open(os.path.join(R, "attribution_e19.json"), "w"), indent=1)
