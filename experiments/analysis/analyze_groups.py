"""Per-configuration flips, attribution, and guard verdicts for a directory of explorer specs (same
procedure as E26; used for the H800 replication). Every spec directory is one identically configured
group. Usage: python analyze_groups.py GROUP_ROOT OUT_JSON"""
import itertools, json, math, os, re, sys, collections
from lockstep import decisions as D, guard as G


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n; d = 1 + z * z / n
    c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((c - h) / d, 3), round((c + h) / d, 3)]


root, out_path = sys.argv[1], sys.argv[2]
res, tot = {}, collections.Counter()
kernels = collections.Counter()
for spec in sorted(os.listdir(root)):
    sd = os.path.join(root, spec)
    if not os.path.exists(os.path.join(sd, "report.json")):
        continue
    rep = json.load(open(os.path.join(sd, "report.json")))
    cls = [[r for r in c if os.path.exists(os.path.join(sd, r, "outputs.json"))] for c in rep["classes"]]
    cls = [c for c in cls if c]
    cid = {r: i for i, c in enumerate(cls) for r in c}
    runs = list(cid)
    major = max(cls, key=len); ref = major[0]
    dref = D.load(os.path.join(sd, ref))
    outl = [o for c in cls if c is not major for o in c]
    attributed = 0
    for o in outl:
        pair = rep["pairs"].get(f"{ref}|{o}") or rep["pairs"].get(f"{o}|{ref}") or {}
        site = pair.get("earliest_site")
        dd = D.diff(dref, D.load(os.path.join(sd, o)))
        subs = sorted({x["subgraph"] for x in dd if x.get("subgraph") is not None})
        if site:  # fingerprints captured: earliest differing decision must sit in the producing subgraph
            prod = site[0] - 1 if site[1] == "in" else site[0]
            ok = bool(subs) and subs[0] == prod
        else:     # trace-only runs: attribution means some traced decision differs
            ok = bool(dd)
        attributed += ok
        for x in dd[:1]:
            kernels[re.sub(r"sub\w+/", "", x["kernel"]).split("#")[0]] += 1
    ids = {r: G.identity(os.path.join(sd, r))[0] for r in runs}
    pairs = list(itertools.combinations(runs, 2))
    div = [(a, b) for a, b in pairs if cid[a] != cid[b]]
    miss = sum(1 for a, b in div if ids[a] == ids[b])
    fblock = sum(1 for a, b in pairs if cid[a] == cid[b] and ids[a] != ids[b])
    res[spec] = {"processes": len(runs), "classes": len(cls), "largest": len(major), "outliers": len(outl),
                 "attributed": attributed, "divergent_pairs": len(div), "guard_missed_pairs": miss,
                 "guard_false_blocks": fblock, "decisions_per_process": len(dref),
                 "benchmarked_per_process": sum(1 for v in dref.values() if v.get("benchmarked"))}
    for k in ("outliers", "attributed", "divergent_pairs", "guard_missed_pairs", "guard_false_blocks"):
        tot[k] += res[spec][k]
    tot["outliers_missed_by_guard"] += sum(1 for o in outl if ids[o] == ids[ref])
summary = dict(tot)
summary["attribution_ci95"] = wilson(tot["attributed"], tot["outliers"])
summary["guard_separated"] = tot["outliers"] - tot["outliers_missed_by_guard"]
summary["guard_ci95"] = wilson(summary["guard_separated"], tot["outliers"])
summary["first_differing_kernel_families"] = dict(kernels)
json.dump({"groups": res, "summary": summary}, open(out_path, "w"), indent=1)
print(json.dumps({"groups": res, "summary": summary}, indent=1))
