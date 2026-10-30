"""E22b analysis (pre-registered in results_prereg/E22b_PREREGISTRATION.md).
Per process: per-layer decision vector (decisions_*.json, per-object keys); flips = number of decisions
that differ from the most common vector over all warm processes. Primary: fraction of processes whose
vector differs from the most common one, load vs noload, one-sided Fisher exact test.
Usage: python analyze_e22b.py RUNS_ROOT [SPEC_SUBDIR]   (default e22b/e22b_q4b_load_interleaved;
       the original E22 is e22/e22_load)"""
import collections, json, math, os, sys
from lockstep import decisions as D

R = os.path.join(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "e22b/e22b_q4b_load_interleaved")


def fisher_one_sided(a, b, c, d):
    """P(X >= a) for the 2x2 table [[a, b], [c, d]] (rows: load, noload; cols: differs, same)."""
    n1, n2, k = a + b, c + d, a + c
    tot = math.comb(n1 + n2, k)
    return sum(math.comb(n1, x) * math.comb(n2, k - x) for x in range(a, min(n1, k) + 1)) / tot


def mann_whitney_one_sided(x, y):
    """P-value for x > y via normal approximation with tie correction (small samples: indicative)."""
    allv = sorted((v, g) for g, vs in ((0, x), (1, y)) for v in vs)
    ranks, i = {}, 0
    vals = [v for v, _ in allv]
    r = [0.0] * len(vals)
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1] == vals[i]:
            j += 1
        for t in range(i, j + 1):
            r[t] = (i + j) / 2 + 1
        i = j + 1
    rx = sum(rr for rr, (_, g) in zip(r, allv) if g == 0)
    n1, n2 = len(x), len(y)
    u = rx - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    ties = collections.Counter(vals).values()
    sd = math.sqrt(n1 * n2 / 12 * ((n1 + n2 + 1) - sum(t ** 3 - t for t in ties) / ((n1 + n2) * (n1 + n2 - 1))))
    return u, (0.5 * math.erfc((u - mu) / sd / math.sqrt(2)) if sd else 1.0)


runs = [r for r in sorted(os.listdir(R)) if (r.startswith("load") or r.startswith("noload"))
        and os.path.exists(os.path.join(R, r, "outputs.json"))]
vec = {}
for r in runs:
    d = D.load(os.path.join(R, r))
    vec[r] = tuple(sorted((k, json.dumps(v.get("chosen"), sort_keys=True)) for k, v in d.items()
                          if not k.startswith("__runtime__")))
mode, _ = collections.Counter(vec.values()).most_common(1)[0]
md = dict(mode)
flips = {r: sum(1 for k, c in vec[r] if md.get(k) != c) for r in runs}
grp = {"load": [r for r in runs if r.startswith("load")], "noload": [r for r in runs if r.startswith("noload")]}
diff = {g: sum(1 for r in rs if vec[r] != mode) for g, rs in grp.items()}
a, b = diff["load"], len(grp["load"]) - diff["load"]
c, d_ = diff["noload"], len(grp["noload"]) - diff["noload"]
p = fisher_one_sided(a, b, c, d_)
u, pmw = mann_whitney_one_sided([flips[r] for r in grp["load"]], [flips[r] for r in grp["noload"]])
rep = json.load(open(os.path.join(R, "report.json")))
cid = {r: i for i, cl in enumerate(rep["classes"]) for r in cl}
out = {"processes": {g: len(rs) for g, rs in grp.items()}, "differs_from_mode": diff,
       "fisher_one_sided_p": p, "flips": {g: [flips[r] for r in rs] for g, rs in grp.items()},
       "mann_whitney_u": u, "mann_whitney_p_one_sided": pmw,
       "numerical_classes": {g: len({cid[r] for r in rs}) for g, rs in grp.items()},
       "vector_vs_class_consistent": all((vec[x] == vec[y]) == (cid[x] == cid[y]) for x in runs for y in runs)}
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(R, "load_report.json"), "w"), indent=1)
