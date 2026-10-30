"""Automatic root-cause attribution: first causal divergence site + hidden-decision diff.

python -m lockstep.attribute RUN_A RUN_B [--ignore-out-slots 3]

1. Finds the first (subgraph, io, slot) whose fingerprints differ on rows whose inputs are
   identical (rows downstream of an earlier token divergence are excluded).
2. Diffs the two runs' hidden-decision traces (lockstep.decisions).
3. Reports the decisions in subgraphs <= the divergent subgraph, closest first. A decision in the
   divergent subgraph itself is the prime suspect; if none exists, the cause lies outside the traced
   decisions (e.g., an extern library call), which is reported explicitly.
"""
import argparse
import json

import numpy as np

from lockstep import align, decisions


def first_divergence(run_a, run_b, ignore_out_slots=(3,)):
    keep = align.prompt_keys(run_a)
    ta, _ = align.kappa_table(align.load_records(run_a), keep)
    tb, _ = align.kappa_table(align.load_records(run_b), keep)
    c, ia, ib = np.intersect1d(ta.keys, tb.keys, assume_unique=True, return_indices=True)
    ok = ta.widths[ia] == tb.widths[ib]
    c, ia, ib = c[ok], ia[ok], ib[ok]
    sub = ((c >> np.uint64(7)) & np.uint64(0xFF)).astype(int)
    io = ((c >> np.uint64(6)) & np.uint64(1)).astype(int)
    slot = (c & np.uint64(0x3F)).astype(int)
    w = ta.widths[ia]
    bad = ta.vals[ia] != tb.vals[ib]
    pp = c & np.uint64(~np.uint64(0x7FFF))
    tainted = np.unique(pp[(sub == 0) & (io == 0) & (w == 1) & bad])
    m = ~np.isin(pp, tainted) & ~((io == 1) & np.isin(slot, list(ignore_out_slots)))
    sub, io, slot, bad = sub[m], io[m], slot[m], bad[m]
    if not bad.any():
        return None
    order = np.lexsort((slot[bad], io[bad], sub[bad]))
    i = order[0]
    return {"subgraph": int(sub[bad][i]), "io": "out" if io[bad][i] else "in", "slot": int(slot[bad][i]),
            "rows_differing": int(((sub == sub[bad][i]) & (io == io[bad][i]) & (slot == slot[bad][i]) & bad).sum())}


def attribute(run_a, run_b):
    site = first_divergence(run_a, run_b)
    da, db = decisions.load(run_a), decisions.load(run_b)
    ddiff = decisions.diff(da, db)
    res = {"first_divergence": site, "decisions_traced": (len(da), len(db)), "differing_decisions": len(ddiff)}
    if site is None:
        res["verdict"] = "no numerical difference"
        return res
    # a divergence first visible at the input of subgraph k was produced in subgraph k-1 (or the op between)
    producer = site["subgraph"] if site["io"] == "out" else site["subgraph"] - 1

    def sg(d):
        return d.get("subgraph") if d.get("subgraph") is not None else int(str(d["kernel"]).split("/")[0][3:] or -1)
    cands = [d for d in ddiff if "only_in" not in d and sg(d) is not None and sg(d) <= producer]
    cands.sort(key=lambda d: producer - sg(d))
    structural = [d for d in ddiff if "only_in" in d]
    res["suspects"] = cands[:5]
    res["structural_differences"] = structural[:5]
    if cands and sg(cands[0]) == producer:
        res["verdict"] = (f"root cause: run-time decision of {cands[0]['kernel']} in subgraph {producer} "
                          f"({cands[0]['a']} vs {cands[0]['b']})")
    elif structural:
        res["verdict"] = "root cause candidate: different kernel/fusion structure (see structural_differences)"
    else:
        res["verdict"] = (f"no traced decision differs in subgraph {producer}: cause is outside the traced "
                          f"autotuner decisions (e.g., extern library call, attention, or compile-time choice)")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    a = ap.parse_args()
    print(json.dumps(attribute(a.run_a, a.run_b), indent=1, default=str))


if __name__ == "__main__":
    main()
