"""Fine-grained view of the first divergent region, including intermediate subgraph outputs.

For two runs, list per (subgraph, io, slot) the number of differing rows around the first
divergence, so one can see whether e.g. q/k/v (outputs of subgraph k) are identical while the
attention output (input of subgraph k+1) differs -> the attention op is the source.

python -m lockstep.locate_layer RUN_A RUN_B [--window 3] [--ignore-out-slots 3]
"""
import argparse
from collections import defaultdict

import numpy as np

from lockstep import align


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--ignore-out-slots", default="3", help="comma list of scratch output slots to ignore")
    a = ap.parse_args()
    ign = {int(x) for x in a.ignore_out_slots.split(",") if x}
    keep = align.prompt_keys(a.run_a)
    ta, _ = align.kappa_table(align.load_records(a.run_a), keep)
    tb, _ = align.kappa_table(align.load_records(a.run_b), keep)
    c, ia, ib = np.intersect1d(ta.keys, tb.keys, assume_unique=True, return_indices=True)
    same_w = ta.widths[ia] == tb.widths[ib]
    c, ia, ib = c[same_w], ia[same_w], ib[same_w]
    sub = ((c >> np.uint64(7)) & np.uint64(0xFF)).astype(int)
    io = ((c >> np.uint64(6)) & np.uint64(1)).astype(int)
    slot = (c & np.uint64(0x3F)).astype(int)
    bad = ta.vals[ia] != tb.vals[ib]
    # rows whose input token/position already differ are downstream of an earlier divergence
    # (a different generated token is fed back); exclude them so the reported site is causal
    pp = c & np.uint64(~np.uint64(0x7FFF))
    w = ta.widths[ia]
    tainted = np.unique(pp[(sub == 0) & (io == 0) & (w == 1) & bad])
    clean = ~np.isin(pp, tainted)
    print(f"rows excluded as downstream of earlier divergence: {len(tainted)}")
    c, sub, io, slot, bad = c[clean], sub[clean], io[clean], slot[clean], bad[clean]
    mask = ~((io == 1) & np.isin(slot, list(ign)))
    stats = defaultdict(lambda: [0, 0])
    for s, i, sl, b in zip(sub[mask], io[mask], slot[mask], bad[mask]):
        stats[(s, "out" if i else "in", sl)][0] += int(b)
        stats[(s, "out" if i else "in", sl)][1] += 1
    order = sorted(stats, key=lambda k: (k[0], 0 if k[1] == "in" else 1, k[2]))
    first = next((k for k in order if stats[k][0]), None)
    if first is None:
        print("no differences (excluding ignored scratch slots)")
        return
    lo, hi = first[0] - a.window, first[0] + 1
    print(f"first differing site (incl. intermediate outputs): {first}")
    for k in order:
        if lo <= k[0] <= hi:
            d, n = stats[k]
            print(f"  sub={k[0]:3d} {k[1]:3s} slot={k[2]}  differing rows {d:>7}/{n:<7}{'  <-- first' if k == first else ''}")


if __name__ == "__main__":
    main()
