"""Baseline: OpGuard-style alignment (whole-tensor fingerprints + schedule-tolerant stream mapping).

OpGuard [OSDI'26] records one fingerprint per tensor at semantic operator boundaries and aligns the
two executions' fingerprint streams with a schedule-tolerant mapper that tolerates inserted or
reordered events, then reports the first boundary of the longest bitwise-identical prefix. We
re-implement that comparison on Lockstep's captures, so both methods see the same observations:
  * event      = one observation (subgraph, io, slot) in one engine step, in execution order;
  * fingerprint = digest of the whole tensor, i.e., of all its rows in layout order;
  * mapper     = walk both streams; an event of A matches the next event of B with the same label and
                 fingerprint within a window of W events (events skipped inside the window count as
                 tolerated insertions); the first event of A without a match is the reported divergence.
Only consumed values are compared, as in Lockstep (subgraph inputs and the final subgraph's outputs).

python -m lockstep.opguard_like e4 RUNS_DIR TRUTH_JSON      # planted faults: exact-site accuracy
python -m lockstep.opguard_like pairs ROOT A B [A2 B2 ...]  # equivalent configurations: false reports
"""
import hashlib
import json
import os
import sys

import numpy as np

from lockstep import align


def stream(recs, consumed=True):
    maxsub = max((c[0] for r in recs for c in r["calls"]), default=0)
    ev = []
    for r in recs:
        for c in r["calls"]:
            sub, _rng, io, slot, h = c[:5]
            if consumed and not (io == "in" or sub == maxsub):
                continue
            if io == "out" and slot == 3:  # attention scratch buffer, not consumed (same rule as Lockstep)
                continue
            fp = hashlib.sha1(np.ascontiguousarray(np.asarray(h, dtype=np.uint64)).tobytes()).hexdigest()[:16]
            ev.append(((sub, io, slot), fp))
    return ev


def first_divergence(a, b, window=64):
    """Returns (index in A, label) of the first unmatched event, or None if A is fully matched."""
    j = 0
    for i, (lab, fp) in enumerate(a):
        hit = None
        for k in range(j, min(len(b), j + window)):
            if b[k][0] == lab and b[k][1] == fp:
                hit = k
                break
        if hit is None:
            return i, lab
        j = hit + 1
    return None


def run_e4(runs, truth_f):
    truth = json.load(open(truth_f))
    base = stream(align.load_records(os.path.join(runs, "base")))
    ok, rows = 0, []
    for name, t in sorted(truth.items()):
        d = os.path.join(runs, name)
        if not os.path.exists(os.path.join(d, "outputs.json")):
            continue
        r = first_divergence(base, stream(align.load_records(d)))
        found = None if r is None else (r[1][0], "out" if r[1][1] == "out" else "in", r[1][2])
        want = (t["sub"], t["io"], t["slot"])
        ok += found == want
        rows.append({"run": name, "truth": want, "found": found, "event_index": None if r is None else r[0]})
        print(f"{name:22s} truth={want} opguard-like found={found}")
    ctrl = os.path.join(runs, "serial_noinj")
    if os.path.exists(os.path.join(ctrl, "outputs.json")):
        print("control (no injection):", first_divergence(base, stream(align.load_records(ctrl))))
    print(f"opguard-like exact-site accuracy {ok}/{len(rows)}")
    json.dump({"rows": rows, "exact": ok, "n": len(rows)}, open(os.path.join(runs, "e4_baseline_opguard_like.json"), "w"),
              indent=1, default=str)


def run_pairs(root, pairs):
    out = {}
    for a, b in pairs:
        sa = stream(align.load_records(os.path.join(root, a)))
        sb = stream(align.load_records(os.path.join(root, b)))
        r = first_divergence(sa, sb)
        out[f"{a}|{b}"] = {"events_a": len(sa), "events_b": len(sb),
                           "first_divergence": None if r is None else {"index": r[0], "site": r[1]}}
        print(f"{a:10s} vs {b:14s} events {len(sa)}/{len(sb)} opguard-like first divergence: {out[f'{a}|{b}']['first_divergence']}")
    json.dump(out, open(os.path.join(root, "e5_opguard_like.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    if sys.argv[1] == "e4":
        run_e4(sys.argv[2], sys.argv[3])
    else:
        run_pairs(sys.argv[2], list(zip(sys.argv[3::2], sys.argv[4::2])))
