"""E5 (ablation): schedule-agnostic kappa alignment vs. call-order alignment.

Call-order alignment is the assumption made by training-side tools: the i-th invocation of
subgraph s in run A corresponds to the i-th invocation in run B, row r to row r. Under
inference schedules (different batching, chunking, prefix hits) this pairs unrelated tokens.

python -m lockstep.e5_alignment ROOT RUN_A RUN_B [RUN_A2 RUN_B2 ...]
For each pair: mismatch counts and first-divergence site under both alignments.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

from lockstep import align


def call_order_table(recs, consumed=True):
    """key = (subgraph, io, slot, invocation index of that subgraph, row) -> fingerprint."""
    tab = {}
    inv = defaultdict(int)
    maxsub = max((c[0] for r in recs for c in r["calls"]), default=0)
    for r in recs:
        seen = set()
        for c in r["calls"]:
            sub, rng, io, slot, h = c[:5]
            if consumed and not (io == "in" or sub == maxsub):
                continue
            k_inv = (sub, io, slot)
            i = inv[k_inv]
            seen.add(k_inv)
            for row, v in enumerate(np.asarray(h).tolist()):
                tab[(sub, io, slot, i, row)] = v
        for k in seen:
            inv[k] += 1
    return tab


def compare_call_order(ta, tb):
    common = ta.keys() & tb.keys()
    bad = [k for k in common if ta[k] != tb[k]]
    first = min(bad, key=lambda k: (k[0], 0 if k[1] == "in" else 1, k[2])) if bad else None
    return {"common": len(common), "mismatched": len(bad), "first_site": first[:3] if first else None,
            "only_a": len(ta.keys() - tb.keys()), "only_b": len(tb.keys() - ta.keys())}


def main():
    root = sys.argv[1]
    pairs = list(zip(sys.argv[2::2], sys.argv[3::2]))
    out = {}
    for a, b in pairs:
        ra, rb = align.load_records(os.path.join(root, a)), align.load_records(os.path.join(root, b))
        keep = align.prompt_keys(os.path.join(root, a))
        ka = align.compare(align.kappa_table(ra, keep)[0], align.kappa_table(rb, keep)[0])
        co = compare_call_order(call_order_table(ra), call_order_table(rb))
        out[f"{a}|{b}"] = {"kappa": {k: ka[k] for k in ("common", "mismatched_keys", "prompts_diverged", "earliest_site")},
                           "call_order": co}
        print(f"{a:10s} vs {b:16s} kappa: mism={ka['mismatched_keys']:>7}/{ka['common']:<7} first={ka['earliest_site']}"
              f" | call-order: mism={co['mismatched']:>7}/{co['common']:<7} first={co['first_site']} unmatched={co['only_a']}+{co['only_b']}")
    json.dump(out, open(os.path.join(root, "e5_alignment.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
