"""Schedule-agnostic alignment of boundary-capture records and first-divergence search.

Key kappa = (prompt_key, position, subgraph_index, io, slot). Each (prompt, position) is
computed exactly once per run regardless of chunking/batching, so runs with different
schedules can be joined on kappa. Positions served from a prefix cache are absent in a
run and are reported as coverage gaps rather than mismatches.

Implementation: kappa is packed into one uint64
    pkey(32) | pos(16) | sub(8) | io(1) | slot(6)   (63 bits)
so joins are sorted-array intersections (numpy), fast enough for all-pairs analysis.

python -m lockstep.align RUN_A RUN_B [--rank 0] [--json out.json]
"""
import argparse
import glob
import json
import os
import pickle
from collections import defaultdict

import numpy as np

_IO = {"in": 0, "out": 1}


def load_records(run_dir, rank=0):
    recs = []
    for f in sorted(glob.glob(os.path.join(run_dir, "capture", f"rank{rank}_pid*.pkl"))):
        with open(f, "rb") as fh:
            while True:
                try:
                    recs.append(pickle.load(fh))
                except EOFError:
                    break
    return recs


def prompt_keys(run_dir):
    """Keys of the user prompts of a run (excludes engine warmup / dummy requests)."""
    f = os.path.join(run_dir, "outputs.json")
    if not os.path.exists(f):
        return None
    ks = {o.get("pkey") for o in json.load(open(f))}
    return None if None in ks else ks


def pack(pkey, pos, sub, io, slot):
    return ((np.asarray(pkey, np.uint64) & np.uint64(0xFFFFFFFF)) << np.uint64(31)) | \
           ((np.asarray(pos, np.uint64) & np.uint64(0xFFFF)) << np.uint64(15)) | \
           ((np.asarray(sub, np.uint64) & np.uint64(0xFF)) << np.uint64(7)) | \
           ((np.asarray(io, np.uint64) & np.uint64(1)) << np.uint64(6)) | \
           (np.asarray(slot, np.uint64) & np.uint64(0x3F))


def unpack(k):
    k = int(k)
    return ((k >> 31) & 0xFFFFFFFF, (k >> 15) & 0xFFFF, (k >> 7) & 0xFF, "out" if (k >> 6) & 1 else "in", k & 0x3F)


class Table:
    """Sorted arrays: keys (uint64), vals (int64 fingerprints), widths (int64)."""

    def __init__(self, keys, vals, widths, ranges):
        order = np.argsort(keys, kind="stable")
        keys, vals, widths = keys[order], vals[order], widths[order]
        # a kappa recomputed in a later step (e.g. after preemption) keeps the last value
        last = np.r_[keys[1:] != keys[:-1], True] if len(keys) else np.zeros(0, bool)
        self.keys, self.vals, self.widths = keys[last], vals[last], widths[last]
        self.ranges = ranges

    def __len__(self):
        return len(self.keys)


def kappa_table(recs, keep=None):
    ks, vs, ws = [], [], []
    ranges = defaultdict(set)
    keep_arr = np.array(sorted(keep), dtype=np.int64) if keep is not None else None
    for r in recs:
        if not r["layout"]:
            continue
        if "rows_pos" in r:  # exact per-row labels (vLLM V2 plugin >= it4)
            pk, pos = np.asarray(r["rows_pk"], np.int64), np.asarray(r["rows_pos"], np.int64)
        else:
            pk = np.concatenate([np.full(n, p, np.int64) for p, s, n in r["layout"]])
            pos = np.concatenate([np.arange(s, s + n, dtype=np.int64) for p, s, n in r["layout"]])
        m = np.isin(pk, keep_arr) if keep_arr is not None else np.ones(len(pk), bool)
        for c in r["calls"]:
            sub, rng, io, slot, h = c[:5]
            w = c[5] if len(c) > 5 else -1
            ranges[sub].add(str(rng))
            n = min(len(h), len(pk))
            mm = m[:n]
            ks.append(pack(pk[:n][mm], pos[:n][mm], sub, _IO[io], slot))
            vs.append(np.asarray(h[:n], np.int64)[mm])
            ws.append(np.full(int(mm.sum()), w, np.int64))
    if not ks:
        return Table(np.zeros(0, np.uint64), np.zeros(0, np.int64), np.zeros(0, np.int64), ranges), ranges
    return Table(np.concatenate(ks), np.concatenate(vs), np.concatenate(ws), ranges), ranges


def consumed_only(t):
    """Keep only values that are consumed: subgraph inputs, plus outputs of the last subgraph.
    Intermediate subgraph outputs can include scratch buffers (e.g. torch.empty outputs handed to
    the attention op before it writes them), whose contents are allocator garbage."""
    if not len(t):
        return t
    sub = (t.keys >> np.uint64(7)) & np.uint64(0xFF)
    io = (t.keys >> np.uint64(6)) & np.uint64(1)
    m = (io == 0) | (sub == sub.max())
    out = Table.__new__(Table)
    out.keys, out.vals, out.widths, out.ranges = t.keys[m], t.vals[m], t.widths[m], t.ranges
    return out


def compare(ta, tb, consumed=True):
    if consumed:
        ta, tb = consumed_only(ta), consumed_only(tb)
    common, ia, ib = np.intersect1d(ta.keys, tb.keys, assume_unique=True, return_indices=True)
    same_w = ta.widths[ia] == tb.widths[ib]
    common, ia, ib = common[same_w], ia[same_w], ib[same_w]
    bad = ta.vals[ia] != tb.vals[ib]
    # alignment self-check: 1-element rows entering subgraph 0 are input token ids / positions,
    # which are identical for the same (prompt, position) unless the labels are wrong
    sub = (common >> np.uint64(7)) & np.uint64(0xFF)
    io = (common >> np.uint64(6)) & np.uint64(1)
    selfchk = (sub == 0) & (io == 0) & (ta.widths[ia] == 1)
    selfcheck_bad = int((bad & selfchk).sum())
    selfcheck_checked = int(selfchk.sum())
    if selfcheck_bad:
        # drop every kappa of a (prompt, position) whose labels failed the self-check
        pp_mask = np.uint64(~np.uint64(0x7FFF))
        bad_pp = np.unique(common[bad & selfchk] & pp_mask)
        keep = ~np.isin(common & pp_mask, bad_pp)
        common, ia, ib, bad = common[keep], ia[keep], ib[keep], bad[keep]
    diff_keys = common[bad]
    prompts = np.unique(common >> np.uint64(31))
    first_by_prompt = {}
    if len(diff_keys):
        # dataflow order within a prompt: (sub, io, slot) = low 15 bits; position ignored
        p = diff_keys >> np.uint64(31)
        site = diff_keys & np.uint64(0x7FFF)
        order = np.lexsort((site, p))
        p, site = p[order], site[order]
        firsts = np.r_[True, p[1:] != p[:-1]]
        for pp, ss in zip(p[firsts], site[firsts]):
            ss = int(ss)
            first_by_prompt[int(pp)] = ((ss >> 7) & 0xFF, "out" if (ss >> 6) & 1 else "in", ss & 0x3F)
    hist = defaultdict(int)
    for s in first_by_prompt.values():
        hist[s] += 1
    ordered = sorted(hist.items(), key=lambda x: (x[0][0], 0 if x[0][1] == "in" else 1, x[0][2]))
    return {
        "keys_a": len(ta), "keys_b": len(tb), "common": int(len(common)),
        "only_a": int(len(ta) - len(common)), "only_b": int(len(tb) - len(common)),
        "mismatched_keys": int(bad.sum()), "prompts": int(len(prompts)),
        "prompts_diverged": len(first_by_prompt),
        "align_selfcheck_checked": selfcheck_checked, "align_selfcheck_bad": selfcheck_bad,
        "earliest_site": ordered[0][0] if ordered else None,
        "first_site_histogram": ordered[:10],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--json")
    a = ap.parse_args()
    ka, kb = prompt_keys(a.run_a), prompt_keys(a.run_b)
    keep = (ka | kb) if ka is not None and kb is not None else None
    ta, ra = kappa_table(load_records(a.run_a, a.rank), keep)
    tb, rb = kappa_table(load_records(a.run_b, a.rank), keep)
    res = compare(ta, tb)
    if res["earliest_site"]:
        s = res["earliest_site"][0]
        res["ranges_a"] = sorted(ra.get(s, []))
        res["ranges_b"] = sorted(rb.get(s, []))
    print(json.dumps(res, indent=1, default=str))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
