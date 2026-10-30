"""Timing margins of autotuning decisions that two processes of one deployment made differently (review question:
how close were the timings when the two instances picked different block sizes?). Offline, from decision traces.

python -m lockstep.timing_margin --root RUNS [--same-host "*"] [--out timing_margin.json]

Inputs: every deployment (inst_a and inst_b of one pd_harness directory, or two
operator-supplied cross-host capture directories) on vLLM 0.32.0 with the batch_invariant Inductor flag
off on both sides (cfg.json engine_kwargs.ibi false, stack vllm 0.32.0). Smoke and failed attempts are skipped.

Per process, a decision counts as benchmarked if lockstep.decisions recorded it from autotune_to_one_config with
at least two candidates and two timings, and it was not forced (planted-decision experiments). Decisions of the two
processes are matched as lockstep.decisions.diff does (same key: subgraph + kernel name + instance order; per TP
rank), and additionally require the same kernel source digest. A flip is a matched decision whose chosen
configuration differs.

Per benchmarked decision and side: the relative margin between the best and second-best timing, (t2 - t1) / t1.
For a flip in which both chosen configurations were timed on both sides: whether each side's own timings ranked
the two configurations oppositely, and the margin between those two configurations on each side.
Warm processes load their configurations from the cache and record no timings: their decisions enter only the
"one side benchmarked" section (flip or not, with the cold side's margin).
"""
import argparse
import glob
import json
import os
import re
import sys

from lockstep import decisions

EXP = os.path.abspath("runs")
SAME_HOST_RUNS = ["*"]
CROSS_HOST_RUNS = []
VLLM = "0.32.0"
BINS = [(0.0, 0.01, "0-1%"), (0.01, 0.02, "1-2%"), (0.02, 0.05, "2-5%"), (0.05, 0.10, "5-10%"),
        (0.10, 0.20, "10-20%"), (0.20, float("inf"), ">20%")]
# absolute best-to-second gap; the timer's resolution is about 1 us (most timings are multiples of 1.024 us)
US_BINS = [(0.0, 1e-9, "0"), (1e-9, 1.1, "(0,1.1]us"), (1.1, 2.1, "(1.1,2.1]us"), (2.1, 5.0, "(2.1,5]us"),
           (5.0, float("inf"), ">5us")]
SCENARIO = re.compile(r"(homo|hetero|big)")
SKIP = re.compile(r"smoke|failed")


# ---- inputs -----------------------------------------------------------------------------------------------------

def _cfg(inst):
    try:
        return json.load(open(os.path.join(inst, "cfg.json")))
    except Exception:
        return {}


def _flag_on(inst):
    return bool(_cfg(inst).get("engine_kwargs", {}).get("ibi"))


def _ranks(inst):
    fs = glob.glob(os.path.join(inst, "capture", "decisions_rank*_pid*.json"))
    return {int(re.search(r"decisions_rank(\d+)_pid", f).group(1)) for f in fs}


def _load_rank(inst, rank):
    """All decision files of one rank merged (rank 0 equals lockstep.decisions.load)."""
    if rank == 0:
        return decisions.load(inst)
    dec = {}
    for f in glob.glob(os.path.join(inst, "capture", f"decisions_rank{rank}_pid*.json")):
        dec.update(json.load(open(f)))
    return dec


def deployments():
    """(run label, deployment name, inst_a, inst_b) of every candidate deployment, before filtering."""
    out = []
    for pat in SAME_HOST_RUNS:
        for run in sorted(glob.glob(os.path.join(EXP, pat))):
            for a in sorted(glob.glob(os.path.join(run, "*", "inst_a"))):
                d = os.path.dirname(a)
                out.append((os.path.relpath(run, EXP).replace(os.sep, "/"), os.path.basename(d), a,
                            os.path.join(d, "inst_b")))
    for ha, hb in CROSS_HOST_RUNS:
        for a in sorted(glob.glob(os.path.join(EXP, ha, "*", "inst_a"))):
            name = os.path.basename(os.path.dirname(a))
            out.append((f"{ha}+{os.path.basename(hb)}", name, a, os.path.join(EXP, hb, name, "inst_b")))
    return out


def eligible(name, a, b):
    """None if the deployment enters, else the reason it does not."""
    if SKIP.search(name):
        return "smoke/failed"
    if not (os.path.isdir(a) and os.path.isdir(b)):
        return "missing inst_a or inst_b"
    va, vb = decisions.load_stack(a).get("vllm"), decisions.load_stack(b).get("vllm")
    if va != VLLM or vb != VLLM:
        return f"vllm {va}/{vb}"
    if _flag_on(a) or _flag_on(b):
        return "batch_invariant flag on"
    if not (_ranks(a) & _ranks(b)):
        return "no decision traces"
    return None


# ---- per-decision quantities ------------------------------------------------------------------------------------

def benchmarked(v):
    t = v.get("timings_ms") or {}
    return bool(v.get("benchmarked")) and (v.get("candidates") or 0) >= 2 and len(t) >= 2 and "forced" not in v


def margin(timings):
    """(t2 - t1) / t1 of the best and second-best timing."""
    t = sorted(timings.values())
    return (t[1] - t[0]) / t[0] if t[0] > 0 else None


def ranking(ta, tb, ca, cb):
    """For a flip (side a chose ca, side b chose cb): how each side's own timings order the two configurations.
    'opposite': each side measured its own choice strictly faster; 'tie_a'/'tie_b'/'tie_both': equal timings on
    that side (tie broken by candidate order); 'not_comparable': a configuration was not timed on some side
    (the candidate sets differ)."""
    if not all(c in t for c in (ca, cb) for t in (ta, tb)):
        return "not_comparable", None, None
    da, db = ta[cb] - ta[ca], tb[ca] - tb[cb]  # > 0: the side measured its own choice faster
    ma, mb = abs(da) / min(ta[ca], ta[cb]), abs(db) / min(tb[ca], tb[cb])
    if da > 0 and db > 0:
        kind = "opposite"
    elif da == 0 and db == 0:
        kind = "tie_both"
    elif da == 0 and db > 0:
        kind = "tie_a"
    elif db == 0 and da > 0:
        kind = "tie_b"
    else:
        kind = "inconsistent"  # a side chose a configuration its own timings rank slower
    return kind, ma, mb


def differs_in(ca, cb):
    """Fields in which two launch configurations differ, e.g. 'R0_BLOCK' or 'XBLOCK+num_warps'."""
    ka, kb = ca.get("kwargs") or {}, cb.get("kwargs") or {}
    d = sorted(k for k in set(ka) | set(kb) if ka.get(k) != kb.get(k))
    return "+".join(d + [k for k in ("num_warps", "num_stages") if ca.get(k) != cb.get(k)])


def gap_us(timings):
    t = sorted(timings.values())
    return 1000.0 * (t[1] - t[0])


def _chosen_is_min(v):
    t = v["timings_ms"]
    c = str(v.get("chosen"))
    return c in t and t[c] == min(t.values())


# ---- one deployment ---------------------------------------------------------------------------------------------

def compare(run, name, a, b):
    row = {"run": run, "name": name, "gpu": decisions.load_stack(a).get("gpu"), "ranks": sorted(_ranks(a) & _ranks(b)),
           "n_bench_a": 0, "n_bench_b": 0, "matched_both": 0, "flips_both": 0, "matched_one": 0, "flips_one": 0,
           "src_mismatch": 0, "unmatched": 0}
    items = []
    for r in row["ranks"]:
        A, B = _load_rank(a, r), _load_rank(b, r)
        row["n_bench_a"] += sum(benchmarked(v) for v in A.values())
        row["n_bench_b"] += sum(benchmarked(v) for v in B.values())
        for k in sorted(set(A) | set(B)):
            if k.startswith("__runtime__"):
                continue
            x, y = A.get(k), B.get(k)
            bx, by = x is not None and benchmarked(x), y is not None and benchmarked(y)
            if not (bx or by):
                continue
            if x is None or y is None:
                row["unmatched"] += 1
                continue
            if x.get("kernel") != y.get("kernel") or x.get("src") is None or x.get("src") != y.get("src"):
                row["src_mismatch"] += 1
                continue
            flip = x.get("chosen") != y.get("chosen")
            it = {"run": run, "name": name, "rank": r, "key": k, "src": x.get("src"), "flip": flip,
                  "both": bx and by, "margin_a": margin(x["timings_ms"]) if bx else None,
                  "margin_b": margin(y["timings_ms"]) if by else None,
                  "gap_us_a": gap_us(x["timings_ms"]) if bx else None, "gap_us_b": gap_us(y["timings_ms"]) if by else None,
                  "n_cand_a": len(x["timings_ms"]) if bx else None, "n_cand_b": len(y["timings_ms"]) if by else None,
                  "chosen_is_min": [_chosen_is_min(v) for v, ok in ((x, bx), (y, by)) if ok]}
            if bx and by:
                it["same_candidates"] = set(x["timings_ms"]) == set(y["timings_ms"])
            if flip:
                it["chosen_a"], it["chosen_b"] = x.get("chosen"), y.get("chosen")
                it["differs_in"] = differs_in(x.get("chosen") or {}, y.get("chosen") or {})
                if bx and by:
                    # was the other side's choice among this side's timed candidates at all?
                    it["b_choice_timed_on_a"] = str(y["chosen"]) in x["timings_ms"]
                    it["a_choice_timed_on_b"] = str(x["chosen"]) in y["timings_ms"]
                    kind, ma, mb = ranking(x["timings_ms"], y["timings_ms"], str(x["chosen"]), str(y["chosen"]))
                    it.update({"ranking": kind, "pair_margin_a": ma, "pair_margin_b": mb})
            tag = "both" if bx and by else "one"
            row["matched_" + tag] += 1
            row["flips_" + tag] += flip
            items.append(it)
    return row, items


# ---- summaries --------------------------------------------------------------------------------------------------

def _pct(xs, q):
    """Linear-interpolation percentile of a sorted list."""
    if not xs:
        return None
    p = (len(xs) - 1) * q
    lo = int(p)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (p - lo)


def dist(ms, bins=BINS):
    xs = sorted(m for m in ms if m is not None)
    hist = {label: sum(lo <= m < hi for m in xs) for lo, hi, label in bins}
    return {"count": len(xs), "exact_ties": sum(m == 0 for m in xs), "median": _pct(xs, 0.5), "p10": _pct(xs, 0.1),
            "p90": _pct(xs, 0.9), "max": xs[-1] if xs else None, "histogram": hist}


def _side_margins(items):
    return [i["margin_a"] for i in items] + [i["margin_b"] for i in items]


def summarize_both(items):
    """Decisions benchmarked on both sides: margins of flips and non-flips, per side and pooled."""
    flips = [i for i in items if i["flip"]]
    stay = [i for i in items if not i["flip"]]
    same = [i for i in flips if i["same_candidates"]]
    kinds = {}
    for i in flips:
        kinds[i["ranking"]] = kinds.get(i["ranking"], 0) + 1
    comp = [i for i in flips if i["ranking"] != "not_comparable"]
    fields = {}
    for i in flips:
        f = f"{i['differs_in']} ({'same' if i['same_candidates'] else 'different'} candidate set)"
        fields[f] = fields.get(f, 0) + 1
    return {
        "matched": len(items), "flips": len(flips), "non_flips": len(stay),
        "flips_same_candidate_set": len(same), "flips_different_candidate_set": len(flips) - len(same),
        "non_flips_same_candidate_set": sum(i["same_candidates"] for i in stay),
        "distinct_flipped_kernels": len({(i["run"], i["src"]) for i in flips}),
        "flip_margin_pooled": dist(_side_margins(flips)),
        "flip_margin_side_a": dist([i["margin_a"] for i in flips]),
        "flip_margin_side_b": dist([i["margin_b"] for i in flips]),
        "flip_margin_larger_side": dist([max(i["margin_a"], i["margin_b"]) for i in flips]),
        "flip_margin_pooled_same_candidates": dist(_side_margins(same)),
        "flip_margin_pooled_different_candidates": dist(_side_margins([i for i in flips if not i["same_candidates"]])),
        "nonflip_margin_pooled": dist(_side_margins(stay)),
        "nonflip_margin_pooled_same_candidates": dist(_side_margins([i for i in stay if i["same_candidates"]])),
        "flip_gap_us_pooled": dist([g for i in flips for g in (i["gap_us_a"], i["gap_us_b"])], US_BINS),
        "flip_gap_us_pooled_same_candidates": dist([g for i in same for g in (i["gap_us_a"], i["gap_us_b"])], US_BINS),
        "nonflip_gap_us_pooled": dist([g for i in stay for g in (i["gap_us_a"], i["gap_us_b"])], US_BINS),
        "flip_differs_in": dict(sorted(fields.items(), key=lambda kv: -kv[1])),
        "flips_other_choice_never_timed": sum(not (i["a_choice_timed_on_b"] or i["b_choice_timed_on_a"]) for i in flips),
        "flip_ranking": kinds,
        "flip_pair_margin_pooled": dist([i["pair_margin_a"] for i in comp] + [i["pair_margin_b"] for i in comp]),
        "flip_pair_margin_opposite_only": dist([m for i in comp if i["ranking"] == "opposite"
                                                for m in (i["pair_margin_a"], i["pair_margin_b"])]),
        "chosen_not_fastest": sum(not all(i["chosen_is_min"]) for i in items),
    }


def summarize_one(items):
    """Decisions benchmarked on one side only (the other process loaded its configuration from the cache)."""
    flips = [i for i in items if i["flip"]]
    stay = [i for i in items if not i["flip"]]
    timed = lambda xs: [i["margin_a"] if i["margin_a"] is not None else i["margin_b"] for i in xs]
    return {"matched": len(items), "flips": len(flips), "non_flips": len(stay),
            "flip_margin_timed_side": dist(timed(flips)), "nonflip_margin_timed_side": dist(timed(stay))}


def _scenario(name):
    """homo (same engine arguments), hetero (different batching arguments), big, or other (TP14, A8X names)."""
    m = SCENARIO.search(name)
    return m.group(1) if m else "other"


def _brief(s):
    out = {k: s[k] for k in ("matched", "flips", "flips_same_candidate_set", "flip_ranking")}
    out.update({"flip_margin_pooled_median": s["flip_margin_pooled"]["median"],
                "flip_margin_pooled_same_candidates_median": s["flip_margin_pooled_same_candidates"]["median"],
                "nonflip_margin_pooled_median": s["nonflip_margin_pooled"]["median"]})
    return out


def by_run(rows, items):
    out = {}
    for r in rows:
        o = out.setdefault(r["run"], {"deployments": 0, "gpu": r["gpu"], "matched_both": 0, "flips_both": 0,
                                      "matched_one": 0, "flips_one": 0})
        o["deployments"] += 1
        for f in ("matched_both", "flips_both", "matched_one", "flips_one"):
            o[f] += r[f]
    for run, o in out.items():
        fl = [i for i in items if i["run"] == run and i["both"] and i["flip"]]
        o["flips_both_same_candidates"] = sum(i["same_candidates"] for i in fl)
        o["flip_margin_pooled_median"] = dist(_side_margins(fl))["median"]
    return out


def main(out_path):
    rows, items, skipped = [], [], []
    for run, name, a, b in deployments():
        why = eligible(name, a, b)
        if why:
            skipped.append({"run": run, "name": name, "reason": why})
            continue
        row, its = compare(run, name, a, b)
        rows.append(row)
        items += its
    both = [i for i in items if i["both"]]
    one = [i for i in items if not i["both"]]
    rep = {
        "vllm": VLLM, "flag": "off",
        "runs_used": sorted({r["run"] for r in rows}),
        "deployments": len(rows),
        "deployments_with_both_benchmarked": sum(r["matched_both"] > 0 for r in rows),
        "processes_with_timings": sum((r["n_bench_a"] > 0) + (r["n_bench_b"] > 0) for r in rows),
        "benchmarked_decisions_total": sum(r["n_bench_a"] + r["n_bench_b"] for r in rows),
        "excluded_src_mismatch": sum(r["src_mismatch"] for r in rows),
        "excluded_unmatched": sum(r["unmatched"] for r in rows),
        "both_sides_benchmarked": summarize_both(both),
        "one_side_benchmarked": summarize_one(one),
        "by_gpu": {g: summarize_both([i for i in both if any(r["gpu"] == g and r["name"] == i["name"]
                                                             and r["run"] == i["run"] for r in rows)])
                   for g in sorted({r["gpu"] for r in rows})},
        "by_scenario": {sc: _brief(summarize_both([i for i in both if _scenario(i["name"]) == sc]))
                        for sc in sorted({_scenario(i["name"]) for i in both})},
        "by_run": by_run(rows, items),
        "per_deployment": rows,
        "flips": [i for i in items if i["flip"]],
        "skipped": skipped,
        "bins": [b[2] for b in BINS],
    }
    with open(out_path, "w") as f:
        json.dump(rep, f, indent=1, default=str)
    return rep


def _fmt(d):
    f = lambda x: "-" if x is None else f"{100 * x:.2f}%"
    return (f"n={d['count']} ties={d['exact_ties']} median={f(d['median'])} p10={f(d['p10'])} p90={f(d['p90'])} "
            f"max={f(d['max'])} hist={d['histogram']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="root of the captured experiment directories")
    ap.add_argument("--same-host", action="append", help="relative run glob; repeatable; default: *")
    ap.add_argument("--cross-host", action="append", nargs=2, metavar=("PRODUCER", "CONSUMER"),
                    help="relative producer/consumer capture directories; repeatable")
    ap.add_argument("--out", default="timing_margin.json")
    args = ap.parse_args()
    EXP = os.path.abspath(args.root)
    SAME_HOST_RUNS = args.same_host or ["*"]
    CROSS_HOST_RUNS = args.cross_host or []
    rep = main(ap.parse_args().out)
    s, o = rep["both_sides_benchmarked"], rep["one_side_benchmarked"]
    print(f"runs: {', '.join(rep['runs_used'])}")
    print(f"deployments: {rep['deployments']} ({rep['deployments_with_both_benchmarked']} with decisions timed on "
          f"both sides); benchmarked decisions: {rep['benchmarked_decisions_total']}")
    print(f"both sides timed: matched {s['matched']}, flips {s['flips']} (same candidate set {s['flips_same_candidate_set']}), "
          f"ranking {s['flip_ranking']}")
    for k in ("flip_margin_pooled", "flip_margin_pooled_same_candidates", "flip_margin_larger_side",
              "flip_pair_margin_pooled", "nonflip_margin_pooled", "nonflip_margin_pooled_same_candidates"):
        print(f"  {k}: {_fmt(s[k])}")
    for k in ("flip_gap_us_pooled", "flip_gap_us_pooled_same_candidates", "nonflip_gap_us_pooled"):
        d = s[k]
        print(f"  {k}: n={d['count']} median={d['median']:.3f}us p90={d['p90']:.3f}us hist={d['histogram']}")
    print(f"  flip_differs_in: {s['flip_differs_in']}")
    print(f"  flips where neither side timed the other's choice: {s['flips_other_choice_never_timed']}")
    for sc, b in rep["by_scenario"].items():
        print(f"  scenario {sc}: {b}")
    print(f"one side timed: matched {o['matched']}, flips {o['flips']}")
    print(f"  flip_margin_timed_side: {_fmt(o['flip_margin_timed_side'])}")
    print(f"  nonflip_margin_timed_side: {_fmt(o['nonflip_margin_timed_side'])}")
    print(f"written: {ap.parse_args().out}", file=sys.stderr)
