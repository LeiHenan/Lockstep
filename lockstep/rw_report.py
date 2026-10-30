"""RW report (results_prereg/RW_RACE_WINDOW_PREREGISTRATION.md): reads around a decision made after the handshake.

Per deployment: when the producer changed its decision (test_retuned, first republish), every read the consumer
checked against the producer's transfer tag (xfer_check: verdict, time), when the consumer invalidated the path,
and the disaggregated outputs of every round against the decode instance's own generation (local_b).
  python -m lockstep.rw_report ROOT [--out rw_report.json]
"""
import argparse
import glob
import json
import os


def _events(path):
    try:
        return [json.loads(l) for l in open(path)]
    except FileNotFoundError:
        return []


def _pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def deployment(d):
    ea = _events(os.path.join(d, "inst_a", "capture", "live_guard.jsonl"))
    eb = _events(os.path.join(d, "inst_b", "capture", "live_guard.jsonl"))
    mode = next((e.get("mode") for e in eb if e["event"] == "xfer_tag_installed"), None)
    ret = [e for e in ea if e["event"] == "test_retuned"]
    t_change = ret[0]["t"] if ret else None
    rep = [e for e in ea if e["event"] == "republished" and t_change and e["t"] >= t_change]
    checks = [e for e in eb if e["event"] == "xfer_check"]
    hs = [e for e in eb if e["event"] == "check"]
    inval = [e for e in eb if e["event"] == "invalidated" and t_change and e["t"] >= t_change]
    bad = [c for c in checks if c["verdict"] != "ok"]
    before = [c for c in bad if t_change is None or c["t"] < t_change]
    after = [c for c in bad if t_change is not None and c["t"] >= t_change]
    issued_bad = after if mode == "observe" else []  # enforce never reads on a non-ok verdict
    reqs = json.load(open(os.path.join(d, "requests.json")))
    rounds = json.load(open(os.path.join(d, "outputs_rounds.json")))
    lb = json.load(open(os.path.join(d, "local_b.json")))
    la = json.load(open(os.path.join(d, "local_a.json")))
    exact = sum(o == lb[i] for r in rounds for i, o in enumerate(r))
    late = [q for q in reqs if t_change and q["t_prefill_done"] and q["t_prefill_done"] >= t_change]
    late_mis = sum(rounds[q["round"]][q["i"]] != lb[q["i"]] for q in late)
    lat = [q["latency_s"] for q in reqs]
    meta = json.load(open(os.path.join(d, "meta.json")))
    verdicts = {}
    for c in checks:
        verdicts[c["verdict"]] = verdicts.get(c["verdict"], 0) + 1
    return {
        "mode": mode,
        "retuned": sum(isinstance(c, list) for c in (ret[0].get("changed") or [])) if ret else 0,
        "retune_errors": sum(isinstance(c, dict) for c in (ret[0].get("changed") or [])) if ret else 0,
        "t_change_after_first_request_s": (t_change - min(q["t_send"] for q in reqs)) if t_change else None,
        "first_republish_after_change_s": (rep[0]["t"] - t_change) if rep else None,
        "invalidated_after_change_s": (inval[0]["t"] - t_change) if inval else None,
        "handshakes": {r: sum(h.get("result") == r for h in hs) for r in {h.get("result") for h in hs}},
        "reads_checked": len(checks), "verdicts": verdicts,
        "non_ok_before_change": len(before), "non_ok_after_change": len(after),
        "stale_reads_issued": len(issued_bad),
        "last_stale_read_after_change_s": (issued_bad[-1]["t"] - t_change) if issued_bad else None,
        "producer_local_differs": sum(x != y for x, y in zip(la, lb)),
        "outputs": len(rounds) * len(lb), "outputs_exact": exact,
        "requests_after_change": len(late), "after_change_not_exact": late_mis,
        "wall_s": meta.get("wall_s"), "latency_p50_s": _pct(lat, 0.5), "latency_p99_s": _pct(lat, 0.99),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out")
    a = ap.parse_args()
    out = {}
    for d in sorted(glob.glob(os.path.join(a.root, "rw_*"))):
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "local_b.json")):
            out[os.path.basename(d)] = deployment(d)
    js = json.dumps(out, indent=1)
    print(js)
    if a.out:
        open(a.out, "w").write(js)


if __name__ == "__main__":
    main()
