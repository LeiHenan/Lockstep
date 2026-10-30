"""HB / HB2 analysis (results_prereg/HB_SK_PREREGISTRATION.md and addendum 1).

python -m lockstep.hb_report RUN_DIR [RUN_DIR ...] > hb_report.json

Per deployment: the strict policy's verdicts at the decode side (first check, and all checks), the producer's
and consumer's own generations compared (local_a vs local_b; first token), the disaggregated outputs against the
decode instance's own (local_b), and versions 6 and 6p recomputed offline from the same captures.
"""
import glob
import json
import os
import sys

from lockstep import ablate_identity


def _checks(d):
    f = os.path.join(d, "inst_b", "capture", "live_guard.jsonl")
    return [e for e in (json.loads(l) for l in open(f)) if e.get("event") == "check"]


def _ident(d, side, variant):
    try:
        return ablate_identity.ident(os.path.join(d, side), variant)
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"


def row(d):
    la, lb = json.load(open(os.path.join(d, "local_a.json"))), json.load(open(os.path.join(d, "local_b.json")))
    pd = json.load(open(os.path.join(d, "outputs.json")))
    ch = _checks(d)
    first = lambda x: (x["logprobs"] or [None])[0]  # noqa: E731
    return {"name": os.path.basename(d), "n": len(lb),
            "first_check": ch[0]["result"] if ch else None,
            "results": sorted({c["result"] for c in ch}),
            "local_equal": sum(a == b for a, b in zip(la, lb)),
            "first_token_equal": sum(first(a) == first(b) for a, b in zip(la, lb)),
            "pd_equal_decode_local": sum(p == b for p, b in zip(pd, lb)),
            **{f"{v}_equal": _same(d, v) for v in ("v6", "v6p")}}


def _same(d, v):
    a, b = _ident(d, "inst_a", v), _ident(d, "inst_b", v)
    return None if a.startswith("error") or b.startswith("error") else a == b


def main(runs):
    rows = [row(d) for r in runs for d in sorted(glob.glob(os.path.join(r, "hb*")))
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "local_b.json"))]
    unsafe = [r["name"] for r in rows if r["first_check"] == "admitted" and r["pd_equal_decode_local"] < r["n"]]
    print(json.dumps({"rows": rows, "unsafe_admits": unsafe}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
