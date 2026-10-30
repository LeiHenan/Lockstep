"""V7S analysis (results_prereg/V7S_V032_PREREGISTRATION.md).

python -m lockstep.v7s_report RUN_DIR PROMPTS_META.json > v7s_report.json

The rows of lockstep.analyze_g7 (live verdicts, ground truth per prompt group), plus per deployment: whether the
two version-7 identities were equal (the decode side's first check: its identity against the producer's served
one), their coverage, whether version 6 recomputed offline from the same captures is equal, and canary and
identity timings. Checks V-1..V-5.
"""
import contextlib
import glob
import io
import json
import os
import sys

from lockstep import ablate_identity, analyze_g7


def _checks(d):
    f = os.path.join(d, "inst_b", "capture", "live_guard.jsonl")
    return [e for e in (json.loads(l) for l in open(f)) if e.get("event") == "check"]


def _from_producer(d):
    """Prompts whose instances' own generations differ and whose disaggregated output equals the producer's own
    and not the consumer's: the consumer's decode steps reproduce the producer's result once given its KV, so
    the difference lies in the transferred K/V even where the first-token proxy agrees."""
    la, lb = json.load(open(os.path.join(d, "local_a.json"))), json.load(open(os.path.join(d, "local_b.json")))
    pd = json.load(open(os.path.join(d, "outputs.json")))
    return {"n": sum(a != b and p == a for a, b, p in zip(la, lb, pd)),
            "first_token_differs": sum((a["logprobs"] or [None])[0] != (b["logprobs"] or [None])[0] for a, b in zip(la, lb)),
            "pd_ne_consumer": sum(p != b for p, b in zip(pd, lb))}


def main(run, meta):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        analyze_g7.main(run, meta)
    rep = json.loads(buf.getvalue())
    for r in rep["rows"]:
        d = os.path.join(run, r["name"])
        ch = _checks(d)
        first = ch[0] if ch else {}
        # equal identities are admitted on the first try (vLLM's check passes with both folded in); only a
        # mismatch goes on to the canary, whose record names the producer's served identity
        r["v7_equal"] = first.get("result") == "admitted"
        r["prefill_from_producer"] = _from_producer(d)
        r["v7_coverage"] = [first.get("coverage"), first.get("remote_coverage")]
        r["canary_checked"] = any(e.get("canary_checked") for e in ch)
        r["canary_ms"] = [e.get("canary_ms") for e in ch if e.get("canary_ms") is not None]
        r["identity_ms"] = [e.get("identity_ms") for e in ch]
        try:
            r["v6_equal"] = (ablate_identity.ident(os.path.join(d, "inst_a"), "v6")
                             == ablate_identity.ident(os.path.join(d, "inst_b"), "v6"))
        except Exception as e:
            r["v6_equal"] = None
            r["v6_error"] = repr(e)
        r["admitted"] = "blocked" not in r["verdicts"] and any(v.startswith("admitted") for v in r["verdicts"])
    main_ = [r for r in rep["rows"] if not r["prime"]]
    rep["V"] = {
        "V-1_unsafe_admits": [r["name"] for r in main_ if r["admitted"] and r["kv_unsafe"]],
        "V-2_not_exact": [r["name"] for r in main_ if r["pd_eq_local_b"] != r["n"]],
        "V-3_kv_unsafe": sum(r["kv_unsafe"] for r in main_),
        "V-4_unneeded_blocks": [r["name"] for r in main_ if not r["admitted"] and not r["kv_unsafe"]],
        "V-4_v7_differs_prefill_equal": [r["name"] for r in main_ if not r["v7_equal"] and not r["kv_unsafe"]],
        "V-5_v7_equal_v6_differs": [r["name"] for r in rep["rows"] if r["v7_equal"] and r["v6_equal"] is False],
        "V-5_v7_differs_v6_equal": [r["name"] for r in rep["rows"] if not r["v7_equal"] and r["v6_equal"]],
        "main": len(main_), "divergent": sum(r["outputs_differ"] for r in main_),
        "blocked": sum(not r["admitted"] for r in main_),
        "admitted_by_canary": sum("admitted_by_canary" in r["verdicts"] for r in main_),
        "v7_equal": sum(r["v7_equal"] for r in main_),
        "partial_coverage": [r["name"] for r in rep["rows"] if any(c for c in r["v7_coverage"] if c)],
    }
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
