"""FH report (results_prereg/FH_FINAL_HELDOUT_MAIN_PREREGISTRATION.md): the final frozen version on vLLM main.

Per guarded deployment (fh_*): live verdicts of the consumer's handshakes, coverage on both sides, KV-unsafe by
the first-token proxy, exact outputs, the offline static hashes (as offline_r13 o2), start-up time, and identity
time. Per K/V pair (fhk_*): bit-identical K/V, and the identity recomputed from the traces with the frozen code's
identity_v7 (all parts but the resolved options and the full environment, which kv_truth does not record) and as
version 6 with the pointwise rule. Checks FH-1..FH-5.

python -m lockstep.fh_report RUN_DIR > fh_report.json
"""
import glob
import json
import os
import sys

from lockstep import ablate_identity, decisions, guard, offline_r13


def _events(d, side):
    out = []
    for f in glob.glob(os.path.join(d, f"inst_{side}", "capture", "live_guard.jsonl")):
        out += [json.loads(l) for l in open(f)]
    return out


def _first_lp(o):
    lp = o.get("logprobs") or []
    return lp[0] if lp else None


def _v7_from_trace(inst):
    cfg = json.load(open(os.path.join(inst, "cfg.json")))
    cap = os.path.join(inst, "capture")
    use, dec = decisions.load_usage(inst), decisions.load(inst)
    ident, _ = guard.identity_v7(cfg, use, dec, decisions.load_stack(inst), cfg.get("env") or {}, None, None,
                                 decisions.load_structure(inst))
    return ident


def deployment(d):
    ea, eb = _events(d, "a"), _events(d, "b")
    res = {}
    for e in eb:
        if e["event"] == "check":
            res[e.get("result")] = res.get(e.get("result"), 0) + 1
    cov = sorted({",".join(e.get("coverage") or []) or "complete" for e in ea + eb if e["event"] in ("serve", "check")})
    la, lb = json.load(open(os.path.join(d, "local_a.json"))), json.load(open(os.path.join(d, "local_b.json")))
    out = json.load(open(os.path.join(d, "outputs.json")))
    ia, ib = os.path.join(d, "inst_a"), os.path.join(d, "inst_b")
    h = lambda x, drop: ablate_identity._h(offline_r13.static_part(x, drop))
    st, stx = h(ia, False) == h(ib, False), h(ia, True) == h(ib, True)
    fa, fb = offline_r13.flag_on(ia), offline_r13.flag_on(ib)
    meta = json.load(open(os.path.join(d, "meta.json")))
    ms = [e.get("identity_ms") for e in ea + eb if e["event"] in ("serve", "check") and e.get("identity_ms") is not None]
    return {"handshakes": res, "coverage": cov,
            "kv_unsafe": any(_first_lp(x) != _first_lp(y) for x, y in zip(la, lb)),
            "divergent": any(x != y for x, y in zip(la, lb)),
            "outputs_exact": sum(o == y for o, y in zip(out, lb)), "n": len(lb),
            "admitted_on_identity": res.get("admitted", 0) > 0 and not res.get("blocked"),
            "static": st, "static_flag_required": st and fa and fb, "static_no_budget_flag_required": stx and fa and fb,
            "flag_on": [fa, fb], "ready_s": meta.get("ready_s"), "identity_ms": [min(ms), max(ms)] if ms else None}


def kv_pair(d):
    kt = json.load(open(os.path.join(d, "kv_truth.json")))
    ia, ib = os.path.join(d, "inst_a"), os.path.join(d, "inst_b")
    try:
        v7 = _v7_from_trace(ia) == _v7_from_trace(ib)
    except Exception as e:
        v7 = f"error: {e!r}"[:200]
    try:
        v6p = ablate_identity.ident(ia, "v6p") == ablate_identity.ident(ib, "v6p")
    except Exception as e:
        v6p = f"error: {e!r}"[:200]
    return {"v7_from_trace_equal": v7, "v6p_equal": v6p, "kv_all_equal": kt["kv_all_equal"],
            "kv_equal_prompts": kt["kv_equal_prompts"], "prompts_kv": kt["prompts_kv"]}


def main(run):
    deps = {os.path.basename(d): deployment(d) for d in sorted(glob.glob(os.path.join(run, "fh_*")))
            if os.path.isdir(d) and "smoke" not in d and os.path.exists(os.path.join(d, "local_b.json"))}
    kvs = {os.path.basename(d): kv_pair(d) for d in sorted(glob.glob(os.path.join(run, "fhk_*")))
           if os.path.exists(os.path.join(d, "kv_truth.json"))}
    ident = [k for k, v in deps.items() if not v["kv_unsafe"]]
    same = [k for k, v in kvs.items() if v["v7_from_trace_equal"] is True]
    out = {"deployments": deps, "kv_pairs": kvs,
           "FH-1_kv_unsafe": [k for k, v in deps.items() if v["kv_unsafe"]],
           "FH-2_unsafe_admits": [k for k, v in deps.items() if v["kv_unsafe"] and v["admitted_on_identity"]],
           "FH-2_all_exact": all(v["outputs_exact"] == v["n"] for v in deps.values()),
           "FH-2_coverage": sorted({c for v in deps.values() for c in v["coverage"]}),
           "FH-3_unneeded_blocks_identity": [k for k in ident if not deps[k]["admitted_on_identity"]],
           "FH-4_static_flag_required_blocks": [k for k in ident if not deps[k]["static_flag_required"]],
           "FH-4_static_no_budget_differs_from_identity": [k for k in deps if deps[k]["static_no_budget_flag_required"] != deps[k]["admitted_on_identity"]],
           "FH-5_equal_identity_pairs": len(same), "FH-5_all_kv_equal": all(kvs[k]["kv_all_equal"] for k in same)}
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
