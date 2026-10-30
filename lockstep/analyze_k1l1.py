"""K1/L1 analysis (results_prereg/K1_L1_KV_TRUTH_LIFECYCLE_PREREGISTRATION.md).

python -m lockstep.analyze_k1l1 RUN_DIR [RUN_DIR ...] > k1l1_report.json

K1 (k1*_ dirs with kv_truth.json): per pair the three ground-truth labels (K/V bit-identical on every prompt,
all prompt-token log-probabilities equal, generated outputs equal, first-token proxy equal) and whether the v6
identities of the two instances are equal (recomputed offline from the captures). Checks K-1 to K-3.
L1 (l1_ dirs): guard events of both instances, canaries computed on demand, start-up and request latencies.
"""
import glob
import json
import os
import statistics as st
import sys

from lockstep import guard


def _scenario(name):
    return name.split("_")[1].upper()


def k1(d):
    r = json.load(open(os.path.join(d, "kv_truth.json")))
    try:
        v6 = guard.identity(os.path.join(d, "inst_a"), version=6)[0] == guard.identity(os.path.join(d, "inst_b"), version=6)[0]
    except Exception as e:
        v6 = f"error: {e}"
    n = r["n"]
    firsts = [p["first"] for p in r["per_prompt_kv"] if p["first"]]
    return {"name": os.path.basename(d), "scenario": _scenario(os.path.basename(d)), "n": n,
            "kv_equal": r["kv_all_equal"], "kv_equal_prompts": r["kv_equal_prompts"], "prompts_kv": r["prompts_kv"],
            "prompt_logprobs_equal": r["prompt_logprobs_equal"] == n, "outputs_equal": r["outputs_equal"] == n,
            "first_token_equal": r["first_token_equal"] == n, "v6_equal": v6,
            "first_kv_difference": min(firsts, key=lambda f: (len(f["layer"]), f["layer"])) if firsts else None}


def l1(d):
    ev = {nm: [json.loads(l) for l in open(f)] for nm in ("a", "b")
          for f in [os.path.join(d, f"inst_{nm}", "capture", "live_guard.jsonl")] if os.path.exists(f)}
    meta = json.load(open(os.path.join(d, "meta.json")))
    la, lb = json.load(open(os.path.join(d, "local_a.json"))), json.load(open(os.path.join(d, "local_b.json")))
    pd = json.load(open(os.path.join(d, "outputs.json")))
    lat = [x[1] for x in meta.get("request_latency_s", [])]
    checks = [e for e in ev.get("b", []) if e.get("event") == "check"]
    kinds = lambda nm: sorted({e["event"] for e in ev.get(nm, [])})
    return {"name": os.path.basename(d), "scenario": _scenario(os.path.basename(d)),
            "results": sorted({e.get("result") for e in checks}), "n_checks": len(checks),
            "events_a": kinds("a"), "events_b": kinds("b"),
            "invalidated": sum(e["event"] == "invalidated" for e in ev.get("b", [])),
            "republished": sum(e["event"] == "republished" for e in ev.get("a", [])),
            "injected": [nm for nm in ev if any(e["event"] == "test_injected" for e in ev[nm])],
            "canaries_computed": len(meta.get("canary_on_demand") or []),
            "canary_s": [c["s"] for c in (meta.get("canary_on_demand") or [])],
            "identity_ms_median": st.median([e["identity_ms"] for e in checks]) if checks else None,
            "ready_s": meta.get("ready_s"), "latency_p50": st.median(lat) if lat else None,
            "latency_p99": sorted(lat)[int(0.99 * (len(lat) - 1))] if lat else None,
            "instances_equal": la == lb, "pd_eq_local_b": sum(x == y for x, y in zip(pd, lb)), "n": len(lb)}


def main(roots):
    out = {"k1": [], "l1": []}
    for root in roots:
        for d in sorted(glob.glob(os.path.join(root, "k1*_*"))):
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "kv_truth.json")):
                out["k1"].append(k1(d))
        for d in sorted(glob.glob(os.path.join(root, "l1_*"))):
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "local_b.json")):
                out["l1"].append(l1(d))
    K = out["k1"]
    out["K1_checks"] = {
        "K-1 v6 equal but KV differs": [r["name"] for r in K if r["v6_equal"] is True and not r["kv_equal"]],
        "K-2 first-token proxy differs but KV equal": [r["name"] for r in K if not r["first_token_equal"] and r["kv_equal"]],
        "K-3 T1 KV equal": [(r["name"], r["kv_equal"]) for r in K if r["scenario"] == "T1"],
        "K-4 proxy equal but KV differs": [r["name"] for r in K if r["first_token_equal"] and not r["kv_equal"]],
        "KV equal but outputs differ": [r["name"] for r in K if r["kv_equal"] and not r["outputs_equal"]],
        "v6 differs but KV equal (unneeded block)": [r["name"] for r in K if r["v6_equal"] is False and r["kv_equal"]],
    }
    out["L1_checks"] = {"not exact": [r["name"] for r in out["l1"] if r["pd_eq_local_b"] != r["n"]]}
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1:])
