"""Offline evaluation of a hybrid guard (review round 4): identity first, canary only on a mismatch.

python -m lockstep.hybrid_offline ROOT [ROOT ...] > hybrid_report.json

For every live deployment with ground truth (local_a.json, local_b.json) and a live verdict: the identity
admits or blocks (live_guard.jsonl). The hybrid admits when the identity admits; on a mismatch it runs a
canary, i.e. compares the first prompt's prefill (first generated token's log-probability) on both sides,
and admits if they agree. KV safety is judged at the prefill: a reuse is unsafe if the two instances'
prefills differ on any prompt (the decode instance does its own decoding, so only prefill numerics are
reused). Reported: unsafe admits, unneeded blocks, and how many canaries the hybrid runs.
"""
import glob
import json
import os
import sys


def verdict(d):
    ev = [json.loads(l) for f in glob.glob(os.path.join(d, "inst_*", "capture", "live_guard.jsonl")) for l in open(f)]
    res = {e.get("result") for e in ev if e.get("event") == "check"}
    if not res:
        return None
    return "admitted" if res == {"admitted"} or ("admitted" in res and "blocked" not in res) else "blocked"


def main(roots):
    rows = []
    for root in roots:
        for f in sorted(glob.glob(os.path.join(root, "**", "local_b.json"), recursive=True)):
            d = os.path.dirname(f)
            if "failed_attempts" in d or "plant_not_applied" in d or not os.path.exists(os.path.join(d, "local_a.json")):
                continue
            v = verdict(d)
            if v is None:
                continue
            a, b = json.load(open(os.path.join(d, "local_a.json"))), json.load(open(os.path.join(d, "local_b.json")))
            pre = [(x["logprobs"] or [None])[0] != (y["logprobs"] or [None])[0] for x, y in zip(a, b)]
            kv_unsafe = any(pre)
            canary_differs = pre[0]
            hybrid = "admitted" if v == "admitted" or not canary_differs else "blocked"
            rows.append({"dir": d, "identity": v, "hybrid": hybrid, "kv_unsafe": kv_unsafe})
    def cm(key):
        return {"unsafe_admits": sum(1 for r in rows if r[key] == "admitted" and r["kv_unsafe"]),
                "unneeded_blocks": sum(1 for r in rows if r[key] == "blocked" and not r["kv_unsafe"]),
                "blocks": sum(1 for r in rows if r[key] == "blocked")}
    out = {"deployments": len(rows), "kv_unsafe": sum(r["kv_unsafe"] for r in rows),
           "identity": cm("identity"), "hybrid": cm("hybrid"),
           "hybrid_canaries_run": sum(1 for r in rows if r["identity"] == "blocked")}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
