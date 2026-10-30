"""Guard policies compared offline on the same live deployments (review round 5).

python -m lockstep.baselines_offline ROOT [ROOT ...] > baselines_report.json

For every deployment with ground truth (local_a.json, local_b.json, inst_a/, inst_b/), KV reuse is unsafe
if the two instances' prefills differ on any prompt (first generated token's log-probability), as in
hybrid_offline. Policies (admit = allow the transfer):
  budget          - configuration incl. token budget + stack (ablate_identity variant "budget")
  canary          - precomputed canary only: admit iff the canary prompt's prefill agrees
  budget+canary   - budget hash; canary only on a mismatch
  v6              - identity version 6, recomputed offline
  v6+canary       - identity version 6; canary only on a mismatch
  live+canary     - the identity the live guard used (version 3 or 6); canary only on a mismatch
Reported per policy: unsafe admits, unneeded blocks, canaries run; overall and for deployments whose
captures have per-subgraph usage records (where version 6 is defined).
"""
import glob
import json
import os
import sys

from lockstep import ablate_identity
from lockstep.hybrid_offline import verdict

POLICIES = ("budget", "canary", "budget+canary", "v6", "v6+canary", "live+canary", "v6p")


def rows(roots):
    for root in roots:
        for f in sorted(glob.glob(os.path.join(root, "**", "local_b.json"), recursive=True)):
            d = os.path.dirname(f)
            if "failed_attempts" in d or "plant_not_applied" in d:
                continue
            if not all(os.path.exists(os.path.join(d, n)) for n in ("local_a.json", "inst_a", "inst_b")):
                continue
            a, b = json.load(open(os.path.join(d, "local_a.json"))), json.load(open(os.path.join(d, "local_b.json")))
            pre = [(x["logprobs"] or [None])[0] != (y["logprobs"] or [None])[0] for x, y in zip(a, b)]
            ia, ib = os.path.join(d, "inst_a"), os.path.join(d, "inst_b")
            try:
                same = {v: ablate_identity.ident(ia, v) == ablate_identity.ident(ib, v) for v in ("budget", "v6", "v6p")}
            except Exception as e:  # missing cfg.json etc.
                print(f"skip {d}: {e}", file=sys.stderr)
                continue
            usage = bool(glob.glob(os.path.join(ia, "capture", "usage_*.json")))
            live = verdict(d)
            canary_ok = not pre[0]
            adm = {"budget": same["budget"], "canary": canary_ok,
                   "budget+canary": same["budget"] or canary_ok,
                   "v6": same["v6"], "v6+canary": same["v6"] or canary_ok, "v6p": same["v6p"],
                   "live+canary": None if live is None else (live == "admitted" or canary_ok)}
            ran = {"budget": 0, "canary": 1, "budget+canary": int(not same["budget"]), "v6": 0, "v6p": 0,
                   "v6+canary": int(not same["v6"]), "live+canary": None if live is None else int(live != "admitted")}
            yield {"dir": d, "kv_unsafe": any(pre), "usage": usage, "live": live, "admit": adm, "canaries": ran}


def summarize(rs):
    out = {"deployments": len(rs), "kv_unsafe": sum(r["kv_unsafe"] for r in rs)}
    for p in POLICIES:
        sub = [r for r in rs if r["admit"][p] is not None]
        out[p] = {"n": len(sub),
                  "unsafe_admits": sum(r["admit"][p] and r["kv_unsafe"] for r in sub),
                  "unneeded_blocks": sum((not r["admit"][p]) and not r["kv_unsafe"] for r in sub),
                  "canaries_run": sum(r["canaries"][p] for r in sub)}
    return out


def main(roots):
    rs = list(rows(roots))
    out = {"all": summarize(rs), "with_usage_records": summarize([r for r in rs if r["usage"]]),
           "unsafe_examples": {p: [r["dir"] for r in rs if r["admit"][p] and r["kv_unsafe"]][:20] for p in POLICIES}}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
