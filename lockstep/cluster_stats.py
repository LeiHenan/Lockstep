"""Clustered counts for the live guard's natural KV-unsafe deployments (review round 9, question 3).

python -m lockstep.cluster_stats ROOT [ROOT ...] > cluster_report.json

Live deployments (a live_guard.jsonl verdict, ground truth local_a/local_b) under each ROOT. A deployment is
KV-unsafe if the instances' prefills differ on any prompt (first generated token's log-probability), planted if
either instance replays a plant file. Deployments of one configuration share warm compiler caches and a server,
so they are not independent; we group them
  cell    - mirror root, model, both instances' arguments and flag, and which instances are cold;
  family  - mirror root, model, both instances' arguments and flag (cold and warm combinations together).
For each grouping: the number of groups containing a natural KV-unsafe deployment, the number of groups with an
unsafe admit, and a one-sided 95% exact bound on the per-group rate of unsafe admits (treating groups as
independent), next to the per-deployment bound.
"""
import glob
import json
import os
import sys


def verdict(d):  # as hybrid_offline.verdict, with "admitted_by_canary" counted as admitted
    ev = [json.loads(l) for f in glob.glob(os.path.join(d, "inst_*", "capture", "live_guard.jsonl")) for l in open(f)]
    res = {"admitted" if str(e.get("result")).startswith("admitted") else e.get("result")
           for e in ev if e.get("event") == "check"}
    if not res:
        return None
    return "admitted" if "admitted" in res and "blocked" not in res else "blocked"


def _first_blocked(d):
    f = os.path.join(d, "inst_b", "capture", "live_guard.jsonl")
    if not os.path.exists(f):
        return False
    res = [json.loads(l).get("result") for l in open(f) if '"check"' in l]
    return bool(res) and res[0] == "blocked"


def _bound(n):
    return None if not n else 1 - 0.05 ** (1 / n)


def _spec(meta, side):
    s = meta.get(side) or {}
    env = s.get("env") or {}
    a = s.get("args", [])
    # the served model name is a label, also when unique per server (m-<pid>-a, since R14 addendum 3)
    args = [x for i, x in enumerate(a) if x != "--served-model-name" and (i == 0 or a[i - 1] != "--served-model-name")]
    return {"args": " ".join(args), "ibi": bool(s.get("ibi")), "cold": "VLLM_CACHE_ROOT" in env,
            # planted: a replayed plant file, or the LD test hook that changes a decision after the handshake
            "plant": "plant_" in os.path.basename(str(env.get("LOCKSTEP_FORCE", ""))) or
                     bool(env.get("LOCKSTEP_TEST_RETUNE_AFTER_S")),
            "version": env.get("LOCKSTEP_GUARD_VERSION")}


def rows(roots):
    for root in roots:
        for f in sorted(glob.glob(os.path.join(root, "**", "local_b.json"), recursive=True)):
            d = os.path.dirname(f)
            if "failed_attempts" in d or "plant_not_applied" in d or not os.path.exists(os.path.join(d, "meta.json")):
                continue
            live = verdict(d)
            if live is None or not os.path.exists(os.path.join(d, "local_a.json")):
                continue
            meta = json.load(open(os.path.join(d, "meta.json")))
            if meta.get("mode") != "pd":
                continue
            a, b = json.load(open(os.path.join(d, "local_a.json"))), json.load(open(os.path.join(d, "local_b.json")))
            unsafe = any((x["logprobs"] or [None])[0] != (y["logprobs"] or [None])[0] for x, y in zip(a, b))
            sa, sb = _spec(meta, "a"), _spec(meta, "b")
            try:
                model = json.load(open(os.path.join(d, "inst_b", "cfg.json")))["engine_kwargs"]["model"]
            except Exception:
                model = meta.get("model") or ""
            fam = (root, model, sa["args"], sb["args"], sa["ibi"], sb["ibi"])
            plant_file = any("plant_" in os.path.basename(str(((meta.get(s) or {}).get("env") or {}).get("LOCKSTEP_FORCE", "")))
                             for s in ("a", "b"))
            planted = sa["plant"] or sb["plant"]
            if planted and not plant_file and _first_blocked(d):
                planted = False  # LD hook, but the pair diverged on its own before the change (LD addendum 4)
            yield {"dir": d, "unsafe": unsafe, "admitted": live == "admitted", "planted": planted,
                   "family": "|".join(map(str, fam)), "cell": "|".join(map(str, fam + (sa["cold"], sb["cold"])))}


def main(roots):
    rs = [r for r in rows(roots)]
    nat = [r for r in rs if r["unsafe"] and not r["planted"]]
    out = {"deployments": len(rs), "kv_unsafe": sum(r["unsafe"] for r in rs), "natural_kv_unsafe": len(nat),
           "unsafe_admits": sum(r["unsafe"] and r["admitted"] for r in rs),
           "per_deployment_bound": _bound(len(nat))}
    for g in ("cell", "family"):
        groups = sorted({r[g] for r in nat})
        bad = {r[g] for r in nat if r["admitted"]}
        out[g] = {"groups_with_natural_unsafe": len(groups), "groups_with_unsafe_admit": len(bad),
                  "bound": _bound(len(groups)), "groups": groups}
    out["unsafe_admit_dirs"] = [r["dir"] for r in rs if r["unsafe"] and r["admitted"]]
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
