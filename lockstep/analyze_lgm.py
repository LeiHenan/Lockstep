"""LG-M analysis (results_prereg/LG_MATRIX_PREREGISTRATION.md): the live guard over a deployment matrix.

python -m lockstep.analyze_lgm RUN_DIR [RUN_DIR ...] > lgm_report.json

Deployment names are lgm_<model>_<homo|hetero>_<on|off>_<ww|cw|wc|cc>_r<k> (prime_* are warm-up deployments,
reported separately). Per deployment: ground truth (the two instances compute identically when each serves
every prompt alone), the live verdict, and whether the disaggregated outputs equal the decode instance's own.
M1: no unsafe admit. M2: blocked deployments have exact outputs. M3: with the flag on, unneeded blocks only
in deployments with a cold instance. M4: flag on, ww, hetero -> identical and admitted. M5: live = offline.
"""
import collections
import json
import os
import re
import sys

from lockstep.analyze_liveguard import scenario

PAT = re.compile(r"(lgm|prime)_([a-z0-9]+)_(homo|hetero)_(on|off)(?:_(ww|cw|wc|cc)_r(\d+))?$")


def rows(root):
    off = {}
    p = os.path.join(root, "guard_pd_offline.json")
    if os.path.exists(p):
        try:
            off = {s["scenario"]: s["allow"] for s in json.load(open(p))["scenarios"]}
        except Exception:
            off = {}
    out = []
    for d in sorted(os.listdir(root)):
        m = PAT.match(d)
        full = os.path.join(root, d)
        if not m or not os.path.exists(os.path.join(full, "local_b.json")):
            continue
        r = scenario(full)
        n, k = r["pd_eq_local_b"].split("/")
        admitted = r["verdicts"] == ["admitted"]
        r.update(kind=m[1], model=m[2], cfg=m[3], flag=m[4], combo=m[5] or "ww", rep=m[6],
                 admitted=admitted, blocked="blocked" in r["verdicts"], outputs_exact=n == k,
                 unsafe=admitted and not r["instances_identical"],
                 unneeded=(not admitted) and r["instances_identical"],
                 offline_allow=off.get(d))
        out.append(r)
    return out


def summarize(rs):
    s = {"deployments": len(rs), "diverged": sum(not r["instances_identical"] for r in rs),
         "blocked": sum(r["blocked"] for r in rs), "unsafe": sum(r["unsafe"] for r in rs),
         "unneeded": sum(r["unneeded"] for r in rs), "outputs_exact": sum(r["outputs_exact"] for r in rs),
         "no_verdict": sum(not r["verdicts"] for r in rs)}
    return s


def main(roots):
    allr = [r for root in roots for r in rows(root)]
    main_ = [r for r in allr if r["kind"] == "lgm"]
    by = collections.defaultdict(list)
    for r in main_:
        by[(r["model"], r["flag"])].append(r)
        by[(r["model"], r["flag"], r["cfg"])].append(r)
        by[("all", r["flag"], r["combo"])].append(r)
    report = {
        "total": summarize(main_),
        "by_group": {"/".join(k): summarize(v) for k, v in sorted(by.items())},
        "M1_unsafe": [r["name"] for r in main_ if r["unsafe"]],
        "M2_blocked_not_exact": [r["name"] for r in main_ if r["blocked"] and not r["outputs_exact"]],
        "M3_unneeded_flag_on_ww": [r["name"] for r in main_ if r["flag"] == "on" and r["combo"] == "ww" and r["unneeded"]],
        "M4_hetero_on_ww": {r["name"]: (r["instances_identical"], r["verdicts"]) for r in main_
                            if r["flag"] == "on" and r["cfg"] == "hetero" and r["combo"] == "ww"},
        "M5_live_vs_offline": [r["name"] for r in main_ if r["offline_allow"] is not None
                               and r["offline_allow"] != r["admitted"]],
        "offline_missing": sum(r["offline_allow"] is None for r in main_),
        "prime": summarize([r for r in allr if r["kind"] == "prime"]),
        "deployments": [{k: r[k] for k in ("name", "model", "cfg", "flag", "combo", "instances_identical",
                                            "local_a_eq_b", "verdicts", "pd_eq_local_b", "offline_allow")}
                        for r in allr],
    }
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
