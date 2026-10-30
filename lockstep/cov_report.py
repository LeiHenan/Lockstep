"""COV analysis (results_prereg/R13_KVC_COV_PREREGISTRATION.md).

python -m lockstep.cov_report RUN_DIR > cov_report.json

Per process (a TP4 single instance): its outputs, each rank's version-6 identity recomputed from the rank's capture
files, and the NCCL settings its log reports. Then every pair of processes: outputs equal (all, first token) and
per-rank identities equal. Checks C-1, C-2.
"""
import glob
import itertools
import json
import os
import re
import sys

from lockstep import decisions, guard


def proc(d):
    cap = os.path.join(d, "inst_a", "capture")
    cfg = json.load(open(os.path.join(d, "inst_a", "cfg.json")))
    ids = {}
    for r in range(4):
        use, dec, st, _ = decisions.load_rank(cap, r)
        ids[r] = guard.identity_v6(cfg, use, dec, st) if use else None
    nccl = set()
    for line in open(os.path.join(d, "server_a.log"), errors="replace"):
        m = re.search(r"NCCL INFO (NCCL_[A-Z_]+ set by environment to .*)$", line)
        if m:
            nccl.add(m.group(1).strip())
    return {"outputs": json.load(open(os.path.join(d, "outputs.json"))), "ids": ids, "nccl": sorted(nccl)}


def main(run):
    P = {os.path.basename(d): proc(d) for d in sorted(glob.glob(os.path.join(run, "*")))
         if os.path.exists(os.path.join(d, "outputs.json"))}
    first = lambda x: (x["logprobs"] or [None])[0]  # noqa: E731
    pairs = []
    for a, b in itertools.combinations(sorted(P), 2):
        oa, ob = P[a]["outputs"], P[b]["outputs"]
        pairs.append({"a": a, "b": b, "outputs_equal": sum(x == y for x, y in zip(oa, ob)),
                      "first_token_equal": sum(first(x) == first(y) for x, y in zip(oa, ob)), "n": len(oa),
                      "rank_ids_equal": all(P[a]["ids"][r] == P[b]["ids"][r] and P[a]["ids"][r] for r in range(4))})
    print(json.dumps({"processes": {k: {"ids": v["ids"], "nccl": v["nccl"]} for k, v in P.items()}, "pairs": pairs,
                      "C-1_all_equal": all(p["outputs_equal"] == p["n"] and p["rank_ids_equal"] for p in pairs)},
                     indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
