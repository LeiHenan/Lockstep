"""FN report (results_prereg/FN_FINAL_NATURAL_PREREGISTRATION.md): the final frozen version on vLLM 0.32.0 with the
flag off (natural divergence).

Per guarded deployment (fn_*), as fh_report: live verdicts, coverage, KV-unsafe by the first-token proxy, exact
outputs, the offline static hashes. Per K/V pair (fnk_*): bit-identical K/V and the identity recomputed from the
traces (as fh_report). Checks FN-1..FN-6 and gives the one-sided 95% bounds of the report-only items.

python -m lockstep.fn_report RUN_DIR > fn_report.json
"""
import glob
import json
import os
import sys

from lockstep import fh_report


def _bound(n):  # one-sided exact 95% upper bound with no event in n trials
    return round(1 - 0.05 ** (1 / n), 4) if n else None


def _cell(name):  # fn_qwen8b_homo_cw_r1 -> qwen8b_homo_cw; family: model and configuration
    p = name.split("_")
    return "_".join(p[1:4]), "_".join(p[1:3])


def main(run):
    deps = {os.path.basename(d): fh_report.deployment(d) for d in sorted(glob.glob(os.path.join(run, "fn_*")))
            if os.path.isdir(d) and "smoke" not in d and os.path.exists(os.path.join(d, "local_b.json"))}
    kvs = {os.path.basename(d): fh_report.kv_pair(d) for d in sorted(glob.glob(os.path.join(run, "fnk_*")))
           if os.path.exists(os.path.join(d, "kv_truth.json"))}
    unsafe = [k for k, v in deps.items() if v["kv_unsafe"]]
    safe = [k for k in deps if k not in unsafe]
    cells = {_cell(k)[0] for k in unsafe}
    fams = {_cell(k)[1] for k in unsafe}
    same = [k for k, v in kvs.items() if v["v7_from_trace_equal"] is True]
    differ = [k for k, v in kvs.items() if v["kv_all_equal"] is False]
    out = {"deployments": deps, "kv_pairs": kvs,
           "FN-1_kv_unsafe": unsafe, "FN-1_held": len(unsafe) >= 8,
           "FN-2_unsafe_admits": [k for k in unsafe if deps[k]["admitted_on_identity"]],
           "FN-2_coverage": sorted({c for v in deps.values() for c in v["coverage"]}),
           "FN-3_inexact": [k for k, v in deps.items() if v["outputs_exact"] != v["n"]],
           "FN-4_static_full_unsafe_admits": [k for k in unsafe if deps[k]["static"]],
           "FN-5_equal_identity_pairs": same, "FN-5_all_kv_equal": all(kvs[k]["kv_all_equal"] for k in same),
           "FN-6_kv_differs": differ, "FN-6_all_identities_differ": all(kvs[k]["v7_from_trace_equal"] is False for k in differ),
           "unneeded_blocks_identity": [k for k in safe if not deps[k]["admitted_on_identity"]],
           "static_full_unneeded_blocks": [k for k in safe if not deps[k]["static"]],
           "static_full_unsafe_admits": [k for k in unsafe if deps[k]["static"]],
           "bounds_unsafe_admit": {"deployments": [len(unsafe), _bound(len(unsafe))], "cells": [len(cells), _bound(len(cells))],
                                   "families": [len(fams), _bound(len(fams))]},
           "kv_equal_identity_bound": [len(same), _bound(len(same))],
           "first_handshake_identity_ms": sorted(v["identity_ms"][0] for v in deps.values() if v["identity_ms"])}
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
