"""KVE analysis (results_prereg/CF_KVE_PREREGISTRATION.md): for each K/V-digest pair, whether the two instances'
version-6 identities (recomputed offline from the captures) are equal, and whether their K/V is bit-identical on
every prompt, layer, and prompt token. Checks KE-1..KE-3.

python -m lockstep.kve_report RUN_DIR > kve_report.json
"""
import glob
import json
import os
import sys

from lockstep import ablate_identity


def main(run):
    rows = []
    for f in sorted(glob.glob(os.path.join(run, "kve_*", "kv_truth.json"))):
        d = os.path.dirname(f)
        kt = json.load(open(f))
        try:
            eq = ablate_identity.ident(os.path.join(d, "inst_a"), "v6") == ablate_identity.ident(os.path.join(d, "inst_b"), "v6")
        except Exception as e:
            eq = None
        rows.append({"name": os.path.basename(d), "v6_equal": eq, "kv_all_equal": kt["kv_all_equal"],
                     "kv_equal_prompts": kt["kv_equal_prompts"], "prompts_kv": kt["prompts_kv"],
                     "outputs_equal": kt["outputs_equal"], "first_token_equal": kt["first_token_equal"], "n": kt["n"]})
    # addendum 2: the first heterogeneous pairs recorded only the first chunk of chunked prefills; judge that
    # group by its reruns with the fixed connector (kve_q8_hetero_on_chunk_*), and list the first runs apart
    voided = [r for r in rows if r["name"].startswith("kve_q8_hetero_on_r")]
    valid = [r for r in rows if r not in voided]
    eqs = [r for r in valid if r["v6_equal"]]
    checks = {"pairs": len(valid), "v6_equal_pairs": len(eqs),
              "KE-1_equal_identity_equal_kv": all(r["kv_all_equal"] for r in eqs),
              "KE-2_at_least_15_equal": len(eqs) >= 15,
              "KE-3_kv_differs_identity_differs": all(r["v6_equal"] is False for r in valid if not r["kv_all_equal"]),
              "voided_first_chunk_only": [r["name"] for r in voided]}
    print(json.dumps({"checks": checks, "rows": valid, "voided_rows": voided}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
