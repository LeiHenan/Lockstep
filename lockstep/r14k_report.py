"""R14 part K (results_prereg/R14_HELDOUT_MAIN_KV_PREREGISTRATION.md): K/V-digest pairs on 96 unseen UltraChat prompts.
For each pair, whether the two instances' identities (v6 and v6p, recomputed offline from the captures) are equal,
and whether their K/V is bit-identical on every prompt, layer, and prompt token. Checks K-1, K-2; pools with the 31
earlier equal-identity pairs for the one-sided 95% bound.

python -m lockstep.r14k_report RUN_DIR > r14k_report.json
"""
import glob
import json
import os
import sys

from lockstep import ablate_identity

EARLIER_EQUAL_PAIRS = 31  # K1/KVE/CF pairs with equal identities, all bit-identical (paper, Section 5.1)


def upper95(k, n):
    """One-sided exact 95% upper bound for k = 0 events in n trials."""
    assert k == 0
    return 1 - 0.05 ** (1 / n)


def main(run):
    rows = []
    for f in sorted(glob.glob(os.path.join(run, "k_*", "kv_truth.json"))):
        d = os.path.dirname(f)
        kt = json.load(open(f))
        eq = {}
        for v in ("v6", "v6p"):
            try:
                eq[v] = ablate_identity.ident(os.path.join(d, "inst_a"), v) == ablate_identity.ident(os.path.join(d, "inst_b"), v)
            except Exception as e:
                eq[v] = f"error: {e!r}"[:200]
        rows.append({"name": os.path.basename(d), "v6_equal": eq["v6"], "v6p_equal": eq["v6p"],
                     "kv_all_equal": kt["kv_all_equal"], "kv_equal_prompts": kt["kv_equal_prompts"],
                     "prompts_kv": kt["prompts_kv"], "outputs_equal": kt["outputs_equal"],
                     "first_token_equal": kt["first_token_equal"], "n": kt["n"]})
    same = [r for r in rows if r["v6p_equal"] is True]
    out = {"pairs": rows,
           "K-1_equal_identity_implies_equal_kv": all(r["kv_all_equal"] for r in same),
           "equal_identity_pairs": len(same),
           "K-2_at_least_8": len(same) >= 8,
           "differing_identity_pairs": [{"name": r["name"], "kv_all_equal": r["kv_all_equal"]} for r in rows
                                        if r["v6p_equal"] is False],
           "pooled_equal_pairs": len(same) + EARLIER_EQUAL_PAIRS}
    if all(r["kv_all_equal"] for r in same):
        out["pooled_upper95"] = upper95(0, out["pooled_equal_pairs"])
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
