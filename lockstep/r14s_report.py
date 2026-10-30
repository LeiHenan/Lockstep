"""R14 addendum 1, part S: what an unguarded KV substitution does. Guard off; the decode instance uses the prefill
instance's KV. Per deployment: KV-unsafe by the first-token proxy (local_a vs local_b), divergent (generations
differ), how many disaggregated outputs differ from the decode instance's own generation, and the offline verdicts
of the identity (v6p) and the static hash ("budget" variant of ablate_identity). Checks S-1, S-2.

python -m lockstep.r14s_report RUN_DIR > r14s_report.json
"""
import glob
import json
import os
import sys

from lockstep import ablate_identity


def _first_lp(o):
    lp = o.get("logprobs") or []
    return lp[0] if lp else None


def main(run):
    rows = []
    for d in sorted(glob.glob(os.path.join(run, "s_*"))):
        if not os.path.exists(os.path.join(d, "local_b.json")):
            continue
        la, lb = json.load(open(os.path.join(d, "local_a.json"))), json.load(open(os.path.join(d, "local_b.json")))
        out = json.load(open(os.path.join(d, "outputs.json")))
        unsafe = any(_first_lp(x) != _first_lp(y) for x, y in zip(la, lb))
        divergent = any(x != y for x, y in zip(la, lb))
        changed = sum(o != y for o, y in zip(out, lb))
        changed_tokens = sum(o["token_ids"] != y["token_ids"] for o, y in zip(out, lb))
        ver = {}
        for v in ("v6p", "budget"):
            try:
                ver[v] = ablate_identity.ident(os.path.join(d, "inst_a"), v) == ablate_identity.ident(os.path.join(d, "inst_b"), v)
            except Exception as e:
                ver[v] = f"error: {e!r}"[:200]
        rows.append({"name": os.path.basename(d), "kv_unsafe": unsafe, "divergent": divergent,
                     "outputs_changed": changed, "outputs_changed_tokens": changed_tokens, "n": len(lb),
                     "identity_equal": ver["v6p"], "static_hash_equal": ver["budget"]})
    unsafe = [r for r in rows if r["kv_unsafe"]]
    decode_only = [r for r in rows if r["divergent"] and not r["kv_unsafe"]]
    harmful = [r for r in rows if r["outputs_changed"] > 0]
    res = {"deployments": rows,
           "S-1_every_kv_unsafe_changes_outputs": all(r["outputs_changed"] > 0 for r in unsafe),
           "kv_unsafe": len(unsafe),
           "S-2_decode_only_divergent_unchanged": all(r["outputs_changed"] == 0 for r in decode_only),
           "decode_only_divergent": len(decode_only),
           "by_actual_harm": {
               "harmful": len(harmful),
               "identity_admits_harmful": sum(r["identity_equal"] is True for r in harmful),
               "identity_blocks_harmless": sum(r["identity_equal"] is False and r["outputs_changed"] == 0 for r in rows),
               "static_admits_harmful": sum(r["static_hash_equal"] is True for r in harmful),
               "static_blocks_harmless": sum(r["static_hash_equal"] is False and r["outputs_changed"] == 0 for r in rows)}}
    json.dump(res, sys.stdout, indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
