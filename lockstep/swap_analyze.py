"""Analyze autotune-swap intervention runs against the two reference runs.

python -m lockstep.swap_analyze --runs runs/e4/e4_swap_q8 --manifest cache/swap_q8_f1f2/manifest.json \
       --ref-a runs/e1v2/e1v2_qwen3_8b/x_fresh1 --ref-b runs/e1v2/e1v2_qwen3_8b/x_fresh2
"""
import argparse
import json
import os

from lockstep import align


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ref-a", required=True)
    ap.add_argument("--ref-b", required=True)
    a = ap.parse_args()
    man = json.load(open(a.manifest))
    keep = align.prompt_keys(a.ref_a)
    refs = {"A": align.kappa_table(align.load_records(a.ref_a), keep)[0],
            "B": align.kappa_table(align.load_records(a.ref_b), keep)[0]}
    out = {}
    names = ["cacheA", "cacheB"] + sorted(k for k in man)
    for nm in names:
        d = os.path.join(a.runs, nm)
        if not os.path.exists(os.path.join(d, "outputs.json")):
            continue
        t = align.kappa_table(align.load_records(d), keep)[0]
        res = {}
        for k, ref in refs.items():
            c = align.compare(ref, t)
            res[k] = {"mismatched": c["mismatched_keys"], "common": c["common"], "first": c["earliest_site"]}
        out[nm] = {"kernel": man.get(nm, {}).get("kernel"), "A": man.get(nm, {}).get("A"),
                   "B": man.get(nm, {}).get("B"), **res}
        print(f"{nm:7s} {str(out[nm]['kernel']):38s} vs A: {res['A']['mismatched']:>7}/{res['A']['common']} "
              f"first={res['A']['first']} | vs B: {res['B']['mismatched']:>7} first={res['B']['first']}")
    json.dump(out, open(os.path.join(a.runs, "swap_analysis.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
