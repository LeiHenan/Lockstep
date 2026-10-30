"""E4 baseline: localization by invocation-order alignment (training-tool assumption) on planted faults."""
import json
import os
import sys

from lockstep import align
from lockstep.e5_alignment import call_order_table, compare_call_order


def main():
    runs, truth_f = sys.argv[1], sys.argv[2]
    truth = json.load(open(truth_f))
    base = call_order_table(align.load_records(os.path.join(runs, "base")))
    ok = 0
    out = []
    for name, t in sorted(truth.items()):
        tb = call_order_table(align.load_records(os.path.join(runs, name)))
        c = compare_call_order(base, tb)
        found = tuple(c["first_site"]) if c["first_site"] else None
        want = (t["sub"], t["io"], t["slot"])
        ok += found == want
        out.append({"run": name, "truth": want, "found": found, "mismatched": c["mismatched"]})
        print(f"{name:22s} truth={want} call-order found={found} mismatched={c['mismatched']}")
    print(f"call-order exact-site accuracy {ok}/{len(truth)}")
    json.dump({"rows": out, "exact": ok, "n": len(truth)}, open(os.path.join(runs, "e4_baseline_callorder.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
