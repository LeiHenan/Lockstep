"""E4: localization accuracy against planted faults.

python -m lockstep.e4_accuracy RUNS_DIR TRUTH_JSON [--base base] [--control serial_noinj]
TRUTH_JSON maps run name -> {"sub": k, "io": "out", "slot": s}. Each injected run is compared with
the un-injected base; the causal first-divergence site must equal the planted site.
"""
import argparse
import json
import os

from lockstep.attribute import first_divergence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("truth")
    ap.add_argument("--base", default="base")
    ap.add_argument("--control", default="serial_noinj")
    a = ap.parse_args()
    truth = json.load(open(a.truth))
    rows, exact, subok = [], 0, 0
    for name, t in sorted(truth.items()):
        if not os.path.exists(os.path.join(a.runs, name, "outputs.json")):
            print(f"{name:22s} MISSING")
            continue
        s = first_divergence(os.path.join(a.runs, a.base), os.path.join(a.runs, name))
        found = None if s is None else (s["subgraph"], s["io"], s["slot"])
        want = (t["sub"], t["io"], t["slot"])
        e = found == want
        so = s is not None and s["subgraph"] == t["sub"]
        exact += e
        subok += so
        rows.append({"run": name, "truth": want, "found": found, "rows": None if s is None else s["rows_differing"], "exact": e})
        print(f"{name:22s} truth={want} found={found} rows={None if s is None else s['rows_differing']} exact={e}")
    ctrl = first_divergence(os.path.join(a.runs, a.base), os.path.join(a.runs, a.control)) \
        if os.path.exists(os.path.join(a.runs, a.control, "outputs.json")) else "missing"
    print("control (no injection):", ctrl)
    print(f"exact-site accuracy {exact}/{len(rows)}; subgraph accuracy {subok}/{len(rows)}")
    json.dump({"rows": rows, "exact": exact, "subgraph": subok, "n": len(rows), "control": ctrl},
              open(os.path.join(a.runs, "e4_accuracy.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
