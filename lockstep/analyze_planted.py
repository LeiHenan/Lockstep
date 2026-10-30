"""Experiment B analysis: for every planted decision, did it take effect, did numerics change, and
did Lockstep's attribution name the planted decision?

python -m lockstep.analyze_planted SPEC_RUN_DIR PLANTS_JSON   (SPEC_RUN_DIR holds base/, plant_XXX/, report.json)
"""
import json
import os
import sys
from collections import Counter

from lockstep import decisions
from lockstep.attribute import attribute


def family(kernel):
    return kernel.split("#")[0]


def main(run_dir, plants_path):
    plants = json.load(open(plants_path))
    rep = json.load(open(os.path.join(run_dir, "report.json")))
    cls_of = {n: i for i, c in enumerate(rep["classes"]) for n in c}
    base_cls = cls_of.get("base")
    replays = [n for n in ("base", "base_rep", "replay_r1", "replay_r2") if n in cls_of]
    out = {"replay_processes": len(replays), "replay_classes": len({cls_of[n] for n in replays}), "plants": []}
    for p in plants:
        name = f"plant_{p['i']:03d}"
        d = os.path.join(run_dir, name)
        if name not in cls_of:
            out["plants"].append({"name": name, "status": "missing"})
            continue
        dec = decisions.load(d)
        rec = dec.get(p["key"], {})
        applied = bool((rec.get("forced") or {}).get("applied")) and str(rec.get("chosen")) == p["to"]
        others = [k for k, v in dec.items() if v.get("forced") and k != p["key"] and not v["forced"].get("applied")]
        changed = cls_of[name] != base_cls
        row = {"name": name, "key": p["key"], "family": family(p["kernel"]), "subgraph": p["subgraph"],
               "applied": applied, "other_forcings_not_applied": len(others), "numerics_changed": changed}
        if changed:
            a = attribute(os.path.join(run_dir, "base"), d)
            site = a.get("first_divergence") or {}
            producer = None if not site else (site["subgraph"] if site["io"] == "out" else site["subgraph"] - 1)
            sus = (a.get("suspects") or [{}])[0]
            row.update({"first_divergence": site, "producer": producer, "verdict": a.get("verdict"),
                        "attributed_key": sus.get("kernel"),
                        "attribution_exact": sus.get("kernel") == p["key"] and str(a.get("verdict", "")).startswith("root cause"),
                        "differing_decisions": a.get("differing_decisions"),
                        "first_divergence_in_planted_subgraph": producer == p["subgraph"]})
        out["plants"].append(row)
    ok = [r for r in out["plants"] if r.get("applied")]
    ch = [r for r in ok if r.get("numerics_changed")]
    out["summary"] = {"planted": len(plants), "applied": len(ok), "changed_numerics": len(ch),
                      "attributed_exactly": sum(bool(r.get("attribution_exact")) for r in ch),
                      "first_div_in_planted_subgraph": sum(bool(r.get("first_divergence_in_planted_subgraph")) for r in ch),
                      "families_changed": dict(Counter(r["family"] for r in ch)),
                      "families_unchanged": dict(Counter(r["family"] for r in ok if not r.get("numerics_changed")))}
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
