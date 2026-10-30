"""Screen phase 1 -> predictions (same rule as E7d/E24: tau = 5%, exact ties count as risk).
Usage: python predict_screen.py SCREEN_ROOT OUT_JSON   (one traced base process per spec dir)"""
import datetime, json, os, sys
from lockstep import decisions as D
from lockstep import guard as G

root, out = sys.argv[1], sys.argv[2]
pred = {}
for spec in sorted(os.listdir(root)):
    base = os.path.join(root, spec, "base")
    if not os.path.exists(os.path.join(base, "outputs.json")):
        continue
    risky, nb = D.risk(D.load(base), 0.05)
    pred[spec] = {"benchmarked": nb, "near_ties": len(risky), "exact_ties": sum(1 for r in risky if r["margin"] == 0),
                  "min_margin": risky[0]["margin"] if risky else None,
                  "prediction": "UNSTABLE" if risky else "STABLE", "identity": G.identity(base)[0]}
json.dump({"made_at": datetime.datetime.utcnow().isoformat() + "Z", "tau": 0.05, "platform": "H800 PCIe",
           "source": "base run only (single cold process)", "predictions": pred}, open(out, "w"), indent=1)
print(json.dumps(pred, indent=1))
