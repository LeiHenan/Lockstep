"""Clustered counts restricted to deployments guarded by identity version 7 (offline_r13.classify_version)."""
import json
from lockstep import cluster_stats as c, offline_r13 as o

import argparse

_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument("roots", nargs="+", help="capture directories to analyze")
roots = _parser.parse_args().roots
rs = [r for r in c.rows(roots) if o.classify_version(r["dir"])[0] == 7]
nat = [r for r in rs if r["unsafe"] and not r["planted"]]
out = {"deployments": len(rs), "kv_unsafe": sum(r["unsafe"] for r in rs), "natural_kv_unsafe": len(nat),
       "planted_kv_unsafe": sum(r["unsafe"] and r["planted"] for r in rs),
       "natural_unsafe_admits": sum(r["admitted"] for r in nat), "per_deployment_bound": c._bound(len(nat))}
for g in ("cell", "family"):
    groups = sorted({r[g] for r in nat})
    out[g] = {"groups": len(groups), "bound": c._bound(len(groups))}
print(json.dumps(out, indent=1))
