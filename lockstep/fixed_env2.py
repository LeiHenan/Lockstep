"""One-cohort baseline comparison: both instances with the flag on and traces, natural deployments only (the FC
injection tests are planted failures and excluded). "without_rule_design_runs" also leaves out the runs on which
the pointwise rule was found and first checked (hb_sk, hb2)."""
import json, os, sys
from lockstep import offline_r13 as o

import argparse

_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument("roots", nargs="+", help="capture directories to analyze")
roots = _parser.parse_args().roots


def run_of(d):
    return d.replace("\\", "/").split("/")[-2]


rows = [r for r in o.o2_rows(roots) if run_of(r["dir"]) != "fc"]


def bud(c):
    args = c.get("engine_kwargs", {}).get("args", []) or []
    return args[args.index("--max-num-batched-tokens") + 1] if "--max-num-batched-tokens" in args else None


def budget_differs(d):
    a = json.load(open(os.path.join(d, "inst_a", "cfg.json"))); b = json.load(open(os.path.join(d, "inst_b", "cfg.json")))
    return bud(a) != bud(b)


out = {}
sels = {"all_traced": lambda r: r["usage"],
        "fixed": lambda r: r["usage"] and r["flag"] == "both_on",
        "fixed_without_rule_design_runs": lambda r: r["usage"] and r["flag"] == "both_on" and run_of(r["dir"]) not in ("hb_sk", "hb2")}
for name, sel in sels.items():
    rs = [r for r in rows if sel(r) and not r["planted"]]
    res = {"deployments": len(rs), "kv_unsafe": sum(r["unsafe"] for r in rs), "identical": sum(not r["unsafe"] for r in rs)}
    for pol in ("static", "static_flag_enforced", "static_budget_excluded_flag_enforced", "live_identity", "live_final", "v6p"):
        adm = [r for r in rs if r["admit"][pol]]
        blk = [r for r in rs if not r["admit"][pol] and not r["unsafe"]]
        res[pol] = {"unsafe_admits": sum(r["unsafe"] for r in adm), "unneeded_blocks": len(blk),
                    "unneeded_with_budget_diff": sum(budget_differs(r["dir"]) for r in blk),
                    "unneeded_runs": sorted({run_of(r["dir"]) for r in blk})}
    out[name] = res
json.dump(out, sys.stdout, indent=1)
