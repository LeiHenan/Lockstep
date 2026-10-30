"""Would adding the lowering structure to identity version 6 change any verdict? (review round 6)

python -m lockstep.lowering_offline ROOT [ROOT ...]

v6 hashes, per subgraph, the Triton kernels that ran (name-normalized source digest + launch configuration),
runtime choices, configuration, and stack. It does not hash the lowering structure that the tracer also
records (lockstep.decisions structure_*.json: per compiled module, the calls into vendor libraries and the
number of Triton kernels). Variant v6l = v6 + the multiset of per-module lowering records. Ground truth as in
baselines_offline (KV unsafe if the prefills differ on any prompt).
"""
import glob
import hashlib
import json
import os
import sys

from lockstep import ablate_identity, decisions


def lowering(inst):
    try:
        st = decisions.load_structure(inst) or {}
    except Exception:
        return None
    return sorted(json.dumps({"extern": v.get("extern", {}), "n_triton": v.get("n_triton"),
                              "user": sorted(v.get("user_kernels", []))}, sort_keys=True) for v in st.values())


def main(roots):
    c = {"deployments": 0, "verdict_changes": 0, "v6_admits_lowering_differs": [], "unsafe_v6": 0, "unsafe_v6l": 0,
         "unneeded_v6": 0, "unneeded_v6l": 0}
    for root in roots:
        for f in sorted(glob.glob(os.path.join(root, "**", "local_b.json"), recursive=True)):
            d = os.path.dirname(f)
            if "failed_attempts" in d or not glob.glob(os.path.join(d, "inst_a", "capture", "usage_*.json")):
                continue
            a, b = json.load(open(os.path.join(d, "local_a.json"))), json.load(open(os.path.join(d, "local_b.json")))
            unsafe = any((x["logprobs"] or [None])[0] != (y["logprobs"] or [None])[0] for x, y in zip(a, b))
            ia, ib = os.path.join(d, "inst_a"), os.path.join(d, "inst_b")
            v6 = ablate_identity.ident(ia, "v6") == ablate_identity.ident(ib, "v6")
            la, lb = lowering(ia), lowering(ib)
            v6l = v6 and la == lb
            c["deployments"] += 1
            c["verdict_changes"] += v6 != v6l
            if v6 and la != lb:
                c["v6_admits_lowering_differs"].append(d)
            c["unsafe_v6"] += v6 and unsafe
            c["unsafe_v6l"] += v6l and unsafe
            c["unneeded_v6"] += (not v6) and not unsafe
            c["unneeded_v6l"] += (not v6l) and not unsafe
    print(json.dumps(c, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
