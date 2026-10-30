"""Identity ablation and version audit (offline; review items E1, E3, E7, E8).

For the same pairs of instances, compare four definitions of an instance's numerical identity:
  config  - the configuration under the default-deny rule only, with every option counted (what a
            config hash can know; close to, but stricter than, vLLM's NIXL compatibility hash);
  static  - config + static kernel identity: the source digest of every generated kernel and the
            lowering structure of every compiled module (no run-time choices);
  v3      - lockstep.guard identity version 3 (config + run-time decision trace), used by the live guard
            in the 2026-10-19 experiments;
  v4      - post-hoc revision (not pre-registered): v3 without the decisions of kernels that never ran
            inside a piecewise subgraph (keys "subNone/..."), i.e. kernels that only run while compiling.
A pair is admitted iff the two identities are equal. Ground truth: bit-identical outputs.

Pair sets
  pd       - prefill/decode deployments (a directory per scenario with inst_a/, inst_b/, local_a.json,
             local_b.json): ground truth = the two instances' own generations agree bit for bit.
  explorer - repeated processes of one configuration (a spec directory with report.json): all pairs within
             the spec; ground truth = same numerical class. Pairs share processes, so they are not independent.

python -m lockstep.ablate_identity pd ROOT [ROOT ...]  |  explorer ROOT [ROOT ...]
"""
import hashlib
import itertools
import json
import os
import sys

from lockstep import decisions, guard


def _cfg(d):
    return json.load(open(os.path.join(d, "cfg.json")))


def _h(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def ident(d, variant):
    if variant == "v3":
        return guard.identity(d, version=3)[0]
    if variant == "budget":  # review baseline: configuration incl. the token budget (+ stack when recorded),
        # without run-time decisions and without the compilation-cache location (a cold/warm difference)
        part = guard.config_part(_cfg(d), False)
        part["env"] = {k: v for k, v in part["env"].items() if k not in guard.TRACE_DEPENDENT_ENV}
        return _h({"cfg": part, "stack": decisions.load_stack(d)})
    if variant == "v6":
        return guard.identity(d, version=6)[0]
    if variant == "v6p":
        return guard.identity(d, version="6p")[0]
    if variant == "v5":
        return guard.identity(d, version=5)[0]
    cfg = _cfg(d)
    dec = decisions.load(d)
    if variant == "config":
        return _h({"cfg": guard.config_part(cfg, False)})
    if variant == "static":
        src = sorted((k, v.get("src")) for k, v in dec.items() if not k.startswith("__runtime__"))
        try:
            st = decisions.load_structure(d)
        except Exception:
            st = {}
        lowering = sorted(json.dumps(v.get("extern", {}), sort_keys=True) for v in (st or {}).values())
        return _h({"cfg": guard.config_part(cfg, False), "src": src, "lowering": lowering})
    if variant == "v4":
        kept = {k: v for k, v in dec.items() if not k.startswith("subNone/")}
        items = sorted(("run:" + k, json.dumps(v.get("chosen"), sort_keys=True)) for k, v in kept.items())
        traced = any(not k.startswith("__runtime__") for k in kept)
        return hashlib.sha256(json.dumps({"cfg": guard.config_part(cfg, traced), "kernels": items},
                                         sort_keys=True, default=str).encode()).hexdigest()[:16]
    raise ValueError(variant)


VARIANTS = ("config", "budget", "static", "v3", "v4", "v5", "v6")


def _cm(pairs):
    """pairs: list of (same_numerics, {variant: admitted}) -> per-variant confusion counts."""
    out = {}
    for v in VARIANTS:
        c = {"admit_same": 0, "UNSAFE_admit_diff": 0, "block_same": 0, "block_diff": 0}
        for same, adm in pairs:
            key = ("admit_same" if same else "UNSAFE_admit_diff") if adm[v] else ("block_same" if same else "block_diff")
            c[key] += 1
        out[v] = c
    return out


def pd_sets(roots):
    rows = []
    key = lambda r: (r["token_ids"], r["logprobs"])
    for root in roots:
        for sc in sorted(os.listdir(root)):
            d = os.path.join(root, sc)
            need = ("local_a.json", "local_b.json", "inst_a", "inst_b")
            if not all(os.path.exists(os.path.join(d, n)) for n in need):
                continue
            a = json.load(open(os.path.join(d, "local_a.json")))
            b = json.load(open(os.path.join(d, "local_b.json")))
            same = all(key(x) == key(y) for x, y in zip(a, b))
            ids = {v: (ident(os.path.join(d, "inst_a"), v), ident(os.path.join(d, "inst_b"), v)) for v in VARIANTS}
            live = None
            lg = os.path.join(d, "inst_b", "capture", "live_guard.jsonl")
            if os.path.exists(lg):  # E8: identity the live guard computed at handshake time
                ev = [json.loads(l) for l in open(lg)]
                live = sorted({e["identity"] for e in ev if e.get("event") in ("check", "serve") and "identity" in e})
            rows.append({"set": os.path.basename(root.rstrip("/")), "scenario": sc, "same": same,
                         "admit": {v: ids[v][0] == ids[v][1] for v in VARIANTS},
                         "v3_b_offline": ids["v3"][1], "v3_b_live": live})
    return rows


def explorer_sets(roots):
    rows = []
    for root in roots:
        for spec in sorted(os.listdir(root)):
            rep = os.path.join(root, spec, "report.json")
            if not os.path.exists(rep):
                continue
            cls = {n: i for i, c in enumerate(json.load(open(rep)).get("classes", [])) for n in c}
            runs = [n for n in cls if os.path.exists(os.path.join(root, spec, n, "cfg.json"))]
            ids = {n: {v: ident(os.path.join(root, spec, n), v) for v in VARIANTS} for n in runs}
            for x, y in itertools.combinations(sorted(runs), 2):
                rows.append({"set": spec, "pair": (x, y), "same": cls[x] == cls[y],
                             "admit": {v: ids[x][v] == ids[y][v] for v in VARIANTS}})
    return rows


def main():
    kind, roots = sys.argv[1], sys.argv[2:]
    rows = pd_sets(roots) if kind == "pd" else explorer_sets(roots)
    by = {}
    for r in rows:
        by.setdefault(r["set"], []).append((r["same"], r["admit"]))
    out = {"kind": kind, "pairs": len(rows), "per_set": {s: {"pairs": len(p), "divergent": sum(not x for x, _ in p),
                                                              "confusion": _cm(p)} for s, p in by.items()},
           "pooled": _cm([(r["same"], r["admit"]) for r in rows])}
    if kind == "pd":
        aud = [r for r in rows if r["v3_b_live"] is not None]
        out["identity_audit"] = {"scenarios_with_live_guard": len(aud),
                                 "live_equals_offline": sum(1 for r in aud if r["v3_b_offline"] in r["v3_b_live"]),
                                 "mismatch": [r["scenario"] for r in aud if r["v3_b_offline"] not in r["v3_b_live"]]}
        out["scenarios"] = [{k: r[k] for k in ("set", "scenario", "same", "admit")} for r in rows]
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
