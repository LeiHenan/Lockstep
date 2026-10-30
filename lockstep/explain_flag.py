"""Why did heterogeneous A800 pairs diverge with the batch_invariant flag set by hand on vLLM 0.31, but not on
vLLM main? (review round 9, item B). Offline, from the decision traces.

python -m lockstep.explain_flag RUN_DIR [RUN_DIR ...] > explain_flag.json

Per deployment (inst_a = prefill, inst_b = decode): whether the two instances computed identically; how many
kernel decisions were chosen by timing ("benchmarked") in each process; and, for the per-subgraph usage
records (identity version 6), which (source digest, launch configuration) entries exist on only one side,
grouped by kernel name and by what differs (only the launch configuration, or the source itself).
"""
import collections
import glob
import json
import os
import sys

from lockstep import decisions


def _same(d):
    a, b = json.load(open(os.path.join(d, "local_a.json"))), json.load(open(os.path.join(d, "local_b.json")))
    return a == b


def _use(inst):
    u = decisions.load_usage(inst)
    by = collections.Counter()
    meta = {}
    for v in u.values():
        k = (int(v["subgraph"]), v.get("srcn"), json.dumps(v.get("chosen"), sort_keys=True))
        by[k] += 1
        meta[k] = v.get("kernel")
    return by, meta


def _bench(inst):
    dec = decisions.load(inst)
    return sum(1 for v in dec.values() if v.get("benchmarked")), sum(1 for v in dec.values() if (v.get("candidates") or 0) > 1)


def main(roots):
    out = []
    for root in roots:
        for f in sorted(glob.glob(os.path.join(root, "*", "local_b.json"))):
            d = os.path.dirname(f)
            ia, ib = os.path.join(d, "inst_a"), os.path.join(d, "inst_b")
            try:
                ua, ma = _use(ia)
                ub, mb = _use(ib)
            except Exception as e:
                out.append({"dir": d, "error": repr(e)})
                continue
            only_a, only_b = ua - ub, ub - ua
            # classify one-sided entries: same (subgraph, source) on the other side with another config = config-only
            src_b = {(s, src) for (s, src, _) in ub}
            src_a = {(s, src) for (s, src, _) in ua}
            kinds = collections.Counter()
            names = collections.Counter()
            for (s, src, ch), n in only_a.items():
                kinds["config_only" if (s, src) in src_b else "source_only_on_a"] += n
                names[ma[(s, src, ch)]] += n
            for (s, src, ch), n in only_b.items():
                kinds["config_only" if (s, src) in src_a else "source_only_on_b"] += n
                names[mb[(s, src, ch)]] += n
            out.append({"dir": d, "same": _same(d), "bench_a": _bench(ia), "bench_b": _bench(ib),
                        "only_a": sum(only_a.values()), "only_b": sum(only_b.values()), "kinds": dict(kinds),
                        "kernels": dict(names.most_common(6))})
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
