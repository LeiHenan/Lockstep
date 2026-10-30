"""How much does a canary's accuracy depend on which prompt it uses? (review round 8)

python -m lockstep.canary_choice ROOT [ROOT ...] [--groups META.json] > canary_choice.json

For every KV-unsafe deployment (prefills differ on some prompt), the share of prompts whose prefill
differs; a single-prompt canary chosen uniformly from the prompts misses the deployment with probability
1 - that share. With --groups (the G7 prompt groups), the same per group, for deployments that ran
those prompts.
"""
import glob
import json
import os
import sys


def main(argv):
    groups = None
    if "--groups" in argv:
        k = argv.index("--groups")
        groups = json.load(open(argv[k + 1]))["groups"]
        argv = argv[:k] + argv[k + 2:]
    rows = []
    for root in argv:
        for f in sorted(glob.glob(os.path.join(root, "**", "local_b.json"), recursive=True)):
            d = os.path.dirname(f)
            if "failed_attempts" in d or not os.path.exists(os.path.join(d, "local_a.json")):
                continue
            a, b = json.load(open(os.path.join(d, "local_a.json"))), json.load(open(os.path.join(d, "local_b.json")))
            pre = [(x["logprobs"] or [None])[0] != (y["logprobs"] or [None])[0] for x, y in zip(a, b)]
            if not any(pre):
                continue
            r = {"dir": d, "n": len(pre), "share": sum(pre) / len(pre)}
            if groups and len(pre) == 56:
                r["groups"] = {g: sum(pre[x:y]) / (y - x) for g, (x, y) in groups.items()}
            rows.append(r)
    out = {"kv_unsafe": len(rows),
           "partial": sum(r["share"] < 1 for r in rows),
           "expected_miss_uniform_prompt": sum(1 - r["share"] for r in rows) / len(rows) if rows else None,
           "deployments_with_share_below_half": sum(r["share"] < 0.5 for r in rows)}
    g56 = [r for r in rows if "groups" in r]
    if g56:
        out["with_56_prompts"] = len(g56)
        out["expected_miss_by_group"] = {g: sum(1 - r["groups"][g] for r in g56) / len(g56) for g in groups}
        out["missed_by_some_prompt_in_group"] = {g: sum(r["groups"][g] < 1 for r in g56) for g in groups}
    out["rows"] = rows
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
