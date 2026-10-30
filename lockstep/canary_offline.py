"""Offline evaluation of a canary baseline for the guard (review item: "why not compare a canary prompt?").

python -m lockstep.canary_offline ROOT [ROOT ...] > canary_report.json

A canary check prefills one fixed prompt on both instances at the handshake and compares the result bit for
bit. Every live deployment already holds each instance's own generation of the same 32 prompts (local_a.json,
local_b.json: token ids and per-token log-probabilities), so a canary built from prompt k is simulated by
comparing prompt k only. Two variants: "prefill" compares the first generated token's log-probability (one
prefill per side, what a handshake-time canary costs); "gen" compares the whole 32-token generation.
Ground truth: the two instances differ on any prompt (token ids or log-probabilities).
"""
import glob
import json
import os
import statistics as st
import sys


def deployments(roots):
    for root in roots:
        for d in sorted(glob.glob(os.path.join(root, "**", "local_b.json"), recursive=True)):
            d = os.path.dirname(d)
            if "failed_attempts" in d or "plant_not_applied" in d:
                continue
            if os.path.exists(os.path.join(d, "local_a.json")):
                yield d


def main(roots):
    rows = []
    for d in deployments(roots):
        a = json.load(open(os.path.join(d, "local_a.json")))
        b = json.load(open(os.path.join(d, "local_b.json")))
        n = min(len(a), len(b))
        gen = [a[k]["token_ids"] != b[k]["token_ids"] or a[k]["logprobs"] != b[k]["logprobs"] for k in range(n)]
        pre = [(a[k]["logprobs"] or [None])[0] != (b[k]["logprobs"] or [None])[0] for k in range(n)]
        if not any(gen):
            continue
        rows.append({"dir": d, "n": n, "differing_prompts_gen": sum(gen), "differing_prompts_prefill": sum(pre),
                     "gen": gen, "pre": pre})
    out = {"divergent_deployments": len(rows)}
    for var in ("pre", "gen"):
        per_k = [sum(r[var][k] for r in rows if k < r["n"]) for k in range(32)]
        frac = [x / len(rows) for x in per_k] if rows else []
        out[var] = {"canary_prompt_detects_fraction": {"min": min(frac), "median": st.median(frac), "max": max(frac)},
                    "best_prompt": per_k.index(max(per_k)), "worst_prompt": per_k.index(min(per_k)),
                    "deployments_missed_by_every_single_prompt": sum(1 for r in rows if not any(r[var])),
                    "deployments_caught_by_prompt0": sum(1 for r in rows if r[var][0])}
    out["deployments"] = [{k: r[k] for k in ("dir", "n", "differing_prompts_gen", "differing_prompts_prefill")}
                          for r in rows]
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
