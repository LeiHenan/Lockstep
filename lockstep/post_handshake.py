"""Did any instance record a new kernel usage after its first handshake? (review round 6)

python -m lockstep.post_handshake ROOT [ROOT ...]

The live guard logs n_records (usage records at that moment) with every check/serve event. A usage record
added later, e.g. a kernel first run or tuned after the handshake, would not be covered by the identity
that the handshake compared. Compares the n_records at the first handshake event with the final number of
usage records that the instance flushed at exit.
"""
import glob
import json
import os
import sys


def main(roots):
    n, later, rows = 0, 0, []
    for root in roots:
        for lg in sorted(glob.glob(os.path.join(root, "**", "inst_*", "capture", "live_guard.jsonl"), recursive=True)):
            if "failed_attempts" in lg:
                continue
            cap = os.path.dirname(lg)
            ev = [json.loads(l) for l in open(lg)]
            first = next((e for e in ev if e.get("event") in ("check", "serve") and "n_records" in e), None)
            uf = glob.glob(os.path.join(cap, "usage_*.json"))
            if first is None or not uf:
                continue
            final = max(len(json.load(open(f))) for f in uf)
            n += 1
            if final > first["n_records"]:
                later += 1
                rows.append({"capture": cap, "at_handshake": first["n_records"], "final": final})
    print(json.dumps({"instances": n, "with_records_after_handshake": later, "examples": rows[:20]}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
