"""SK (results_prereg/HB_SK_PREREGISTRATION.md): does a serving process make kernel decisions after warm-up under
long, varied traffic? One instance, mixed traffic for --minutes; the per-subgraph usage records (the identity's
kernel part) are counted from the capture every 20 s.

python -m lockstep.soak --model M --spec '{...}' --prompts P.json [P2.json ...] --minutes 30 --out DIR
"""
import argparse
import glob
import json
import os
import random
import signal
import sys
import threading
import time

from lockstep import pd_harness as H


def _records(cap):
    n = 0
    for f in glob.glob(os.path.join(cap, "usage_*.json")):
        try:
            n = max(n, len(json.load(open(f))))
        except Exception:
            pass
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--prompts", nargs="+", required=True)
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    prompts = [p for f in a.prompts for p in json.load(open(f))]
    rng = random.Random(20261023)
    # also prompts longer than any in the files, up to the context limit, by repeating text
    prompts += [" ".join(rng.sample(prompts, 3))[: rng.randint(8000, 20000)] for _ in range(32)]
    sp = json.loads(a.spec)
    sp.setdefault("args", []).extend(["--served-model-name", "m"])
    sp.setdefault("env", {})["LOCKSTEP_CAPTURE"] = "1"
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    proc = H.start_server(a.model, a.port, sp, "single", a.port - 2500, a.out, "a", False)
    cap = os.path.join(a.out, "inst_a", "capture")
    timeline, errors, done = [], [0], threading.Event()
    try:
        H.wait_ready(a.port, proc)
        t0 = time.time()
        timeline.append({"t": 0.0, "records": _records(cap), "phase": "ready"})

        def client(i):
            r = random.Random(1000 + i)
            while not done.is_set():
                p = r.choice(prompts)
                extra = {"cache_salt": f"soak-{i}-{r.random()}"} if r.random() < 0.5 else {}
                if r.random() < 0.3:
                    extra.update({"temperature": 1.0})
                try:
                    H.complete(a.port, p, r.choice([1, 8, 32, 128, 256]), extra)
                except Exception:
                    errors[0] += 1
        n_clients = 24
        threads = [threading.Thread(target=client, args=(i,), daemon=True) for i in range(n_clients)]
        active = 0
        while time.time() - t0 < 60 * a.minutes:
            # vary the load: start clients gradually, then keep a varying number busy via bursts
            while active < min(n_clients, 1 + int((time.time() - t0) / 30)):
                threads[active].start()
                active += 1
            time.sleep(20)
            timeline.append({"t": time.time() - t0, "records": _records(cap), "clients": active})
        done.set()
        time.sleep(5)
        timeline.append({"t": time.time() - t0, "records": _records(cap), "phase": "end"})
        rec0 = timeline[0]["records"]
        json.dump({"timeline": timeline, "records_at_ready": rec0, "records_at_end": timeline[-1]["records"],
                   "new_records_after_ready": timeline[-1]["records"] - rec0, "client_errors": errors[0],
                   "spec": json.loads(a.spec), "minutes": a.minutes}, open(os.path.join(a.out, "soak.json"), "w"), indent=1)
        print("SOAK_DONE", a.out)
    finally:
        done.set()
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=60)
        except Exception:
            pass
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass


if __name__ == "__main__":
    main()
