"""Probe P1: host<->GPU bandwidth contention on a PCIe-only multi-GPU server.

Measures pinned-memory H2D/D2H bandwidth for one GPU alone, then with N GPUs
copying concurrently, with the host buffer placed on the local vs remote NUMA
node (via numactl when available, else default allocation). Each GPU runs in
its own process so copies are truly concurrent.

Usage: python pcie_contention.py --gpus 0,1,2 --size-mb 1024 --iters 20 --out res.json
"""
import argparse
import json
import multiprocessing as mp
import os
import time

import torch


def worker(gpu, size_mb, iters, direction, barrier, q):
    torch.cuda.set_device(gpu)
    n = size_mb * 1024 * 1024
    host = torch.empty(n, dtype=torch.uint8, pin_memory=True)
    dev = torch.empty(n, dtype=torch.uint8, device=f"cuda:{gpu}")
    stream = torch.cuda.Stream()
    for _ in range(3):  # warmup
        with torch.cuda.stream(stream):
            (dev.copy_(host, non_blocking=True) if direction == "h2d" else host.copy_(dev, non_blocking=True))
    stream.synchronize()
    barrier.wait()
    t0 = time.perf_counter()
    with torch.cuda.stream(stream):
        for _ in range(iters):
            (dev.copy_(host, non_blocking=True) if direction == "h2d" else host.copy_(dev, non_blocking=True))
    stream.synchronize()
    dt = time.perf_counter() - t0
    q.put((gpu, n * iters / dt / 1e9))


def run(gpus, size_mb, iters, direction):
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(len(gpus))
    q = ctx.Queue()
    ps = [ctx.Process(target=worker, args=(g, size_mb, iters, direction, barrier, q)) for g in gpus]
    for p in ps:
        p.start()
    res = dict(q.get() for _ in gpus)
    for p in ps:
        p.join()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2")
    ap.add_argument("--size-mb", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    gpus = [int(x) for x in a.gpus.split(",")]
    out = {"gpus": gpus, "size_mb": a.size_mb, "iters": a.iters, "numa_node_env": os.environ.get("PROBE_NUMA", "default"), "runs": []}
    for direction in ("h2d", "d2h"):
        for g in gpus:  # each GPU alone
            out["runs"].append({"dir": direction, "set": [g], "gbps": run([g], a.size_mb, a.iters, direction)})
        for k in range(2, len(gpus) + 1):  # growing concurrent sets
            s = gpus[:k]
            out["runs"].append({"dir": direction, "set": s, "gbps": run(s, a.size_mb, a.iters, direction)})
    for r in out["runs"]:
        tot = sum(r["gbps"].values())
        print(f'{r["dir"]} set={r["set"]} per-GPU={ {k: round(v, 2) for k, v in r["gbps"].items()} } total={tot:.2f} GB/s')
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
