"""Concurrent load test of reuse policies on two GPUs.

Policies (both GPUs used in every policy):
  pd           - prefill instance on GPU0, decode instance on GPU1, vLLM's own compatibility check only;
  pd_guard     - same, with the in-connector guard; both instances replay one decision file (admitted);
  pd_block     - same, but the decode instance replays a decision file with one planted change, so the
                 guard blocks and the decode instance recomputes every prefill (the cost of a block);
  colo2        - two colocated instances, one per GPU, requests round-robin (no KV reuse at all).
Requests stream their output, so time to first token (TTFT) and time per output token (TPOT) are
measured at the client. In the PD policies, a request first runs a 1-token prefill on the prefill
instance, then streams from the decode instance; TTFT covers both steps.

python -m lockstep.loadgen --model MODEL --prompts prompts.json --policy pd --concurrency 8 --n 96 --out DIR [--replay F] [--plant F]
"""
import argparse
import concurrent.futures as cf
import json
import os
import statistics as st
import sys
import threading
import time
import urllib.request

from lockstep.pd_harness import HOSTS, _url, start_server, wait_ready

def long_prompts(n, chars, path):
    """Distinct long prompts: consecutive UltraChat conversations joined until `chars` characters."""
    src = json.load(open(path))
    if not isinstance(src, list) or not src or not all(isinstance(p, str) and p for p in src):
        raise ValueError("prompts must be a nonempty JSON list of nonempty strings")
    out, i = [], 0
    while len(out) < n:
        parts, size, j = [], 0, i
        while size < chars:
            p = src[j % len(src)]
            parts.append(p)
            size += len(p)
            j += 1
        out.append(f"[request {len(out)}]\n" + "\n\n".join(parts))
        i += 1
    return out


def _post(port, payload, timeout=900):
    req = urllib.request.Request(_url(port, "/v1/completions"), data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def _stream(port, payload):
    """Returns (first-token time, token times, text) for a streamed completion."""
    payload = dict(payload, stream=True)
    t_first, times, text = None, [], []
    with _post(port, payload) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if not ev.get("choices"):
                continue
            now = time.perf_counter()
            if t_first is None:
                t_first = now
            times.append(now)
            text.append(ev["choices"][0].get("text", ""))
    return t_first, times, "".join(text)


def one_request(policy, prompt, max_tokens, rr):
    base = {"model": "m", "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0, "ignore_eos": True, "seed": 0}
    t0 = time.perf_counter()
    if policy == "colo2":
        port = 8100 if next(rr) % 2 == 0 else 8200
        t_first, times, _ = _stream(port, base)
    else:
        with _post(8100, dict(base, max_tokens=1, stream=False,
                             kv_transfer_params={"do_remote_decode": True, "do_remote_prefill": False,
                                                 "remote_engine_id": None, "remote_block_ids": None,
                                                 "remote_host": None, "remote_port": None})) as r:
            kvp = json.loads(r.read()).get("kv_transfer_params")
        t_first, times, _ = _stream(8200, dict(base, kv_transfer_params=kvp))
    t_end = time.perf_counter()
    tpot = (times[-1] - times[0]) / (len(times) - 1) if len(times) > 1 else None
    return {"ttft_s": t_first - t0 if t_first else None, "tpot_s": tpot, "e2e_s": t_end - t0, "tokens": len(times)}


def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local model snapshot or model identifier")
    ap.add_argument("--prompts", required=True, help="JSON list of source prompt strings")
    ap.add_argument("--policy", choices=["pd", "pd_guard", "pd_block", "colo2"], required=True)
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--chars", type=int, default=10000, help="prompt length in characters (~2.5K tokens)")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--budget", type=int, default=2048)
    ap.add_argument("--replay", default=None, help="decision file both instances replay")
    ap.add_argument("--plant", default=None, help="pd_block: decision file of the decode instance")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ipc", action="store_true", help="PD instances see both GPUs (CUDA IPC KV transfer)")
    ap.add_argument("--a-host", default=None, help="instance a (prefill, or colo2's first) already runs on this "
                    "host (started there with pd_harness --serve-only); start only instance b here")
    ap.add_argument("--side-host", default=None, help="NIXL side-channel host of instance a (its own address, "
                    "needed when the decode instance runs on another machine)")
    ap.add_argument("--b-args", default=None, help="extra vllm serve args (JSON list) for instance b only, e.g. an "
                    "option that changes the guard identity but not the numerics, to force a block")
    ap.add_argument("--compile-config", default=None, help="extra --compilation-config JSON for every instance")
    ap.add_argument("--dump-spec", action="store_true", help="print instance a's spec as JSON and exit")
    ap.add_argument("--kv-buffer", default=None, help='PD: NIXL kv_buffer_device ("cpu" stages KV through host memory)')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    args = ["--max-num-batched-tokens", str(a.budget), "--no-enable-prefix-caching", "--served-model-name", "m"]
    if a.compile_config:
        args += ["--compilation-config", a.compile_config]
    kv = a.policy != "colo2"

    def spec(gpu, force, guard):
        s = {"gpus": gpu, "mem": 0.80, "args": list(args), "env": {}}
        if kv and a.ipc:  # let the two PD instances see each other's GPU (CUDA IPC for KV transfers)
            s["visible"] = "0,1" if gpu == "0" else "1,0"
        if kv and a.kv_buffer:
            s["kv_buffer"] = a.kv_buffer
        if force:
            s["env"]["LOCKSTEP_FORCE"] = force
        if guard:
            s["env"]["LOCKSTEP_LIVE_GUARD"] = "1"
            s["kv_policy"] = "recompute"
        return s
    guard = a.policy in ("pd_guard", "pd_block")
    kv = a.policy != "colo2"
    if not kv:
        pass
    sa = spec("0", a.replay, guard)
    if a.side_host:
        sa["env"]["VLLM_NIXL_SIDE_CHANNEL_HOST"] = a.side_host
    if a.dump_spec:
        print(json.dumps(sa))
        return
    sb = spec("1", a.plant if a.policy == "pd_block" else a.replay, guard)
    if a.b_args:
        sb["args"] = sb["args"] + json.loads(a.b_args)
    procs = []
    try:
        if a.a_host:
            HOSTS[8100] = a.a_host
        else:
            procs.append(start_server(a.model, 8100, sa, "prefill" if kv else "single", 5600, a.out, "a", kv))
        procs.append(start_server(a.model, 8200, sb, "decode" if kv else "single", 5601, a.out, "b", kv))
        wait_ready(8100, None if a.a_host else procs[0])
        wait_ready(8200, procs[-1])
        prompts = long_prompts(a.n + 4, a.chars, a.prompts)
        counter = iter(range(10 ** 9))
        lock = threading.Lock()

        def rr():
            with lock:
                return next(counter)
        rr_iter = iter(rr, None)
        for p in prompts[:4]:  # warm-up (and, for the PD policies, the handshake)
            one_request(a.policy, p, 16, rr_iter)
        t0 = time.perf_counter()
        with cf.ThreadPoolExecutor(max_workers=a.concurrency) as ex:
            res = list(ex.map(lambda p: one_request(a.policy, p, a.max_tokens, rr_iter), prompts[4:]))
        wall = time.perf_counter() - t0
        toks = sum(r["tokens"] for r in res)
        summ = {"policy": a.policy, "concurrency": a.concurrency, "n": a.n, "kv_buffer": a.kv_buffer,
                "a_host": a.a_host, "wall_s": wall,
                "throughput_req_s": a.n / wall, "throughput_tok_s": toks / wall,
                "ttft_p50": pct([r["ttft_s"] for r in res], .5), "ttft_p99": pct([r["ttft_s"] for r in res], .99),
                "tpot_p50": pct([r["tpot_s"] for r in res], .5), "tpot_p99": pct([r["tpot_s"] for r in res], .99),
                "e2e_p50": pct([r["e2e_s"] for r in res], .5), "e2e_p99": pct([r["e2e_s"] for r in res], .99),
                "short_outputs": sum(1 for r in res if r["tokens"] < a.max_tokens * 0.9)}
        json.dump({"summary": summ, "requests": res}, open(os.path.join(a.out, "load.json"), "w"), indent=1)
        print("LOAD_DONE", json.dumps(summ))
    finally:
        import signal
        for p in procs:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        for p in procs:
            try:
                p.wait(timeout=60)
            except Exception:
                pass


if __name__ == "__main__":
    main()
