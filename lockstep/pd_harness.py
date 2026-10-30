"""Deployment scenario harness: colocated vs prefill/decode-disaggregated (NIXL) serving.

python -m lockstep.pd_harness --mode single --model MODEL --a '{"gpus":"0","args":["--max-num-batched-tokens","8192"]}' \
       --prompts p.json --out DIR
python -m lockstep.pd_harness --mode pd --model MODEL --a '{prefill spec}' --b '{decode spec}' --prompts p.json --out DIR

Each spec: {"gpus": "0" | "1,2", "args": [extra `vllm serve` args], "env": {...}, "ibi": bool}.
Both modes use the same OpenAI-compatible client and greedy decoding, and save
generated token ids per prompt, so outputs are directly comparable.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

TEMPERATURE = [0.0]  # --temperature; every request carries seed 0, so sampled outputs stay comparable
HOSTS = {}  # port -> host, for a prefill server on another machine (--a-host); default 127.0.0.1


def _url(port, path):
    return f"http://{HOSTS.get(port, '127.0.0.1')}:{port}{path}"


def post(url, payload, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_ready(port, proc, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"server on :{port} exited rc={proc.returncode}")
        try:
            urllib.request.urlopen(_url(port, "/health"), timeout=5)
            time.sleep(2)  # the answering server must be ours, and still alive
            if proc is not None and proc.poll() is not None:
                raise RuntimeError(f"server on :{port} exited rc={proc.returncode}")
            return
        except RuntimeError:
            raise
        except Exception:
            time.sleep(5)
    raise TimeoutError(f"server :{port} not ready")


def _port_in_use(port, host="127.0.0.1"):
    import socket
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex((host, port)) == 0


def start_server(model, port, spec, role, side_port, out_dir, name, kv):
    # another server on this port would answer our requests (found 2026-10-21: orphaned smoke-test servers
    # answered K1/L1 requests after the new servers had failed to bind); never start into an occupied port
    for p in (port, side_port):
        if _port_in_use(p):
            raise RuntimeError(f"port {p} is already in use; refusing to start {name}")
    env = os.environ.copy()
    env.update({k: str(v) for k, v in spec.get("env", {}).items()})
    # spec["gpus"] indexes into the GPUs this harness was given (e.g. by lockstep.gpurun)
    parent = os.environ.get("CUDA_VISIBLE_DEVICES")
    # spec["visible"] (optional): the GPUs the server sees, its own first. A disaggregated pair on one host
    # must see each other's GPU, or UCX cannot use CUDA IPC and stages KV through host memory (found
    # 2026-10-20: ~0.1 GB/s instead of tens of GB/s).
    vis = spec.get("visible", spec["gpus"])
    if parent:
        plist = parent.split(",")
        env["CUDA_VISIBLE_DEVICES"] = ",".join(plist[int(i)] for i in vis.split(","))
    else:
        env["CUDA_VISIBLE_DEVICES"] = vis
    env["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(side_port)
    tp = len(spec["gpus"].split(","))
    if "LOCKSTEP_CAPTURE" in env:
        # layout understood by lockstep.guard.identity(): inst_<name>/{cfg.json, capture/}
        inst = os.path.join(out_dir, f"inst_{name}")
        os.makedirs(inst, exist_ok=True)
        env["LOCKSTEP_DIR"] = os.path.join(inst, "capture")
        env.setdefault("LOCKSTEP_DEC_EAGER_FLUSH", "1")  # servers are killed at the end; atexit may not run
        json.dump({"engine_kwargs": {"model": model, "tensor_parallel_size": tp, "args": spec.get("args", []),
                                     "ibi": bool(spec.get("ibi"))},
                   "env": {k: env.get(k) for k in ("VLLM_BATCH_INVARIANT",)}},
                  open(os.path.join(inst, "cfg.json"), "w"), indent=1)
    cmd = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", model, "--port", str(port),
           "--tensor-parallel-size", str(tp), "--max-model-len", str(spec.get("max_len", 6144)),  # OPT: 2048
           "--gpu-memory-utilization", str(spec.get("mem", 0.78)),
           "--seed", "0"] + spec.get("args", [])
    if spec.get("ibi"):
        cmd += ["--compilation-config", json.dumps({"inductor_compile_config": {"batch_invariant": True}})]
    if kv:
        ktc = {"kv_connector": "NixlConnector", "kv_role": "kv_both"}
        if spec.get("kv_policy"):  # "recompute": a failed KV load (e.g. a blocked handshake) is recomputed locally
            ktc["kv_load_failure_policy"] = spec["kv_policy"]
        if spec.get("kv_buffer"):  # "cpu": stage KV through pinned host memory (no CUDA IPC needed)
            ktc["kv_buffer_device"] = spec["kv_buffer"]
        cmd += ["--kv-transfer-config", json.dumps(ktc)]
    log = open(os.path.join(out_dir, f"server_{name}.log"), "w")
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


SERVED = {}  # port -> served model name: unique per server, so that a request never reaches another run's server


def complete(port, prompt, max_tokens, extra=None):
    payload = {"model": SERVED.get(port, "m"), "prompt": prompt, "max_tokens": max_tokens, "temperature": TEMPERATURE[0],
               "return_token_ids": True, "seed": 0, "logprobs": 1}
    if extra:
        payload.update(extra)
    return post(_url(port, "/v1/completions"), payload)


def token_ids(resp):
    ch = resp["choices"][0]
    return ch.get("token_ids") or ch.get("text")


def token_logprobs(resp):
    lp = resp["choices"][0].get("logprobs") or {}
    return [repr(x) for x in (lp.get("token_logprobs") or [])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single", "pd"], required=True)
    ap.add_argument("--model", required=True, help="local model snapshot or model identifier")
    ap.add_argument("--a", required=True, help="single: the server; pd: the prefill server")
    ap.add_argument("--b", help="pd: the decode server")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--repeat", type=int, default=1, help="pd/single: send the prompt set this many times")
    ap.add_argument("--concurrency", type=int, default=1, help="requests in flight within a round (RW experiment)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--port-base", type=int, default=8100,
                    help="HTTP ports base and base+100, NIXL side channels base-2500 and base-2499 (lets "
                         "several deployments share a host)")
    ap.add_argument("--a-host", default=None,
                    help="pd: the prefill server already runs on this host (started there with --serve-only a); "
                         "this process starts only the decode server and the client")
    ap.add_argument("--serve-only", action="store_true",
                    help="start only server a (spec --a) and keep it running until SIGTERM")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="sampling temperature of every request (seeded, so runs remain comparable bit for bit)")
    ap.add_argument("--canary", action="store_true",
                    help="pd: before the disaggregated requests, time a 1-token prefill of the first prompt on each "
                         "server (what a canary check at the handshake would cost)")
    ap.add_argument("--canary-on-demand", action="store_true",
                    help="pd: compute an instance's canary only when its guard asks for it (file canary_request in "
                         "its capture directory, written by lockstep.live_guard v7 on an identity mismatch)")
    ap.add_argument("--canary-idx", default=None,
                    help="with --canary-on-demand: a stronger canary over these prompt indices (comma-separated), "
                         "comparing every prompt position's log-probability (prompt_logprobs) instead of only the "
                         "first generated token of prompt 0")
    ap.add_argument("--also-local", action="store_true",
                    help="pd: afterwards, let each instance also generate every prompt on its own (ground truth "
                         "for whether the two instances compute identically)")
    a = ap.parse_args()
    TEMPERATURE[0] = a.temperature
    os.makedirs(a.out, exist_ok=True)
    prompts = json.load(open(a.prompts))
    sa = json.loads(a.a)
    procs = []
    import signal
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # run the cleanup below when stopped
    try:
        # served model names: unique per server (pid and role), so that a server of a concurrent run that answers on
        # one of our ports fails every request instead of serving it (R14 addendum 3: two matrices shared ports);
        # "m" when the prefill server runs on another host (--a-host, --serve-only)
        A, B = a.port_base, a.port_base + 100
        uniq = not (a.a_host or a.serve_only)
        SERVED[A] = f"m-{os.getpid()}-a" if uniq else "m"
        SERVED[B] = f"m-{os.getpid()}-b"
        sa.setdefault("args", []).extend(["--served-model-name", SERVED[A]])
        if a.a_host:
            HOSTS[A] = a.a_host
            pa = None
        else:
            pa = start_server(a.model, A, sa, "prefill" if a.mode == "pd" else "single", A - 2500, a.out, "a",
                              a.mode == "pd")
            procs.append(pa)
        if a.serve_only:
            wait_ready(A, pa)
            print("SERVE_READY", A, flush=True)
            while pa.poll() is None:
                time.sleep(5)
            raise RuntimeError(f"server on :{A} exited rc={pa.returncode}")
        if a.mode == "pd":
            sb = json.loads(a.b)
            sb.setdefault("args", []).extend(["--served-model-name", SERVED[B]])
            pb_ = start_server(a.model, B, sb, "decode", A - 2499, a.out, "b", True)
            procs.append(pb_)
        t_ready0 = time.time()
        wait_ready(A, pa)
        ready_s = {"a": time.time() - t_ready0}
        if a.mode == "pd":
            wait_ready(B, pb_)
            ready_s["b"] = time.time() - t_ready0
        od_stats, od_stop = [], None
        if a.mode == "pd" and a.canary_on_demand:
            import threading
            od_stop = threading.Event()
            caps = {nm: (os.path.join(a.out, f"inst_{nm}", "capture"), port) for nm, port in (("a", A), ("b", B))}
            for cap, _ in caps.values():  # a producer on another host (--a-host) has no local capture directory
                os.makedirs(cap, exist_ok=True)

            def _canary_daemon():
                while not od_stop.is_set():
                    for nm, (cap, port) in caps.items():
                        req, out = os.path.join(cap, "canary_request"), os.path.join(cap, "canary.json")
                        if os.path.exists(req) and not os.path.exists(out):
                            tc = time.perf_counter()
                            if a.canary_idx:  # stronger canary: several prompts, every prompt position's log-prob
                                val = {"mode": "full", "items": []}
                                for i in [int(x) for x in a.canary_idx.split(",")]:
                                    r = complete(port, prompts[i], 1, {"cache_salt": f"canary-{nm}-{time.time_ns()}",
                                                                       "prompt_logprobs": 0})
                                    val["items"].append({"prompt": i, "token_ids": token_ids(r),
                                                         "logprobs": token_logprobs(r),
                                                         "prompt_logprobs": repr(r["choices"][0].get("prompt_logprobs"))})
                            else:
                                r = complete(port, prompts[0], 1, {"cache_salt": f"canary-{nm}-{time.time_ns()}"})
                                val = {"prompt": 0, "token_ids": token_ids(r), "logprobs": token_logprobs(r)}
                            json.dump(val, open(out + ".tmp", "w"))
                            os.replace(out + ".tmp", out)
                            os.replace(req, f"{req}.done.{time.time_ns()}")
                            od_stats.append({"inst": nm, "s": time.perf_counter() - tc, "t": time.time()})
                    time.sleep(0.1)
            threading.Thread(target=_canary_daemon, daemon=True).start()
        canary_s = None
        if a.mode == "pd" and a.canary:
            canary_s = {}
            for port, nm in ((A, "a"), (B, "b")):
                ts = []
                for i in range(3):
                    tc = time.perf_counter()
                    r = complete(port, prompts[0], 1, {"cache_salt": f"canary-{nm}-{i}-{time.time_ns()}"})
                    ts.append(time.perf_counter() - tc)
                canary_s[nm] = ts
                # the canary value (first token and its log-probability), for the hybrid guard: it compares
                # canaries only when the identities differ (lockstep.live_guard, LOCKSTEP_HYBRID_DIR)
                cap = os.path.join(a.out, f"inst_{nm}", "capture")
                if os.path.isdir(cap) and not (nm == "a" and a.a_host):
                    json.dump({"prompt": 0, "token_ids": token_ids(r), "logprobs": token_logprobs(r)},
                              open(os.path.join(cap, "canary.json"), "w"))
        outs, t0, lat, rounds_eq, req_t = [], time.time(), [], [], []
        reqlog, all_rounds = [], []

        def one(rnd, i, p):  # one request; with the prefill response's time and the producer's transfer tag
            tr, ts, tpf, kvp = time.perf_counter(), time.time(), None, None
            if a.mode == "single":
                r = complete(A, p, a.max_tokens)
            else:
                rp = complete(A, p, 1, {"kv_transfer_params": {"do_remote_decode": True, "do_remote_prefill": False,
                                                                   "remote_engine_id": None, "remote_block_ids": None,
                                                                   "remote_host": None, "remote_port": None}})
                tpf = time.time()
                kvp = rp.get("kv_transfer_params")
                r = complete(B, p, a.max_tokens, {"kv_transfer_params": kvp})
            rec = {"round": rnd, "i": i, "t_send": ts, "t_prefill_done": tpf, "t_done": time.time(),
                   "latency_s": time.perf_counter() - tr, "tag": (kvp or {}).get("lockstep_tag")}
            return rec, {"token_ids": token_ids(r), "logprobs": token_logprobs(r)}

        for rnd in range(a.repeat):  # round 0 includes the handshake; later rounds measure steady state
            if a.concurrency > 1:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(a.concurrency) as ex:
                    res = list(ex.map(lambda x: one(rnd, *x), enumerate(prompts)))
            else:
                res = [one(rnd, i, p) for i, p in enumerate(prompts)]
            cur = [o for _, o in res]
            for rec, _ in res:
                req_t.append(rec["t_send"])
                lat.append([rnd, rec["latency_s"]])
                reqlog.append(rec)
            all_rounds.append(cur)
            if rnd == 0:
                outs = cur
            else:
                rounds_eq.append(sum(x == y for x, y in zip(cur, outs)))
        json.dump(reqlog, open(os.path.join(a.out, "requests.json"), "w"))
        if a.repeat > 1:
            json.dump(all_rounds, open(os.path.join(a.out, "outputs_rounds.json"), "w"))
        if a.mode == "pd" and a.also_local:
            local_lat = {}
            salt = f"lockstep-local-{os.getpid()}-{time.time_ns()}"
            for port, nm in ((A, "a"), (B, "b")):
                loc, lt = [], []
                for p in prompts:
                    tr = time.perf_counter()
                    # Ground truth must be each instance's own computation. Without a salt, the decode
                    # instance's prefix cache still holds the KV it received from the prefill instance in the
                    # disaggregated phase and would reuse it here (found 2026-10-19; ~92% hit rate).
                    r = complete(port, p, a.max_tokens, {"cache_salt": f"{salt}-{nm}"})
                    lt.append(time.perf_counter() - tr)  # local prefill + decode: the cost of always recomputing
                    loc.append({"token_ids": token_ids(r), "logprobs": token_logprobs(r)})
                json.dump(loc, open(os.path.join(a.out, f"local_{nm}.json"), "w"))
                local_lat[nm] = lt
        if od_stop is not None:
            od_stop.set()
        meta = {"ready_s": ready_s, "canary_on_demand": od_stats if a.canary_on_demand else None,
                "ground_truth_cache_salt": True, "temperature": a.temperature, "canary_s": canary_s, "a_host": a.a_host, "mode": a.mode, "a": sa, "b": json.loads(a.b) if a.b else None, "wall_s": time.time() - t0,
                "max_tokens": a.max_tokens, "n": len(prompts), "request_latency_s": lat, "request_t": req_t,
                "later_rounds_equal_to_round0": rounds_eq,
                "local_latency_s": local_lat if a.mode == "pd" and a.also_local else None}
        json.dump(outs, open(os.path.join(a.out, "outputs.json"), "w"))
        json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
        print("PD_DONE", a.out)
    finally:
        for p in procs:  # kill each server's whole process group (API server + engine core + workers)
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        for p in procs:
            try:
                p.wait(timeout=60)
            except Exception:
                pass
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass


if __name__ == "__main__":
    main()
