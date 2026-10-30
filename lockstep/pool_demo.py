"""P1 (review round 9): does a shared KV pool keyed by token ids hand a consumer K/V that it would not compute itself,
and does adding the numerical identity to the key prevent it? (results_prereg/P1_KV_POOL_PREREGISTRATION.md)

python -m lockstep.pool_demo --model M --producer '{spec}' --consumers '[{"name": n, "spec": {spec}}, ...]' \
       --prompts P.json --out DIR [--limit 48] [--max-prompt-tokens 2000]
python -m lockstep.pool_demo --compare DIR

Specs as in lockstep.pd_harness (LOCKSTEP_CAPTURE is set). The producer (GPU 0) serves every prompt once and
stores its K/V in DIR/pool under both keys (lockstep.pool_connector, LOCKSTEP_POOL_KEY=both). Every consumer then
runs twice, once per key (tokens, identity), read-only, two instances at a time (GPUs 0 and 1). For each prompt a
consumer serves a pool request (may load the producer's K/V) and then an own request (cache salt "own-", never
loads), both greedy. Ground truth: the consumer's own output. Lookups and hits: inst_*/capture/pool.jsonl.
"""
import argparse
import glob
import json
import os
import signal
import sys
import threading
import time

from lockstep import pd_harness as H

KEYS = ("tokens", "identity")


def _kvcfg(path):
    return json.dumps({"kv_connector": "PoolConnector", "kv_role": "kv_both",
                       "kv_connector_module_path": "lockstep.pool_connector",
                       "kv_connector_extra_config": {"shared_storage_path": path}})


def _prompts(a):
    prompts = json.load(open(a.prompts))
    if a.max_prompt_tokens:  # the producer must prefill every prompt in one step (see pool_connector)
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model)
        prompts = [p for p in prompts if len(tok(p)["input_ids"]) <= a.max_prompt_tokens]
    return prompts[: a.limit] if a.limit else prompts


def _instance(a, name, spec, gpu, port, env, procs):
    sp = json.loads(json.dumps(spec))
    sp["gpus"] = str(gpu)
    sp["env"] = {k: str(v).replace("{tag}", name) for k, v in sp.get("env", {}).items()}  # e.g. per-instance cold caches
    sp["env"].update(dict(env, LOCKSTEP_CAPTURE="1"))
    sp.setdefault("args", []).extend(["--served-model-name", "m", "--kv-transfer-config", _kvcfg(a.pool)])
    t0 = time.time()
    p = H.start_server(a.model, port, sp, name, port - 2500, a.out, name, False)
    procs.append(p)
    H.wait_ready(port, p)
    return p, time.time() - t0


def _stop(p):
    for sig, wait in ((signal.SIGTERM, 60), (signal.SIGKILL, 0)):
        try:
            os.killpg(p.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            p.wait(timeout=wait or 10)
        except Exception:
            pass


def _row(r):
    return {"token_ids": H.token_ids(r), "logprobs": H.token_logprobs(r)}


def _consumer(a, job, gpu, prompts, procs, meta):
    name, spec, key = job
    tag = f"{name}_{key}"
    p, ready = _instance(a, tag, spec, gpu, a.port_base + 100 * (gpu + 1),
                         {"LOCKSTEP_POOL_KEY": key, "LOCKSTEP_POOL_READONLY": "1"}, procs)
    port = a.port_base + 100 * (gpu + 1)
    pool, own = [], []
    try:
        for i, pr in enumerate(prompts):  # pool request first: the identity freezes at the first lookup
            pool.append(_row(H.complete(port, pr, a.max_tokens, {"cache_salt": f"pool-{tag}-{i}"})))
            own.append(_row(H.complete(port, pr, a.max_tokens, {"cache_salt": f"own-{tag}-{i}"})))
        json.dump(pool, open(os.path.join(a.out, f"pool_{tag}.json"), "w"))
        json.dump(own, open(os.path.join(a.out, f"own_{tag}.json"), "w"))
        meta[tag] = {"name": name, "key": key, "spec": spec, "ready_s": ready}
    finally:
        _stop(p)


def run(a):
    os.makedirs(a.out, exist_ok=True)
    a.pool = os.path.join(a.out, "pool")
    os.makedirs(a.pool, exist_ok=True)
    prompts = _prompts(a)
    json.dump(prompts, open(os.path.join(a.out, "prompts.json"), "w"))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    procs, meta = [], {}
    try:
        p, ready = _instance(a, "producer", json.loads(a.producer), 0, a.port_base, {"LOCKSTEP_POOL_KEY": "both"}, procs)
        rows = [_row(H.complete(a.port_base, pr, a.max_tokens, {"cache_salt": f"pool-producer-{i}"}))
                for i, pr in enumerate(prompts)]
        json.dump(rows, open(os.path.join(a.out, "producer.json"), "w"))
        meta["producer"] = {"spec": json.loads(a.producer), "ready_s": ready}
        _stop(p)
        jobs = [(c["name"], c["spec"], k) for c in json.loads(a.consumers) for k in KEYS]
        lock, errors = threading.Lock(), []

        def lane(gpu):
            while True:
                with lock:
                    if not jobs:
                        return
                    job = jobs.pop(0)
                try:
                    _consumer(a, job, gpu, prompts, procs, meta)
                except Exception as e:  # recorded; the run counts as failed
                    errors.append(f"{job[0]}_{job[2]}: {e!r}")

        lanes = [threading.Thread(target=lane, args=(g,)) for g in (0, 1)]
        for t in lanes:
            t.start()
        for t in lanes:
            t.join()
        json.dump({"meta": meta, "errors": errors, "n": len(prompts), "max_tokens": a.max_tokens,
                   "consumers": json.loads(a.consumers)}, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
        if errors:
            raise RuntimeError("; ".join(errors))
        print("POOL_DEMO_DONE", a.out)
    finally:
        for p in procs:
            _stop(p)


def _jsonl(path):
    out = []
    for f in glob.glob(path):
        out += [json.loads(l) for l in open(f) if l.strip()]
    return out


def compare(d):
    m = json.load(open(os.path.join(d, "meta.json")))
    prod = json.load(open(os.path.join(d, "producer.json")))
    pid = [r["identity"] for r in _jsonl(os.path.join(d, "inst_producer", "capture", "pool.jsonl"))
           if r["event"] == "identity"]
    out = {"dir": d, "n": m["n"], "producer_identity": pid[0] if pid else None, "consumers": {}}
    for tag, info in sorted(m["meta"].items()):
        if tag == "producer":
            continue
        pool = json.load(open(os.path.join(d, f"pool_{tag}.json")))
        own = json.load(open(os.path.join(d, f"own_{tag}.json")))
        log = _jsonl(os.path.join(d, f"inst_{tag}", "capture", "pool.jsonl"))
        ident = next((r["identity"] for r in log if r["event"] == "identity"), None)
        hit = {}
        for r in log:
            if r["event"] == "lookup" and str(r.get("salt", "")).startswith("pool-"):
                i = int(r["salt"].rsplit("-", 1)[1])
                hit[i] = hit.get(i, False) or r["hit"]
        rows = []
        for i, (x, y, z) in enumerate(zip(pool, own, prod)):
            rows.append({"hit": hit.get(i, False), "served_eq_own": x == y,
                         "served_tokens_eq_own": x["token_ids"] == y["token_ids"],
                         "own_eq_producer": y == z, "served_eq_producer": x == z})
        hits = [r for r in rows if r["hit"]]
        out["consumers"][tag] = {
            "name": info["name"], "key": info["key"], "identity": ident,
            "identity_eq_producer": ident is not None and ident == out["producer_identity"],
            "hits": len(hits), "lookups": len(hit),
            "served_ne_own": sum(not r["served_eq_own"] for r in rows),
            "served_ne_own_tokens": sum(not r["served_tokens_eq_own"] for r in rows),
            "hit_and_served_ne_own": sum(not r["served_eq_own"] for r in hits),
            "own_ne_producer": sum(not r["own_eq_producer"] for r in rows),
            "ready_s": info["ready_s"], "per_prompt": rows}
    json.dump(out, open(os.path.join(d, "pool_report.json"), "w"), indent=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare")
    ap.add_argument("--model")
    ap.add_argument("--producer")
    ap.add_argument("--consumers")
    ap.add_argument("--prompts")
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int, default=48)
    ap.add_argument("--max-prompt-tokens", type=int, default=2000)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--port-base", type=int, default=8100)
    a = ap.parse_args()
    if a.compare:
        r = compare(a.compare)
        print(json.dumps({"producer_identity": r["producer_identity"],
                          "consumers": {k: {x: y for x, y in v.items() if x != "per_prompt"}
                                        for k, v in r["consumers"].items()}}))
    else:
        run(a)


if __name__ == "__main__":
    main()
