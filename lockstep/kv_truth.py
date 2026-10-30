"""K1 (review round 7): three kinds of ground truth for whether two instances compute the same KV.

python -m lockstep.kv_truth --model M --a '{spec}' --b '{spec}' --prompts P.json --out DIR [--max-tokens 32]

Starts the two instances of a deployment (same specs as lockstep.pd_harness, without NIXL) with
lockstep.kv_digest_connector, sends every prompt to each instance on its own (cache salt per instance, as in
the guard experiments' ground truth), and records
  * per layer and prompt token, a digest of the K/V bytes the instance wrote (kv_a/, kv_b/);
  * the log-probability of every prompt token (prompt_lp_a.json, prompt_lp_b.json);
  * the generated tokens and their log-probabilities (local_a.json, local_b.json, as in pd_harness).
python -m lockstep.kv_truth --compare DIR writes DIR/kv_truth.json.
"""
import argparse
import glob
import json
import os
import sys
import time

from lockstep import pd_harness as H


def _kvcfg(path):
    return json.dumps({"kv_connector": "KVDigestConnector", "kv_role": "kv_producer",
                       "kv_connector_module_path": "lockstep.kv_digest_connector",
                       "kv_connector_extra_config": {"shared_storage_path": path}})


def run(a):
    os.makedirs(a.out, exist_ok=True)
    prompts = json.load(open(a.prompts))
    if a.limit:
        prompts = prompts[: a.limit]
    import signal
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    procs, ports = [], {"a": a.port_base, "b": a.port_base + 100}
    t_start = {}
    try:
        for nm in ("a", "b"):
            sp = json.loads(getattr(a, nm))
            kvdir = os.path.join(a.out, f"kv_{nm}")
            os.makedirs(kvdir, exist_ok=True)
            sp.setdefault("args", []).extend(["--served-model-name", "m", "--kv-transfer-config", _kvcfg(kvdir)])
            t_start[nm] = time.time()
            procs.append(H.start_server(a.model, ports[nm], sp, nm, ports[nm] - 2500, a.out, nm, False))
        ready = {}
        for (nm, port), p in zip(ports.items(), procs):
            H.wait_ready(port, p)
            ready[nm] = time.time() - t_start[nm]
        salt = f"kvtruth-{os.getpid()}-{time.time_ns()}"
        for nm, port in ports.items():
            loc, plp = [], []
            for p in prompts:
                r = H.complete(port, p, a.max_tokens, {"cache_salt": f"{salt}-{nm}", "prompt_logprobs": 0})
                loc.append({"token_ids": H.token_ids(r), "logprobs": H.token_logprobs(r)})
                plp.append(r["choices"][0].get("prompt_logprobs"))
            json.dump(loc, open(os.path.join(a.out, f"local_{nm}.json"), "w"))
            json.dump(plp, open(os.path.join(a.out, f"prompt_lp_{nm}.json"), "w"))
        turns = None
        if a.decode_kv:
            # second turn (D1): resend prompt + the instance's own generated tokens with the same salt and no
            # prompt log-probabilities, so the prefix cache serves the blocks written during decoding; the
            # connector then records the K/V that decode steps produced
            import torch
            from vllm.utils.hashing import safe_hash
            turns = {}
            for nm, port in ports.items():
                loc = json.load(open(os.path.join(a.out, f"local_{nm}.json")))
                rows = []
                for p, out in zip(prompts, loc):
                    first = H.complete(port, p, 1, {"cache_salt": f"{salt}-{nm}", "return_token_ids": True})
                    pids = first["choices"][0].get("prompt_token_ids") or []
                    ids = list(pids) + list(out["token_ids"])
                    H.complete(port, ids, 1, {"cache_salt": f"{salt}-{nm}"})
                    folder = safe_hash(torch.tensor(ids).numpy().tobytes(), usedforsecurity=False).hexdigest()
                    rows.append({"folder": folder, "prompt_len": len(pids), "gen_len": len(out["token_ids"])})
                turns[nm] = rows
            json.dump(turns, open(os.path.join(a.out, "decode_turns.json"), "w"))
        json.dump({"a": json.loads(a.a), "b": json.loads(a.b), "n": len(prompts), "ready_s": ready,
                   "max_tokens": a.max_tokens, "decode_kv": bool(a.decode_kv)},
                  open(os.path.join(a.out, "meta.json"), "w"), indent=1)
        print("KV_TRUTH_DONE", a.out)
    finally:
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
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass


def _kv(d):
    out = {}
    for f in glob.glob(os.path.join(d, "*", "*.json")):
        folder, layer = os.path.basename(os.path.dirname(f)), os.path.basename(f)[:-5]
        out.setdefault(folder, {})[layer] = json.load(open(f))["d"]
    return out


def compare(d):
    ka, kb = _kv(os.path.join(d, "kv_a")), _kv(os.path.join(d, "kv_b"))
    la, lb = json.load(open(os.path.join(d, "local_a.json"))), json.load(open(os.path.join(d, "local_b.json")))
    pa, pb = json.load(open(os.path.join(d, "prompt_lp_a.json"))), json.load(open(os.path.join(d, "prompt_lp_b.json")))
    prompts = []
    for folder in sorted(set(ka) | set(kb)):
        A, B = ka.get(folder, {}), kb.get(folder, {})
        diff_layers, first = [], None
        for layer in sorted(set(A) | set(B), key=lambda s: (len(s), s)):
            x, y = A.get(layer), B.get(layer)
            if x != y:
                diff_layers.append(layer)
                if x and y and first is None:
                    pos = next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), None)
                    first = {"layer": layer, "token": pos, "tokens_differing": sum(u != v for u, v in zip(x, y)),
                             "n": len(x)}
        prompts.append({"folder": folder, "in_both": folder in ka and folder in kb, "layers": len(A),
                        "kv_equal": not diff_layers and folder in ka and folder in kb,
                        "layers_differing": len(diff_layers), "first": first})
    first_tok = [(x["logprobs"] or [None])[0] == (y["logprobs"] or [None])[0] for x, y in zip(la, lb)]
    gen = [x == y for x, y in zip(la, lb)]
    plp = [x == y for x, y in zip(pa, pb)]
    dec = None
    tf = os.path.join(d, "decode_turns.json")
    if os.path.exists(tf):  # D1: K/V written during decoding, for prompts whose generations agree
        turns, bs = json.load(open(tf)), 16
        dec = {"same_generation": 0, "prefill_part_differs": 0, "decode_part_differs": 0, "decode_positions": 0}
        for ra, rb in zip(turns["a"], turns["b"]):
            if ra["folder"] != rb["folder"] or ra["folder"] not in ka or ra["folder"] not in kb:
                continue
            dec["same_generation"] += 1
            pl, cut = ra["prompt_len"], ((ra["prompt_len"] + ra["gen_len"] - 1) // bs) * bs
            A, B = ka[ra["folder"]], kb[ra["folder"]]
            pre = any(A[l][:pl] != B[l][:pl] for l in A if l in B)
            dpart = any(A[l][pl:cut] != B[l][pl:cut] for l in A if l in B)
            dec["prefill_part_differs"] += pre
            dec["decode_part_differs"] += dpart
            dec["decode_positions"] += max(0, cut - pl)
    out = {"dir": d, "decode_kv": dec, "prompts_kv": len(prompts), "kv_equal_prompts": sum(p["kv_equal"] for p in prompts),
           "kv_all_equal": all(p["kv_equal"] for p in prompts) and bool(prompts),
           "prompt_logprobs_equal": sum(plp), "outputs_equal": sum(gen), "first_token_equal": sum(first_tok),
           "n": len(la), "per_prompt_kv": prompts}
    json.dump(out, open(os.path.join(d, "kv_truth.json"), "w"), indent=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare")
    ap.add_argument("--model")
    ap.add_argument("--a")
    ap.add_argument("--b")
    ap.add_argument("--prompts")
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--decode-kv", action="store_true", help="D1: also record the K/V written during decoding")
    ap.add_argument("--port-base", type=int, default=8100)
    a = ap.parse_args()
    if a.compare:
        r = compare(a.compare)
        print(json.dumps({k: v for k, v in r.items() if k != "per_prompt_kv"}))
    else:
        run(a)


if __name__ == "__main__":
    main()
