"""Prepare and launch a small vLLM experiment; --dry-run requires no engine/GPU."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("explore", "pd", "kv"))
    parser.add_argument("--model", required=True, help="local model snapshot directory")
    parser.add_argument("--prompts", type=Path, default=root / "examples/prompts.json")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--gpus", help="physical GPU indices (two required for pd/kv)")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--prefill-budget", type=int, default=4096)
    parser.add_argument("--decode-budget", type=int, default=2048)
    parser.add_argument("--port-base", type=int, default=8100)
    parser.add_argument("--policy", choices=("strict", "canary"), default="strict")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    gpu_text = args.gpus or ("0" if args.mode == "explore" else "0,1")
    try:
        gpu_ids = [int(item.strip()) for item in gpu_text.split(",")]
    except ValueError:
        parser.error("--gpus must contain comma-separated nonnegative GPU indices")
    if any(item < 0 for item in gpu_ids) or len(set(gpu_ids)) != len(gpu_ids):
        parser.error("--gpus must contain distinct nonnegative GPU indices")
    if args.mode != "explore" and len(gpu_ids) != 2:
        parser.error("pd and kv require exactly two GPU indices")
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("--gpu-memory-utilization must be between 0 and 1")
    if min(args.max_tokens, args.max_model_len, args.prefill_budget, args.decode_budget) <= 0:
        parser.error("token counts and budgets must be positive")
    if not 2500 < args.port_base <= 65435:
        parser.error("--port-base must be between 2501 and 65435")
    prompts_path = args.prompts.resolve()
    try:
        prompts = json.loads(prompts_path.read_text())
    except (OSError, ValueError) as exc:
        parser.error(f"cannot read prompts: {exc}")
    if not isinstance(prompts, list) or not prompts or not all(isinstance(p, str) and p for p in prompts):
        parser.error("prompts must be a nonempty JSON list of nonempty strings")
    model = Path(args.model).expanduser().resolve()
    if not args.dry_run and not model.is_dir():
        parser.error("--model must be an existing local model snapshot directory")
    out = (args.out or root / "runs" / f"demo-{args.mode}").resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        parser.error("--out must be absent or an empty directory")
    if not args.dry_run and any(key.startswith("LOCKSTEP_TEST_") for key in os.environ):
        parser.error("unset inherited LOCKSTEP_TEST_* fault-injection variables before running a demo")

    settings = {
        "LOCKSTEP_LOCK_DIR": os.environ.get("LOCKSTEP_LOCK_DIR", str(root / ".cache/gpu-locks")),
        "VLLM_BATCH_INVARIANT": "1",
        "LOCKSTEP_GUARD_VERSION": "7",
        "LOCKSTEP_DEC_EAGER_FLUSH": "1",
        "LOCKSTEP_MODE": "boundary" if args.mode == "explore" else "off",
        "LOCKSTEP_CAPTURE": "1",
        "LOCKSTEP_LIVE_GUARD": "1" if args.mode == "pd" else "0",
        "LOCKSTEP_GUARD_POLICY": args.policy,
        "LOCKSTEP_XFER_TAG": "enforce",
    }
    command = [sys.executable, "-m"]
    spec = None
    if args.mode == "explore":
        spec = {
            "prompts": str(prompts_path),
            "base": {
                "engine_kwargs": {
                    "model": str(model), "max_model_len": args.max_model_len,
                    "gpu_memory_utilization": args.gpu_memory_utilization,
                    "max_num_batched_tokens": args.prefill_budget, "max_num_seqs": 32,
                    "seed": 0,
                },
                "env": {"VLLM_BATCH_INVARIANT": "1"},
                "runner_args": ["--max-tokens", str(args.max_tokens)],
            },
            "variants": [
                {"name": "budget", "engine_kwargs": {"max_num_batched_tokens": args.decode_budget}},
                {"name": "serial", "runner_args": ["--order", "serial"]},
            ],
        }
        command += ["lockstep.explore", "--spec", str(out / "spec.json"),
                    "--out-root", str(out / "runs"), "--gpus", ",".join(map(str, gpu_ids))]
    else:
        # pd_harness maps these indices through the parent's visible device list.
        settings["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        settings["LOCKSTEP_HYBRID_DIR"] = str(out / "registry")

        def side(index, budget):
            result = {
                "gpus": str(index), "visible": "0,1" if index == 0 else "1,0",
                "mem": args.gpu_memory_utilization, "max_len": args.max_model_len,
                "args": ["--max-num-batched-tokens", str(budget), "--max-num-seqs", "32"],
                "env": {}, "ibi": False,
            }
            if args.mode == "pd":
                result["kv_policy"] = "recompute"
            return result

        command += ["lockstep.pd_harness" if args.mode == "pd" else "lockstep.kv_truth"]
        if args.mode == "pd":
            command += ["--mode", "pd", "--also-local"]
            if args.policy == "canary":
                settings["LOCKSTEP_CANARY_ON_DEMAND"] = "1"
                command += ["--canary-on-demand", "--canary-idx", ",".join(map(str, range(min(4, len(prompts)))))]
        command += ["--model", str(model), "--a", json.dumps(side(0, args.prefill_budget)),
                    "--b", json.dumps(side(1, args.decode_budget)), "--prompts", str(prompts_path),
                    "--max-tokens", str(args.max_tokens), "--port-base", str(args.port_base), "--out", str(out)]

    plan = {"mode": args.mode, "command": command, "environment_overrides": settings}
    if spec is not None:
        plan["spec"] = spec
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return
    out.mkdir(parents=True, exist_ok=True)
    if spec is not None:
        (out / "spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    (out / "launch.json").write_text(json.dumps(plan, indent=2) + "\n")
    environment = os.environ.copy()
    environment.update(settings)
    raise SystemExit(subprocess.call(command, env=environment, cwd=root))


if __name__ == "__main__":
    main()
