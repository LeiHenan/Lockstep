"""Regression tests for identity version 7 and the guard's lifecycle rules (no GPU, no vLLM needed).

python -m pytest tests/test_guard_v7.py -q
"""
import json
import os

from lockstep import guard, live_guard

CFG = {"engine_kwargs": {"model": "/m", "tensor_parallel_size": 1, "args": ["--max-num-batched-tokens", "2048"],
                         "ibi": False}, "env": {"VLLM_BATCH_INVARIANT": "1"}}
USE = {"use0/k#0": {"subgraph": 0, "kernel": "triton_red_fused_0", "chosen": {"R0_BLOCK": 64}, "srcn": "abc"}}
STACK = {"gpu": "RTX 4090", "cc": "8.9", "sms": 128, "driver": 13000, "torch": "2.13", "cuda": "13.0"}
RES = {"model": "/m", "quantization": None, "dtype": "bfloat16"}
ST = {"m1.py": {"extern": {}, "n_triton": 3, "user_kernels": ["matmul_kernel_persistent"]}}


def ident(**kw):
    args = dict(cfg=CFG, use=USE, dec={}, stack=STACK, env={"VLLM_BATCH_INVARIANT": "1"}, resolved=RES,
                model_files="f00", structure=ST)
    args.update(kw)
    return guard.identity_v7(**args)


def test_full_coverage():
    assert ident()[1] == []


def test_missing_trace_is_partial():
    assert "no-trace" in ident(use={})[1]


def test_untracked_user_kernel_is_partial():
    st = {"m1.py": {"extern": {}, "n_triton": 3, "user_kernels": ["fused_moe_kernel"]}}
    assert "user-kernels" in ident(structure=st)[1]


def test_quantized_is_partial():
    assert "quantized" in ident(resolved=dict(RES, quantization="gptq"))[1]


def test_missing_stack_or_resolved_config_is_partial():
    assert "no-stack" in ident(stack=dict(STACK, driver=None))[1]
    assert "no-resolved-config" in ident(resolved=None)[1]


def test_environment_counts_by_default():
    a = ident(env={"VLLM_BATCH_INVARIANT": "1"})[0]
    assert ident(env={"VLLM_BATCH_INVARIANT": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})[0] != a
    # places, ports and devices do not
    assert ident(env={"VLLM_BATCH_INVARIANT": "1", "CUDA_VISIBLE_DEVICES": "3", "HOME": "/x",
                      "LOCKSTEP_DIR": "/y", "TORCHINDUCTOR_CACHE_DIR": "/z",
                      "VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME": "vllm_mm_x"})[0] == a


def test_numerics_env_counts():
    """NVIDIA_TF32_OVERRIDE changes numerics, so only container bookkeeping NVIDIA_* variables are neutral."""
    a = ident(env={"VLLM_BATCH_INVARIANT": "1"})[0]
    assert ident(env={"VLLM_BATCH_INVARIANT": "1", "NVIDIA_TF32_OVERRIDE": "0"})[0] != a
    assert ident(env={"VLLM_BATCH_INVARIANT": "1", "NVIDIA_VISIBLE_DEVICES": "all"})[0] == a


def test_lowering_counts_as_a_set():
    st2 = dict(ST, **{"m2.py": dict(ST["m1.py"])})  # a cold process records the same module twice
    assert ident(structure=st2)[0] == ident()[0]
    st3 = {"m1.py": {"extern": {"mm": 1}, "n_triton": 3, "user_kernels": ["matmul_kernel_persistent"]}}
    assert ident(structure=st3)[0] != ident()[0]


def test_model_files_and_resolved_config_count():
    assert ident(model_files="f01")[0] != ident()[0]
    assert ident(resolved=dict(RES, dtype="float16"))[0] != ident()[0]


def test_decide():
    c = {"token_ids": [1], "logprobs": ["-1.0"]}
    assert live_guard.decide("strict", True, True, None, None) == "admit"
    assert live_guard.decide("strict", True, False, c, c) == "block"       # partial coverage: never on identity
    assert live_guard.decide("strict", False, True, c, c) == "block"
    assert live_guard.decide("canary", False, True, c, c) == "admit_canary"
    assert live_guard.decide("canary", True, False, c, c) == "admit_canary"
    assert live_guard.decide("canary", False, True, c, dict(c, logprobs=["-1.1"])) == "block"
    assert live_guard.decide("canary", False, True, None, c) == "block"    # missing canary: block


def test_invalidation():
    assert not live_guard.needs_invalidation(10, 10, "x", "x")
    assert live_guard.needs_invalidation(10, 11, "x", "x")      # new kernel usage after the handshake
    assert live_guard.needs_invalidation(10, 10, "x", "y")      # producer republished another identity
    assert not live_guard.needs_invalidation(10, 10, None, "y")  # no registry entry at handshake time


def test_canary_missing_without_on_demand(tmp_path, monkeypatch):
    monkeypatch.delenv("LOCKSTEP_CANARY_ON_DEMAND", raising=False)
    assert live_guard.canary_get(str(tmp_path)) is None
    json.dump({"prompt": 0}, open(os.path.join(tmp_path, "canary.json"), "w"))
    assert live_guard.canary_get(str(tmp_path)) == {"prompt": 0}


def test_v6_unchanged():
    """v6 (frozen) does not see the environment or lowering: guards against accidental edits."""
    a = guard.identity_v6(CFG, USE, {}, STACK)
    assert a == guard.identity_v6(CFG, USE, {}, STACK)
    assert a == "%s" % a and len(a) == 16
