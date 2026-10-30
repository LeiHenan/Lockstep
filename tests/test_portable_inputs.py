"""CPU checks for public-release path handling; no GPU engines are imported."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lockstep import explore, hold, loadgen


def test_scheduler_and_placeholder_share_lock_directory():
    assert explore.LOCK_DIR == hold.LOCK_DIR


def test_loadgen_uses_supplied_prompts(tmp_path):
    source = tmp_path / "prompts.json"
    source.write_text(json.dumps(["first", "second"]))
    prompts = loadgen.long_prompts(2, 10, source)
    assert prompts == ["[request 0]\nfirst\n\nsecond", "[request 1]\nsecond\n\nfirst"]


@pytest.mark.parametrize("source", [[], [""]])
def test_loadgen_rejects_empty_source(tmp_path, source):
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="nonempty"):
        loadgen.long_prompts(1, 10, path)


def test_sglang_launch_inherits_toolchain_and_uses_configured_interpreter(tmp_path, monkeypatch):
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    monkeypatch.setenv("LOCKSTEP_SGLANG_BOOTSTRAP", str(bootstrap))
    monkeypatch.setenv("LOCKSTEP_SGLANG_PYTHON", "/example/sglang/python")
    monkeypatch.setenv("PYTHONPATH", "/example/existing")
    monkeypatch.setenv("CC", "example-cc")
    monkeypatch.setattr(explore.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=""))
    captured = {}

    def popen(command, **kwargs):
        captured.update(command=command, **kwargs)
        kwargs["stdout"].close()
        return SimpleNamespace(pid=123)

    monkeypatch.setattr(explore.subprocess, "Popen", popen)
    explore.launch({"name": "example", "engine": "sglang", "engine_kwargs": {},
                    "env": {"SGLANG_JIT_CACHE_DIR": str(tmp_path / "cache")}},
                   str(tmp_path), "prompts.json", [0])
    assert captured["command"][:3] == ["/example/sglang/python", "-m", "lockstep.runner_sgl"]
    env = captured["env"]
    assert env["CC"] == "example-cc"
    assert env["PYTHONPATH"].split(":") == [str(bootstrap), "/example/existing"]
    assert Path(env["SGLANG_JIT_CACHE_DIR"]) == tmp_path / "cache"
    assert env["LOCKSTEP_ENGINE"] == "sglang"
