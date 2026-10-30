# Loaded by every Python process whose PYTHONPATH includes this directory.
# Activates Lockstep capture for SGLang processes (vLLM uses its own plugin entry point).
import os

if os.environ.get("LOCKSTEP_DIR") and os.environ.get("LOCKSTEP_ENGINE") == "sglang":
    try:
        from lockstep.sglang_adapter import register
        register()
    except Exception as e:  # never break the host process
        print("[lockstep-sglang] register failed:", e, flush=True)

# Hang diagnosis without ptrace: every LOCKSTEP_STACKDUMP_SEC seconds, each Python process appends
# the tracebacks of all its threads to $LOCKSTEP_STACKDUMP_DIR/stacks_<pid>.txt.
if os.environ.get("LOCKSTEP_STACKDUMP_SEC") and os.environ.get("LOCKSTEP_STACKDUMP_DIR"):
    try:
        import faulthandler
        os.makedirs(os.environ["LOCKSTEP_STACKDUMP_DIR"], exist_ok=True)
        _fh = open(os.path.join(os.environ["LOCKSTEP_STACKDUMP_DIR"], f"stacks_{os.getpid()}.txt"), "a")
        faulthandler.dump_traceback_later(int(os.environ["LOCKSTEP_STACKDUMP_SEC"]), repeat=True, file=_fh)
    except Exception as e:
        print("[lockstep] stackdump setup failed:", e, flush=True)
