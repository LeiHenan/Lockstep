"""ENV (review round 9): does the version-7 identity's default-deny rule over the environment block identically
computing instances that differ only in launcher bookkeeping? Offline, deterministic.

python -m lockstep.env_robustness > env_report.json

Uses entirely synthetic environment values and, per launcher, adds the
variables it documents for a process, with the values two instances (A, B) of one deployment would get. For each
scenario: the environment entries that enter guard.env_part and differ between A and B, with and without an
operator-declared LOCKSTEP_NEUTRAL_ENV.
"""
import json

from lockstep import guard

BASE = {  # synthetic serving shell; no deployment credentials or addresses
    "_": "/usr/bin/env", "AI": "/example/project", "FLASHINFER_WORKSPACE_BASE": "/example/project/cache",
    "HF_HOME": "/example/project/cache/hf", "HF_HUB_OFFLINE": "1", "HOME": "/example/user", "LANG": "C.UTF-8",
    "LOGNAME": "user", "PATH": "/example/project/envs/serve/bin:/usr/bin", "PWD": "/example/project",
    "SHELL": "/bin/bash", "SHLVL": "1", "SSH_CLIENT": "192.0.2.1 50000 12900", "SSH_CONNECTION": "x",
    "TMPDIR": "/example/project/tmp", "TOKENIZERS_PARALLELISM": "false", "USER": "user",
    "TORCHINDUCTOR_CACHE_DIR": "/example/project/cache/inductor", "TRITON_CACHE_DIR": "/example/project/cache/triton",
    "VLLM_CACHE_ROOT": "/example/project/cache/vllm", "VLLM_USE_FLASHINFER_SAMPLER": "0",
    "XDG_CACHE_HOME": "/example/project/cache", "VLLM_BATCH_INVARIANT": "1",
}


def _pair(common, a, b):
    return dict(BASE, **common, **a), dict(BASE, **common, **b)


SLURM_JOB = {"SLURM_CLUSTER_NAME": "c", "SLURM_JOB_NUM_NODES": "2", "SLURM_NNODES": "2", "SLURM_NTASKS": "2",
             "SLURM_JOB_PARTITION": "gpu", "SLURM_SUBMIT_DIR": "/example/user", "SLURM_SUBMIT_HOST": "login1",
             "SLURM_TASKS_PER_NODE": "1(x2)", "SLURM_CPUS_ON_NODE": "16", "SLURM_GPUS_ON_NODE": "1",
             "SLURM_JOB_NODELIST": "gpu[01-02]", "SLURM_NODELIST": "gpu[01-02]", "SLURM_STEP_NODELIST": "gpu[01-02]",
             "SLURM_JOB_ID": "4711", "SLURM_JOBID": "4711", "SLURM_JOB_NAME": "serve", "SLURM_STEP_ID": "0",
             "SLURM_LAUNCH_NODE_IPADDR": "192.0.2.2", "SLURM_SRUN_COMM_PORT": "40001", "SLURM_UMASK": "0022",
             "SLURM_TOPOLOGY_ADDR_PATTERN": "node", "SLURM_PRIO_PROCESS": "0"}


def _slurm_task(node, procid, pid):
    return {"SLURM_NODEID": str(node), "SLURM_PROCID": str(procid), "SLURM_LOCALID": "0", "SLURM_GTIDS": str(procid),
            "SLURMD_NODENAME": f"gpu0{node + 1}", "SLURM_TOPOLOGY_ADDR": f"gpu0{node + 1}", "SLURM_TASK_PID": str(pid),
            "SLURM_JOB_GPUS": "0", "SLURM_STEP_GPUS": "0", "CUDA_VISIBLE_DEVICES": "0", "HOSTNAME": f"gpu0{node + 1}"}


def _slurm_job(jid, name):
    j = dict(SLURM_JOB, SLURM_JOB_ID=str(jid), SLURM_JOBID=str(jid), SLURM_JOB_NAME=name, SLURM_NNODES="1",
             SLURM_JOB_NUM_NODES="1", SLURM_NTASKS="1", SLURM_TASKS_PER_NODE="1", SLURM_JOB_NODELIST="gpu01",
             SLURM_NODELIST="gpu01", SLURM_STEP_NODELIST="gpu01", SLURM_SRUN_COMM_PORT=str(40000 + jid % 1000))
    return dict(j, **_slurm_task(0, 0, 1000 + jid % 1000))


K8S = {"KUBERNETES_SERVICE_HOST": "192.0.2.3", "KUBERNETES_SERVICE_PORT": "443", "KUBERNETES_PORT": "tcp://192.0.2.3:443",
       "VLLM_DECODE_SERVICE_HOST": "192.0.2.4", "VLLM_DECODE_SERVICE_PORT": "8000"}

SCENARIOS = {
    "same host, same shell": _pair({}, {}, {}),
    "two hosts, ssh, same image": _pair({}, {"HOSTNAME": "a", "SSH_CLIENT": "1", "SSH_CONNECTION": "1"},
                                        {"HOSTNAME": "b", "SSH_CLIENT": "2", "SSH_CONNECTION": "2"}),
    "docker containers": _pair({}, {"HOSTNAME": "3f2a9c1e5b7d"}, {"HOSTNAME": "7d4e5f6a8b90"}),
    "kubernetes pods": _pair(K8S, {"HOSTNAME": "vllm-prefill-0"}, {"HOSTNAME": "vllm-decode-0"}),
    "kubernetes pods, downward API": _pair(dict(K8S, POD_NAMESPACE="llm"),
                                           {"HOSTNAME": "p-0", "POD_NAME": "p-0", "POD_IP": "192.0.2.5", "NODE_NAME": "n1"},
                                           {"HOSTNAME": "d-0", "POD_NAME": "d-0", "POD_IP": "192.0.2.6", "NODE_NAME": "n2"}),
    "slurm, one job on two nodes": _pair(SLURM_JOB, _slurm_task(0, 0, 2001), _slurm_task(1, 1, 3002)),
    "slurm, one job per instance": _pair({}, _slurm_job(4711, "prefill"), _slurm_job(4712, "decode")),
    "systemd services": _pair({}, {"INVOCATION_ID": "a1", "JOURNAL_STREAM": "8:1", "SYSTEMD_EXEC_PID": "11"},
                              {"INVOCATION_ID": "b2", "JOURNAL_STREAM": "8:2", "SYSTEMD_EXEC_PID": "12"}),
    "different install paths": _pair({}, {"VIRTUAL_ENV": "/opt/a/venv", "PATH": "/opt/a/venv/bin:/usr/bin", "AI": "/opt/a"},
                                     {"VIRTUAL_ENV": "/opt/b/venv", "PATH": "/opt/b/venv/bin:/usr/bin", "AI": "/opt/b"}),
    "different CPU thread count": _pair({}, {"OMP_NUM_THREADS": "8"}, {"OMP_NUM_THREADS": "16"}),
    "different allocator setting": _pair({}, {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}, {}),
}
DECLARED = "SLURM_,SLURMD_,POD_,NODE_NAME,INVOCATION_ID,JOURNAL_STREAM,SYSTEMD_EXEC_PID,AI"


def _diff(a, b):
    ea, eb = guard.env_part(a), guard.env_part(b)
    return sorted(k for k in set(ea) | set(eb) if ea.get(k) != eb.get(k))


def main():
    out = {}
    for name, (a, b) in SCENARIOS.items():
        d = _diff(a, b)
        dd = _diff(dict(a, LOCKSTEP_NEUTRAL_ENV=DECLARED), dict(b, LOCKSTEP_NEUTRAL_ENV=DECLARED))
        out[name] = {"v7_env_equal": not d, "differing": d, "with_declared_neutral": not dd, "still_differing": dd}
    print(json.dumps({"declared": DECLARED, "scenarios": out}, indent=1))


if __name__ == "__main__":
    main()
