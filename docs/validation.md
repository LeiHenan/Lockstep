# Public release validation

Validated on 2026-10-31 after removing host-specific material from the supplied
source archive. This validation uses CPU checks only.

| Check | Result |
| --- | --- |
| Original v7 regression tests | 13 passed |
| Portable path/input regression tests | 5 passed |
| Identity, live guard, decision tracer, vLLM plugin, original tests | Byte-identical to the supplied archive |
| Python syntax | All sources parsed with the Python 3.10 grammar |
| JSON syntax | All JSON files parsed |
| CLI help | Eight entry points passed without GPU engine imports |
| Demo command previews | Explore, strict PD, canary PD, and K/V passed |
| Editable installation | Passed |
| vLLM plugin entry point | Registered as `lockstep.vllm_plugin:register` |
| Source archive and wheel | Built with the setuptools backend |
| Release integrity | SHA-256 digests recorded in `source-manifest.json` |
| Static inspection | No retained private deployment paths, private network addresses, recognizable access tokens, or private keys found |

All 72 Python package modules are retained. Removed 126 historical launch,
configuration, and experiment-plan files. Internal source path maps and the
recorded host environment matrix are excluded. Model/dataset inputs, interpreter
selection, GPU lock storage, and offline capture roots are now configurable.
The offline environment study uses synthetic paths and documentation addresses.

GPU inference, CUDA compilation, live NIXL transfers, SGLang execution, and the
complete paper workloads were not rerun. Passing CPU tests does not establish
compatibility with every engine or CUDA version. Example prompts are smoke
inputs, not the paper's evaluation workload. Consult CI for GitHub check status.
