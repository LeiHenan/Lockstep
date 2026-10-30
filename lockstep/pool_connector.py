"""P1 (review round 9): a shared KV pool whose keys carry the producer's numerical identity.

A vLLM KV connector (subclass of vLLM's ExampleConnector, which stores each request's prompt K/V in a shared
directory keyed by its token ids). LOCKSTEP_POOL_KEY selects the key:
  tokens    the token ids only (vLLM's behavior): any instance's blocks match;
  identity  the token ids and the instance's v7 identity (lockstep.live_guard.identity_v7_now), frozen at the
            first lookup after start-up: a consumer finds only blocks that an instance with an equal identity
            produced;
  both      (producer) look up and store under the identity key, and link the token-only key to the same blocks.
LOCKSTEP_POOL_READONLY=1 (consumers): never store. Requests whose cache salt starts with "own-" bypass the pool,
so a consumer can also compute a prompt itself (the salt also separates vLLM's in-process prefix cache).
ExampleConnector stores a prompt's K/V in the step that schedules it first, so the producer's token budget must
cover the longest prompt (no chunked prefill). Use with kv_role "kv_both":

  --kv-transfer-config '{"kv_connector": "PoolConnector", "kv_role": "kv_both",
      "kv_connector_module_path": "lockstep.pool_connector",
      "kv_connector_extra_config": {"shared_storage_path": DIR}}'

Every lookup goes to $LOCKSTEP_DIR/pool.jsonl.
"""
import json
import os
import time

from vllm.distributed.kv_transfer.kv_connector.v1.example_connector import ExampleConnector

_IDENT = []


def _identity():
    if not _IDENT:
        from lockstep import live_guard
        ident, why, n, ms = live_guard.identity_v7_now()
        if not n:  # no usage records yet (before warm-up): do not freeze an identity
            return ident
        _IDENT.append(ident)
        _log({"event": "identity", "identity": ident, "coverage": why, "n_records": n, "ms": ms})
    return _IDENT[0]


def _log(rec):
    d = os.environ.get("LOCKSTEP_DIR")
    if d:
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "pool.jsonl"), "a") as f:
            f.write(json.dumps(dict(rec, pid=os.getpid(), t=time.time()), default=str) + "\n")


def _own(request):
    return str(getattr(request, "cache_salt", "") or "").startswith("own-")


class PoolConnector(ExampleConnector):

    def __init__(self, vllm_config, *a, **k):
        from lockstep import live_guard
        if not live_guard._VLLM_CFG:  # the resolved configuration that identity_v7_now covers
            live_guard._VLLM_CFG.append(vllm_config)
        super().__init__(vllm_config, *a, **k)
        mode = os.environ.get("LOCKSTEP_POOL_KEY", "identity")
        self._both = mode == "both"
        self._key = "identity" if self._both else mode

    def _generate_foldername_debug(self, token_ids, mm_hashes, create_folder=False):
        extra = [f"id:{_identity()}"] if self._key == "identity" else []
        return super()._generate_foldername_debug(token_ids, list(mm_hashes) + extra, create_folder=create_folder)

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        if _own(request):
            return 0, False
        ident = _identity()  # also with the token-only key, for the report
        n, async_load = super().get_num_new_matched_tokens(request, num_computed_tokens)
        _log({"event": "lookup", "hit": bool(n), "tokens": len(request.prompt_token_ids or []), "key": self._key,
              "salt": getattr(request, "cache_salt", None), "identity": ident})
        return n, async_load

    def build_connector_meta(self, scheduler_output):
        meta = super().build_connector_meta(scheduler_output)
        if os.environ.get("LOCKSTEP_POOL_READONLY") == "1":
            meta.requests = [r for r in meta.requests if not r.is_store]
        return meta

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        super().save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)
        if not self._both:
            return
        for request in self._get_connector_metadata().requests:  # token-only key -> the same blocks
            if request.is_store:
                target = self._generate_foldername_debug(request.token_ids, request.mm_hashes)
                self._key = "tokens"
                try:
                    link = self._generate_foldername_debug(request.token_ids, request.mm_hashes)
                finally:
                    self._key = "identity"
                if not os.path.lexists(link):
                    os.symlink(target, link)
