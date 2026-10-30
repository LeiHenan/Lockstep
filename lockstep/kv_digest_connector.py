"""KV ground truth (review round 7): record a digest of every prompt token's K/V bytes in every layer.

A vLLM KV connector (subclass of vLLM's ExampleConnector) that, for each new request, writes per layer the
blake2b digest of each token's slice of the paged KV buffer, instead of the tensors themselves (a tensor dump
of our workloads would not fit on the shared server). It never loads KV, so the instance computes everything
itself. Equal digests mean bit-identical K/V. Use with kv_role "kv_producer":

  --kv-transfer-config '{"kv_connector": "KVDigestConnector", "kv_role": "kv_producer",
      "kv_connector_module_path": "lockstep.kv_digest_connector",
      "kv_connector_extra_config": {"shared_storage_path": DIR}}'

A prompt is stored in the step that completes its prefill, so chunked prefill is covered too (see
build_connector_meta). Every prompt token is covered (see the patch of align_to_block_size below). Layout: DIR/<hash of the prompt token ids>/<layer>.json = {"n": tokens, "d": [...]}.
"""
import hashlib
import json
import os

from vllm.distributed.kv_transfer.kv_connector.v1 import example_connector as _ec
from vllm.distributed.kv_transfer.kv_connector.v1.example_connector import ExampleConnector

# The reference connector keeps only whole blocks of a prompt (it aligns the token count down to the block
# size), which would leave out up to block_size - 1 final tokens of every prompt and all of a short one. This
# process uses the module only through this connector, which never loads, so cover every prompt token.
_ec.align_to_block_size = lambda num_tokens, block_size: num_tokens


class KVDigestConnector(ExampleConnector):

    def _found_match_for_prompt(self, *a, **k):  # never load: the instance computes all KV itself
        return False

    def _found_match_for_request(self, *a, **k):
        return False

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        return 0, False

    def start_load_kv(self, forward_context, **kwargs):
        return

    def build_connector_meta(self, scheduler_output):
        """Store each prompt once, in the step that completes its prefill, with all of its blocks. The reference
        connector stores a request in the step that first schedules it, which under chunked prefill covers only
        the first chunk (found 2026-10-23, KVE: the budget-2048 side recorded 2,048 tokens of longer prompts)."""
        meta = _ec.ExampleConnectorMetadata()
        track = self.__dict__.setdefault("_lockstep_chunked", {})  # req id -> [prompt ids, block ids, mm hashes]
        n_sched = scheduler_output.num_scheduled_tokens
        for r in scheduler_output.scheduled_new_reqs:
            ids, blocks = list(r.prompt_token_ids or []), list(r.block_ids[0])
            mm = [f.identifier for f in r.mm_features]
            if r.num_computed_tokens + n_sched.get(r.req_id, 0) >= len(ids):
                meta.add_request(ids, blocks, self._block_size, True, mm)
            else:
                track[r.req_id] = [ids, blocks, mm]
        c = scheduler_output.scheduled_cached_reqs
        for i, rid in enumerate(c.req_ids):
            if rid not in track:
                continue
            ids, blocks, mm = track[rid]
            nb = c.new_block_ids[i]
            if nb is not None:
                blocks = list(nb[0]) if rid in c.resumed_req_ids else blocks + list(nb[0])
                track[rid][1] = blocks
            if c.num_computed_tokens[i] + n_sched.get(rid, 0) >= len(ids):
                meta.add_request(ids, blocks, self._block_size, True, mm)
                del track[rid]
        return meta

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        meta = self._get_connector_metadata()
        for request in meta.requests:
            if not request.is_store:
                continue
            slot = request.slot_mapping.to(kv_layer.device)
            if slot.numel() == 0:
                continue
            kv = kv_layer[slot // self._block_size, :, slot % self._block_size]  # [tokens, heads, channels]
            rows = kv.detach().contiguous().view(kv.shape[0], -1).view(dtype=__import__("torch").uint8).cpu().numpy()
            dig = [hashlib.blake2b(r.tobytes(), digest_size=8).hexdigest() for r in rows]
            folder = self._generate_foldername_debug(request.token_ids, request.mm_hashes, create_folder=True)
            with open(os.path.join(folder, f"{layer_name}.json"), "w") as f:
                json.dump({"n": len(dig), "d": dig}, f)
