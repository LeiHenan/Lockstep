"""F-004 intervention: make the AWQ path's matmul opaque to torch.compile.

Enabled by LOCKSTEP_AWQ_OPAQUE=1 (loaded from the vLLM plugin). Inside torch.compile, Inductor lowers
`torch.matmul(x, dequantized_w)` of AutoAWQLinearMethod to an extern cuBLAS addmm, bypassing vLLM's
batch-invariant matmul, which is registered at the dispatcher. Wrapping the matmul in a custom op keeps
it opaque to Inductor, so at run time it goes through the dispatcher (and thus the BI override).
"""
import torch


@torch.library.custom_op("lockstep::awq_matmul", mutates_args=())
def awq_matmul(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.matmul(x, w)


@awq_matmul.register_fake
def _(x, w):
    return x.new_empty((x.shape[0], w.shape[1]))


def install():
    from vllm.model_executor.layers.quantization import auto_awq as aa
    from vllm import _custom_ops as ops

    def apply(self, layer, x, bias=None):
        qweight, scales, qzeros = layer.qweight, layer.scales, layer.qzeros
        pack_factor = self.quant_config.pack_factor
        out_shape = x.shape[:-1] + (qweight.shape[-1] * pack_factor,)
        reshaped_x = x.reshape(-1, x.shape[-1])
        w = ops.awq_dequantize(qweight, scales, qzeros, 0, 0, 0)
        out = torch.ops.lockstep.awq_matmul(reshaped_x, w)
        if bias is not None:
            out.add_(bias)
        return out.reshape(out_shape)

    aa.AutoAWQLinearMethod.apply = apply
