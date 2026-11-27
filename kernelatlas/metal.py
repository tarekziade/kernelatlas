"""Experimental FP32 MPS RMSNorm tail for decoder prefill."""

from contextlib import ExitStack, contextmanager
from functools import lru_cache, partial
from pathlib import Path
from unittest.mock import patch

import torch
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm


@lru_cache(maxsize=1)
def compile_metal_tail():
    shader = torch.mps.compile_shader(
        Path(__file__).with_name("rmsnorm_tail.metal").read_text()
    )

    def apply(module, hidden_states):
        x = hidden_states
        if (
            x.dtype != torch.float32
            or module.weight.dtype != torch.float32
            or x.device.type != "mps"
        ):
            raise ValueError("Metal experiment supports MPS FP32 inputs and weights")
        width = module.weight.numel()
        if x.shape[-1] != width or not width or x.numel() >= 2**32:
            raise ValueError("Shape outside tested Metal indexing contract")
        inv = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + module.variance_epsilon)
        output = torch.empty_like(x, memory_format=torch.contiguous_format)
        if x.numel():
            shader.rmsnorm_tail(
                output,
                x.contiguous(),
                inv.contiguous(),
                module.weight.contiguous(),
                width,
            )
        return output

    return apply


@contextmanager
def metal_rmsnorm_prefill(model):
    """Run an eager evaluation model with the experimental MPS tail in this scope.

    Use on a dedicated model instance. Forward methods are restored on exit.
    """
    if model.training or getattr(model, "use_kernels", False):
        raise ValueError("Use an eager eval() model with Hub kernels disabled")
    modules = [m for m in model.modules() if type(m) in (LlamaRMSNorm, Qwen3RMSNorm)]
    if not modules or any(
        m.weight.dtype != torch.float32 or m.weight.device.type != "mps"
        for m in modules
    ):
        raise ValueError("Expected eager Llama or Qwen3 RMSNorm modules on MPS in FP32")
    apply = compile_metal_tail()
    with torch.inference_mode(), ExitStack() as stack:
        for module in modules:
            stack.enter_context(patch.object(module, "forward", partial(apply, module)))
        yield
