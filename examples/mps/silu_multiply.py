"""Separate-input BF16 Metal gate for KernelAtlas trials."""

import torch

_SHADER = torch.mps.compile_shader("""
#include <metal_stdlib>
using namespace metal;
kernel void gate(device const bfloat* gate, device const bfloat* up,
                 device bfloat* out, uint i [[thread_position_in_grid]]) {
    float x = float(gate[i]);
    // Match eager's rounding to BF16 between activation and multiplication.
    out[i] = bfloat(x / (1.0f + exp(-x))) * up[i];
}
""")


def candidate(gate, up):
    out = torch.empty_like(gate)
    _SHADER.gate(gate, up, out, threads=gate.numel())
    return out
