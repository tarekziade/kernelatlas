#include <metal_stdlib>
using namespace metal;

// Keep the same two FP32 roundings as the eager normalization and weight scaling.
kernel void rmsnorm_tail(device float* out, device const float* x,
                         device const float* inv, device const float* weight,
                         constant uint& width, uint i [[thread_position_in_grid]]) {
#pragma clang fp reassociate(off)
    float normalized = x[i] * inv[i / width];
    out[i] = weight[i % width] * normalized;
}
