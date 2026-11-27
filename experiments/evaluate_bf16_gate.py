"""Test the split-input BF16 MPS SiLU/multiply lead from a pretrained profile."""

import argparse
from contextlib import ExitStack
from functools import partial
import json
from pathlib import Path
import platform
from statistics import median
from time import perf_counter_ns
from unittest.mock import patch

import torch
import transformers
from transformers import AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaMLP
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

from kernelatlas.discovery import analyze_run


SHADER = """
#include <metal_stdlib>
using namespace metal;
kernel void gate(device const bfloat* gate, device const bfloat* up,
                 device bfloat* out, uint i [[thread_position_in_grid]]) {
    float x = float(gate[i]);
    // Match eager's rounding to BF16 between activation and multiplication.
    out[i] = bfloat(x / (1.0f + exp(-x))) * up[i];
}
"""


def timed(forward):
    torch.mps.synchronize()
    start = perf_counter_ns()
    _output = forward()  # Keep results alive until device work completes.
    torch.mps.synchronize()
    return (perf_counter_ns() - start) / 1e6


def paired(run):
    rows = []
    for idx in range(24):
        order = ("candidate", "baseline") if idx % 2 else ("baseline", "candidate")
        row = {name: run(name) for name in order}
        if idx >= 4:
            rows.append(row)
    deltas = [r["baseline"] - r["candidate"] for r in rows]
    return {
        "pairs_ms": rows,
        "baseline_median_ms": median(r["baseline"] for r in rows),
        "median_paired_savings_ms": median(deltas),
        "faster_pairs": sum(x > 0 for x in deltas),
    }


@torch.inference_mode()
def evaluate(folder):
    report = json.loads((folder / "report.json").read_text())
    matches = [
        c
        for c in analyze_run(folder)["candidates"]
        if c["operators"] == ["aten.silu.default", "aten.mul.Tensor"]
    ]
    if (
        len(matches) != 1
        or report["weight_source"] != "pretrained"
        or report["dtype"] != "torch.bfloat16"
        or report["precision"]["weights"]["mode"] != "none"
        or not report["device"].startswith("mps")
    ):
        raise ValueError(
            "Expected a dense pretrained BF16 MPS run with one SiLU/multiply signature"
        )
    match = matches[0]
    model = (
        AutoModelForCausalLM.from_pretrained(
            report["model_id"],
            revision=report["revision"],
            dtype=torch.bfloat16,
            use_kernels=False,
        )
        .eval()
        .to("mps")
    )
    modules = {
        name: mod
        for name, mod in model.named_modules()
        if type(mod) in (LlamaMLP, Qwen3MLP)
    }
    observed = {o["consumer_scope"] for o in match["observations"]}
    if not modules or set(modules) != observed or model.config.hidden_act != "silu":
        raise ValueError(
            "Observed multiplication sites do not match the supported SwiGLU MLPs"
        )
    shader = torch.mps.compile_shader(SHADER)

    def fused(a, b):
        if (
            a.dtype != torch.bfloat16
            or b.dtype != a.dtype
            or a.shape != b.shape
            or not (a.is_contiguous() and b.is_contiguous())
        ):
            raise ValueError("Expected equal-shape contiguous BF16 inputs")
        out = torch.empty_like(a)
        shader.gate(a, b, out, threads=a.numel())
        return out

    def forward(mod, x):
        return mod.down_proj(fused(mod.gate_proj(x), mod.up_proj(x)))

    def install(stack):
        for mod in modules.values():
            stack.enter_context(patch.object(mod, "forward", partial(forward, mod)))

    shape = match["intermediate"]["shape"]
    torch.manual_seed(42)
    a = torch.randn(shape, device="mps", dtype=torch.bfloat16)
    b = torch.randn_like(a)
    reference = torch.nn.functional.silu(a) * b
    iso_error = (reference - fused(a, b)).abs().max().item()
    prompt = torch.arange(shape[1], device="mps")[None, :]

    def outputs(ids):
        first = model(ids, use_cache=True)
        token = ids[:, -1:] + 1
        second = model(token, past_key_values=first.past_key_values, use_cache=True)
        return [first.logits, second.logits]

    prompts = [
        prompt,
        torch.randint(model.config.vocab_size, prompt.shape, device="mps"),
        prompt[:, :7],
    ]
    baseline = [t for ids in prompts for t in outputs(ids)]
    with ExitStack() as stack:
        install(stack)
        candidate = [t for ids in prompts for t in outputs(ids)]
    errors = [
        (a - b).abs().max().item() for a, b in zip(baseline, candidate, strict=True)
    ]
    passed = iso_error == 0 and all(x == 0 for x in errors)
    result = {
        "model_id": report["model_id"],
        "revision": report["revision"],
        "torch": str(torch.__version__),
        "transformers": transformers.__version__,
        "platform": platform.platform(),
        "hardware": torch.backends.mps.get_name(),
        "dtype": "bfloat16",
        "profile": folder.name,
        "shape": shape,
        "sites": len(modules),
        "synthetic_input_max_abs_error": iso_error,
        "model_max_abs_errors_prefill_decode": errors,
        "exact_on_tested_inputs": passed,
        "isolated": None,
        "prefill": None,
    }
    if passed:
        result["isolated"] = paired(
            lambda name: timed(
                lambda: fused(a, b)
                if name == "candidate"
                else torch.nn.functional.silu(a) * b
            )
        )

        def run(name):
            with ExitStack() as stack:
                if name == "candidate":
                    install(stack)
                return timed(lambda: model(prompt, use_cache=True))

        result["prefill"] = paired(run)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.profile), indent=2, allow_nan=False))
