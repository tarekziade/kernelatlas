"""Replay the saved pretrained FP32 prefill, changing one verified module per run."""

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch
import transformers
from transformers import AutoModelForCausalLM

from evaluate_rmsnorm import replacement


def difference(baseline, candidate, tolerance):
    # Keep the argument order used by the original gate.
    delta = (baseline - candidate).abs()
    return {
        "changed": int(torch.count_nonzero(delta)),
        "outside_tolerance": int(
            (~torch.isclose(baseline, candidate, **tolerance)).sum()
        ),
        "max_abs": delta.max().item(),
        "elements": delta.numel(),
    }


def describe_run(baseline, candidate, tolerance):
    states = []
    for index, (a, b) in enumerate(
        zip(baseline.hidden_states, candidate.hidden_states, strict=True)
    ):
        stats = difference(a, b, tolerance)
        states.append({"index": index, **stats})
    return {
        "logits": difference(baseline.logits, candidate.logits, tolerance),
        "greedy_token_equal": torch.equal(
            baseline.logits[:, -1].argmax(-1), candidate.logits[:, -1].argmax(-1)
        ),
        "hidden_states": states,
    }


def local_error(module, x, tolerance):
    baseline = module(x)
    candidate = replacement(module, x)
    xd = x.cpu().double()
    ms = xd.square().mean(-1, keepdim=True)
    truth = module.weight.cpu().double() * xd * (ms + module.variance_epsilon).rsqrt()
    return {
        "shape": list(x.shape),
        "stride": list(x.stride()),
        "input_sha256_fp32": hashlib.sha256(
            x.cpu().contiguous().numpy().tobytes()
        ).hexdigest(),
        "row_mean_square_min": ms.min().item(),
        "row_mean_square_max": ms.max().item(),
        "eps": module.variance_epsilon,
        "max_abs_vs_cpu_fp64": {
            "eager": (baseline.cpu().double() - truth).abs().max().item(),
            "native": (candidate.cpu().double() - truth).abs().max().item(),
        },
        "eager_vs_native": difference(baseline, candidate, tolerance),
    }


@torch.inference_mode()
def diagnose(folder):
    report = json.loads((folder / "report.json").read_text())
    prior = json.loads((folder / "native.json").read_text())
    if (report["torch"], report["transformers"], report["device"], report["dtype"]) != (
        torch.__version__,
        transformers.__version__,
        "mps:0",
        "torch.float32",
    ):
        raise ValueError("Replay requires the original MPS environment")
    model = (
        AutoModelForCausalLM.from_pretrained(
            report["weights"],
            revision=report["revision"],
            local_files_only=True,
            dtype=torch.float32,
            attn_implementation=report["attention"],
        )
        .eval()
        .to("mps")
    )
    prompt = torch.arange(32, device="mps")[None, :]
    tolerance = prior["logits_tolerance"]
    names = list(prior["occurrences"])
    inputs = {}

    def capture(name, module, args):
        inputs[name] = args[0].detach().clone(memory_format=torch.preserve_format)

    def forward():
        return model(prompt, use_cache=True, output_hidden_states=True)

    with ExitStack() as hooks:
        for name in names:
            hook = model.get_submodule(name).register_forward_pre_hook(
                partial(capture, name)
            )
            hooks.callback(hook.remove)
        captured = forward()
    baseline = forward()
    controls = [
        describe_run(baseline, other, tolerance)
        for other in (captured, forward(), forward())
    ]
    sites = []
    for name in names:
        module = model.get_submodule(name)
        local = local_error(module, inputs[name], tolerance)
        with patch.object(module, "forward", partial(replacement, module)):
            result = describe_run(baseline, forward(), tolerance)
        sites.append({"name": name, "local": local, **result})
    with ExitStack() as patches:
        for name in names:
            module = model.get_submodule(name)
            patches.enter_context(
                patch.object(module, "forward", partial(replacement, module))
            )
        combined = describe_run(baseline, forward(), tolerance)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "workload_report": "report.json",
        "original_evaluation": "native.json",
        "input_token_ids": prompt[0].tolist(),
        "tolerance": tolerance,
        "controls": controls,
        "control_order": ["with_input_hooks", "unmodified", "unmodified"],
        "sites": sites,
        "combined": combined,
        "final_control": describe_run(baseline, forward(), tolerance),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(0)
    data = diagnose(args.run)
    with args.out.open("x") as output:
        json.dump(data, output, indent=2, allow_nan=False)
        output.write("\n")
