"""Evaluate a native FP32 RMSNorm replacement for scopes discovered in a decoder trace."""

import argparse
import json
import sys
from contextlib import ExitStack, nullcontext
from pathlib import Path
from statistics import median
from time import perf_counter_ns
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch.fx.experimental.proxy_tensor import make_fx
from transformers import AutoModelForCausalLM, LlamaConfig, Qwen3Config


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernelatlas.metal import compile_metal_tail  # noqa: E402


def occurrences(events):
    by_id = {event["id"]: event for event in events}
    matches = {}
    for event in events:
        if event["name"] != "aten::rsqrt":
            continue
        parent = event
        while parent["parent"] is not None:
            parent = by_id[parent["parent"]]
            if parent["name"].startswith("module/"):
                matches[parent["name"].removeprefix("module/")] = event["id"]
                break
    return matches


def reference(x, weight, eps):
    inv = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return weight * (x * inv)


def replacement(module, x):
    return F.rms_norm(
        x, tuple(module.weight.shape), module.weight, module.variance_epsilon
    )


def timed(fn):
    torch.mps.synchronize()
    start = perf_counter_ns()
    value = fn()
    torch.mps.synchronize()
    return value, (perf_counter_ns() - start) / 1e6


def paired(a, b):
    samples = {"baseline": [], "candidate": []}
    cases = [("baseline", a), ("candidate", b)]
    for i in range(24):
        for key, fn in cases[:: 1 if i % 2 == 0 else -1]:
            _, sample = timed(fn)
            if i >= 4:
                samples[key].append(sample)
    return samples


@torch.inference_mode()
def evaluate(folder, implementation="native"):
    apply = replacement if implementation == "native" else compile_metal_tail()
    report = json.loads((folder / "report.json").read_text())
    if report["torch"] != torch.__version__ or report["dtype"] != "torch.float32":
        raise ValueError("This experiment requires the trace runtime and FP32 contract")
    import transformers

    if (
        report["transformers"] != transformers.__version__
        or report["device"] != "mps:0"
    ):
        raise ValueError(
            "Trace must match the installed Transformers version and target"
        )
    args = report["config"]
    cls = {"llama": LlamaConfig, "qwen3": Qwen3Config}[args["model_type"]]
    torch.manual_seed(0)
    if report.get("revision"):
        model = (
            AutoModelForCausalLM.from_pretrained(
                report["weights"],
                revision=report["revision"],
                local_files_only=True,
                dtype=torch.float32,
                attn_implementation="sdpa",
            )
            .eval()
            .to("mps")
        )
    else:
        model = (
            AutoModelForCausalLM.from_config(
                cls.from_dict(args), attn_implementation="sdpa"
            )
            .eval()
            .to("mps")
        )
    scopes = occurrences(json.loads((folder / "prefill.events.json").read_text()))
    verified = {}
    isolated = {}
    for name, event_id in scopes.items():
        module = model.get_submodule(name)
        weight = getattr(module, "weight", None)
        if (
            weight is None
            or weight.ndim != 1
            or not hasattr(module, "variance_epsilon")
        ):
            continue
        width = weight.numel()
        x = torch.randn(1, 32, width, device="mps")
        # Trace reference and model with the same captured parameter. FX code retains
        # constants (including epsilon), operands, reduction axes and dependencies.
        expected = make_fx(lambda x: reference(x, weight, module.variance_epsilon))(x)
        actual = make_fx(lambda x: module(x))(x)
        if actual.code != expected.code:
            continue
        for length in (1, 32):
            for step in (1, 2):
                test = torch.randn(2, length, width * step, device="mps")[..., ::step]
                torch.testing.assert_close(
                    apply(module, test), module(test), rtol=1e-5, atol=1e-6
                )
        verified[name] = {
            "trace_event_id": event_id,
            "width": width,
            "graph": actual.code,
        }
        if str(width) not in isolated:
            isolated[str(width)] = paired(lambda: module(x), lambda: apply(module, x))
    if not verified:
        raise ValueError("No exact reference graph matched")

    def patched():
        stack = ExitStack()
        for name in verified:
            mod = model.get_submodule(name)
            stack.enter_context(
                patch.object(mod, "forward", lambda x, m=mod: apply(m, x))
            )
        return stack

    prompt = torch.arange(32, device="mps")[None, :]

    def generate():
        inputs = prompt
        cache = None
        logits = []
        for _ in range(4):
            out = model(inputs, past_key_values=cache, use_cache=True)
            logits.append(out.logits)
            inputs = out.logits[:, -1].argmax(-1, keepdim=True)
            cache = out.past_key_values
        return logits

    baseline = generate()
    with patched():
        candidate = generate()
    pairs = list(zip(baseline, candidate, strict=True))
    mismatches = [
        int((~torch.isclose(a, b, rtol=1e-4, atol=1e-5)).sum()) for a, b in pairs
    ]
    greedy_match = all(
        torch.equal(a[:, -1].argmax(-1), b[:, -1].argmax(-1)) for a, b in pairs
    )
    passed = not any(mismatches) and greedy_match
    evaluation = {
        "prompt_length": prompt.shape[1],
        "pattern": "weighted_last_dim_rmsnorm_fp32",
        "trace": "prefill.events.json",
        "implementation": "torch.nn.functional.rms_norm"
        if implementation == "native"
        else "rmsnorm_tail.metal",
        "occurrences": verified,
        "max_abs_logits_error_by_step": [(a - b).abs().max().item() for a, b in pairs],
        "greedy_tokens_match": greedy_match,
        "logits_tolerance": {"rtol": 1e-4, "atol": 1e-5},
        "mismatched_logits_by_step": mismatches,
        "correctness": "passed_tested_inputs" if passed else "rejected",
        "isolated_ms_by_width": isolated,
        "prefill_ms": None,
    }
    if not passed:
        return evaluation

    # Install replacements outside each timed forward.
    samples = {"baseline": [], "candidate": []}
    for i in range(24):
        for key in (
            ("baseline", "candidate") if i % 2 == 0 else ("candidate", "baseline")
        ):
            with patched() if key == "candidate" else nullcontext():
                _, elapsed = timed(lambda: model(prompt, use_cache=True))
                if i >= 4:
                    samples[key].append(elapsed)
    evaluation["prefill_ms"] = samples
    evaluation["median_prefill_ms"] = {
        key: median(values) for key, values in samples.items()
    }
    return evaluation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument(
        "--implementation", choices=("native", "metal-tail"), default="native"
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Write to a new result file after a successful evaluation",
    )
    args = parser.parse_args()
    result = json.dumps(
        evaluate(args.run, args.implementation), indent=2, allow_nan=False
    )
    if args.out is None:
        print(result)
    else:
        with args.out.open("x") as output:
            output.write(result + "\n")
