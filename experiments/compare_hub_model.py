"""Replay the default Transformers RMSNorm integration against a pinned MPS baseline."""

import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
from functools import partial
import hashlib
import importlib.metadata
import json
from pathlib import Path
from statistics import median
from unittest.mock import patch

import kernels
import torch
import transformers
from huggingface_hub import constants, snapshot_download
from transformers import AutoModelForCausalLM
from transformers.integrations.hub_kernels import get_kernel_mapping_transformers

from evaluate_rmsnorm import compile_metal_tail, paired, timed
from profile_decoder import add_module_scopes


def forward_steps(model, prompt):
    tokens = prompt
    cache = None
    outputs = []
    for _ in range(4):
        result = model(tokens, cache=cache, use_cache=True)
        outputs.append(result.logits)
        tokens = result.logits[:, -1].argmax(-1, keepdim=True)
        cache = result.cache
    return outputs


def trace_sites(model, prompt, operators):
    with ExitStack() as hooks:
        add_module_scopes(hooks, model)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as prof:
            result = model(prompt, use_cache=True)
            torch.mps.synchronize()
    sites = {op: Counter() for op in operators}
    for event in prof.events():
        if event.name in sites:
            scope = event.cpu_parent
            while scope and not scope.name.startswith("module/"):
                scope = scope.cpu_parent
            name = scope.name.removeprefix("module/") if scope else "<unattributed>"
            sites[event.name][name] += 1
    return sites, result.logits


def check(baseline, candidate, tolerance):
    pairs = list(zip(baseline, candidate, strict=True))
    outside = [int((~torch.isclose(a, b, **tolerance)).sum()) for a, b in pairs]
    greedy = [torch.equal(a[:, -1].argmax(-1), b[:, -1].argmax(-1)) for a, b in pairs]
    a, b = pairs[0]
    return {
        "prefill_mismatches_by_token": (~torch.isclose(a, b, **tolerance))
        .sum(-1)
        .flatten()
        .tolist(),
        "prefill_top1_changes": int((a.argmax(-1) != b.argmax(-1)).sum()),
        "mismatched_logits": outside,
        "max_abs_error": [(a - b).abs().max().item() for a, b in pairs],
        "greedy_matches": greedy,
        "passed": not any(outside) and all(greedy),
    }


def measure(model, control, prompt):
    samples = paired(
        lambda: control(prompt, use_cache=True), lambda: model(prompt, use_cache=True)
    )

    def summary(samples):
        deltas = [
            a - b
            for a, b in zip(samples["baseline"], samples["candidate"], strict=True)
        ]
        return {
            "median_paired_saving_ms": median(deltas),
            "faster_pairs": sum(d > 0 for d in deltas),
        }

    decode = {"baseline": [], "candidate": []}
    order = [("baseline", control), ("candidate", model)]
    for i in range(24):
        for label, instance in order if i % 2 == 0 else reversed(order):
            prefill = instance(prompt, use_cache=True)
            token = prefill.logits[:, -1].argmax(-1, keepdim=True)
            _, elapsed = timed(
                lambda: instance(token, cache=prefill.cache, use_cache=True)
            )
            if i >= 4:
                decode[label].append(elapsed)
    return {
        "synchronized_prefill_ms": samples,
        **summary(samples),
        "decode_one_token": {"synchronized_ms": decode, **summary(decode)},
    }


@torch.inference_mode()
def compare(run, length):
    if not constants.HF_HUB_OFFLINE:
        raise ValueError("Set HF_HUB_OFFLINE=1 to resolve the pinned cached v1 build")
    previous = json.loads((run / "report.json").read_text())
    verified = json.loads((run / "native.json").read_text())
    mapping = get_kernel_mapping_transformers()["RMSNorm"]["mps"][
        kernels.Mode.INFERENCE
    ]
    kernel_id = "kernels-community/mlx-rmsnorm"
    if mapping != kernels.LayerRepository(kernel_id, layer_name="RMSNorm", version=1):
        raise ValueError(f"Unexpected default mapping {mapping}")
    snapshot = Path(
        snapshot_download(
            kernel_id,
            revision="v1",
            repo_type="kernel",
            allow_patterns=["build/torch215-metal-aarch64-darwin/*"],
            local_files_only=True,
        )
    )
    if snapshot.name != "d5793d964608e8e1ee2c4de42e83341b72e5f3dc":
        raise ValueError(
            f"Cached v1 resolves to an unexpected revision {snapshot.name}"
        )
    metadata = json.loads(
        (snapshot / "build/torch215-metal-aarch64-darwin/metadata.json").read_text()
    )
    op = metadata["id"] + "::launch_forward_kernel"

    def load():
        return (
            AutoModelForCausalLM.from_pretrained(
                previous["weights"],
                revision=previous["revision"],
                local_files_only=True,
                dtype=torch.float32,
                attn_implementation="sdpa",
                use_kernels=False,
            )
            .eval()
            .to("mps")
        )

    control, candidate = load(), load()
    torch.manual_seed(0)
    prompt = torch.arange(length, device="mps")[None, :]
    expected = forward_steps(control, prompt)
    unchanged = forward_steps(candidate, prompt)
    if not all(torch.equal(a, b) for a, b in zip(expected, unchanged, strict=True)):
        raise ValueError("Independent eager model copies produced different logits")
    eager_sites, eager_logits = trace_sites(control, prompt, (op, "aten::rsqrt"))
    intended = Counter({name: 1 for name in verified["occurrences"]})
    if (
        eager_sites[op]
        or eager_sites["aten::rsqrt"] != intended
        or not torch.equal(eager_logits, expected[0])
    ):
        raise ValueError("Eager execution no longer matches the saved scopes")
    candidate.set_use_kernels(True)
    hub_sites, diagnostic = trace_sites(candidate, prompt, (op, "aten::rsqrt"))
    hub = forward_steps(candidate, prompt)
    if not torch.equal(hub[0], diagnostic):
        raise ValueError("Instrumented and uninstrumented Hub model outputs differ")
    tolerance = verified["logits_tolerance"]
    hub_result = check(expected, hub, tolerance)
    fully_integrated = hub_sites[op] == intended and not hub_sites["aten::rsqrt"]
    shader = compile_metal_tail()
    metal_model = load()
    with ExitStack() as patches:
        for name in intended:
            module = metal_model.get_submodule(name)
            patches.enter_context(
                patch.object(module, "forward", partial(shader, module))
            )
        metal_result = check(expected, forward_steps(metal_model, prompt), tolerance)
        metal_time = (
            measure(metal_model, control, prompt) if metal_result["passed"] else None
        )
    hub_time = (
        measure(candidate, control, prompt)
        if fully_integrated and hub_result["passed"]
        else None
    )
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "weights": previous["weights"],
        "revision": previous["revision"],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "kernels": kernels.__version__,
        "urllib3": importlib.metadata.version("urllib3"),
        "model_source_sha256": hashlib.sha256(
            Path(transformers.models.llama.modeling_llama.__file__).read_bytes()
        ).hexdigest(),
        "device": str(prompt.device),
        "hardware": torch.backends.mps.get_name(),
        "dtype": "float32",
        "attention": "sdpa",
        "cache": "dynamic",
        "batch": 1,
        "input_token_ids": prompt[0].tolist(),
        "decode_steps": 3,
        "trace_method": "PyTorch CPU profiler; counts of registered operator calls inside named module scopes during prefill",
        "timing_method": "Synchronized forward wall clock, alternating order, four warmup pairs and twenty measured pairs per phase. Decode uses a freshly populated cache; prefill and token selection are outside its timed interval.",
        "kernel_id": kernel_id,
        "kernel_revision": snapshot.name,
        "kernel_op": op,
        "tolerance": tolerance,
        "module_source": "default Transformers MPS inference mapping; set_use_kernels(True)",
        "baseline_copies_identical": True,
        "eager_events": eager_sites,
        "hub_events": hub_sites,
        "hub": {
            "all_sites_dispatched": fully_integrated,
            "correctness": hub_result,
            "timing": hub_time,
        },
        "metal_tail": {"correctness": metal_result, "timing": metal_time},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--length", type=int, choices=(32, 128), default=32)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.run, args.length)
    with args.out.open("x") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")
