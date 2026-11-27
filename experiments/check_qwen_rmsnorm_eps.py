"""Trace RMSNorm epsilon on MPS and replay identical Qwen3 inputs in both implementations."""

import argparse
from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path
from unittest.mock import patch

import torch
import transformers
from kernels import get_kernel
from torch.utils._python_dispatch import TorchDispatchMode
from transformers import AutoModelForCausalLM

from kernelatlas.numerics import logits_close, model_logit_policy
from kernelatlas.profiling import sequential_prompt


class LaunchEpsilon(TorchDispatchMode):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights
        self.calls = Counter()
        self.operators = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if "mlx_rmsnorm" in str(func) and "launch_forward_kernel" in str(func):
            self.operators.add(str(func))
            self.calls[self.weights[args[1].data_ptr()], float(args[3])] += 1
        return func(*args, **(kwargs or {}))


@torch.inference_mode()
def audit(folder):
    report = json.loads((folder / "report.json").read_text())
    adapter = json.loads((folder / "kernel-configs.json").read_text())["adapters"][
        "RMSNorm"
    ]
    if (
        report["model_id"],
        report["weight_source"],
        report["dtype"],
        report["device"],
        report["hardware"],
        report["torch"],
        report["transformers"],
    ) != (
        "Qwen/Qwen3-0.6B",
        "pretrained",
        "torch.bfloat16",
        "mps:0",
        torch.backends.mps.get_name(),
        str(torch.__version__),
        transformers.__version__,
    ) or report.get("workload"):
        raise ValueError(
            "Select a matching pretrained Qwen3-0.6B default BF16 MPS profile"
        )

    def load(use_kernels):
        return AutoModelForCausalLM.from_pretrained(
            report["model_id"],
            revision=report["revision"],
            dtype=torch.bfloat16,
            device_map="mps",
            attn_implementation=report["attention"],
            use_kernels=use_kernels,
        ).eval()

    native = load(False)
    length = report["phases"]["prefill"]["cache_length"]
    prompt = sequential_prompt(native, "mps", length, 2)

    def forward(model):
        return model(prompt, use_cache=True).logits.cpu()

    reference = forward(native)
    samples = {}

    def capture(name):
        def hook(module, args, output):
            samples[name] = (args[0].detach(), output.detach())

        return hook

    with ExitStack() as stack:
        for name, module in native.named_modules():
            if type(module).__name__ == "Qwen3RMSNorm":
                stack.callback(module.register_forward_hook(capture(name)).remove)
        assert torch.equal(forward(native), reference)
    hub = load(True)
    bound = {
        name: m
        for name, m in hub.named_modules()
        if getattr(vars(m).get("_kernel_repo"), "_repo_id", None) == adapter["repo"]
    }
    assert set(samples) == set(bound) and bound
    kernel = get_kernel(adapter["repo"], revision=adapter["revision"])
    tracker = LaunchEpsilon({m.weight.data_ptr(): name for name, m in bound.items()})
    observed = forward(hub)
    with tracker:
        traced = forward(hub)
    pinned_op = kernel.rmsnorm_forward.__globals__["ops"].launch_forward_kernel.default
    assert tracker.operators == {str(pinned_op)}
    assert torch.equal(observed, traced)
    assert {name for name, _ in tracker.calls} == set(bound)
    eps_counts = Counter()
    for (_, eps), count in tracker.calls.items():
        eps_counts[str(eps)] += count
    replays = {}
    for name, (x, expected) in samples.items():
        module = native.get_submodule(name)
        assert torch.equal(module.weight, bound[name].weight)
        eps = module.variance_epsilon
        output = kernel.rmsnorm_forward(x, module.weight, eps)
        delta = (output.float() - expected.float()).abs()
        replays[name] = dict(
            shape=list(x.shape),
            strides=list(x.stride()),
            native_eps=eps,
            adapter_eps=bound[name].variance_epsilon,
            differing_values=int(delta.count_nonzero()),
            max_abs_error=delta.max().item(),
        )
    name = next(iter(samples))
    x, _ = samples[name]
    module = native.get_submodule(name)
    control = kernel.rmsnorm_forward(x, module.weight, 1e-5)
    original = kernel.rmsnorm_forward(x, module.weight, module.variance_epsilon)
    policy = model_logit_policy(report["dtype"])
    # Leave the rest of the automatically loaded model intact and swap just the normalizers.
    with ExitStack() as replacements:
        for site, module in bound.items():
            replacements.enter_context(
                patch.object(module, "forward", native.get_submodule(site).forward)
            )
        restored = forward(hub)
    first_difference = next(
        (s for s, r in replays.items() if r["differing_values"]), None
    )
    return {
        "native_norms_restored_in_hub_model": {
            "sites": len(bound),
            "different_logits": int((restored != reference).sum()),
            "outside_tolerance": int(
                (~logits_close(restored, reference, policy)).sum()
            ),
        },
        "model": report["model_id"],
        "revision": report["revision"],
        "torch": report["torch"],
        "transformers": report["transformers"],
        "hardware": report["hardware"],
        "dtype": report["dtype"],
        "execution": "eager",
        "attention": report["attention"],
        "prompt_ids": prompt[0].tolist(),
        "kernel": adapter["repo"],
        "kernel_revision": adapter["revision"],
        "observed_operator": str(pinned_op),
        "config_eps": native.config.rms_norm_eps,
        "sites": len(bound),
        "launch_epsilon_counts": dict(eps_counts),
        "replays_on_native_inputs": replays,
        "first_site_with_difference": first_difference,
        "changed_epsilon_control": {
            "site": name,
            "eps": 1e-5,
            "changed_values": int((control != original).sum()),
            "elements": x.numel(),
        },
        "logit_policy": policy,
        "prefill_logits": observed.numel(),
        "outside_tolerance": int((~logits_close(observed, reference, policy)).sum()),
        "same_next_token": torch.equal(
            reference[:, -1].argmax(-1), observed[:, -1].argmax(-1)
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    print(json.dumps(audit(parser.parse_args().run), indent=2))
