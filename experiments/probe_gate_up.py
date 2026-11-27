"""Feasibility check for one packed BF16 gate/up projection on a saved gate workload."""

import argparse
from contextlib import ExitStack, contextmanager
from functools import partial

from kernelatlas.generation import decode_distribution
from itertools import permutations
import json
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn.functional import linear

from kernelatlas.gate_workload import generation, load_model
from kernelatlas.profiling import _cache_key
from kernelatlas.trials import POLICY, classify


@torch.inference_mode()
def probe(run):
    task = json.loads((run / "optimization/task.json").read_text())
    model, sites = load_model(task)
    if any(
        layer.gate_proj.bias is not None or layer.up_proj.bias is not None
        for layer in sites.values()
    ):
        raise ValueError("This probe handles bias-free projections")
    weights = {
        name: torch.cat([mod.gate_proj.weight, mod.up_proj.weight])
        for name, mod in sites.items()
    }
    ids = torch.tensor([task["prompt_ids"]], device="mps")
    replay = partial(
        generation, cached_tokens=task["cached_tokens"] if "workload" in task else None
    )
    result = {
        "model_id": task["model_id"],
        "revision": task["revision"],
        "hardware": task["hardware"],
        "torch": task["torch"],
        "transformers": task["transformers"],
        "dtype": task["dtype"],
        "sites": len(sites),
        "extra_packed_bytes_during_ab": sum(w.nbytes for w in weights.values()),
        "workload": {
            "prompt_length": ids.shape[1],
            "max_new_tokens": task["cached_tokens"] + 1,
            "logits_to_keep": 1,
        },
        "correctness": "Exact individual projections and last-position logits on the three gate-runner prompts",
        "checked_projection_calls": 0,
        "verdict": "incorrect",
    }

    @contextmanager
    def combined(check=False):
        def forward(name, layer, x):
            packed = linear(x, weights[name])
            gate, up = packed.chunk(2, dim=-1)
            if check:
                for label, actual, op in (
                    ("gate", gate, layer.gate_proj),
                    ("up", up, layer.up_proj),
                ):
                    ref = op(x)
                    if not torch.equal(actual, ref):
                        result["failure"] = {
                            "site": name,
                            "projection": label,
                            "shape": list(x.shape),
                            "different_elements": (actual != ref).sum().item(),
                            "max_abs_error": (actual.float() - ref.float())
                            .abs()
                            .max()
                            .item(),
                        }
                        raise ArithmeticError("Projection mismatch")
                result["checked_projection_calls"] += 1
            return layer.down_proj(layer.act_fn(gate) * up)

        with ExitStack() as stack:
            for name, mod in sites.items():
                stack.enter_context(
                    patch.object(mod, "forward", partial(forward, name, mod))
                )
            yield

    prompts = [
        ids,
        torch.randint(
            model.config.vocab_size,
            ids.shape,
            generator=torch.Generator().manual_seed(42),
        ).to("mps"),
        ids[:, :7],
    ]
    for i, prompt in enumerate(prompts):
        expected = replay(model, prompt)
        with combined(check=True):
            try:
                got = replay(model, prompt)
            except ArithmeticError:
                result["prompt_index"] = i
                return result
        if any(not torch.equal(a, b) for a, b in zip(expected, got, strict=True)):
            result["failure"] = {
                "reason": "last-position logit mismatch",
                "prompt_index": i,
            }
            return result
    key = _cache_key(model)
    rows = []
    orders = list(permutations(("incumbent", "control", "candidate")))
    for i in range(POLICY["warmup_blocks"] + POLICY["measured_blocks"]):
        row = {}
        for name in orders[i % len(orders)]:
            with ExitStack() as stack:
                if name == "candidate":
                    stack.enter_context(combined())
                row[name] = replay(model, ids, clock=True, cache_key=key)
        if i >= POLICY["warmup_blocks"]:
            rows.append(row)
    result["verdict"], result["timing"] = classify(rows)
    if "workload" in task:
        result["per_cached_token"] = {
            name: decode_distribution([row[name] for row in rows]) for name in orders[0]
        }
    result["blocks_ms"] = rows
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    print(json.dumps(probe(parser.parse_args().run), indent=2, allow_nan=False))
