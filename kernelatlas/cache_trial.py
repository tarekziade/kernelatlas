"""Test Transformers StaticCache against the observed growing cache on a saved MPS workload."""

from datetime import datetime, timezone
from uuid import uuid4
from functools import partial
from itertools import permutations
import json
from pathlib import Path

import torch
from transformers import StaticCache
from transformers.cache_utils import DynamicLayer, StaticLayer

from .dataflow import cache_tensors
from .discovery import analyze_run
from .generation import decode_distribution, decode_step_savings, trajectory
from .profiling import _cache_key, load_pinned_bf16_model
from .trials import POLICY, classify, _digest, _write


def reserve_for_decode(source, *, config, capacity):
    """Move dense full-attention KV state using Transformers' native update operation."""
    cache = StaticCache(config=config, max_cache_len=capacity)
    if not hasattr(source, "layers") or len(source.layers) != len(cache.layers):
        raise ValueError("Source and destination cache layer counts differ")
    for index, (old, new) in enumerate(zip(source.layers, cache.layers, strict=True)):
        if (
            type(old) is not DynamicLayer
            or type(new) is not StaticLayer
            or not old.is_initialized
        ):
            raise ValueError(
                "Decode-time migration requires initialized dense full-attention cache layers"
            )
        cache.update(old.keys, old.values, index)
    return cache


@torch.inference_mode()
def probe(run, *, reserve_at="prefill"):
    if reserve_at not in {"prefill", "decode"}:
        raise ValueError(f"Unknown cache reservation point: {reserve_at}")
    report = json.loads((run / "report.json").read_text())
    decode = analyze_run(run).get("decode", {})
    if not decode.get("cache_trial"):
        raise ValueError(
            "Requires a pinned dense BF16 MPS generation profile with observed KV cache appends."
        )
    w = report["workload"]
    steps = w["cached_tokens"]
    model = load_pinned_bf16_model(report)
    ids = torch.tensor([w["prompt_ids"]], device="mps")
    key = _cache_key(model)

    def run_model(prompt, *, static=False, timed=False):
        capacity = prompt.shape[1] + steps
        factory = (
            partial(StaticCache, config=model.config, max_cache_len=capacity)
            if static and reserve_at == "prefill"
            else None
        )
        migrate = (
            partial(reserve_for_decode, config=model.config, capacity=capacity)
            if static and reserve_at == "decode"
            else None
        )
        return trajectory(
            model,
            prompt,
            steps,
            key,
            w["forward_kwargs"],
            cache_factory=factory,
            migrate_cache=migrate,
            synchronize=torch.mps.synchronize if timed else None,
        )

    result = {
        k: report[k]
        for k in (
            "model_id",
            "revision",
            "timestamp",
            "torch",
            "transformers",
            "device",
            "dtype",
            "hardware",
            "platform",
            "attention",
        )
    }
    result.update(
        implementation="transformers.StaticCache",
        reserve_at=reserve_at,
        report_sha256=_digest(run / "report.json"),
        graph_sha256=_digest(run / decode["evidence"]),
        prompt_length=ids.shape[1],
        max_new_tokens=steps + 1,
        retained_appends=len(decode["retained_cache_appends"]),
        cache_append_output_bytes=decode["cache_append_output_bytes"],
        method="Greedy batch one. A fresh cache per trajectory; candidate construction and lazy allocations are inside timing. Require exact last-position logits at every step.",
        policy={
            k: POLICY[k]
            for k in (
                "warmup_blocks",
                "measured_blocks",
                "minimum_fraction",
                "required_wins",
                "phase_guard",
                "noise_floor",
            )
        },
        checks=[],
    )
    prompts = [
        ids,
        torch.randint(
            model.config.vocab_size,
            ids.shape,
            generator=torch.Generator().manual_seed(42),
        ).to("mps"),
        ids[:, :7],
    ]
    for idx, prompt in enumerate(prompts):
        expected = run_model(prompt)
        try:
            got = run_model(prompt, static=True)
        except (RuntimeError, ValueError, TypeError) as error:
            result.update(
                verdict="error",
                reason=f"Static cache on validation prompt {idx}: {type(error).__name__}: {error}",
            )
            return result
        for step, (a, b) in enumerate(zip(expected, got, strict=True)):
            if not torch.equal(a, b):
                result.update(
                    verdict="incorrect",
                    failure={
                        "prompt_index": idx,
                        "generated_token_index": step,
                        "different_logits": (a != b).sum().item(),
                        "max_abs_error": (a.float() - b.float()).abs().max().item(),
                        "greedy_token_match": bool(
                            torch.equal(a.argmax(-1), b.argmax(-1))
                        ),
                    },
                )
                return result
        result["checks"].append(
            {
                "prompt_index": idx,
                "length": prompt.shape[1],
                "exact_logits_positions": len(got),
            }
        )
    # Verify that the candidate reuses the observed KV storage across the same first cached step.
    capacity = ids.shape[1] + steps
    common = {"use_cache": True, **w["forward_kwargs"]}
    if reserve_at == "prefill":
        common[key] = StaticCache(config=model.config, max_cache_len=capacity)
    first = model(ids, **common)
    cache = getattr(first, key)
    if reserve_at == "decode":
        cache = reserve_for_decode(cache, config=model.config, capacity=capacity)
    common[key] = cache
    before = {(layer, slot): tensor for layer, slot, tensor in cache_tensors(cache)}
    second = model(first.logits[:, -1].argmax(-1, keepdim=True), **common)
    stable = {
        (layer, slot)
        for layer, slot, tensor in cache_tensors(getattr(second, key))
        if (layer, slot) in before
        and tensor.untyped_storage().data_ptr()
        == before[layer, slot].untyped_storage().data_ptr()
    }
    observed = {(a["layer"], a["slot"]) for a in decode["retained_cache_appends"]}
    result["static_storage"] = {
        "observed_slots": len(observed),
        "reused_after_first_cached_step": len(observed & stable),
    }
    del cache, common, first, before, second
    rows = []
    orders = list(permutations(("incumbent", "control", "candidate")))
    for i in range(POLICY["warmup_blocks"] + POLICY["measured_blocks"]):
        row = {
            arm: run_model(ids, static=arm == "candidate", timed=True)
            for arm in orders[i % len(orders)]
        }
        if i >= POLICY["warmup_blocks"]:
            rows.append(row)
    result["verdict"], result["timing"] = classify(rows)
    result["per_cached_token"] = {
        arm: decode_distribution([row[arm] for row in rows]) for arm in orders[0]
    }
    result["paired_decode_savings"] = decode_step_savings(rows)
    result["blocks_ms"] = rows
    return result


def run_cache_trial(folder, *, reserve_at="prefill"):
    run = Path(folder)
    result = probe(run, reserve_at=reserve_at)
    folder = run / "optimization" / "cache-trials"
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / f"{uuid4().hex}.json"
    result["completed_at"] = datetime.now(timezone.utc).isoformat()
    result["evidence"] = str(destination.relative_to(run))
    _write(destination, result)
    return result
