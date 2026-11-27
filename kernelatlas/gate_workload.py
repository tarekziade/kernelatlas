"""Generation-shaped gate replay and isolated trial worker for dense BF16 on MPS."""

from collections import Counter
from contextlib import contextmanager, ExitStack
from functools import partial
import inspect
from itertools import permutations
from pathlib import Path
import sys
from time import perf_counter_ns
from unittest.mock import patch

import torch
from transformers.models.llama.modeling_llama import LlamaMLP
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

from .generation import decode_distribution, trajectory
from .dataflow import capture
from .discovery import find_candidates
from .profiling import _cache_key, load_pinned_bf16_model
from .trials import POLICY, classify, _digest, _read, _write


def load_model(scope):
    model = load_pinned_bf16_model(scope)
    modules = {
        n: m for n, m in model.named_modules() if type(m) in (LlamaMLP, Qwen3MLP)
    }
    if sorted(modules) != scope["sites"] or model.config.hidden_act != "silu":
        raise ValueError("Loaded MLPs differ from the observed gate sites")
    if "logits_to_keep" not in inspect.signature(model.forward).parameters:
        raise ValueError(
            "This model does not support generation-shaped final-position logits"
        )
    return model, modules


def generation(
    model, ids, *, clock=False, cache_key=None, cached_tokens=None, on_step=None
):
    common = {"use_cache": True, "logits_to_keep": 1}
    key = cache_key or _cache_key(model)
    if cached_tokens is not None:
        return trajectory(
            model,
            ids,
            cached_tokens,
            key,
            {"logits_to_keep": 1},
            synchronize=torch.mps.synchronize if clock else None,
            on_step=on_step,
        )
    # Replay the timing boundaries originally used for v1 tasks.
    if on_step:
        on_step(0)
    if clock:
        torch.mps.synchronize()
        start = perf_counter_ns()
    first = model(ids, **common)
    token = first.logits[:, -1].argmax(-1, keepdim=True)
    if clock:
        torch.mps.synchronize()
        split = perf_counter_ns()
    if on_step:
        on_step(1)
    second = model(token, **{key: getattr(first, key)}, **common)
    if clock:
        torch.mps.synchronize()
        end = perf_counter_ns()
        return {
            "prefill": (split - start) / 1e6,
            "decode": (end - split) / 1e6,
            "total": (end - start) / 1e6,
        }
    return first.logits, second.logits


@torch.inference_mode()
def prepare(report, sites, staging):
    shape = report["phases"]["prefill"]["output_shape"]
    workload = report.get("workload")
    if shape[0] != 1 or (workload is None and shape[1] != 32):
        raise ValueError(
            "Expected a batch-one generation profile or a legacy 32-token profile"
        )
    prompt_ids = workload["prompt_ids"] if workload else list(range(32))
    steps = workload["cached_tokens"] if workload else 1
    scope = {
        k: report[k]
        for k in (
            "model_id",
            "revision",
            "torch",
            "transformers",
            "platform",
            "hardware",
            "dtype",
            "device",
            "attention",
        )
    }
    scope.update(
        sites=sites,
        prompt_ids=prompt_ids,
        cached_tokens=steps,
        logits_to_keep=1,
        reference="torch.nn.functional.silu(gate) * up",
        contract="Independent contiguous BF16 gate and up tensors of equal shape on MPS; return a fresh tensor with identical shape, dtype and device. Inputs are read-only.",
    )
    if workload:
        scope["workload"] = workload
    model, modules = load_model(scope)
    name, mod = next(iter(modules.items()))
    captured = {"gate": [], "up": []}
    with ExitStack() as stack:
        for label, proj in (("gate", mod.gate_proj), ("up", mod.up_proj)):

            def save(_mod, _args, value, label=label):
                captured[label].append(value.detach().cpu())

            stack.enter_context(proj.register_forward_hook(save))
        generation(
            model, torch.tensor([prompt_ids], device="mps"), cached_tokens=min(steps, 1)
        )
    fixtures = {}
    for idx, phase in enumerate(("prefill", "decode") if steps else ("prefill",)):
        gate, up = (captured[k][idx] for k in ("gate", "up"))
        if not (
            gate.dtype == up.dtype == torch.bfloat16
            and gate.shape == up.shape
            and gate.is_contiguous()
            and up.is_contiguous()
        ):
            raise ValueError("Observed gate inputs violate the supported contract")
        path = staging / f"{phase}.pt"
        expected = torch.nn.functional.silu(gate.to("mps")) * up.to("mps")
        torch.save({"gate": gate, "up": up, "expected": expected.cpu()}, path)
        fixtures[phase] = {
            "site": name,
            "shape": list(gate.shape),
            "stride": list(gate.stride()),
            "file": path.name,
            "sha256": _digest(path),
        }
    scope["fixtures"] = fixtures
    scope["site_source"] = {
        "class": type(mod).__qualname__,
        "file": inspect.getsourcefile(type(mod)),
        "line": inspect.getsourcelines(type(mod).forward)[1],
        "forward": inspect.getsource(type(mod).forward),
    }
    return scope


@contextmanager
def installed(modules, fn, check=None):
    if fn is None:
        yield
        return

    def forward(name, mod, x):
        gate, up = mod.gate_proj(x), mod.up_proj(x)
        y = fn(gate, up) if check is None else check(name, fn, gate, up)
        return mod.down_proj(y)

    with ExitStack() as stack:
        for name, mod in modules.items():
            stack.enter_context(
                patch.object(mod, "forward", partial(forward, name, mod))
            )
        yield


class WrongOutput(Exception):
    pass


def validate(model, modules, fn, ids, trial, result, cached_tokens=None):
    prompts = [
        ids,
        torch.randint(
            model.config.vocab_size,
            ids.shape,
            generator=torch.Generator().manual_seed(42),
        ).to("mps"),
        ids[:, :7],
    ]
    cases = result["checks"] = []
    for prompt in prompts:
        counts = {"prefill": Counter(), "decode": Counter()}
        case = {"prompt_ids": prompt[0].tolist(), "calls": counts}
        cases.append(case)
        baseline = generation(model, prompt, cached_tokens=cached_tokens)
        step = 0

        def mark(index):
            nonlocal step
            step = index

        def check(name, candidate, a, b):
            phase = "decode" if step else "prefill"
            counts[phase][name] += 1
            saved_a, saved_b = a.clone(), b.clone()
            ref = torch.nn.functional.silu(a) * b
            try:
                got = candidate(a, b)
                if (
                    not isinstance(got, torch.Tensor)
                    or got.shape != ref.shape
                    or got.dtype != ref.dtype
                    or got.device != ref.device
                ):
                    raise WrongOutput(
                        "Gate output has the wrong shape, dtype or device"
                    )
                if not torch.equal(a, saved_a) or not torch.equal(b, saved_b):
                    raise WrongOutput("Candidate modified a read-only input")
                if got.untyped_storage().data_ptr() in (
                    a.untyped_storage().data_ptr(),
                    b.untyped_storage().data_ptr(),
                ):
                    raise WrongOutput("Gate output aliases an input")
                if not torch.equal(got, ref):
                    diff = (got.float() - ref.float()).abs().max().item()
                    raise WrongOutput(
                        f"Gate output differs from eager (max absolute error {diff:g})"
                    )
            except Exception:
                torch.save(
                    {"gate": saved_a.cpu(), "up": saved_b.cpu(), "expected": ref.cpu()},
                    trial / "failure.pt",
                )
                result["failure"] = {
                    "site": name,
                    "phase": phase,
                    "generated_token_index": step,
                    "prompt_ids": case["prompt_ids"],
                    "fixture": "failure.pt",
                }
                raise
            return got

        with installed(modules, fn, check):
            outputs = generation(
                model, prompt, cached_tokens=cached_tokens, on_step=mark
            )
        steps = 1 if cached_tokens is None else cached_tokens
        if any(
            counts[phase] != Counter({name: count for name in modules})
            for phase, count in (("prefill", 1), ("decode", steps))
        ):
            raise WrongOutput("Candidate call counts differ from the recorded workload")
        for step, (expected, got) in enumerate(zip(baseline, outputs, strict=True)):
            phase = "decode" if step else "prefill"
            if not torch.equal(expected, got):
                torch.save(
                    {
                        "input_ids": prompt.cpu(),
                        "expected_logits": expected.cpu(),
                        "actual_logits": got.cpu(),
                    },
                    trial / "failure.pt",
                )
                result["failure"] = {
                    "phase": phase,
                    "generated_token_index": step,
                    "prompt_ids": case["prompt_ids"],
                    "fixture": "failure.pt",
                }
                raise WrongOutput(f"{phase} logits differ from the original model")
        case["exact_logits"] = True


def load_candidate(path, fingerprint):
    if _digest(path) != fingerprint:
        raise ValueError("Candidate fingerprint does not match its archived source")
    namespace = {"__file__": str(path), "__name__": "kernelatlas_local_candidate"}
    exec(compile(path.read_bytes(), str(path), "exec"), namespace)
    fn = namespace.get("candidate")
    if not callable(fn):
        raise ValueError("Provide a Python function named candidate(gate, up)")
    return fn


@torch.inference_mode()
def measure(root, ident, result):
    task = _read(root / "task.json")
    trial = root / "trials" / ident
    req = _read(trial / "request.json")
    torch.manual_seed(0)
    model, modules = load_model(task)
    candidate = load_candidate(trial / "candidate.py", req["sha256"])
    incumbent = None
    if req["incumbent"] != "eager":
        prior = root / "trials" / req["incumbent"]
        incumbent = load_candidate(
            prior / "candidate.py", _read(prior / "result.json")["sha256"]
        )
    ids = torch.tensor([task["prompt_ids"]], device="mps")
    steps = task["cached_tokens"] if "workload" in task else None
    validate(model, modules, candidate, ids, trial, result, steps)
    if incumbent is not None:
        incumbent_checks = trial / "incumbent-check"
        incumbent_checks.mkdir()
        try:
            validate(model, modules, incumbent, ids, incumbent_checks, {}, steps)
        except WrongOutput as exc:
            raise ValueError(f"Archived incumbent failed revalidation: {exc}") from exc
    functions = {"incumbent": incumbent, "control": incumbent, "candidate": candidate}
    orders = list(permutations(functions))
    key = _cache_key(model)
    rows = []
    for i in range(POLICY["warmup_blocks"] + POLICY["measured_blocks"]):
        row = {}
        for name in orders[i % len(orders)]:
            with installed(modules, functions[name]):
                row[name] = generation(
                    model, ids, clock=True, cache_key=key, cached_tokens=steps
                )
        if i >= POLICY["warmup_blocks"]:
            rows.append(row)
    verdict, summary = classify(rows)
    result.update(verdict=verdict, timing=summary, blocks_ms=rows)
    if steps is not None:
        result["per_cached_token"] = {
            arm: decode_distribution([row[arm] for row in rows]) for arm in functions
        }
        result["workload"] = {
            "prompt_length": len(task["prompt_ids"]),
            "max_new_tokens": steps + 1,
        }
    if verdict == "faster":
        with installed(modules, candidate):
            graph = capture(model, ids, logits_to_keep=1)
        _write(trial / "winner.dataflow.json", graph)
        result["winner_prefill_trace"] = "winner.dataflow.json"
        result["remaining_pairs"] = [
            {
                "operators": c["operators"],
                "occurrences": c["occurrences"],
                "shape": c["intermediate"]["shape"],
            }
            for c in find_candidates(graph)
        ]


if __name__ == "__main__":
    root = Path(sys.argv[1])
    ident = sys.argv[2]
    result = {}
    try:
        measure(root, ident, result)
    except WrongOutput as error:
        result.update(verdict="incorrect", reason=str(error))
    except Exception as error:
        import traceback

        traceback.print_exc()
        result.update(verdict="error", reason=f"{type(error).__name__}: {error}")
    _write(root / "trials" / ident / "measurement.json", result)
