"""Select an Inductor recipe using paired model timings on the profiled workload."""

from contextlib import ExitStack, contextmanager
from copy import copy
from itertools import permutations
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

from .generation import trajectory
from .numerics import logits_close, model_logit_policy
from .trials import POLICY, _write, classify


def recipe_for(model, mode):
    if mode == "whole_model":
        return {"mode": mode, "paths": [""], "options": {}}
    paths = [
        name for name, _ in model.named_modules() if name.rsplit(".", 1)[-1] == "mlp"
    ]
    return {"mode": mode, "paths": paths, "options": {"emulate_precision_casts": True}}


def apply_recipe(model, recipe):
    for path in recipe["paths"]:
        model.get_submodule(path).compile(backend="inductor", options=recipe["options"])


def recipe_python(recipe):
    call = f'.compile(backend="inductor", options={recipe["options"]!r})'
    if recipe["mode"] == "whole_model":
        return "model" + call
    return (
        'for name, module in model.named_modules():\n    if name.rsplit(".", 1)[-1] == "mlp":\n        module'
        + call
    )


def candidate_context(model, recipe):
    # Share the loaded parameters. Only the candidate modules receive Module.compile().
    replacements = []
    for path in recipe["paths"]:
        original = model.get_submodule(path)
        candidate = copy(original)
        candidate.compile(backend="inductor", options=recipe["options"])
        replacements.append((path, original, candidate))

    @contextmanager
    def installed():
        with ExitStack() as stack:
            root = model
            for path, original, candidate in replacements:
                if path:
                    model.set_submodule(path, candidate)
                    stack.callback(model.set_submodule, path, original)
                else:
                    root = candidate
            yield root

    return installed


def _logits(model, ids, steps, key, kwargs):
    inputs = {"input_ids": ids}
    for step in range(steps + 1):
        out = model(**inputs, use_cache=True, **kwargs)
        yield out.logits.detach().cpu()
        if step < steps:
            inputs = {
                "input_ids": out.logits[:, -1].argmax(-1, keepdim=True),
                key: getattr(out, key),
            }


def measure(report, mode):
    import torch
    from torch._dynamo.utils import counters
    from .profiling import _cache_key, load_hub_workload, sequential_prompt

    source = report["config_source"]
    device = report["device"].split(":")[0]
    w = report.get("workload", {})
    length = len(w["prompt_ids"]) if w else report["phases"]["prefill"]["cache_length"]
    steps = w.get("cached_tokens", 1)
    kwargs = w.get("forward_kwargs", {})
    model, commit, _, proxy, precision = load_hub_workload(
        source["model_id"],
        revision=source["revision"],
        device=device,
        random_weights=report["weight_source"] == "random",
        dtype=report["precision"]["requested_dtype"],
        quantization=report["precision"]["requested_quantization"],
        prompt_length=length if w else None,
        max_new_tokens=steps + 1 if w else None,
    )
    if (
        commit != source["revision"]
        or precision != report["precision"]
        or proxy != report.get("proxy")
        or model.config._attn_implementation != report["attention"]
    ):
        raise ValueError("Reloaded model differs from the saved profile")
    recipe = recipe_for(model, mode)
    if not recipe["paths"]:
        return {"verdict": "unavailable", "reason": "No MLP modules identified"}
    install = candidate_context(model, recipe)
    key = _cache_key(model)
    ids = (
        torch.tensor([w["prompt_ids"]], device=device)
        if w
        else sequential_prompt(model, device, length, steps + 1)
    )
    sync = torch.mps.synchronize if device == "mps" else torch.cuda.synchronize
    policy = model_logit_policy(model.dtype)
    recorded = {
        "recipe": recipe,
        "checks": [],
        **policy,
        "prompt_length": length,
        "cached_tokens": steps,
        "forward_kwargs": kwargs,
        "cache": report["phases"]["prefill"].get("cache_class"),
    }
    initial_graphs = counters["stats"]["unique_graphs"]
    for i, prompt in enumerate(
        (
            ids,
            torch.randint(
                model.config.vocab_size,
                ids.shape,
                generator=torch.Generator().manual_seed(42),
            ).to(device),
            ids[:, :7],
        )
    ):
        reference = list(_logits(model, prompt, steps, key, kwargs))
        sync()
        start = perf_counter()
        with install() as compiled:
            for step, actual in enumerate(
                _logits(compiled, prompt, steps, key, kwargs)
            ):
                expected = reference[step]
                misses = int((~logits_close(actual, expected, policy)).sum())
                greedy = torch.equal(
                    expected[:, -1].argmax(-1), actual[:, -1].argmax(-1)
                )
                if misses or not greedy:
                    return {
                        **recorded,
                        "verdict": "failed_logits",
                        "compiled_graphs": counters["stats"]["unique_graphs"]
                        - initial_graphs,
                        "prompt": i,
                        "step": step,
                        "outside_tolerance": misses,
                        "logits": actual.numel(),
                        "greedy_match": greedy,
                    }
        if i == 0:
            recorded["first_checked_trajectory_seconds"] = perf_counter() - start
        recorded["checks"].append(
            {"prompt_length": prompt.shape[1], "logit_steps": len(reference)}
        )
    recorded["compiled_graphs"] = counters["stats"]["unique_graphs"] - initial_graphs
    if not recorded["compiled_graphs"]:
        return {
            **recorded,
            "verdict": "unavailable",
            "reason": "No compiled graph observed",
        }
    orders = list(permutations(("incumbent", "control", "candidate")))
    blocks = []
    for i in range(POLICY["warmup_blocks"] + POLICY["measured_blocks"]):
        row = {}
        for arm in orders[i % len(orders)]:
            if arm == "candidate":
                with install() as compiled:
                    row[arm] = trajectory(
                        compiled, ids, steps, key, kwargs, synchronize=sync
                    )
            else:
                row[arm] = trajectory(model, ids, steps, key, kwargs, synchronize=sync)
        if i >= POLICY["warmup_blocks"]:
            blocks.append(row)
    verdict, timing = classify(blocks)
    return {**recorded, "verdict": verdict, "timing": timing, "blocks_ms": blocks}


def compare_compilation(folder, *, progress):
    folder = Path(folder)
    outputs = []
    directory = folder / "compile-trials"
    directory.mkdir()
    for mode in ("whole_model", "mlp"):
        progress(f"Comparing Inductor ({mode}) with eager")
        path = directory / f"{mode}.json"
        with path.with_suffix(".log").open("w") as log:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "kernelatlas.compilation",
                    str(folder.resolve()),
                    mode,
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if proc.returncode or not path.is_file():
            _write(
                path,
                {
                    "verdict": "error",
                    "reason": f"Worker exited {proc.returncode}; see {mode}.log",
                },
            )
        raw = json.loads(path.read_text())
        outputs.append(
            {
                "mode": mode,
                "evidence": str(path.relative_to(folder)),
                **{k: v for k, v in raw.items() if k != "blocks_ms"},
            }
        )
    winner = max(
        (r for r in outputs if r["verdict"] == "faster"),
        key=lambda r: r["timing"]["total"]["median_paired_saving_ms"],
        default=None,
    )
    result = {
        "selected": winner["mode"] if winner else "eager",
        "recipe": winner["recipe"] if winner else None,
        "trials": outputs,
        "backend": "inductor",
        "warmup_blocks": POLICY["warmup_blocks"],
        "measured_blocks": POLICY["measured_blocks"],
        "required_wins": POLICY["required_wins"],
        "objective": "median paired whole-workload latency saving against eager",
    }
    if winner:
        result["python"] = recipe_python(winner["recipe"])
        (directory / "apply_to_model.py").write_text(
            "def apply_to_model(model):\n"
            + "\n".join("    " + line for line in result["python"].splitlines())
            + "\n    return model\n"
        )
    return result


if __name__ == "__main__":
    import torch
    import traceback

    root, mode = sys.argv[1:]
    folder = Path(root)
    try:
        with torch.inference_mode():
            record = measure(json.loads((folder / "report.json").read_text()), mode)
    except Exception as exc:
        traceback.print_exc()
        record = {"verdict": "error", "reason": f"{type(exc).__name__}: {exc}"[:1200]}
    _write(folder / "compile-trials" / f"{mode}.json", record)
