"""Compare published Transformers loading options in independent local processes."""

from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
from importlib.metadata import version
from itertools import permutations
import json
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

from .numerics import logits_close, model_logit_policy
from .trials import POLICY, classify, _write


def scope(report, options):
    w = report.get("workload")
    return {
        **{
            k: report.get(k)
            for k in (
                "model_id",
                "revision",
                "torch",
                "transformers",
                "hardware",
                "device",
                "platform",
                "dtype",
                "attention",
                "precision",
            )
        },
        **(
            {"compile_recipe": options["compile_recipe"]}
            if options.get("compile_recipe")
            else {}
        ),
        "logit_policy": model_logit_policy(report["dtype"]),
        "kernels": version("kernels"),
        "workload": w
        or {
            "mode": "full_prompt_logits_plus_one_cached_forward",
            "prompt_length": report["phases"]["prefill"]["cache_length"],
        },
        "configs": [
            {k: v for k, v in c.items() if k != "python"} for c in options["configs"]
        ],
        "adapter_revisions": {k: a["revision"] for k, a in options["adapters"].items()},
    }


def matching_measurement(report, options, evidence):
    return matching_scope(json.loads(json.dumps(scope(report, options))), evidence)


def matching_scope(current, evidence):
    if not current["hardware"] or not current["revision"]:
        return None
    matches = []
    for path in (Path(evidence) / "hub-loading").glob("*.json"):
        record = json.loads(path.read_text())
        if record.get("scope") == current:
            matches.append(record)
    return max(matches, key=lambda r: r["completed_at"]) if matches else None


def saved_guidance(root, model_id, family=None, torch_version=None, prompt_length=None):
    """Look up a portable observation when a full local trace is unavailable."""
    records = []
    for path in (Path(root) / "hub-loading").glob("*.json"):
        record = json.loads(path.read_text())
        scope = record["scope"]
        if (
            scope["model_id"] != model_id
            or family
            and record.get("family") != family
            or torch_version
            and scope["torch"] != torch_version
            or prompt_length
            and scope["workload"].get("prompt_length") != prompt_length
        ):
            continue
        if "development_tasks" in record:
            records.append((record["completed_at"], str(path), record))
    if records:
        _, path, record = max(records)
        return {
            "evidence": path,
            "comparison": record,
            "hub_summary": record.get("hub_summary"),
            "development_tasks": record["development_tasks"],
        }


def _measure(report, options, config, *, prepare_model=None):
    import torch
    from transformers import AutoModelForCausalLM, KernelConfig, __version__
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from .generation import trajectory
    from .profiling import _cache_key, sequential_prompt

    device = options["device"]
    if (
        str(torch.__version__) != report["torch"]
        or __version__ != report["transformers"]
    ):
        raise ValueError("Installed runtime differs from the saved profile")
    if device == "mps" and report["hardware"] != torch.backends.mps.get_name():
        raise ValueError("MPS device differs from the saved profile")
    w = report.get("workload") or {}
    steps = w.get("cached_tokens", 1)
    length = len(w["prompt_ids"]) if w else report["phases"]["prefill"]["cache_length"]
    kwargs = w.get("forward_kwargs", {})
    common = dict(
        revision=report["revision"],
        device_map=device,
        dtype=getattr(torch, options["dtype"]),
    )

    def load(**settings):
        model = AutoModelForCausalLM.from_pretrained(
            options["model_id"], **common, **settings
        ).eval()

        if prepare_model is not None:
            prepare_model(model)
        elif options.get("compile_recipe"):
            from .compilation import apply_recipe

            apply_recipe(model, options["compile_recipe"])
        return model

    eager = load(use_kernels=False, attn_implementation=report["attention"])
    ids = (
        torch.tensor([w["prompt_ids"]], device=device)
        if w
        else sequential_prompt(eager, device, length, steps + 1)
    )
    key = _cache_key(eager)
    prompts = [
        ids,
        torch.randint(
            eager.config.vocab_size,
            ids.shape,
            generator=torch.Generator().manual_seed(42),
        ).to(device),
        ids[:, :7],
    ]

    def outputs(model, prompt):
        inputs = {"input_ids": prompt}
        saved = []
        for i in range(steps + 1):
            out = model(**inputs, use_cache=True, **kwargs)
            saved.append(out.logits.cpu())
            if i < steps:
                inputs = {
                    "input_ids": out.logits[:, -1].argmax(-1, keepdim=True),
                    key: getattr(out, key),
                }
        return saved

    settings = {
        k: config[k] for k in ("use_kernels", "attn_implementation") if k in config
    }
    if "kernel_mapping" in config:
        settings["kernel_config"] = KernelConfig(
            kernel_mapping={k: tuple(v) for k, v in config["kernel_mapping"].items()},
            inherit_mapping=False,
        )
    reference = outputs(eager, ids)
    model = load(**settings)
    bindings = {}
    calls = Counter()
    for name, mod in model.named_modules():
        for path, site in [
            (name, mod),
            *(
                (f"{name}.{k}", v)
                for k, v in vars(mod).get("_kernel_funcs", {}).items()
            ),
        ]:
            repo = vars(site).get("_kernel_repo")
            if repo is not None:
                bindings[path] = (
                    site,
                    getattr(site, "kernel_layer_name", None),
                    f"{repo._repo_id}:{repo.layer_name}",
                )
    attention = model.config._attn_implementation
    attn_calls = []
    with ExitStack() as stack:
        for site, layer, target in bindings.values():

            def hook(_mod, _args, label=target):
                calls[label] += 1

            stack.callback(site.register_forward_pre_hook(hook).remove)
        if "attn_implementation" in settings:
            original = ALL_ATTENTION_FUNCTIONS[attention]

            def record(*a, **kw):
                attn_calls.append(1)
                return original(*a, **kw)

            ALL_ATTENTION_FUNCTIONS.register(attention, record)
            stack.callback(ALL_ATTENTION_FUNCTIONS.register, attention, original)
        observed = outputs(model, ids)
    result = {
        "bindings": dict(Counter(v[2] for v in bindings.values())),
        "bound_module_calls": dict(calls),
        "attention": attention,
        "attention_interface_calls": len(attn_calls),
        "checks": [],
    }
    if settings.get("attn_implementation") and (
        attention != settings["attn_implementation"] or not attn_calls
    ):
        return {
            **result,
            "verdict": "unbound",
            "reason": "Requested attention path was not invoked",
        }
    requested = config.get("kernel_mapping", {})
    for layer, (target, _) in requested.items():
        if (
            sum(k == layer and t == target for _, k, t in bindings.values())
            != options["registered_sites"][layer]
            or not calls[target]
        ):
            return {
                **result,
                "verdict": "unbound",
                "reason": f"Binding or execution incomplete for {layer}",
            }
    if requested and any(
        requested.get(layer, [None])[0] != target
        for _, layer, target in bindings.values()
    ):
        return {
            **result,
            "verdict": "unbound",
            "reason": "Found an unselected module binding",
        }
    if not calls and "attn_implementation" not in settings:
        return {**result, "verdict": "unbound", "reason": "No Hub layer calls observed"}
    policy = model_logit_policy(options["dtype"])
    if options.get("compile_recipe"):
        observed = outputs(
            model, ids
        )  # Validate the same hook-free forward used during timing.
    for i, prompt in enumerate(prompts):
        if i:
            reference, observed = outputs(eager, prompt), outputs(model, prompt)
        for step, (a, b) in enumerate(zip(reference, observed, strict=True)):
            ok = logits_close(b, a, policy)
            greedy = torch.equal(a[:, -1].argmax(-1), b[:, -1].argmax(-1))
            if not bool(ok.all()) or not greedy:
                return {
                    **result,
                    "verdict": "incorrect",
                    "failure": {
                        "prompt_index": i,
                        "step": step,
                        "outside_tolerance": int((~ok).sum()),
                        "logits": ok.numel(),
                        "max_abs_error": (a.float() - b.float()).abs().max().item(),
                        "greedy_token_match": greedy,
                        **policy,
                    },
                }
        result["checks"].append(
            {
                "prompt_index": i,
                "length": prompt.shape[1],
                "positions": len(reference),
                **policy,
            }
        )
    sync = torch.mps.synchronize if device == "mps" else torch.cuda.synchronize
    orders = list(permutations(("incumbent", "control", "candidate")))
    blocks = []
    for i in range(POLICY["warmup_blocks"] + POLICY["measured_blocks"]):
        row = {
            arm: trajectory(
                model if arm == "candidate" else eager,
                ids,
                steps,
                key,
                kwargs,
                synchronize=sync,
            )
            for arm in orders[i % len(orders)]
        }
        if i >= POLICY["warmup_blocks"]:
            blocks.append(row)
    result["verdict"], result["timing"] = classify(blocks)
    return {**result, "blocks_ms": blocks}


def compare(folder, *, progress=None):
    run = Path(folder).resolve()
    report = json.loads((run / "report.json").read_text())
    options = json.loads((run / "kernel-configs.json").read_text())
    directory = run / "hub-trials" / uuid4().hex
    directory.mkdir(parents=True)
    results = []
    for config in options["configs"]:
        name = config["name"]
        if progress:
            progress(f"Validating Hub loading ({name}) on the selected runtime")
        path = directory / f"{name}.json"
        with (directory / f"{name}.log").open("w") as log:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "kernelatlas.loading_trial",
                    str(run),
                    name,
                    str(path),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if proc.returncode or not path.is_file():
            _write(
                path,
                {
                    "verdict": "error",
                    "reason": f"Worker exited {proc.returncode}. See {name}.log",
                },
            )
        record = json.loads(path.read_text())
        results.append(
            {
                "name": name,
                "evidence": str(path.relative_to(run)),
                **{k: v for k, v in record.items() if k != "blocks_ms"},
            }
        )
    result = {
        "scope": scope(report, options),
        "policy": {
            **POLICY,
            "correctness": "all returned logits within the scoped model tolerance and identical next greedy tokens",
        },
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    from .trials import _digest

    result["report_sha256"] = _digest(run / "report.json")
    _write(directory / "summary.json", result)
    from .example import write_best_combo

    write_best_combo(run, report, {**options, "comparison": result})
    return result


if __name__ == "__main__":
    import torch

    root, name, path = sys.argv[1:]
    options = json.loads((Path(root) / "kernel-configs.json").read_text())
    config = next(c for c in options["configs"] if c["name"] == name)
    try:
        with torch.inference_mode():
            result = _measure(
                json.loads((Path(root) / "report.json").read_text()), options, config
            )
    except Exception as exc:
        import traceback

        traceback.print_exc()
        result = {"verdict": "error", "reason": f"{type(exc).__name__}: {exc}"}
    _write(Path(path), result)
