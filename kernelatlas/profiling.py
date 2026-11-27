"""Capture a reproducible offline decoder trace and a separate latency sample."""

from contextlib import ExitStack
from datetime import datetime, timezone
import inspect
import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import torch
import transformers
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM


from .generation import decode_distribution, trajectory
from .precision import quantize_linears, resolve_dtype
from .proxy import REVISIONS as PROXY_REVISIONS, text_proxy


def add_module_scopes(stack, model):
    for name, module in model.named_modules():
        scopes = []

        def enter(mod, args, label=name or "model", active=scopes):
            scope = torch.profiler.record_function("module/" + label)
            scope.__enter__()
            active.append(scope)

        def leave(mod, args, output, active=scopes):
            active.pop().__exit__(None, None, None)

        stack.callback(module.register_forward_pre_hook(enter).remove)
        stack.callback(module.register_forward_hook(leave, always_call=True).remove)


def _quiet(*args):
    pass


def _cache_key(model):
    return (
        "cache"
        if "cache" in inspect.signature(model.forward).parameters
        else "past_key_values"
    )


def decode_inputs(model, prompt, cache_key, **kwargs):
    result = model(prompt, use_cache=True, **kwargs)
    return {
        "input_ids": result.logits[:, -1].argmax(-1, keepdim=True),
        cache_key: getattr(result, cache_key),
    }


def sequential_prompt(model, device, length, budget):
    if any(type(n) is not int or n < 1 for n in (length, budget)):
        raise ValueError(
            "Prompt length and generated token count must both be positive integers"
        )
    limit = getattr(model.config, "max_position_embeddings", None)
    if limit and length + budget - 1 > limit:
        raise ValueError(
            f"Workload needs {length + budget - 1} input positions; model context limit is {limit}"
        )
    return torch.arange(length, device=device)[None, :] % model.config.vocab_size


def generation_kwargs(model):
    if "logits_to_keep" in inspect.signature(model.forward).parameters:
        return {"logits_to_keep": 1}
    return {}


@torch.inference_mode()
def pack_for_profile(model, mode, device, prompt_length=32, max_new_tokens=None):
    prompt = sequential_prompt(
        model, device, prompt_length, 2 if max_new_tokens is None else max_new_tokens
    )

    def exercise():
        key = _cache_key(model)
        if max_new_tokens is None:
            model(**decode_inputs(model, prompt, key), use_cache=True)
        else:
            trajectory(model, prompt, max_new_tokens - 1, key, generation_kwargs(model))

    return quantize_linears(model, mode, exercise=exercise)


def collect(
    model,
    prompt,
    phase,
    device,
    out,
    progress=_quiet,
    *,
    timings=None,
    forward_kwargs=None,
):
    forward_kwargs = forward_kwargs or {}
    synchronize = torch.mps.synchronize if device == "mps" else torch.cuda.synchronize
    cache_key = _cache_key(model)

    def prepare():
        if phase == "prefill":
            return {"input_ids": prompt}
        return decode_inputs(model, prompt, cache_key, **forward_kwargs)

    samples = []
    label = (
        "Measuring prompt processing"
        if phase == "prefill"
        else "Measuring one cached token"
    )
    if timings is None:
        progress(label, 0, 12)
        for idx in range(12):
            inputs = prepare()
            synchronize()
            start = perf_counter_ns()
            result = model(**inputs, use_cache=True, **forward_kwargs)
            synchronize()
            elapsed = (perf_counter_ns() - start) / 1e6
            if idx >= 3:
                samples.append(elapsed)
            del result, inputs
            progress(label, idx + 1, 12)

    else:
        samples = [row[phase] for row in timings]

    progress(f"Capturing {phase} trace")
    inputs = prepare()
    synchronize()
    activities = [torch.profiler.ProfilerActivity.CPU]
    if (
        device == "cuda"
        and torch.profiler.ProfilerActivity.CUDA
        in torch.profiler.supported_activities()
    ):
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with ExitStack() as stack:
        add_module_scopes(stack, model)
        with torch.profiler.profile(
            activities=activities, record_shapes=True
        ) as profile:
            result = model(**inputs, use_cache=True, **forward_kwargs)
            synchronize()
    raw = out / f"{phase}.trace.json"
    profile.export_chrome_trace(str(raw))
    nodes = [
        {
            "id": event.id,
            "name": event.name,
            "parent": event.cpu_parent.id if event.cpu_parent else None,
            "device_type": str(event.device_type),
            "start_us": event.time_range.start,
            "end_us": event.time_range.end,
            "input_shapes": event.input_shapes,
        }
        for event in profile.events()
    ]
    (out / f"{phase}.events.json").write_text(json.dumps(nodes))
    return {
        "raw_trace": raw.name,
        "event_file": f"{phase}.events.json",
        "aten_host_events": sum(
            n["name"].startswith("aten::") and n["device_type"] == "DeviceType.CPU"
            for n in nodes
        ),
        "module_calls": sum(n["name"].startswith("module/") for n in nodes),
        "latency_method": "host perf_counter_ns around forward with device synchronization at boundaries",
        "warmup_forwards": 3,
        **(
            {
                "latency_method": "synchronized greedy trajectory including argmax",
                "trace_scope": "prompt"
                if phase == "prefill"
                else "first cached forward",
                "sample_scope": "prompt"
                if phase == "prefill"
                else "sum of all cached steps",
            }
            if timings is not None
            else {}
        ),
        "samples_ms": samples,
        "median_ms": median(samples),
        "output_shape": list(result.logits.shape),
        "cache_class": type(getattr(result, cache_key)).__name__,
        "cache_length": int(getattr(result, cache_key).get_seq_length()),
        "profiler_activities": [a.name for a in activities],
    }


@torch.inference_mode()
def profile_model(
    model,
    device,
    out,
    *,
    model_id=None,
    revision=None,
    config_source=None,
    proxy=None,
    precision=None,
    discover=True,
    prompt_length=None,
    max_new_tokens=None,
    progress=_quiet,
):
    generation_mode = prompt_length is not None or max_new_tokens is not None
    length = 32 if prompt_length is None else prompt_length
    budget = 2 if max_new_tokens is None else max_new_tokens
    prompt = sequential_prompt(model, device, length, budget)
    out.mkdir(parents=True, exist_ok=False)
    kwargs = {}
    timings = None
    phases = ["prefill", "decode"]
    if generation_mode:
        kwargs = generation_kwargs(model)
        if budget == 1:
            phases.remove("decode")
        timings = []
        sync = torch.mps.synchronize if device == "mps" else torch.cuda.synchronize
        cache_key = _cache_key(model)
        for i in range(12):
            progress("Measuring fixed-length greedy generation", i, 12)
            sample = trajectory(
                model, prompt, budget - 1, cache_key, kwargs, synchronize=sync
            )
            if i >= 3:
                timings.append(sample)
        progress("Measuring fixed-length greedy generation", 12, 12)
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "weights": model_id or "random seed 0",
        "weight_source": "pretrained" if model_id else "random",
        "config_source": config_source,
        "revision": revision,
        "model_id": model_id,
        "config": model.config.to_dict(),
        "attention": model.config._attn_implementation,
        "dtype": str(model.dtype),
        "device": str(prompt.device),
        "platform": platform.platform(),
        "hardware": torch.backends.mps.get_name()
        if device == "mps" and hasattr(torch.backends.mps, "get_name")
        else None,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu_dispatch_attribution": "unvalidated",
        "phases": {
            phase: collect(
                model,
                prompt,
                phase,
                device,
                out,
                progress,
                timings=timings,
                forward_kwargs=kwargs,
            )
            for phase in phases
        },
    }
    if generation_mode:
        report["workload"] = {
            "mode": "fixed_length_greedy",
            "prompt_ids": prompt[0].tolist(),
            "max_new_tokens": budget,
            "cached_tokens": budget - 1,
            "forward_kwargs": kwargs,
        }
        report["generation"] = {
            "samples_ms": timings,
            "total_median_ms": median(s["total"] for s in timings),
            "per_cached_token": decode_distribution(timings),
        }
    if precision is not None:
        report["precision"] = precision
    if proxy is not None:
        report["proxy"] = proxy
    if discover:
        from .dataflow import capture
        from .recipes import registrations

        if transformers.utils.is_kernels_available():
            report["kernel_registrations"] = registrations(model)

        progress("Capturing prefill tensor dependencies")
        graph = capture(model, prompt, **kwargs)
        report["prefill_dataflow"] = "prefill.dataflow.json"
        (out / report["prefill_dataflow"]).write_text(json.dumps(graph))
        if "decode" in phases:
            progress("Capturing tensor dependencies for the first cached token")
            key = _cache_key(model)
            inputs = decode_inputs(model, prompt, key, **kwargs)
            graph = capture(
                model,
                inputs["input_ids"],
                cache_argument=key,
                **{key: inputs[key]},
                **kwargs,
            )
            report["decode_dataflow"] = "decode.dataflow.json"
            (out / report["decode_dataflow"]).write_text(json.dumps(graph))
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    return report


def load_pinned_bf16_model(scope):
    if not torch.backends.mps.is_available():
        raise ValueError("This trial requires its recorded MPS device")
    current = {
        "torch": str(torch.__version__),
        "transformers": transformers.__version__,
        "platform": platform.platform(),
        "hardware": torch.backends.mps.get_name(),
    }
    for key, value in current.items():
        if scope[key] != value:
            raise ValueError(f"Trial requires {key}={scope[key]}, found {value}")
    model = (
        AutoModelForCausalLM.from_pretrained(
            scope["model_id"],
            revision=scope["revision"],
            dtype=torch.bfloat16,
            attn_implementation=scope["attention"],
            use_kernels=False,
        )
        .eval()
        .to("mps")
    )
    return model


def load_hub_workload(
    model_id,
    *,
    revision,
    device,
    random_weights,
    dtype,
    quantization,
    prompt_length,
    max_new_tokens,
    progress=_quiet,
):
    if device not in ("mps", "cuda"):
        raise ValueError(f"Unsupported profiling device {device}")
    backend = torch.backends.mps if device == "mps" else torch.cuda
    if not backend.is_available():
        raise ValueError(f"PyTorch cannot access the requested {device} device")
    execution_dtype = resolve_dtype(device, dtype)
    pack_weights = quantization in ("int4", "int8")
    if quantization not in ("auto", "none", "int4", "int8"):
        raise ValueError(f"Unknown quantization {quantization}")
    if pack_weights and device != "mps":
        raise ValueError(
            "Explicit INT4/INT8 weight-only profiles currently require --device mps."
        )
    if any(n is not None and n < 1 for n in (prompt_length, max_new_tokens)):
        raise ValueError(
            "Prompt length and generated token count must both be positive"
        )
    torch.manual_seed(0)
    progress(f"Resolving {model_id}")
    snapshot = snapshot_download(
        model_id, revision=revision, allow_patterns=["config.json"]
    )
    commit = Path(snapshot).name
    proxy = None
    if device == "mps" and model_id in PROXY_REVISIONS:
        config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
        config, proxy = text_proxy(model_id, config.text_config)
        if execution_dtype != torch.bfloat16 or pack_weights:
            proxy["scope"] = "random text decoder only, reduced layers and vocabulary"
        progress(
            f"Profiling a reduced {config.num_hidden_layers}-layer text decoder with random {execution_dtype} weights on MPS"
        )
        model = AutoModelForCausalLM.from_config(config, dtype=execution_dtype)
        proxy["model_parameter_elements"] = sum(p.numel() for p in model.parameters())
        random_weights = True
    elif random_weights:
        progress(f"Initializing random weights from config for {device}")
        config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
        model = AutoModelForCausalLM.from_config(config, dtype=execution_dtype)
    else:
        if quantization != "auto":
            config_json = json.loads((Path(snapshot) / "config.json").read_text())
            if config_json.get("quantization_config"):
                raise ValueError(
                    "This checkpoint already declares quantization. Omit --quantization to use its loader."
                )
        progress(f"Loading pretrained weights for {device} with {execution_dtype}")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=commit, dtype=execution_dtype
        )
    model = model.eval().to(device)
    weights = {"mode": "none"}
    if pack_weights:
        progress(f"Packing eligible Linear weights as {quantization.upper()}")
        budget = (
            2
            if prompt_length is not None and max_new_tokens is None
            else max_new_tokens
        )
        weights = pack_for_profile(
            model, quantization, device, prompt_length or 32, budget
        )
    elif not random_weights and getattr(model, "hf_quantizer", None) is not None:
        weights = {
            "mode": "checkpoint",
            "config": model.hf_quantizer.quantization_config.to_dict(),
        }
    precision = {
        "requested_dtype": dtype,
        "dtype": str(execution_dtype),
        "requested_quantization": quantization,
        "weights": weights,
    }
    return model, commit, random_weights, proxy, precision


def profile_hub_model(
    model_id,
    *,
    revision=None,
    device="mps",
    out=None,
    random_weights=False,
    dtype="auto",
    quantization="auto",
    discover=True,
    prompt_length=None,
    max_new_tokens=None,
    progress=_quiet,
):
    """Profile a Hub causal LM. Either length option selects fixed-length greedy generation."""
    model, commit, random_weights, proxy, precision = load_hub_workload(
        model_id,
        revision=revision,
        device=device,
        random_weights=random_weights,
        dtype=dtype,
        quantization=quantization,
        prompt_length=prompt_length,
        max_new_tokens=max_new_tokens,
        progress=progress,
    )
    if out is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        out = Path("runs") / f"{model_id.replace('/', '--')}-{stamp}"
    out = Path(out)
    return out, profile_model(
        model,
        device,
        out,
        model_id=None if random_weights else model_id,
        revision=None if random_weights else commit,
        config_source={"model_id": model_id, "revision": commit},
        proxy=proxy,
        precision=precision,
        discover=discover,
        prompt_length=prompt_length,
        max_new_tokens=max_new_tokens,
        progress=progress,
    )
