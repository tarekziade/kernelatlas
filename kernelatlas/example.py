"""Write an opt-in, revision-pinned Transformers trial from inspected adapters."""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from textwrap import indent
from pprint import pformat

from .numerics import model_logit_policy


def write_example(folder, report, snapshot, metal_tail=False):
    picks = [
        lead
        for lead in snapshot["layers"]
        if lead["example_status"] == "picked_for_trial"
    ]
    if not picks and not metal_tail:
        return None
    config = report["config_source"]
    adapters = [
        (
            p["layer"],
            p["repo_id"],
            p["pinned_revision"],
            p["symbol"],
            p["sites"]["original_forward"],
        )
        for p in picks
    ]
    dtype = report.get("precision", {}).get(
        "dtype", report.get("dtype", "torch.float32")
    )
    mode = report.get("precision", {}).get("weights", {}).get("mode", "none")
    policy = model_logit_policy(dtype)
    loader = (
        "config = AutoConfig.from_pretrained(MODEL, revision=REVISION)\n"
        "model = AutoModelForCausalLM.from_config(config, dtype=DTYPE)"
        if report["weight_source"] == "random"
        else "model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=DTYPE, use_kernels=False)"
    )
    imports = []
    if "proxy" in report:
        imports.extend(["import json", "from pathlib import Path"])
        loader = (
            'config = json.loads(Path(__file__).with_name("report.json").read_text())["config"]\n'
            'text_config = AutoConfig.for_model(config.pop("model_type"), **config)\n'
            "model = AutoModelForCausalLM.from_config(text_config, dtype=DTYPE)"
        )
    workload = report.get("workload", {})
    length = len(workload.get("prompt_ids", range(32)))
    forward_kwargs = workload.get("forward_kwargs", {})
    packing = ""
    if mode in ("int4", "int8"):
        imports.append("from kernelatlas.profiling import pack_for_profile")
        budget = workload.get("max_new_tokens")
        packing = f"pack_for_profile(model, {mode!r}, DEVICE, {length}, {budget!r})\n"
    wiring = ""
    if picks:
        imports.append(
            "from kernels import LayerRepository, Mode, kernelize, use_kernel_mapping"
        )
        wiring = """
    for key, repo, revision, symbol, sites in ADAPTERS:
        replacement = LayerRepository(repo, revision=revision, layer_name=symbol)
        mapping = {key: {torch.device(DEVICE).type: replacement}}
        with use_kernel_mapping(mapping, inherit_mapping=False):
            for site in sites:
                module = model.get_submodule(site)
                if vars(module).get("_kernel_repo") is not None:
                    raise RuntimeError(f"Another kernel is already bound at {site}")
                kernelize(module, mode=Mode.INFERENCE, device=torch.device(DEVICE).type, use_fallback=False)
                if vars(module).get("_kernel_repo") != replacement:
                    raise RuntimeError(f"Kernel binding failed at {site}")
"""
    if metal_tail:
        imports.append("from kernelatlas.metal import metal_rmsnorm_prefill")
    evaluate = (
        "with metal_rmsnorm_prefill(model):\n        candidate = model(prompt, use_cache=True, **FORWARD_KWARGS).logits"
        if metal_tail
        else "candidate = model(prompt, use_cache=True, **FORWARD_KWARGS).logits"
    )
    compile_code = report.get("compilation", {}).get("python", "")
    path = folder / "try_kernels.py"
    path.write_text(f'''"""Trial on this profile's prefill. Check the selected kernels against the native forward."""
import torch
from transformers import AutoConfig, AutoModelForCausalLM
{chr(10).join(imports)}

MODEL = {config["model_id"]!r}
REVISION = {config["revision"]!r}
DEVICE = {report["device"]!r}
DTYPE = getattr(torch, {dtype.removeprefix("torch.")!r})
FORWARD_KWARGS = {forward_kwargs!r}
ADAPTERS = {pformat(adapters, width=92)}

torch.manual_seed(0)
{loader}
model = model.eval().to(DEVICE)
{packing}prompt = torch.arange({length}, device=DEVICE)[None, :] % model.config.vocab_size

{compile_code}
with torch.inference_mode():
    eager = model(prompt, use_cache=True, **FORWARD_KWARGS).logits
{wiring}
    {evaluate}
    torch.testing.assert_close(candidate, eager, rtol={policy["rtol"]!r}, atol={policy["atol"]!r})
print("Picked kernels passed the {length}-token prefill logit check.")
''')
    return path.name


def _apply_proxy_checks(root, report, layers):
    for path in sorted(root.glob("*-proxy.json")):
        evidence = json.loads(path.read_text())
        check = evidence.get("kernel_check", {})
        if check.get("status") != "rejected" or len(check.get("selected", [])) != 1:
            continue
        if evidence.get("workload") != report["proxy"] or any(
            evidence.get(k) != report.get(k)
            for k in ("config_source", "torch", "transformers", "hardware", "dtype")
        ):
            continue
        candidate = check["selected"][0]
        for lead in layers:
            if (
                lead["repo_id"] == candidate["repo_id"]
                and lead["symbol"] == candidate["symbol"]
                and lead.get("pinned_revision") == candidate["revision"]
                and set(lead["sites"]["original_forward"]) == set(candidate["sites"])
            ):
                lead.update(
                    example_status="failed_trial",
                    failure=f"Failed the saved correctness check for this reduced workload ({path.name}).",
                )


def prepare_example(out, root, report, discovery, snapshot, trials):
    weight_mode = report.get("precision", {}).get("weights", {}).get("mode", "none")
    compiled = bool(report.get("compilation", {}).get("recipe"))
    if (
        not compiled
        and "workload" not in report
        and weight_mode == "none"
        and "proxy" in report
    ):
        _apply_proxy_checks(root, report, snapshot["layers"])
    failure_file = root / "hub-failures.json"
    if not compiled and failure_file.is_file():
        for failure in json.loads(failure_file.read_text()):
            if failure.get("generation_workload") != report.get("workload"):
                continue
            if failure.get("weight_mode", "none") != weight_mode:
                continue
            if all(report.get(k) == v for k, v in failure["workload"].items()):
                for lead in snapshot["layers"]:
                    if (
                        lead["repo_id"],
                        lead["symbol"],
                        lead.get("pinned_revision"),
                    ) == (
                        failure["repo_id"],
                        failure["symbol"],
                        failure["revision"],
                    ):
                        lead.update(
                            example_status="failed_trial",
                            failure=failure["outcome"],
                            failure_summary=failure.get(
                                "summary", "A previous trial failed on this setup."
                            ),
                        )
    sites = (discovery["transformers_integrations"] or {}).get("sites", {})
    rms_sites = {name for name, site in sites.items() if site["layer"] == "RMSNorm"}
    metal_tail = (
        bool(rms_sites)
        and all(sites[n]["binding"] == "original_forward" for n in rms_sites)
        and any(
            row["candidate"] == "rmsnorm_tail.metal"
            and row["correctness"] == "passed_tested_inputs"
            and (row["median_paired_prefill_savings_ms"] or 0) > 0
            and set(row["occurrences"]) == rms_sites
            for row in trials
        )
    )
    if metal_tail:
        for lead in snapshot["layers"]:
            if (
                lead["layer"] == "RMSNorm"
                and lead["example_status"] == "picked_for_trial"
            ):
                lead["example_status"] = "measured_local_alternative_selected"
    snapshot["local_prefill_kernels"] = ["rmsnorm_tail.metal"] if metal_tail else []
    snapshot["example"] = write_example(out, report, snapshot, metal_tail)


def write_best_combo(folder, report, options=None):
    """Export the selected runtime and at most one jointly benchmarked Hub config."""
    from .recipes import snippet, winning_hub_config

    options = options or {}
    comparison = options.get("comparison") or {}
    winner = winning_hub_config(comparison)
    config = {"name": "native", "use_kernels": False}
    if winner is not None:
        config = dict(
            next(c for c in options["configs"] if c["name"] == winner["name"])
        )
        if "kernel_mapping" in config:
            config["kernel_mapping"] = {
                k: tuple(v) for k, v in config["kernel_mapping"].items()
            }

    config.setdefault("attn_implementation", report["attention"])
    source = report["config_source"]
    device = report["device"]
    workload = report.get("workload", {})
    length = (
        len(workload["prompt_ids"])
        if workload
        else report["phases"]["prefill"]["cache_length"]
    )
    budget = workload.get("max_new_tokens", 2)
    precision = report["precision"]
    if (
        report["weight_source"] == "pretrained"
        and precision["weights"]["mode"] == "none"
    ):
        code = snippet(
            source["model_id"],
            source["revision"],
            device,
            report["dtype"].removeprefix("torch."),
            config,
        )
        imports, loader = code.split("\n\n", 1)
    else:
        # Replay the same config-only, reduced-decoder or quantized loader as profile.
        imports = "import torch\nfrom kernelatlas.profiling import load_hub_workload"
        loader = f"""model, *_ = load_hub_workload(
    {source["model_id"]!r}, revision={source["revision"]!r}, device={device.split(":")[0]!r},
    random_weights={report["weight_source"] == "random"!r}, dtype={precision["requested_dtype"]!r},
    quantization={precision["requested_quantization"]!r}, prompt_length={length if workload else None!r},
    max_new_tokens={budget if workload else None!r},
)"""
    compilation = report.get("compilation") or {}
    if compilation.get("python"):
        loader += "\n\n" + compilation["python"]
    prompt = (
        f"torch.arange({length}, device={device!r})[None, :] % model.config.vocab_size"
    )
    if workload:
        prompt = f"torch.tensor([{workload['prompt_ids']!r}], device={device!r})"
    kwargs = workload.get("forward_kwargs", {})
    code = f'''"""Replay the selected profile configuration for {source["model_id"]}.
Hub loading: {config["name"]}. Compilation: {compilation.get("selected", "eager")}.
Profiled with PyTorch {report["torch"]} and Transformers {report["transformers"]}.
"""
import inspect

{imports}


def load_model():
{indent(loader, "    ")}
    return model


@torch.inference_mode()
def main():
    model = load_model()
    # Same input IDs and fixed-length greedy decoding as the profile.
    prompt = {prompt}
    inputs = {{"input_ids": prompt}}
    cache_key = "cache" if "cache" in inspect.signature(model.forward).parameters else "past_key_values"
    tokens = []
    for _ in range({budget}):
        output = model(**inputs, use_cache=True{"".join(f", {key}={value!r}" for key, value in kwargs.items())})
        token = output.logits[:, -1].argmax(-1, keepdim=True)
        tokens.append(token)
        inputs = {{"input_ids": token, cache_key: getattr(output, cache_key)}}
    print("Generated token IDs", torch.cat(tokens, dim=1)[0].tolist())


if __name__ == "__main__":
    main()
'''
    path = folder / "best_combo.py"
    with NamedTemporaryFile("w", dir=folder, delete=False) as tmp:
        tmp.write(code)
    Path(tmp.name).replace(path)
    return path.name
