"""Build Transformers loading examples from registered model hooks and published Hub artifacts."""

import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from pprint import pformat

from .integrations import mapping_declarations


def registrations(model):
    from transformers import KernelConfig

    keys = KernelConfig()
    keys.store_registered_layer_names(model)
    return {
        "layers": dict(sorted(Counter(keys.registered_layer_names.values()).items())),
        "flash_attention": bool(getattr(model, "_supports_flash_attn", False)),
    }


def inspect_adapter(spec, device):
    import torch
    from huggingface_hub import HfApi, hf_hub_download
    from kernels.variants import get_variants, resolve_variant
    from packaging.version import Version

    repo, ref = spec["repo"], spec["ref"]
    api = HfApi()
    info = api.repo_info(repo, revision=ref, repo_type="kernel")
    variants = get_variants(api, repo_id=repo, revision=info.sha)
    local = (
        torch.backends.mps.is_available()
        if device == "mps"
        else torch.cuda.is_available() and not torch.version.hip
    )
    if local:
        variant, _ = resolve_variant(
            variants, backend="metal" if device == "mps" else "cuda"
        )
    else:
        # Inspect a published target example only; remote driver and GPU capability are unknown.
        torch_version = Version(".".join(torch.__version__.split(".")[:2]))
        eligible = []
        for v in variants:
            tag = v.variant_str
            if (
                device == "cuda"
                and (
                    "-cu" in tag
                    and tag.endswith("-x86_64-linux")
                    or tag == "torch-cuda"
                )
                or device == "mps"
                and tag.endswith("-metal-aarch64-darwin")
            ):
                version = getattr(v.framework, "version", None)
                if (
                    version == torch_version
                    or tag == "torch-cuda"
                    or ("stable-abi" in tag and version <= torch_version)
                ):
                    eligible.append(v)
        variant = next(iter(sorted(eligible, key=lambda v: v.variant_str)), None)
    if variant is None:
        raise ValueError(f"No matching published {device} build for this torch version")
    filename = "__init__.py" if spec["key"] == "attention" else "layers.py"
    path = f"build/{variant.variant_str}/{filename}"
    source = Path(hf_hub_download(repo, path, revision=info.sha, repo_type="kernel"))
    nodes = ast.parse(source.read_text()).body
    if spec["key"] == "attention":
        names = {
            a.asname or a.name
            for n in nodes
            if isinstance(n, ast.ImportFrom)
            for a in n.names
        }
        names.update(n.name for n in nodes if isinstance(n, ast.FunctionDef))
        valid = (
            "flash_attn_varlen_func" in names
            or {"flash_attn_forward", "supports_flash_attn"} <= names
        )
    else:
        valid = any(
            isinstance(n, ast.ClassDef)
            and n.name == spec["symbol"]
            and any(
                isinstance(m, ast.FunctionDef) and m.name == "forward" for m in n.body
            )
            for n in nodes
        )
    if not valid:
        raise ValueError(f"Required Transformers adapter is missing from {path}")
    return {
        **spec,
        "revision": info.sha,
        "build": variant.variant_str,
        "local_target": bool(local),
        "source": f"https://huggingface.co/kernels/{repo}/blob/{info.sha}/{path}",
    }


def loading_options(model_id, revision, device, dtype, registry, *, excluded=()):
    from transformers.integrations import hub_kernels
    from transformers import modeling_flash_attention_utils as flash

    declarations = mapping_declarations(Path(hub_kernels.__file__), registry["layers"])
    specs = {}
    for d in declarations:
        if (
            d["device"] == device
            and d["mode"] in (None, "Mode.INFERENCE")
            and d["repo_id"].startswith("kernels-community/")
        ):
            ref = d.get("revision") or (
                f"v{d['version']}" if d.get("version") else None
            )
            if ref:
                specs[d["layer"]] = {
                    "key": d["layer"],
                    "repo": d["repo_id"],
                    "ref": ref,
                    "symbol": d["layer_name"],
                    "sites": registry["layers"][d["layer"]],
                }
    attention = (
        "kernels-community/metal-flash-sdpa"
        if device == "mps"
        else "kernels-community/flash-attn2"
    )
    if registry["flash_attention"] and device in getattr(
        flash, "FLASH_ATTN_KERNEL_DEVICES", {}
    ).get(attention, ()):
        specs["attention"] = {
            "key": "attention",
            "repo": attention,
            "ref": f"v{flash.FLASH_ATTN_KERNEL_VERSIONS[attention]}",
        }
    inspected, skipped = {}, []

    def inspect_one(spec):
        try:
            return inspect_adapter(spec, device), None
        except (
            ImportError,
            OSError,
            RuntimeError,
            ValueError,
            SyntaxError,
        ) as exc:
            return None, str(exc)

    with ThreadPoolExecutor(max_workers=4) as pool:
        for (key, spec), (item, error) in zip(
            specs.items(), pool.map(inspect_one, specs.values()), strict=True
        ):
            if item and (item["repo"], item["revision"]) in excluded:
                error = "Rejected in a matching saved correctness check"
                item = None
            if item:
                inspected[key] = item
            else:
                skipped.append({**spec, "reason": error})
    mapping = {
        k: (f"{v['repo']}:{v['symbol']}", {"revision": v["revision"]})
        for k, v in inspected.items()
        if k != "attention"
    }
    attn = inspected.get("attention")
    attn_ref = f"{attn['repo']}@{attn['revision']}" if attn else None
    configs = [{"name": "automatic", "use_kernels": True}]
    if mapping:
        configs.append(
            {
                "name": "selective",
                "use_kernels": True,
                "kernel_mapping": mapping,
                "inherit_mapping": False,
            }
        )
    if attn_ref:
        configs.append(
            {
                "name": "attention_only",
                "use_kernels": False,
                "attn_implementation": attn_ref,
            }
        )
        if mapping:
            configs.append(
                {**configs[1], "name": "combined", "attn_implementation": attn_ref}
            )
    for c in configs:
        c["python"] = snippet(model_id, revision, device, dtype, c)
    return {
        "model_id": model_id,
        "revision": revision,
        "device": device,
        "dtype": dtype,
        "registered_sites": registry["layers"],
        "adapters": inspected,
        "excluded": skipped,
        "configs": configs,
        "status": "candidates_to_test",
        "measured_speedup": None,
    }


def snippet(model_id, revision, device, dtype, config):
    imports = "from transformers import AutoModelForCausalLM"
    lines = [
        "import torch",
        imports + (", KernelConfig" if "kernel_mapping" in config else ""),
        "",
    ]
    if "kernel_mapping" in config:
        lines += [
            "selection = KernelConfig(",
            f"    kernel_mapping={pformat(config['kernel_mapping'], width=88)},",
            "    inherit_mapping=False,",
            ")",
            "",
        ]
    lines += [
        "model = AutoModelForCausalLM.from_pretrained(",
        f"    {model_id!r},",
        f"    revision={revision!r},",
        f"    dtype=torch.{dtype},",
        f"    device_map={device!r},",
        f"    use_kernels={config['use_kernels']},",
    ]
    if "kernel_mapping" in config:
        lines += ["    kernel_config=selection,"]
    if "attn_implementation" in config:
        lines += [f"    attn_implementation={config['attn_implementation']!r},"]
    return "\n".join([*lines, ").eval()"])


def recommend(model_id, revision, device, dtype="bfloat16"):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.utils import is_kernels_available

    if not is_kernels_available():
        raise ValueError(
            "Install a compatible kernels library to inspect Transformers registrations"
        )

    if device == "auto":
        device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else None
        )
    if device is None:
        raise ValueError(
            "Choose --device mps or --device cuda to inspect configurations for a target GPU"
        )
    from huggingface_hub import hf_hub_download

    snapshot = Path(hf_hub_download(model_id, "config.json", revision=revision)).parent
    config = AutoConfig.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False
    )
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=False)
    return loading_options(model_id, snapshot.name, device, dtype, registrations(model))


def show_options(options, *, full=True, historical_runtime=True):
    if not full and options.get("comparison"):
        show_comparison(options["comparison"], historical_runtime=historical_runtime)
        return
    print(
        f"Hub kernel loading · {options['model_id']} · {options['device']} · {options['dtype']}"
    )
    print(
        "Automatic: use_kernels=True lets Transformers apply its device and inference defaults."
    )
    layers = [
        f"{k} → {a['repo']}:{a['symbol']}"
        for k, a in options["adapters"].items()
        if k != "attention"
    ]
    print(
        "Selective: "
        + (
            ", ".join(layers) + " (KernelConfig, inherit_mapping=False)."
            if layers
            else "No inspected layer adapter to select for this target."
        )
    )
    if a := options["adapters"].get("attention"):
        print(f"Attention override: {a['repo']} via attn_implementation.")
    comparison = options.get("comparison")
    if comparison:
        show_comparison(comparison, historical_runtime=historical_runtime)
    else:
        print("Speedup is unmeasured for these configurations.")

    if any(not a["local_target"] for a in options["adapters"].values()):
        print(
            "Published builds inspected. Target GPU compatibility needs a hardware check."
        )

    if full:
        for c in options["configs"]:
            print(f"\n# {c['name']}\n{c['python']}")
        for e in options["excluded"]:
            print(f"# Skipped {e['repo']}: {e['reason']}")


def winning_hub_config(comparison):
    """Pick a tested configuration using its paired end-to-end saving."""
    return max(
        (r for r in comparison.get("results", []) if r["verdict"] == "faster"),
        key=lambda r: r["timing"]["total"]["median_paired_saving_ms"],
        default=None,
    )


def show_comparison(comparison, *, historical_runtime=True):
    scope = comparison["scope"]
    print(
        f"Checked against the default model on {scope['hardware'] or scope['device']} · {scope['dtype'].removeprefix('torch.')}"
    )

    if recipe := scope.get("compile_recipe"):
        print(
            f"Both the native and Hub models use the selected Inductor recipe ({recipe['mode']})."
        )
    tolerances = {
        (check["rtol"], check["atol"])
        for row in comparison["results"]
        for check in [*row.get("checks", []), row.get("failure") or {}]
        if "rtol" in check and "atol" in check
    }
    if len(tolerances) == 1:
        rtol, atol = tolerances.pop()
        relative_to = scope.get("logit_policy", {}).get("relative_to", "candidate")
        print(
            f"Logits check: every absolute difference must be ≤ {atol:g} + {rtol:g} × |{relative_to} logit|, with the same greedy token."
        )
        if scope["dtype"].removeprefix("torch.") == "bfloat16" and rtol == 0.016:
            print(
                f"The relative allowance matches PyTorch’s BF16 assert_close default. The {atol:g} allowance near zero is a KernelAtlas heuristic."
            )
    for r in comparison["results"]:
        name = r["name"]
        active = [f"{n} × {target}" for target, n in r.get("bindings", {}).items()]
        if r.get("attention_interface_calls"):
            active.append(r["attention"].split("@")[0])
        if active:
            label = "automatic (use_kernels=True)" if name == "automatic" else name
            print(f"  {label} used {', '.join(active)}")
        if "timing" in r:
            total = r["timing"]["total"]
            print(
                f"  {name}: {r['verdict']}, {total['median_paired_saving_ms']:+.2f} ms paired generation saving"
            )
        elif "failure" in r:
            f = r["failure"]
            token = ", same greedy token" if f.get("greedy_token_match") else ""
            print(
                f"  {name}: failed the logits check at output step {f['step'] + 1} ({f['outside_tolerance']:,}/{f['logits']:,} outside tolerance{token})"
            )
        else:
            print(f"  {name}: {r['verdict']} · {r.get('reason', '')}")
    winner = winning_hub_config(comparison)
    if winner:
        config = next(c for c in scope["configs"] if c["name"] == winner["name"])
        print(
            f"Selected {winner['name']} (use_kernels={config['use_kernels']}) for its measured end-to-end gain."
        )
        if not config["use_kernels"]:
            print(
                "This keeps native layers and selects the tested attention backend separately."
            )
    else:
        print(
            "Keep use_kernels=False. Transformers keeps the native layer implementations."
        )
        if scope["device"].split(":")[0] in ("cuda", "mps"):
            print(f"PyTorch still runs GPU kernels on {scope['device']}.")
        trials = comparison["results"]
        if trials and all(
            t["verdict"] == "incorrect"
            and t.get("failure", {}).get("outside_tolerance", 0) > 0
            for t in trials
        ):
            print(
                "Every tested Hub configuration exceeded the logits tolerance, so timing was skipped."
            )
        else:
            print(
                "No tested Hub configuration passed validation and demonstrated a speedup on this workload."
            )
    if audit := comparison.get("rmsnorm_epsilon_audit"):
        print(
            f"RMSNorm epsilon check ({audit['execution']} prefill): {audit['matching_launches']}/{audit['total_launches']} "
            f"Metal launches used the model’s eps={audit['epsilon']:g}. Restoring native RMSNorm left "
            f"{audit['restored_different_logits']:,} logits different from baseline."
        )
    if historical_runtime and (trial := comparison.get("runtime_comparison")):
        savings = ", ".join(f"{ms:.2f} ms" for ms in trial["saved_ms"])
        runs = (
            "on this workload"
            if len(trial["saved_ms"]) == 1
            else "across independent runs of this workload"
        )
        print(
            f"{trial['implementation']} with {trial['cache']} saved {savings} {runs}."
        )
        print(f"Correctness and phase timings: {trial['evidence']}")
