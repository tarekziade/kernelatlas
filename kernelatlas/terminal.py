"""Terminal views shared by the profile and Atlas commands."""

from collections import defaultdict
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
from textwrap import wrap

from .guidance import guidance


def profile_progress(stage, completed=0, total=None):
    stream = sys.stderr
    interactive = stream.isatty() and os.environ.get("TERM") != "dumb"
    if total is None or not interactive:
        if completed == 0:
            print(stage, file=stream, flush=True)
        return
    width = shutil.get_terminal_size().columns
    size = min(24, max(0, width - len(stage) - 12))
    filled = size * completed // total
    count = f"{completed}/{total}"
    line = f"{stage}  [{'#' * filled}{'-' * (size - filled)}] {count}"
    print(
        "\r" + line[: max(1, width - 1)],
        end="\n" if completed == total else "",
        file=stream,
        flush=True,
    )


def saving(value):
    return "—" if value is None else f"{value:+.3f}"


def _show_trials(rows, reports):
    groups = defaultdict(list)
    for row in rows:
        key = (row["run"], row["torch"], row["device"], row["dtype"])
        groups[key].append(row)

    sources = {}
    width = shutil.get_terminal_size().columns
    for (run, torch, device, dtype), trials in groups.items():
        print()
        _line(f"{run} · PyTorch {torch} · {device} · {dtype.removeprefix('torch.')}")
        header = ("Candidate", "Tokens", "Check", "Prefill ms", "Decode ms", "Ref")
        data = []
        for row in trials:
            ref = sources.setdefault(row["evaluation"], len(sources) + 1)
            status = row["correctness"]
            data.append(
                (
                    row["candidate"],
                    str(row["prompt_length"]),
                    "passed"
                    if status == "passed_tested_inputs"
                    else status.replace("_", " "),
                    saving(row["median_paired_prefill_savings_ms"]),
                    saving(row["median_paired_decode_savings_ms"]),
                    str(ref),
                )
            )
        sizes = [max(map(len, col)) for col in zip(header, *data)]
        if sum(sizes) + 2 * (len(header) - 1) <= width:
            for line in (header, tuple("─" * size for size in sizes), *data):
                print(
                    "  ".join(
                        cell.ljust(size) if i in (0, 2) else cell.rjust(size)
                        for i, (cell, size) in enumerate(zip(line, sizes))
                    )
                )
        else:
            for candidate, tokens, status, prefill, decode, ref in data:
                _line(f"{candidate} [{ref}]", "  ")
                _line(f"{tokens} tokens · {status}", "    ")
                _line(f"Prefill {prefill} ms · Decode {decode} ms", "    ")
    print()
    _line("Savings vs eager · positive = faster · — = unmeasured")
    _line(f"Evidence in {reports}")
    for path, ref in sources.items():
        print(
            "\n".join(
                wrap(f"[{ref}] {path}", width=max(12, width), subsequent_indent="    ")
            )
        )


def _line(text, indent=""):
    width = max(1, shutil.get_terminal_size().columns)
    padding = indent[: width - 1]
    print(
        "\n".join(
            wrap(
                text,
                width,
                initial_indent=padding,
                subsequent_indent=padding,
                break_on_hyphens=False,
            )
        )
    )


def _heading(text):
    # Styling is confined to headings on interactive terminals.
    if (
        sys.stdout.isatty()
        and os.environ.get("TERM") != "dumb"
        and "NO_COLOR" not in os.environ
        and len(text) <= shutil.get_terminal_size().columns
    ):
        print(f"\033[1m{text}\033[0m")
    else:
        _line(text)


def _measurements(measurements):
    groups = defaultdict(list)
    for trial in measurements:
        groups[trial["torch"], trial["prompt_length"]].append(trial)
    for (version, tokens), trials in sorted(groups.items()):
        parts = []
        for phase in ("prefill", "decode"):
            values = [
                r[f"median_paired_{phase}_savings_ms"]
                for r in trials
                if r[f"median_paired_{phase}_savings_ms"] is not None
            ]
            if not values:
                continue
            low, high = min(values), max(values)
            if low * high > 0:
                lo, hi = sorted((abs(low), abs(high)))
                amount = f"{lo:.3f}" if lo == hi else f"{lo:.3f}–{hi:.3f}"
                parts.append(f"{phase} {amount} ms {'faster' if low > 0 else 'slower'}")
            else:
                parts.append(f"{phase} saving {low:+.3f} to {high:+.3f} ms")
        if parts:
            _line(f"{version} / {tokens} tokens   " + ", ".join(parts), "  ")
        for row in trials:
            rep = row.get("repeated_prefill")
            if rep:
                _line(
                    f"Across {rep['processes']} processes: {rep['median_of_process_medians_ms']:.3f} ms median prefill saving.",
                    "    ",
                )


def _run_view(run):
    _line(f"{run['run']}  /  {run['device']}  /  {run['dtype'].removeprefix('torch.')}")
    print()
    _heading("Existing kernels to use")
    existing = run["existing_kernels"]
    if not existing:
        _line("No evaluated replacements in this selection.")
    for decision in existing:
        if decision["action"] == "keep_eager":
            continue
        print()
        label = "TRY FOR PREFILL" if decision["action"] == "try_prefill" else "TESTED"
        _line(f"{label}   {decision['candidate']}")
        _line(decision["reason"])
        code = decision["python"]
        if code:
            print()
            if max(map(len, code.splitlines())) <= shutil.get_terminal_size().columns:
                print(code)
            else:
                _line(
                    "Activation example: README.md or the python field in --format json."
                )
        print()
        _heading("Measured against eager")
        _measurements(decision["measurements"])
    rejected = [d for d in existing if d["action"] == "keep_eager"]
    if rejected:
        print()
        _heading("Keep eager for these replacements")
        by_reason = defaultdict(list)
        for decision in rejected:
            by_reason[decision["reason"]].append(decision)
        for reason, skipped in by_reason.items():
            _line(reason)
            for decision in skipped:
                versions = ", ".join(
                    sorted({m["torch"] for m in decision["measurements"]})
                )
                _line(f"- {decision['candidate']} (Torch {versions})")
    print()
    _heading("Kernels to create")
    for task in run["development_tasks"]:
        print()
        _line("• " + task["title"] + "  [untested]")
        _line(task["reason"], "  ")
        if "shape" in task:
            shape = " × ".join(map(str, task["shape"]))
            _line(f"Observed at {shape}, Torch {task['torch']}.", "  ")
        _line("Next: " + task["next_step"], "  ")
    if not run["development_tasks"]:
        _line("No evidence-backed proposals for this selection.")


def _hub_summary(summary):
    if not summary:
        return
    _heading("Kernel Hub search")
    status = summary["status"]
    if status != "searched":
        _line(
            {
                "offline": "Search skipped in offline mode.",
                "not_needed": "No operator pairs or layer mappings to look up.",
                "unavailable": "The public catalog could not be reached.",
            }.get(status, status)
        )
        return
    size = f"{summary['catalog_size']:,}"
    if summary["limit_reached"]:
        size += " (listing limit reached)"
    _line(
        f"Searched {size} public repositories for {summary['operator_pairs']} operator pairs."
    )
    repos = summary["target_repositories"]
    driver = {"metal": "Metal", "cuda": "CUDA"}.get(
        summary["target_driver"], summary["target_driver"]
    )
    count = summary["matching_repositories"]
    text = f"{count} {'repository' if count == 1 else 'repositories'} matched by name, tag or Transformers mapping."
    text += f" {len(repos)} {'advertises' if len(repos) == 1 else 'advertise'} {driver} support"
    if repos:
        names = ", ".join(repos[:3])
        if len(repos) > 3:
            names += f" and {len(repos) - 3} more"
        text += f" ({names})"
    _line(text + ".")


def _show_hub(search, run_dir):
    _heading("Kernel Hub search")
    for lead in search.get("layers", []):
        sites = lead["sites"]
        devices = ", ".join(sorted({m["device"] for m in lead["mappings"]}))
        _line(f"{lead['layer']} → {lead['repo_id']} / {lead['symbol']}")
        _line(
            f"Transformers mapping on {devices}. At {len(sites['original_forward'])} sites the original forward ran. This adapter was already bound at {len(sites['already_bound'])} sites. Unknown at {len(sites['unknown'])}.",
            "  ",
        )
        _line("Example: " + lead["example_status"].replace("_", " "), "  ")
        if lead.get("failure"):
            _line(lead["failure"], "  ")
    if search.get("local_prefill_kernels"):
        _line("Selected for prefill: " + ", ".join(search["local_prefill_kernels"]))
    if search.get("example"):
        _line(
            f"Generated {search['example']} in the run directory for the picked layers."
        )
        if run_dir is not None:
            print(shlex.join([sys.executable, str(run_dir / search["example"])]))
    if search["status"] != "searched":
        messages = {
            "not_needed": "No candidate pairs to search.",
            "offline": "Skipped because HF_HUB_OFFLINE is set.",
            "unavailable": "Catalog unavailable. " + search.get("error", ""),
        }
        _line(messages[search["status"]])
        return
    _line(
        f"Searched {search['catalog_size']} public kernel repositories using names, tags and Transformers mappings."
    )
    if search["limit_reached"]:
        _line(f"Catalog listing capped at {search['catalog_limit']} repositories.")
    driver = search["target_driver"]
    for q in search["queries"]:
        label = " → ".join(op.split(".")[1] for op in q["operators"])
        hits = q["hits"]
        if not hits:
            _line(f"{label} · no metadata hits")
            continue
        target = [h for h in hits if driver in h["advertised_drivers"]]
        _line(f"{label} · {len(hits)} leads · {len(target)} advertise {driver}")
        for hit in (target or hits)[:2]:
            symbols = hit.get("symbols", {}).get("matches", [])
            names = sorted(
                symbols, key=lambda m: (m["kind"] != "pair", len(m["name"]), m["name"])
            )
            detail = ", ".join(m["name"] for m in names[:2])
            basis = (
                f"symbols: {detail}"
                if detail
                else f"matched {hit['catalog_match']['field']}"
            )
            _line(f"{hit['repo_id']} ({basis})", "  ")
    _line(
        "Catalog hits alone leave usage unknown. Forward bindings above refer to observed module sites."
    )


def _kernel_choices(search, run_dir):
    _heading("Which kernels should I use?")
    if not search:
        _line("This trace has no saved Kernel Hub search.")
        return
    messages = {
        "offline": "Kernel Hub search was offline.",
        "unavailable": "Kernel Hub could not be reached.",
        "not_needed": "The trace produced no candidates for a Hub search.",
    }
    if search["status"] in messages:
        _line(messages[search["status"]])

    choices = []
    selected_layers = {
        lead["layer"]
        for lead in search.get("layers", [])
        if lead["example_status"]
        in ("picked_for_trial", "measured_local_alternative_selected")
    }
    for kernel in search.get("local_prefill_kernels", []):
        choices.append(
            f"Try {kernel} for prompt processing. It passed the saved correctness check and reduced prompt time in the matching trial."
        )
    for lead in search.get("layers", []):
        layer, repo = lead["layer"], lead["repo_id"]
        status = lead["example_status"]
        sites = lead["sites"]
        if status in ("previously_rejected", "failed_trial"):
            reason = lead.get(
                "failure_summary", "A previous trial failed on this setup."
            )
            keep = "" if layer in selected_layers else f"Keep the current {layer}. "
            choices.append(f"{keep}Skip {repo}. {reason}")
        elif status == "picked_for_trial":
            choices.append(
                f"Test {repo} at {len(sites['original_forward'])} {layer} sites. The example checks its outputs against the current model."
            )
        if sites["already_bound"]:
            choices.append(
                f"{repo} is already in use at {len(sites['already_bound'])} {layer} sites."
            )
    if not choices and search["status"] == "searched":
        _line(
            "Keep the current kernels. The search found no replacement ready to try on this setup."
        )
    for choice in dict.fromkeys(choices):
        _line(choice)
    for lead in search.get("layers", []):
        if lead["example_status"] == "requires_device_integration":
            _line(
                f"{lead['repo_id']} has a {lead['layer']} adapter. Using it here would require a Transformers integration for this device."
            )
    if search.get("example") and run_dir is not None:
        print(shlex.join([sys.executable, str(Path(run_dir) / search["example"])]))


def _fusion_idea(result):
    if "development_tasks" in result:
        _heading("Which kernel could I build?")
        for task in result["development_tasks"]:
            _line(task["title"])
            _line(
                f"{task['occurrences']} occurrences per prefill · {task['dtype'].removeprefix('torch.')} {task['shape']} · e.g. {task['scope']}",
                "  ",
            )
            if leads := task.get("hub_leads"):
                lead = leads[0]
                target = lead["repo_id"]
                if lead["symbols"]:
                    target += " / " + ", ".join(lead["symbols"])
                    _line(
                        f"First test {target} from the Hub. Check layout and correctness.",
                        "  ",
                    )
                else:
                    _line(
                        f"Inspect {target} first. Its catalog metadata matches this pair.",
                        "  ",
                    )
            if task.get("prior_trials"):
                _line(
                    "Earlier implementations in this run: "
                    + ", ".join(r["verdict"] for r in task["prior_trials"]),
                    "  ",
                )
        if not result["development_tasks"]:
            _line("No repeated single-use operation pair qualified in this capture.")
        else:
            _line(
                "These are prototype ideas. Check backend fusion, correctness and whole-model latency."
            )
        return
    _heading("What could be faster?")
    candidates = result["candidates"]
    if not candidates:
        _line(
            "The prompt trace showed no repeated operation pairs that met the fusion criteria."
        )
        return
    names = {
        "silu": "SiLU activation",
        "mul": "multiply",
        "pow": "power",
        "mean": "average",
        "linear": "linear projection",
        "relu": "ReLU",
        "gelu": "GELU",
        "add": "add",
    }
    _line("These pairs pass a temporary result from one operation to the next.")
    rows = []
    seen = set()
    for candidate in candidates:
        pair = tuple(candidate["operators"])
        if pair in seen:
            continue
        seen.add(pair)
        ops = [op.split(".")[1] for op in pair]
        label = " then ".join(names.get(op, op) for op in ops)
        size = candidate["total_logical_intermediate_bytes"]
        amount = f"{size / 2**20:,.1f} MiB" if size >= 2**20 else f"{size:,} bytes"
        rows.append((label, str(candidate["occurrences"]), amount))
        if len(rows) == 4:
            break
    header = ("Operations", "Calls", "Total temporary data")
    sizes = [max(map(len, column)) for column in zip(header, *rows)]
    if sum(sizes) + 4 <= shutil.get_terminal_size().columns:
        for name, count, size in (header, *rows):
            print(f"{name:<{sizes[0]}}  {count:>{sizes[1]}}  {size:>{sizes[2]}}")
    else:
        for name, count, size in rows:
            _line(f"{name} ({count} calls, {size} total)")
    _line(
        "Check whether the GPU already combines a pair before writing a fused kernel. Speedup is unmeasured."
    )


def show_cache_trial(result):
    _line(
        "Static cache from the first cached step"
        if result.get("reserve_at") == "decode"
        else "Static cache from prefill"
    )
    if result["verdict"] == "error":
        _line(result["reason"])
        return
    keep = (
        "Use the static cache for this workload."
        if result["verdict"] == "faster"
        else "Keep the current cache for this workload."
    )
    _line(keep)
    if "failure" in result:
        _line(
            f"The static cache first changed logits at generated token {result['failure']['generated_token_index'] + 1}."
        )
    if "timing" in result:
        deltas = [
            (label, result["timing"][phase]["median_paired_saving_ms"])
            for phase, label in (
                ("prefill", "Prompt"),
                ("decode", "Cached steps combined"),
                ("total", "Whole generation"),
            )
        ]
        split = result.get("paired_decode_savings")
        if result.get("reserve_at") == "decode" and split:
            deltas.append(
                ("First cached step including migration", split["first_cached_step_ms"])
            )
            if split["remaining_cached_steps_ms"] is not None:
                deltas.append(
                    (
                        "Remaining cached steps combined",
                        split["remaining_cached_steps_ms"],
                    )
                )
        for label, delta in deltas:
            direction = "faster" if delta >= 0 else "slower"
            _line(
                f"{label}: static cache {abs(delta):.2f} ms {direction} (paired median)."
            )
        if result["verdict"] == "slower":
            phase, values = max(
                result["timing"].items(), key=lambda item: item[1]["losses_above_floor"]
            )
            _line(
                f"The {phase} regression exceeded its margin in {values['losses_above_floor']} paired blocks."
            )
        total = result["timing"]["total"]
        if policy := result.get("policy"):
            _line(
                f"{total['wins_above_floor']} of {policy['measured_blocks']} blocks exceeded the {total['floor_ms']:.2f} ms whole-generation margin ({policy['required_wins']} required)."
            )
        else:
            _line(f"Required whole-generation margin {total['floor_ms']:.2f} ms.")


def show_compilation(result, run_dir):
    _heading("Compilation vs eager")
    if result.get("disabled"):
        _line("Skipped by request.")
        return
    labels = {"mlp": "MLP modules", "whole_model": "Whole model", "eager": "Eager"}
    for trial in result["trials"]:
        if timing := trial.get("timing"):
            total = timing["total"]
            detail = f"{total['incumbent_ms']:.2f} → {total['candidate_ms']:.2f} ms total ({trial['verdict']})"
        elif trial["verdict"] == "failed_logits":
            detail = f"failed logits check at step {trial['step'] + 1}"
        else:
            detail = trial.get("reason", trial["verdict"]).splitlines()[0]
        _line(f"{labels[trial['mode']]}: {detail}")
    _line(f"Selected for this workload: {labels[result['selected']]}.")
    if result.get("recipe") and run_dir:
        print(
            f"Apply to your model: {Path(run_dir) / 'compile-trials' / 'apply_to_model.py'}"
        )


def show_discovery(result, run_dir=None, *, details=False):
    if "compilation" in result:
        show_compilation(result["compilation"], run_dir)
        if result.get("trace_status") == "unavailable":
            _line("This run has no saved dependency trace.")
            return
        print()

    options = (result.get("hub_search") or {}).get("loading_options")
    if options:
        from .recipes import show_options

        show_options(
            options, full=False, historical_runtime="compilation" not in result
        )
        for trial in (options.get("comparison") or {}).get("results", []):
            if trial["verdict"] == "faster" and run_dir:
                print(Path(run_dir) / "kernel-configs" / (trial["name"] + ".py"))
        if run_dir and (details or not options.get("comparison")):
            print(f"Python examples: {Path(run_dir) / 'kernel-configs'}/")
            print(shlex.join(["kernelatlas", "compare-kernels", str(run_dir)]))
        print()
    _hub_summary(result.get("hub_summary"))
    if not (options and options.get("comparison")):
        if result.get("hub_summary"):
            print()
        _kernel_choices(result.get("hub_search"), run_dir)
    print()
    if result.get("compilation", {}).get("recipe"):
        _line(
            "Build ideas below come from the eager trace. Fusion in the compiled model is unverified."
        )
    _fusion_idea(result)
    if "decode" in result:
        decode = result["decode"]
        _heading("First cached token")
        _line(
            f"Observed {len(decode['candidates'])} repeated operation pairs during decode."
        )
        if decode["retained_cache_appends"]:
            count = len(decode["retained_cache_appends"])
            _line(
                f"{count} key/value cache tensors grew by concatenation. Their outputs totaled {decode['cache_append_output_bytes'] / 2**20:.1f} MiB. GPU cost is unmeasured."
            )
        if "cache_trial" in decode:
            history = decode.get("cache_results", [])
            if history:
                latest = {r.get("reserve_at", "prefill"): r for r in history}
                for measurement in latest.values():
                    show_cache_trial(measurement)
            else:
                _line("Test a preallocated cache against the full generation workload.")
            print(shlex.join(decode["cache_trial"]))
            if "decode_cache_trial" in decode:
                print(shlex.join(decode["decode_cache_trial"]))
    elif result.get("decode_status") == "dependency_trace_unavailable":
        _line(
            "This saved profile has prefill dependencies only. Profile it again to inspect cached decoding."
        )
    if "trial" in result:
        print()
        _heading("Test your own SiLU and multiply kernel")
        print(shlex.join(result["trial"]["command"]))
    if details:
        print()
        _discovery_details(result, run_dir)


def _discovery_details(result, run_dir):
    _heading("Discovered prefill candidates · unvalidated")
    if not result["candidates"]:
        _line("No repeated eligible single-use edges in this capture.")
        if "hub_search" in result:
            _show_hub(result["hub_search"], run_dir)
        return
    for candidate in result["candidates"][:5]:
        pair = " → ".join(candidate["operators"])
        _line(pair)
        _line(
            f"{candidate['occurrences']} occurrences · shape {candidate['intermediate']['shape']} · {candidate['total_logical_intermediate_bytes']:,} logical intermediate bytes",
            "  ",
        )
    _line(
        "Ranked by tensor size across calls. Backend fusion and speedup are unverified."
    )
    if len(result["candidates"]) > 5:
        _line("Use --format json for every candidate and its source nodes.")
    if "hub_search" in result:
        print()
        _show_hub(result["hub_search"], run_dir)


def _best_example(folder, name):
    print()
    _heading("Run the selected configuration")
    print(shlex.join([sys.executable, str(folder / name)]))


def show_evaluations(
    rows,
    model_id,
    reports,
    *,
    opportunities=(),
    details=False,
    output_format="text",
    baseline=None,
    saved_profile=None,
):
    doc = guidance(rows, model_id, reports, opportunities, details)
    if saved_profile:
        doc["saved_profile"] = saved_profile
    if baseline is not None:
        doc["baseline"] = baseline
    if output_format == "json":
        print(json.dumps(doc, indent=2, allow_nan=False))
        return
    _heading(model_id or "KernelAtlas")
    if saved_profile and "evidence" in saved_profile:
        from .recipes import show_comparison

        _line(f"Saved evidence: {saved_profile['evidence']}")
        show_comparison(saved_profile["comparison"])
        print()
        if saved_profile.get("hub_summary"):
            _hub_summary(saved_profile["hub_summary"])
            print()
        _fusion_idea(saved_profile)
        if not doc["runs"]:
            return
    elif saved_profile:
        if saved_profile.get("best_example"):
            _best_example(
                Path(saved_profile["directory"]), saved_profile["best_example"]
            )
        _line(f"Latest matching profile: {saved_profile['directory']}")
        show_discovery(
            saved_profile["discovery"],
            Path(saved_profile["directory"]),
            details=details,
        )
        if not doc["runs"]:
            return
    if baseline is not None:
        precision = baseline["precision"]
        weights = precision["weights"]
        dtype = precision["dtype"].removeprefix("torch.")
        dtype = {"float32": "FP32", "float16": "FP16", "bfloat16": "BF16"}.get(
            dtype, dtype
        )
        context = [baseline.get("hardware") or baseline["device"], dtype]
        if weights["mode"] != "none":
            context.append(
                f"{weights['mode'].upper()} weights"
                if weights["mode"] != "checkpoint"
                else "checkpoint quantization"
            )
        context.append(
            "pretrained weights"
            if baseline["weight_source"] == "pretrained"
            else "random weights (seed 0)"
        )
        if baseline.get("torch_compile") is False:
            context.append("eager")
        if baseline.get("cache_class"):
            context.append(baseline["cache_class"])
        _line(" · ".join(context))
        if "proxy" in baseline:
            proxy = baseline["proxy"]
            _line(
                f"Reduced text decoder: {proxy['effective']['num_hidden_layers']} of {proxy['original']['num_hidden_layers']} layers, vocabulary {proxy['effective']['vocab_size']}."
            )
        if "converted_linears" in weights:
            converted = len(weights["converted_linears"])
            total = converted + len(weights["skipped_linears"])
            fraction = (
                weights["packed_weight_elements"]
                / weights["original_parameter_elements"]
            )
            _line(
                f"Packed {converted}/{total} linear modules ({fraction:.1%} of model parameters)."
            )
        run_dir = Path(baseline["report"]).parent
        if "discovery" in baseline:
            print()
            show_discovery(baseline["discovery"], run_dir)
        else:
            if "compilation" in baseline:
                show_compilation(baseline["compilation"], run_dir)
            _line("Kernel discovery was disabled for this run.")
        print()
        _heading(
            "Eager baseline timing"
            if baseline.get("compilation", {}).get("trials")
            else "How fast is the current model?"
        )
        _line(
            f"Process the {baseline['prompt_tokens']}-token prompt   {baseline['prefill_ms']:.1f} ms"
        )
        if "generation" in baseline:
            gen = baseline["generation"]
            steps = gen["per_cached_token"]
            if steps["count"]:
                _line(
                    f"Each cached token             {steps['median_ms']:.1f} ms median · {steps['p90_ms']:.1f} ms p90"
                )
            _line(
                f"Generate {baseline['max_new_tokens']} tokens in total   {gen['total_median_ms']:.1f} ms"
            )
        else:
            _line(f"Generate one more token       {baseline['decode_ms']:.1f} ms")
        _line(
            f"Median model time, batch size {baseline['batch_size']}, with device synchronization."
        )
        print()
        print(f"Full report: {baseline['report']}")
        if baseline.get("best_example"):
            _best_example(run_dir, baseline["best_example"])
        if details:
            _line(
                f"PyTorch {baseline['torch']} · {baseline['device']} · revision {baseline['config_source']['revision']}"
            )
            if "discovery" in baseline:
                print()
                _discovery_details(baseline["discovery"], run_dir)
        if not doc["runs"]:
            if not details:
                _line("Use --details for the evidence, or --format json for tools.")
            return
        print()
        _heading("Earlier measurements for this model and precision")
        if not details:
            for run in doc["runs"]:
                for choice in run["existing_kernels"]:
                    if choice["action"] == "try_prefill":
                        _line(
                            f"{choice['candidate']} reduced prompt time in a saved {run['dtype'].removeprefix('torch.')} trial on {run['device']}."
                        )
                for task in run["development_tasks"]:
                    _line(f"Proposed experiment: {task['title']}.")
            _line(
                "Use --details for trial results and example code, or --format json for tools."
            )
            return
    if not doc["runs"]:
        _line(
            f"No saved evaluations or kernel proposals match this query in {reports}."
        )
        return
    for run in doc["runs"]:
        _run_view(run)
        print()
    if details and rows:
        _heading("Trial details")
        _show_trials(rows, reports)
    else:
        _line("--details  trial evidence    --format json  agent output")


def show_trial(result, folder):
    root = Path(folder) / "optimization"
    command = ["kernelatlas", "trial", str(folder)]
    if "task" in result:
        task = result["task"]
        _heading(f"{task['model_id']} · SiLU and multiply")
        _line(
            f"{len(task['sites'])} sites · {task['hardware']} · BF16 · {len(task['prompt_ids'])}-token prompt + {task['cached_tokens']} cached steps"
        )
        _line(
            f"Current best: {result['best']}. Completed trials: {len(result['history'])}."
        )
        for entry in result["history"]:
            label = f"{entry['id']}  {entry['verdict']}  {entry['sha256'][:12]}"
            if "total_timing" in entry:
                total = entry["total_timing"]
                label += f"  ({total['median_paired_saving_ms']:+.3f} ms, margin {total['floor_ms']:.3f} ms)"
            _line(label)
            if "workload" in entry and "timing" in entry:
                timing = entry["timing"]
                _line(
                    f"Candidate prompt {timing['prefill']['candidate_ms']:.1f} ms · whole generation {timing['total']['candidate_ms']:.1f} ms",
                    "  ",
                )
            steps = entry.get("per_cached_token", {}).get("candidate", {})
            if steps.get("count"):
                _line(
                    f"Cached tokens: {steps['median_ms']:.1f} ms median · {steps['p90_ms']:.1f} ms p90",
                    "  ",
                )
            if "reason" in entry:
                _line(entry["reason"], "  ")
        _line("Copy reference.py, edit candidate(gate, up), and submit the new file.")
        print(shlex.join(["cp", str(root / task["template"]), "my_gate.py"]))
        print(shlex.join(command + ["my_gate.py"]))
        _line("The task includes source sites and real gate inputs.")
        print(root / "task.json")
        return
    _heading(f"Trial {result['id']} · {result['verdict']}")
    if result.get("cached"):
        _line("Reused a saved result for this implementation.")
    if "reason" in result:
        _line(result["reason"])
    if "failure" in result:
        failure = result["failure"]
        _line(
            f"First failure during {failure['phase']} at {failure.get('site', 'model logits')}."
        )
        print(root / "trials" / result["id"] / failure["fixture"])
    if "timing" in result:
        if "per_cached_token" in result:
            steps = result["per_cached_token"]["candidate"]
            _line(
                f"Candidate prompt: {result['timing']['prefill']['candidate_ms']:.1f} ms"
            )
            if steps["count"]:
                _line(
                    f"Candidate cached token: {steps['median_ms']:.1f} ms median · {steps['p90_ms']:.1f} ms p90"
                )
        total = result["timing"]["total"]
        _line(
            f"Whole workload: {total['median_paired_saving_ms']:+.3f} ms paired saving. Required margin {total['floor_ms']:.3f} ms."
        )
    _line(f"Current best: {result['best']}")
    print(root / result["evidence"])
    print(shlex.join(command + ["--replay", result["id"]]))
