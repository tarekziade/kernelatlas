"""Profile a model, discover prefill fusion candidates and show kernel guidance."""

import argparse
import json
from pathlib import Path

from .atlas import Atlas
from .discovery import analyze_run
from .hub import search, summarize_search
from .example import prepare_example, write_best_combo
from .terminal import (
    profile_progress,
    show_evaluations,
    show_discovery,
    show_trial,
    show_cache_trial,
)
from .trials import run_trial


def main():
    parser = argparse.ArgumentParser(prog="kernelatlas", description=__doc__)
    from . import __version__

    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    compare = commands.add_parser(
        "compare-kernels",
        help="Check and benchmark automatic and selective loading from a saved profile",
    )
    compare.add_argument("run", type=Path)
    compare.add_argument("--format", choices=("text", "json"), default="text")
    suggest = commands.add_parser(
        "recommend",
        help="Show automatic and selective Transformers Hub configs without loading model weights",
    )
    suggest.add_argument("model_id")
    suggest.add_argument("--revision")
    suggest.add_argument("--device", choices=("auto", "mps", "cuda"), default="auto")
    suggest.add_argument("--format", choices=("text", "json"), default="text")
    profile = commands.add_parser(
        "profile",
        help="Profile a Hub model and show discovered candidates and saved kernel guidance",
    )
    profile.add_argument("model_id", metavar="MODEL_ID")
    profile.add_argument(
        "--random-weights",
        action="store_true",
        help="Download only the Hub config and initialize random weights locally (seed 0)",
    )
    profile.add_argument(
        "--dtype",
        choices=("auto", "bf16", "fp16", "fp32", "bfloat16", "float16", "float32"),
        default="auto",
        help="Floating-point dtype (default: BF16 when supported, otherwise FP16)",
    )
    profile.add_argument(
        "--quantization",
        choices=("auto", "none", "int8", "int4"),
        default="auto",
        help="Weight format (default: keep checkpoint quantization; int8/int4 pack eligible MPS Linear weights)",
    )
    profile.add_argument(
        "--prompt-length",
        type=int,
        help="Greedy generation prompt length (default 32 tokens)",
    )
    profile.add_argument(
        "--max-new-tokens",
        type=int,
        help="Generated tokens including the first from prefill (default 2 with --prompt-length)",
    )
    profile.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compare eager, whole-model Inductor and compiled MLPs on this workload (enabled by default)",
    )
    profile.add_argument("--revision", help="Hub branch, tag or commit (default: main)")
    profile.add_argument("--device", choices=("mps", "cuda"), default="mps")
    profile.add_argument(
        "--out",
        help="Fresh output directory (default: timestamped directory under runs/)",
    )
    profile.add_argument(
        "--reports",
        default="evidence",
        help="Saved evaluations to display after profiling (default: evidence)",
    )
    profile.add_argument(
        "--discover",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Discover repeated prefill and cached-decode operations (enabled by default)",
    )
    discover = commands.add_parser(
        "discover", help="Analyze saved execution dependencies without a GPU"
    )
    discover.add_argument(
        "run",
        type=Path,
        help="Saved profile directory containing a prefill dataflow trace",
    )
    discover.add_argument(
        "--details",
        action="store_true",
        help="Include operation pairs and Kernel Hub search results",
    )
    discover.add_argument("--format", choices=("text", "json"), default="text")
    trial = commands.add_parser(
        "trial", help="Inspect a saved opportunity or benchmark one local candidate"
    )
    trial.add_argument("run", type=Path, help="Saved profile directory")
    trial.add_argument(
        "candidate",
        type=Path,
        nargs="?",
        help="Single Python file defining candidate(gate, up)",
    )
    trial.add_argument(
        "--repeat", action="store_true", help="Measure again against the current best"
    )
    trial.add_argument(
        "--replay",
        metavar="TRIAL_ID",
        help="Rerun archived source against its original incumbent",
    )
    cache_options = trial.add_mutually_exclusive_group()
    cache_options.add_argument(
        "--decode-static-cache",
        dest="static_cache",
        action="store_const",
        const="decode",
        help="Reserve and populate StaticCache at the first decode step",
    )
    cache_options.add_argument(
        "--static-cache",
        action="store_const",
        const="prefill",
        help="Compare Transformers StaticCache with the observed growing KV cache",
    )
    trial.add_argument("--format", choices=("text", "json"), default="text")
    atlas = commands.add_parser("atlas")
    actions = atlas.add_subparsers(dest="action", required=True)
    build = actions.add_parser("build", help="Write the complete JSON Atlas to stdout")
    build.add_argument("reports")
    show = actions.add_parser(
        "show", help="Show measured candidates and correctness decisions"
    )
    show.add_argument(
        "reports",
        nargs="?",
        default="evidence",
        help="Saved report directory (default: evidence)",
    )
    show.add_argument(
        "--model-id", help="Exact Hub model ID of the profiled checkpoint"
    )
    show.add_argument("--family")
    show.add_argument("--torch-version")
    show.add_argument("--prompt-length", type=int)
    for command in (show, profile):
        command.add_argument(
            "--format",
            choices=("text", "json"),
            default="text",
            help="Result format on stdout",
        )
        command.add_argument(
            "--details",
            action="store_true",
            help="Include operation pairs, kernel search evidence and saved trial measurements",
        )
    args = parser.parse_args()
    if args.command == "compare-kernels":
        from .loading_trial import compare
        from .recipes import show_comparison

        try:
            comparison = compare(args.run)
        except (ImportError, OSError, ValueError) as error:
            parser.exit(1, f"{error}\n")
        if args.format == "json":
            print(json.dumps(comparison, indent=2))
        else:
            show_comparison(comparison)
        return
    if args.command == "recommend":
        from .recipes import recommend, show_options

        try:
            options = recommend(args.model_id, args.revision, args.device)
        except (ImportError, ValueError, OSError) as error:
            parser.exit(1, f"{error}\n")
        if args.format == "json":
            print(json.dumps(options, indent=2))
        else:
            show_options(options)
        return
    if args.command == "trial" and args.static_cache:
        if args.candidate or args.repeat or args.replay:
            parser.error(
                "A cache option measures a fresh cache trial and takes only a profile directory"
            )
        try:
            from .cache_trial import run_cache_trial

            result = run_cache_trial(args.run, reserve_at=args.static_cache)
        except (ValueError, OSError, ImportError) as error:
            parser.exit(1, f"{error}\n")
        if args.format == "json":
            print(json.dumps(result, indent=2, allow_nan=False))
        else:
            show_cache_trial(result)
            print(args.run / result["evidence"])
        return
    if args.command == "trial":
        if (args.replay and (args.candidate or args.repeat)) or (
            args.repeat and not args.candidate
        ):
            parser.error("Use a candidate with optional --repeat, or --replay TRIAL_ID")
        try:
            result = run_trial(
                args.run, args.candidate, replay=args.replay, repeat=args.repeat
            )
        except (ValueError, OSError, ImportError) as error:
            parser.exit(1, f"{error}\n")
        if args.format == "json":
            print(json.dumps(result, indent=2, allow_nan=False))
        else:
            show_trial(result, args.run)
        return
    if args.command == "discover":
        try:
            discovery = analyze_run(args.run)
        except (OSError, ValueError) as error:
            parser.exit(1, f"{error}\n")
        if args.format == "json":
            print(json.dumps(discovery, indent=2, allow_nan=False))
        else:
            show_discovery(discovery, args.run, details=args.details)
        return
    baseline = None
    if args.command == "profile":
        try:
            from . import profile
        except ModuleNotFoundError as error:
            if error.name not in ("torch", "transformers"):
                raise
            parser.exit(
                1,
                "Install the profiling dependencies with python -m pip install 'kernelatlas[profile]'\n",
            )
        try:
            out, report = profile(
                args.model_id,
                compile=args.compile,
                revision=args.revision,
                device=args.device,
                out=args.out,
                random_weights=args.random_weights,
                dtype=args.dtype,
                quantization=args.quantization,
                discover=args.discover,
                prompt_length=args.prompt_length,
                max_new_tokens=args.max_new_tokens,
                progress=profile_progress,
            )
        except (ValueError, OSError) as error:
            parser.exit(1, f"{error}\n")
        baseline = {
            "compilation": report["compilation"],
            "model_id": report["model_id"],
            "weight_source": report["weight_source"],
            "torch_compile": False,
            "cache_class": report["phases"]["prefill"].get("cache_class"),
            "weights": report["weights"],
            "precision": report["precision"],
            "config_source": report["config_source"],
            "revision": report["revision"],
            "torch": report["torch"],
            "device": report["device"],
            "hardware": report["hardware"],
            "batch_size": report["phases"]["prefill"]["output_shape"][0],
            "prompt_tokens": len(report["workload"]["prompt_ids"])
            if "workload" in report
            else report["phases"]["prefill"]["cache_length"],
            "report": str(out / "report.json"),
            "prefill_ms": report["phases"]["prefill"]["median_ms"],
            "decode_ms": report["phases"].get("decode", {}).get("median_ms"),
        }
        if "generation" in report:
            baseline.update(
                generation=report["generation"],
                max_new_tokens=report["workload"]["max_new_tokens"],
            )
        if "proxy" in report:
            baseline["proxy"] = report["proxy"]
    root = Path(args.reports)
    if (
        (
            args.command == "profile"
            or (args.command == "atlas" and args.action == "show")
        )
        and not (root / "report.json").is_file()
        and not any(root.glob("*/report.json"))
    ):
        result = Atlas({})
    else:
        try:
            result = Atlas.from_reports(root)
        except (ValueError, OSError) as error:
            parser.exit(1, f"{error}\n")
    if args.command == "profile" and args.discover:
        discovery = analyze_run(out)
        profile_progress(
            "Checking Transformers bindings and searching the public Kernel Hub"
        )
        # Saved Atlas kernel checks were run with full prompt logits.
        trials = result.evaluations(
            model_id=args.model_id,
            torch_version=report["torch"],
            prompt_length=baseline["prompt_tokens"],
        )
        trials = [
            row
            for row in trials
            if not report["compilation"]["recipe"]
            and "workload" not in report
            and report["revision"]
            and row["model_revision"] == report["revision"]
            and row["device"] == report["device"]
            and row.get("weight_mode", "none") == report["precision"]["weights"]["mode"]
            and row["dtype"] == report["dtype"]
            and row.get("hardware") == report["hardware"]
        ]
        rejected = {
            (row["candidate"], row.get("kernel_revision"))
            for row in trials
            if row["correctness"] == "rejected"
        }
        snapshot = search(discovery, rejected)
        prepare_example(out, root, report, discovery, snapshot, trials)
        if (
            report.get("kernel_registrations")
            and report["weight_source"] == "pretrained"
            and report["precision"]["weights"]["mode"] == "none"
        ):
            from .recipes import loading_options

            excluded = {
                (lead["repo_id"], lead.get("pinned_revision"))
                for lead in snapshot["layers"]
                if lead["example_status"] in ("previously_rejected", "failed_trial")
            }
            snapshot["loading_options"] = loading_options(
                args.model_id,
                report["revision"],
                args.device,
                report["dtype"].removeprefix("torch."),
                report["kernel_registrations"],
                excluded=excluded,
            )
            from .loading_trial import matching_measurement

            options = snapshot["loading_options"]
            if compilation := report["compilation"]["recipe"]:
                options["compile_recipe"] = compilation
                for config in options["configs"]:
                    config["python"] += "\n\n" + report["compilation"]["python"]
            measured = matching_measurement(report, options, root)
            if measured:
                options["comparison"] = measured
                options["status"] = "compared"
            code_dir = out / "kernel-configs"
            code_dir.mkdir(exist_ok=True)
            for config in options["configs"]:
                (code_dir / f"{config['name']}.py").write_text(config["python"] + "\n")
            (out / "kernel-configs.json").write_text(
                json.dumps(snapshot["loading_options"], indent=2) + "\n"
            )
            if options.get("compile_recipe") and not measured:
                from .loading_trial import compare

                comparison = compare(out, progress=profile_progress)
                options["comparison"] = comparison
                options["status"] = "compared"
                (out / "kernel-configs.json").write_text(
                    json.dumps(options, indent=2) + "\n"
                )
        (out / "hub-search.json").write_text(json.dumps(snapshot, indent=2) + "\n")
        discovery["hub_summary"] = summarize_search(
            snapshot, discovery["development_tasks"]
        )
        discovery["hub_search"] = snapshot
        baseline["discovery"] = discovery
    if args.command == "profile":
        loading = (
            baseline.get("discovery", {}).get("hub_search", {}).get("loading_options")
        )
        baseline["best_example"] = write_best_combo(out, report, loading)
    if args.command == "atlas" and args.action == "build":
        print(json.dumps(result.data, indent=2, allow_nan=False))
        return
    filters = {"model_id": args.model_id}
    if args.command == "profile":
        filters.update(
            dtype=report["dtype"], weight_mode=report["precision"]["weights"]["mode"]
        )
    if args.command == "atlas":
        filters.update(
            family=args.family,
            torch_version=args.torch_version,
            prompt_length=args.prompt_length,
        )
    saved_profile = None
    if args.command == "atlas" and (args.model_id or (root / "report.json").is_file()):
        from .discovery import latest_profile

        folder = latest_profile(
            Path("runs") if args.reports == "evidence" else root,
            args.model_id,
            family=args.family,
            torch_version=args.torch_version,
            prompt_length=args.prompt_length,
        )
        if folder:
            discovery = analyze_run(folder)
            comparison = (
                discovery.get("hub_search", {})
                .get("loading_options", {})
                .get("comparison")
            )
            if comparison:
                from .loading_trial import matching_scope

                saved = matching_scope(comparison["scope"], root)
                if saved:
                    for field in ("runtime_comparison", "rmsnorm_epsilon_audit"):
                        if field in saved:
                            comparison.setdefault(field, saved[field])
            saved_profile = {"directory": str(folder), "discovery": discovery}
            if (folder / "best_combo.py").is_file():
                saved_profile["best_example"] = "best_combo.py"
        elif args.model_id:
            from .loading_trial import saved_guidance

            saved_profile = saved_guidance(
                root, args.model_id, args.family, args.torch_version, args.prompt_length
            )
    show_evaluations(
        result.evaluations(**filters),
        args.model_id,
        args.reports,
        opportunities=result.opportunities(**filters),
        details=args.details,
        output_format=args.format,
        baseline=baseline,
        saved_profile=saved_profile,
    )


if __name__ == "__main__":
    main()
