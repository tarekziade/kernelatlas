"""Compare eager and Inductor for the default Qwen MPS prefill + one cached step."""

import argparse
from functools import partial
from itertools import permutations
import json
from pathlib import Path
from time import perf_counter

import torch
import transformers
from torch._dynamo.utils import counters
from transformers import AutoModelForCausalLM, StaticCache

from kernelatlas.generation import trajectory
from kernelatlas.numerics import logits_close, model_logit_policy
from kernelatlas.profiling import _cache_key, sequential_prompt
from kernelatlas.trials import POLICY, classify, _write


def run(args, result):
    report = json.loads((args.run / "report.json").read_text())
    length = report["phases"]["prefill"]["cache_length"]
    if (
        "workload" in report
        or report["weight_source"] != "pretrained"
        or report["dtype"] != "torch.bfloat16"
        or report["precision"]["weights"]["mode"] != "none"
        or report["hardware"] != torch.backends.mps.get_name()
        or not report["revision"]
        or report["torch"] != str(torch.__version__)
        or report["transformers"] != transformers.__version__
    ):
        raise ValueError(
            "Select a matching pretrained BF16 MPS profile with default prompt and decode workload"
        )
    result.update(
        model=report["model_id"],
        revision=report["revision"],
        hardware=report["hardware"],
        torch=report["torch"],
        transformers=report["transformers"],
        dtype=report["dtype"],
        attention=report["attention"],
        prompt_length=length,
        cached_steps=1,
        use_kernels=False,
        **model_logit_policy(report["dtype"]),
    )

    def compile_mlps(model):
        if model.config.model_type != "qwen3":
            raise ValueError("Regional MLP experiment requires a Qwen3 decoder")
        for layer in model.model.layers:
            layer.mlp.compile(backend="inductor", options=options)
        result["compiled_mlp_sites"] = len(model.model.layers)

    options = {"emulate_precision_casts": True} if args.preserve_casts else None
    if args.hub:
        from kernelatlas.loading_trial import _measure

        if not (args.compile and args.mlp_only and args.cache == "dynamic"):
            raise ValueError(
                "The Hub comparison requires --compile --mlp-only --cache dynamic"
            )
        recipes = json.loads((args.run / "kernel-configs.json").read_text())
        selection = next((c for c in recipes["configs"] if c["name"] == args.hub), None)
        if selection is None:
            raise ValueError(f"No {args.hub} configuration in the saved profile")
        result.update(
            hub_config=selection,
            reference="native with the same regional compilation",
            preserve_casts=args.preserve_casts,
            mlp_only=True,
            use_kernels=selection["use_kernels"],
        )
        result.update(_measure(report, recipes, selection, prepare_model=compile_mlps))
        return

    def load_model():
        return AutoModelForCausalLM.from_pretrained(
            report["model_id"],
            revision=report["revision"],
            device_map="mps",
            dtype=torch.bfloat16,
            use_kernels=False,
            attn_implementation=report["attention"],
        ).eval()

    native = load_model()
    key = _cache_key(native)
    ids = sequential_prompt(native, "mps", length, 2)
    factory = (
        partial(StaticCache, config=native.config, max_cache_len=length + 1)
        if args.cache == "static"
        else None
    )
    if (args.decode_only or args.preserve_casts or args.mlp_only) and not args.compile:
        raise ValueError("The requested compile options require --compile")
    if args.decode_only and length <= 1:
        raise ValueError("Decode-only comparison requires a multi-token prompt")
    result.update(
        decode_only=args.decode_only,
        preserve_casts=args.preserve_casts,
        mlp_only=args.mlp_only,
    )
    candidate = native
    if args.compile:
        if args.mlp_only:
            candidate = load_model()
            compile_mlps(candidate)
        elif args.decode_only:
            compiled = torch.compile(native, backend="inductor", options=options)

            def candidate(**inputs):
                forward = compiled if inputs["input_ids"].shape[1] == 1 else native
                return forward(**inputs)
        else:
            candidate = torch.compile(native, backend="inductor", options=options)
    # Only vary cache for the eager static control. Compare compile against eager on identical cache.
    reference_factory = factory if args.compile else None

    def outputs(model, inputs, factory):
        cache_args = {key: factory()} if factory else {}
        observed = []
        for _ in range(2):
            out = model(input_ids=inputs, **cache_args, use_cache=True)
            observed.append(out.logits.cpu())
            cache = getattr(out, key)
            inputs = out.logits[:, -1].argmax(-1, keepdim=True)
            cache_args = {key: cache}
        return observed, type(cache).__name__

    result["checks"] = []
    for i, prompt in enumerate(
        [
            ids,
            torch.randint(
                native.config.vocab_size,
                ids.shape,
                generator=torch.Generator().manual_seed(42),
            ).to("mps"),
            ids[:, :7],
        ]
    ):
        expected, reference_cache = outputs(native, prompt, reference_factory)
        start = perf_counter()
        actual, actual_cache = outputs(candidate, prompt, factory)
        if i == 0:
            result.update(
                first_candidate_check_seconds=perf_counter() - start,
                reference_cache=reference_cache,
                actual_cache=actual_cache,
            )
        passed = True
        for step, (a, b) in enumerate(zip(expected, actual, strict=True)):
            misses = int(
                (~logits_close(b, a, result)).sum()
            )
            token_match = torch.equal(a[:, -1].argmax(-1), b[:, -1].argmax(-1))
            passed &= misses == 0 and token_match
            result["checks"].append(
                dict(
                    prompt=i,
                    step=step,
                    logits=a.numel(),
                    outside_tolerance=misses,
                    max_abs_error=(a.float() - b.float()).abs().max().item(),
                    greedy_match=token_match,
                )
            )
        if not passed:
            result["verdict"] = "failed_logits"
            return
    if args.compile and not counters["stats"]["unique_graphs"]:
        raise RuntimeError("No compiled graph observed")
    blocks = []
    orders = list(permutations(("incumbent", "control", "candidate")))
    for i in range(POLICY["warmup_blocks"] + POLICY["measured_blocks"]):
        row = {}
        for arm in orders[i % len(orders)]:
            test = arm == "candidate"
            row[arm] = trajectory(
                candidate if test else native,
                ids,
                1,
                key,
                {},
                synchronize=torch.mps.synchronize,
                cache_factory=factory if test else reference_factory,
            )
        if i >= POLICY["warmup_blocks"]:
            blocks.append(row)
    result["verdict"], result["timing"] = classify(blocks)
    result["blocks_ms"] = blocks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--cache", choices=("dynamic", "static"), required=True)
    parser.add_argument("--compile", action="store_true")
    region = parser.add_mutually_exclusive_group()
    region.add_argument("--decode-only", action="store_true")
    region.add_argument("--mlp-only", action="store_true")
    parser.add_argument("--preserve-casts", action="store_true")
    parser.add_argument(
        "--hub", choices=("automatic", "selective", "attention_only", "combined")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"cache": args.cache, "backend": "inductor" if args.compile else "eager"}
    try:
        with torch.inference_mode():
            run(args, result)
    except Exception as exc:
        import traceback

        traceback.print_exc()
        result.update(verdict="error", reason=f"{type(exc).__name__}: {exc}"[:2000])
    result["graphs"] = counters["stats"]["unique_graphs"]
    result["graph_breaks"] = {str(k): v for k, v in counters["graph_break"].items()}
    _write(args.output, result)
