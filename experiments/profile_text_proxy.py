"""Profile a reduced text decoder on Apple Silicon using only the checkpoint config."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModelForCausalLM

from kernelatlas.discovery import analyze_run
from kernelatlas.hub import search
from kernelatlas.numerics import logits_close, model_logit_policy
from kernelatlas.profiling import profile_model
from kernelatlas.proxy import REVISIONS, text_proxy
from kernelatlas.terminal import profile_progress, show_discovery


@torch.inference_mode()
def check_picked_layers(model, snapshot):
    from kernels import LayerRepository, Mode, kernelize, use_kernel_mapping

    picks = [
        layer
        for layer in snapshot["layers"]
        if layer["example_status"] == "picked_for_trial"
    ]
    if not picks:
        return {"status": "no_picks"}
    prompt = torch.arange(32, device="mps")[None, :]
    tokens = [torch.tensor([[token]], device="mps") for token in range(32, 36)]

    def logits():
        result = model(prompt, use_cache=True)
        outputs = [result.logits.float()]
        for token in tokens:
            result = model(token, cache=result.cache, use_cache=True)
            outputs.append(result.logits.float())
        return outputs

    reference = logits()
    repeat = logits()
    eager_repeat_equal = all(torch.equal(a, b) for a, b in zip(reference, repeat))
    selected = []
    for layer in picks:
        repo = LayerRepository(
            layer["repo_id"],
            revision=layer["pinned_revision"],
            layer_name=layer["symbol"],
        )
        with use_kernel_mapping({layer["layer"]: {"mps": repo}}, inherit_mapping=False):
            for site in layer["sites"]["original_forward"]:
                module = model.get_submodule(site)
                kernelize(module, mode=Mode.INFERENCE, device="mps", use_fallback=False)
                if vars(module).get("_kernel_repo") != repo:
                    raise RuntimeError(f"Could not bind {repo} at {site}")
        selected.append(
            {
                "repo_id": layer["repo_id"],
                "revision": layer["pinned_revision"],
                "symbol": layer["symbol"],
                "sites": layer["sites"]["original_forward"],
            }
        )
    policy = model_logit_policy(model.dtype)
    results = []
    for phase, a, b in zip(
        ["prefill", "decode_1", "decode_2", "decode_3", "decode_4"],
        reference,
        logits(),
        strict=True,
    ):
        results.append(
            {
                "phase": phase,
                "max_abs_error": (a - b).abs().max().item(),
                "within_tolerance": bool(logits_close(b, a, policy).all()),
                "last_token_top1_equal": torch.equal(
                    a[:, -1].argmax(-1), b[:, -1].argmax(-1)
                ),
            }
        )
    passed = all(r["within_tolerance"] and r["last_token_top1_equal"] for r in results)
    return {
        "status": ("passed" if passed else "rejected")
        if eager_repeat_equal
        else "inconclusive",
        "eager_repeat_equal": eager_repeat_equal,
        **policy,
        "selected": selected,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_id", choices=REVISIONS)
    parser.add_argument("--out", type=Path, help="Fresh directory for results")
    parser.add_argument(
        "--check-kernels",
        action="store_true",
        help="Load picked Hub adapters and check prefill and decode logits against eager",
    )
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        parser.error("This experiment requires MPS.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = args.out or Path("runs") / f"{args.model_id.split('/')[-1]}-proxy-{stamp}"
    if out.exists():
        parser.error(f"Output directory already exists: {out}")
    path = hf_hub_download(
        args.model_id, "config.json", revision=REVISIONS[args.model_id]
    )
    original = AutoConfig.from_pretrained(
        Path(path).parent, local_files_only=True
    ).text_config
    text, proxy = text_proxy(args.model_id, original)
    print(
        f"{args.model_id} · random BF16 · {text.num_hidden_layers} text layers · MPS",
        flush=True,
    )
    torch.manual_seed(0)
    model = (
        AutoModelForCausalLM.from_config(text, dtype=torch.bfloat16).eval().to("mps")
    )
    proxy["model_parameter_elements"] = sum(p.numel() for p in model.parameters())
    report = profile_model(
        model,
        "mps",
        out,
        config_source={"model_id": args.model_id, "revision": REVISIONS[args.model_id]},
        progress=profile_progress,
        proxy=proxy,
    )
    discovery = analyze_run(out)
    snapshot = search(discovery)
    (out / "hub-search.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    discovery["hub_search"] = snapshot
    phases = report["phases"]
    print(
        f"Proxy median · 32-token prefill {phases['prefill']['median_ms']:.2f} ms · cached decode {phases['decode']['median_ms']:.2f} ms"
    )
    print(f"Results in {out}")
    show_discovery(discovery, out)
    if args.check_kernels:
        trial = check_picked_layers(model, snapshot)
        (out / "kernel-check.json").write_text(json.dumps(trial, indent=2) + "\n")
        print(f"Proxy kernel check: {trial['status']}")


if __name__ == "__main__":
    main()
