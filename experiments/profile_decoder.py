"""Reproduce the pinned offline decoder experiments, including tiny random weights."""

import argparse
from pathlib import Path
import sys

import torch
from transformers import AutoModelForCausalLM, LlamaConfig, Qwen3Config

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernelatlas.profiling import add_module_scopes, profile_model  # noqa: E402, F401


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=("llama", "qwen3"))
    parser.add_argument(
        "--model-id", help="Load pinned weights from the local Hub cache"
    )
    parser.add_argument("--revision", help="Full Hub commit for --model-id")
    parser.add_argument("--device", choices=("mps", "cuda"), default="mps")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if bool(args.model_id) != bool(args.revision):
        parser.error("--model-id and --revision are required together")
    if args.revision and (
        len(args.revision) != 40
        or any(c not in "0123456789abcdef" for c in args.revision)
    ):
        parser.error("--revision must be a full commit hash")
    torch.manual_seed(0)
    if args.model_id:
        model = (
            AutoModelForCausalLM.from_pretrained(
                args.model_id,
                revision=args.revision,
                local_files_only=True,
                dtype=torch.float32,
                attn_implementation="sdpa",
            )
            .eval()
            .to(args.device)
        )
        config = model.config
        if config.model_type != args.family:
            raise ValueError("Requested family differs from checkpoint config")
    else:
        config_class = LlamaConfig if args.family == "llama" else Qwen3Config
        config = config_class(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
        )
        model = (
            AutoModelForCausalLM.from_config(config, attn_implementation="sdpa")
            .eval()
            .to(args.device)
        )
    profile_model(
        model, args.device, args.out, model_id=args.model_id, revision=args.revision
    )


if __name__ == "__main__":
    main()
