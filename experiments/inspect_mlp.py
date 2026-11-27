"""Check whether the saved MLP events support a SiLU/multiply fusion hypothesis."""

import argparse
import inspect
import json
from pathlib import Path

import torch
import transformers
from torch.fx.experimental.proxy_tensor import make_fx
from transformers import LlamaConfig, Qwen3Config
from transformers.models.llama.modeling_llama import LlamaMLP
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP


def inspect_run(folder):
    report = json.loads((folder / "report.json").read_text())
    if (torch.__version__, transformers.__version__) != (
        report["torch"],
        report["transformers"],
    ):
        raise ValueError(
            "Replay the same model source and PyTorch version as the trace"
        )
    if report["device"] != "mps:0" or report["dtype"] != "torch.float32":
        raise ValueError("This proposal evaluates the FP32 MPS workload")
    config_type, mlp_type = {
        "llama": (LlamaConfig, LlamaMLP),
        "qwen3": (Qwen3Config, Qwen3MLP),
    }[report["config"]["model_type"]]
    config = config_type.from_dict(report["config"])
    events = json.loads((folder / "prefill.events.json").read_text())
    by_id = {event["id"]: event for event in events}
    sites = {}
    for event in events:
        if event["name"] not in ("aten::silu", "aten::mul"):
            continue
        parent = event
        while parent["parent"] is not None:
            parent = by_id[parent["parent"]]
            if parent["name"].startswith("module/") and parent["name"].endswith(".mlp"):
                sites.setdefault(parent["name"], {}).setdefault(
                    event["name"], []
                ).append(event)
                break
    length = (
        len(report["workload"]["prompt_ids"])
        if "workload" in report
        else report["phases"]["prefill"]["output_shape"][1]
    )
    shape = [1, length, config.intermediate_size]
    matched = {}
    for name, ops in sites.items():
        silu, mul = ops.get("aten::silu", []), ops.get("aten::mul", [])
        if (
            len(silu) == len(mul) == 1
            and silu[0]["input_shapes"] == [shape]
            and mul[0]["input_shapes"] == [shape, shape]
        ):
            matched[name] = [silu[0]["id"], mul[0]["id"]]
    # Capture tensor dependencies separately; trace adjacency alone cannot prove them.
    with torch.device("meta"):
        mlp = mlp_type(config).eval()
        graph = make_fx(mlp)(torch.empty(1, length, config.hidden_size))
    nodes = [n for n in graph.graph.nodes if n.target == torch.ops.aten.silu.default]
    if (
        not matched
        or len(nodes) != 1
        or len(nodes[0].users) != 1
        or next(iter(nodes[0].users)).target != torch.ops.aten.mul.Tensor
    ):
        raise ValueError(
            "Expected SiLU with one multiply consumer and matching runtime scopes"
        )
    return {
        "pattern": "silu_mul",
        "shape": shape,
        "torch": report["torch"],
        "prompt_length": length,
        "device": report["device"],
        "dtype": report["dtype"],
        "sites": matched,
        "runtime_evidence": "prefill.events.json",
        "graph_method": "make_fx on the configured MLP with meta tensors",
        "forward_source": inspect.getsource(mlp_type.forward),
        "graph": graph.code,
        "status": "hypothesis",
        "dispatch_fusion": "unknown",
        "registry_compatibility": "unknown",
        "measured_savings_ms": None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    folder = parser.parse_args().run
    print(json.dumps(inspect_run(folder), indent=2))
