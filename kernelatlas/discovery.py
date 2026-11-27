"""Mine repeated, directly connected eager tensor operations offline."""

from collections import Counter
import json
from pathlib import Path

from .hub import summarize_search


GATE = "silu_multiply"


def gate_sites(report, discovery):
    precision = report.get("precision", {}).get("weights", {})
    if not (
        report.get("weight_source") == "pretrained"
        and report.get("revision")
        and report.get("device", "").startswith("mps")
        and report.get("dtype") == "torch.bfloat16"
        and precision.get("mode") == "none"
        and report.get("config", {}).get("model_type") in ("llama", "qwen3")
        and report["config"].get("hidden_act") == "silu"
    ):
        return []
    matches = [
        c
        for c in discovery["candidates"]
        if c["operators"] == ["aten.silu.default", "aten.mul.Tensor"]
    ]
    if len(matches) != 1:
        return []
    return sorted({o["consumer_scope"] for o in matches[0]["observations"]})


def role(node):
    if not node["pure_tensor_result"]:
        return None
    if node["pointwise"]:
        return "pointwise"
    if node["op"] in ("aten.mm.default", "aten.addmm.default", "aten.linear.default"):
        return "matrix_product"
    if node["op"] in ("aten.native_layer_norm.default", "aten.mean.dim"):
        return "reduction"
    return None


def find_candidates(graph):
    if graph["schema_version"] != 1:
        raise ValueError("Unknown dataflow trace version")
    nodes = graph["nodes"]
    uses = Counter(tuple(edge) for node in nodes for edge in node["inputs"])
    uses.update(tuple(edge) for edge in graph["returned"])
    groups = {}
    for end, consumer in enumerate(nodes):
        target = role(consumer)
        if target not in ("pointwise", "reduction"):
            continue
        for start, index in consumer["inputs"]:
            if uses[start, index] != 1:
                continue
            producer = nodes[start]
            source = role(producer)
            if source not in ("pointwise", "matrix_product") or (
                source == "matrix_product" and target != "pointwise"
            ):
                continue
            tensor = producer["outputs"][index]
            if (
                not tensor["floating"]
                or not tensor["logical_bytes"]
                or tensor["stride"] is None
            ):
                continue
            # Require a device-local same-precision primary output. Other signatures
            # can be added once their numerical and layout contracts are understood.
            if not consumer["outputs"] or any(
                consumer["outputs"][0][k] != tensor[k] for k in ("device", "dtype")
            ):
                continue
            key = (
                producer["op"],
                consumer["op"],
                tuple(tensor["shape"]),
                tuple(tensor["stride"]),
                tensor["dtype"],
                tensor["device"],
            )
            if key not in groups:
                groups[key] = {
                    "operators": list(key[:2]),
                    "intermediate": tensor,
                    "observations": [],
                    "status": "hypothesis",
                    "backend_fusion": "unknown",
                    "kernel_compatibility": "unverified",
                    "measured_savings_ms": None,
                }
            groups[key]["observations"].append(
                {
                    "producer": start,
                    "consumer": end,
                    "output": index,
                    "producer_scope": producer["scope"],
                    "consumer_scope": consumer["scope"],
                }
            )
    repeated = []
    for candidate in groups.values():
        count = len(candidate["observations"])
        if count >= 2:
            candidate["occurrences"] = count
            candidate["total_logical_intermediate_bytes"] = (
                count * candidate["intermediate"]["logical_bytes"]
            )
            repeated.append(candidate)
    return sorted(
        repeated,
        key=lambda c: (
            -c["total_logical_intermediate_bytes"],
            -c["occurrences"],
            c["operators"],
        ),
    )


def cache_results(folder, report_file, graph_file):
    """Read only trials for these exact captures, including repeat measurements."""
    from .generation import decode_step_savings
    from .trials import _digest

    files = sorted((folder / "optimization" / "cache-trials").glob("*.json"))
    if not files:
        return []
    fingerprints = (_digest(report_file), _digest(graph_file))
    history = []
    for path in files:
        record = json.loads(path.read_text())
        if (record.get("report_sha256"), record.get("graph_sha256")) != fingerprints:
            continue
        entry = {
            k: record[k]
            for k in (
                "completed_at",
                "verdict",
                "evidence",
                "timing",
                "failure",
                "reason",
                "paired_decode_savings",
                "policy",
            )
            if k in record
        }
        entry["reserve_at"] = record.get("reserve_at", "prefill")
        if "paired_decode_savings" not in entry and record.get("blocks_ms"):
            entry["paired_decode_savings"] = decode_step_savings(record["blocks_ms"])
        history.append(entry)
    return sorted(history, key=lambda entry: entry["completed_at"])


def analyze_run(folder):
    folder = Path(folder)
    report = json.loads((folder / "report.json").read_text())
    trace = report.get("prefill_dataflow")
    if not trace or not (folder / trace).is_file():
        if "compilation" in report:
            return {"compilation": report["compilation"], "trace_status": "unavailable"}
        raise ValueError(
            "This report lacks tensor dependency edges. Run kernelatlas profile MODEL_ID again to capture them."
        )
    graph = json.loads((folder / trace).read_text())
    candidates = find_candidates(graph)
    result = {
        "schema_version": 1,
        "phase": "prefill",
        "weight_source": report["weight_source"],
        "config_source": report["config_source"],
        "torch": report["torch"],
        "device": report["device"],
        "evidence": trace,
        "ranking": "total logical bytes of observed single-use intermediates",
        "limitations": "Eager ATen edges only. Alias escape, backend fusion, legality, actual memory traffic and performance benefit require validation.",
        "candidates": candidates,
        "transformers_integrations": graph.get("integrations"),
    }
    if report.get("decode_dataflow"):
        decode = json.loads((folder / report["decode_dataflow"]).read_text())
        appends = decode.get("retained_cache_appends", [])
        result["decode"] = {
            "phase": "first_cached_forward",
            "evidence": report["decode_dataflow"],
            "candidates": find_candidates(decode),
            "retained_cache_appends": appends,
            "cache_append_output_bytes": sum(
                decode["nodes"][a["node"]]["outputs"][0]["logical_bytes"]
                for a in appends
            ),
            "limitations": "One cached forward. Logical output sizes do not measure GPU traffic or time. Cache append detection requires exposed keys and values.",
        }
        if (
            appends
            and report.get("workload", {}).get("mode") == "fixed_length_greedy"
            and report.get("revision")
            and report.get("weight_source") == "pretrained"
            and report["device"].split(":")[0] == "mps"
            and report.get("dtype") == "torch.bfloat16"
            and report.get("precision", {}).get("weights", {}).get("mode") == "none"
        ):
            result["decode"]["cache_trial"] = [
                "kernelatlas",
                "trial",
                str(folder),
                "--static-cache",
            ]
            config = report.get("config", {})
            layers = config.get("num_hidden_layers", 0)
            observed = {(a["layer"], a["slot"]) for a in appends}
            full_slots = {
                (i, slot) for i in range(layers) for slot in ("keys", "values")
            }
            if (
                observed == full_slots
                and not config.get("sliding_window")
                and all(
                    kind == "full_attention" for kind in config.get("layer_types") or []
                )
            ):
                result["decode"]["decode_cache_trial"] = [
                    "kernelatlas",
                    "trial",
                    str(folder),
                    "--decode-static-cache",
                ]
            result["decode"]["cache_results"] = cache_results(
                folder, folder / "report.json", folder / report["decode_dataflow"]
            )
    elif "decode" in report.get("phases", {}):
        result["decode_status"] = "dependency_trace_unavailable"
    if "compilation" in report:
        result["compilation"] = report["compilation"]
    if "precision" in report:
        result["precision"] = report["precision"]
    if "proxy" in report:
        result["proxy"] = report["proxy"]
    if gate_sites(report, result):
        result["trial"] = {
            "opportunity": GATE,
            "command": ["kernelatlas", "trial", str(folder)],
        }
    snapshot = folder / "hub-search.json"
    if snapshot.is_file():
        result["hub_search"] = json.loads(snapshot.read_text())
    options = result.get("hub_search", {}).get("loading_options")
    if options:
        from .trials import _digest

        fingerprint = _digest(folder / "report.json")
        saved = [
            json.loads(p.read_text())
            for p in (folder / "hub-trials").glob("*/summary.json")
        ]
        matches = [r for r in saved if r.get("report_sha256") == fingerprint]
        if matches:
            options["status"] = "compared"
            options["comparison"] = max(matches, key=lambda r: r["completed_at"])
    result["development_tasks"] = build_ideas(folder, result)
    if "hub_search" in result:
        result["hub_summary"] = summarize_search(
            result["hub_search"], result["development_tasks"]
        )
    return result


def build_ideas(folder, result):
    """Present distinct observed operations as kernel development hypotheses."""
    target = {"mps": "Metal", "cuda": "CUDA"}.get(
        result["device"].split(":")[0], result["device"]
    )
    ideas = []
    seen = set()
    for c in result["candidates"]:
        ops = tuple(c["operators"])
        if ops in seen:
            continue
        seen.add(ops)
        names = [op.split(".")[1] for op in ops]
        gate = names == ["silu", "mul"]
        if len(ideas) >= 3 and not gate:
            continue
        labels = {
            ("silu", "mul"): "fused SiLU and multiply (MLP gate)",
            ("pow", "mean"): "fused power and mean reduction",
            ("linear", "mul"): "linear projection with a multiply epilogue",
            ("linear", "silu"): "linear projection with a SiLU epilogue",
        }
        label = labels.get(tuple(names), f"fused {' and '.join(names)}")
        obs = c["observations"][0]
        task = {
            "title": f"{target} {label} kernel",
            "status": "hypothesis",
            "phase": "prefill",
            "operators": list(ops),
            "shape": c["intermediate"]["shape"],
            "dtype": c["intermediate"]["dtype"],
            "occurrences": c["occurrences"],
            "logical_bytes": c["total_logical_intermediate_bytes"],
            "scope": obs["consumer_scope"],
            "nodes": [obs["producer"], obs["consumer"]],
            "evidence": result["evidence"],
            "measured_speedup": None,
            "next_step": "Confirm the exact arguments and backend fusion, then compare a prototype with the model baseline.",
        }
        if gate and result.get("trial"):
            task["command"] = result["trial"]["command"]
            task["prior_trials"] = [
                {
                    "verdict": r["verdict"],
                    "candidate": r.get("candidate"),
                    "evidence": str(p.relative_to(folder)),
                }
                for p in sorted(
                    (folder / "optimization" / "trials").glob("*/result.json")
                )
                for r in [json.loads(p.read_text())]
            ]
        ideas.append(task)
    return sorted(ideas, key=lambda t: "command" not in t)


def latest_profile(
    root, model_id, *, family=None, torch_version=None, prompt_length=None
):
    root = Path(root)
    paths = (
        [root / "report.json"]
        if (root / "report.json").is_file()
        else [*root.glob("*/report.json"), *root.glob("*/*/report.json")]
    )
    matches = []
    for path in paths:
        try:
            r = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if "compilation" not in r and (
            not r.get("prefill_dataflow")
            or not (path.parent / r["prefill_dataflow"]).is_file()
        ):
            continue
        if (
            model_id
            and (r.get("config_source") or {}).get("model_id", r.get("model_id"))
            != model_id
        ):
            continue
        if (
            family
            and r["config"]["model_type"] != family
            or torch_version
            and r["torch"] != torch_version
        ):
            continue
        if prompt_length and r["phases"]["prefill"]["cache_length"] != prompt_length:
            continue
        matches.append((r["timestamp"], path.parent))
    return max(matches)[1] if matches else None
