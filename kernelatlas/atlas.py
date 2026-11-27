"""Aggregate measured RMSNorm evaluations from local decoder experiment folders."""

import json
from pathlib import Path
from statistics import median


def evaluation_row(directory, run, evaluation):
    result = json.loads(evaluation.read_text())
    times = result["prefill_ms"] or {"baseline": [], "candidate": []}
    deltas = [a - b for a, b in zip(times["baseline"], times["candidate"], strict=True)]
    return {
        "family": run["config"]["model_type"],
        "torch": run["torch"],
        "transformers": run["transformers"],
        "model_id": run.get(
            "model_id", run["weights"] if run.get("revision") else None
        ),
        "model_revision": run.get("revision"),
        "prompt_length": result.get(
            "prompt_length", run["phases"]["prefill"]["output_shape"][1]
        ),
        "median_paired_decode_savings_ms": None,
        "run": directory.name,
        "timestamp": run["timestamp"],
        "pattern": result["pattern"],
        "candidate": result["implementation"],
        "device": run["device"],
        "dtype": run["dtype"],
        "weights": run["weights"],
        "occurrences": list(result["occurrences"]),
        "max_abs_logits_error": max(result["max_abs_logits_error_by_step"]),
        "correctness": result.get("correctness", "passed_tested_inputs"),
        "median_paired_prefill_savings_ms": median(deltas) if deltas else None,
        "positive_pairs": sum(delta > 0 for delta in deltas),
        "pairs": len(deltas),
        "report": f"{directory.name}/report.json",
        "evaluation": f"{directory.name}/{evaluation.name}",
    }


def integration_rows(directory, run, path):
    trial = json.loads(path.read_text())
    op = trial["kernel_op"]
    for label in ("hub", "metal_tail"):
        evaluation = trial[label]
        hub = label == "hub"
        check = evaluation["correctness"]
        timing = evaluation["timing"] or {}
        decode = timing.get("decode_one_token") or {}
        sites = trial["hub_events"][op] if hub else trial["eager_events"]["aten::rsqrt"]
        status = "passed_tested_inputs" if check["passed"] else "rejected"
        if hub and not evaluation["all_sites_dispatched"]:
            status = "dispatch_incomplete"
        yield {
            "family": run["config"]["model_type"],
            "run": directory.name,
            "timestamp": trial["timestamp"],
            "torch": trial["torch"],
            "transformers": trial["transformers"],
            "model_id": trial["weights"],
            "model_revision": trial["revision"],
            "prompt_length": len(trial["input_token_ids"]),
            "pattern": "weighted_last_dim_rmsnorm_fp32",
            "candidate": trial["kernel_id"] if hub else "rmsnorm_tail.metal",
            "kernel_revision": trial["kernel_revision"] if hub else None,
            "device": trial["device"],
            "dtype": "torch." + trial["dtype"],
            "hardware": trial["hardware"],
            "weights": trial["weights"],
            "occurrences": list(sites),
            "max_abs_logits_error": max(check["max_abs_error"]),
            "correctness": status,
            "integration": "confirmed"
            if hub and evaluation["all_sites_dispatched"]
            else ("incomplete" if hub else "experimental"),
            "median_paired_prefill_savings_ms": timing.get("median_paired_saving_ms"),
            "median_paired_decode_savings_ms": decode.get("median_paired_saving_ms"),
            "positive_pairs": timing.get("faster_pairs", 0),
            "pairs": len(timing.get("synchronized_prefill_ms", {}).get("baseline", [])),
            "evaluation": f"{directory.name}/{path.name}",
            "report": f"{directory.name}/{path.name}",
        }


def build(folders):
    rows = []
    opportunities = []
    for directory in sorted({path.resolve() for path in folders}):
        run = json.loads((directory / "report.json").read_text())
        first_row, first_proposal = len(rows), len(opportunities)
        proposal = directory / "mlp-fusion.json"
        if proposal.is_file():
            data = json.loads(proposal.read_text())
            opportunities.append(
                {
                    "pattern": data["pattern"],
                    "run": directory.name,
                    "family": run["config"]["model_type"],
                    "model_id": run["weights"] if run.get("revision") else None,
                    "torch": data["torch"],
                    "dtype": data["dtype"],
                    "device": data["device"],
                    "prompt_length": data["prompt_length"],
                    "shape": data["shape"],
                    "sites": len(data["sites"]),
                    "status": data["status"],
                    "evidence": f"{directory.name}/{proposal.name}",
                }
            )
        native = directory / "native.json"
        if native.is_file():
            rows.append(evaluation_row(directory, run, native))
        metal = directory / "metal-tail.json"
        if metal.is_file():
            row = evaluation_row(directory, run, metal)
            repeats = directory / "metal-tail-repeats.json"
            if repeats.is_file():
                summary = json.loads(repeats.read_text())
                row["repeated_prefill"] = {
                    "processes": len(summary["trials"]),
                    "median_of_process_medians_ms": summary[
                        "median_of_process_medians_ms"
                    ],
                    "positive_process_medians": summary["positive_process_medians"],
                    "range_of_process_medians_ms": summary[
                        "range_of_process_medians_ms"
                    ],
                    "evidence": f"{directory.name}/{repeats.name}",
                }
            rows.append(row)
        for path in sorted(directory.glob("hub-integration-*.json")):
            rows.extend(integration_rows(directory, run, path))
        if "precision" in run:
            for item in rows[first_row:] + opportunities[first_proposal:]:
                item["weight_mode"] = run["precision"]["weights"]["mode"]
    return {
        "architectures": sorted({row["family"] for row in rows}),
        "evaluations": rows,
        "opportunities": opportunities,
    }


class Atlas:
    """Index saved KernelAtlas evaluations without importing a GPU runtime."""

    def __init__(self, data):
        self.data = data

    @classmethod
    def from_reports(cls, root):
        root = Path(root)
        if (root / "report.json").is_file():
            return cls(build([root]))
        reports = sorted(root.glob("*/report.json"))
        if not reports:
            raise ValueError(f"No run reports found under {root}")
        return cls(build([p.parent for p in reports]))

    def evaluations(self, **filters):
        return self._select("evaluations", **filters)

    def opportunities(self, **filters):
        return self._select("opportunities", **filters)

    def _select(
        self,
        collection,
        *,
        model_id=None,
        family=None,
        torch_version=None,
        prompt_length=None,
        dtype=None,
        weight_mode=None,
    ):
        filters = {
            "model_id": model_id,
            "family": family,
            "torch": torch_version,
            "prompt_length": prompt_length,
            "dtype": dtype,
        }
        return [
            row
            for row in self.data.get(collection, [])
            if (weight_mode is None or row.get("weight_mode", "none") == weight_mode)
            and all(
                value is None or row.get(key) == value for key, value in filters.items()
            )
        ]
