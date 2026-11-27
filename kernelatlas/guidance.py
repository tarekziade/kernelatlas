"""Derive scoped, serializable decisions from recorded evidence."""

from collections import defaultdict


def existing_kernels(trials):
    groups = defaultdict(list)
    for row in trials:
        groups[row["candidate"], row["correctness"]].append(row)
    decisions = []
    for (candidate, status), rows in groups.items():
        passed = status == "passed_tested_inputs"
        action = "keep_eager"
        reason = {
            "rejected": "Failed the model correctness check.",
            "dispatch_incomplete": "Kernel dispatch was incomplete.",
        }.get(status, status)
        code = None
        if passed:
            action, reason = "evaluated", "Passed correctness on the tested inputs."
            if any((r["median_paired_prefill_savings_ms"] or 0) > 0 for r in rows):
                if candidate == "rmsnorm_tail.metal":
                    action, reason = (
                        "try_prefill",
                        "Use an eager FP32 MPS model in eval mode. Leave cached decode eager.",
                    )
                    code = "import kernelatlas.metal as ka\n\nwith ka.metal_rmsnorm_prefill(model):\n    out = model(input_ids)"
                elif candidate == "torch.nn.functional.rms_norm":
                    action, reason = (
                        "try_prefill",
                        "Use at verified FP32 normalization sites.",
                    )
                    code = "import torch.nn.functional as F\n\nout = F.rms_norm(\n    x, norm.weight.shape,\n    norm.weight, norm.variance_epsilon,\n)"
        measurements = [
            {
                key: r[key]
                for key in (
                    "torch",
                    "transformers",
                    "prompt_length",
                    "median_paired_prefill_savings_ms",
                    "median_paired_decode_savings_ms",
                    "evaluation",
                )
            }
            for r in rows
        ]
        for measurement, row in zip(measurements, rows):
            if "repeated_prefill" in row:
                measurement["repeated_prefill"] = row["repeated_prefill"]
        decisions.append(
            dict(
                candidate=candidate,
                correctness=status,
                action=action,
                reason=reason,
                python=code,
                measurements=measurements,
            )
        )
    return sorted(
        decisions, key=lambda d: (d["action"] == "keep_eager", d["candidate"])
    )


def development_tasks(trials, ideas):
    tasks = []
    for idea in ideas:
        tasks.append(
            {
                **idea,
                "title": "Fuse SiLU and multiply",
                "reason": f"{idea['sites']} MLP blocks. The multiply consumes the SiLU output.",
                "next_step": "Check existing kernels and backend fusion, then benchmark eliminating the intermediate.",
                "measured_savings_ms": None,
            }
        )
    passing = [
        r
        for r in trials
        if r["candidate"] == "rmsnorm_tail.metal"
        and r["correctness"] == "passed_tested_inputs"
    ]
    rejected = [
        r
        for r in trials
        if r["pattern"] == "weighted_last_dim_rmsnorm_fp32"
        and r["correctness"] == "rejected"
    ]
    evidence = {
        r["evaluation"]
        for a in passing
        for b in rejected
        if (a["torch"], a["transformers"], a["prompt_length"])
        == (b["torch"], b["transformers"], b["prompt_length"])
        for r in (a, b)
    }
    if evidence:
        tasks.append(
            {
                "pattern": "weighted_last_dim_rmsnorm_fp32",
                "title": "Prototype full Metal RMSNorm",
                "status": "hypothesis",
                "reason": "The tail passed. A full replacement failed the logits check on the same workload.",
                "next_step": "Preserve eager FP32 rounding through the reduction. Check logits before timing.",
                "evidence_files": sorted(evidence),
                "measured_savings_ms": None,
            }
        )
    return tasks


def guidance(rows, model_id, reports, opportunities=(), details=False):
    trials = defaultdict(list)
    ideas = defaultdict(list)
    for row in rows:
        trials[row["run"]].append(row)
    for idea in opportunities:
        ideas[idea["run"]].append(idea)
    runs = []
    for name in sorted(trials.keys() | ideas.keys()):
        scope = (trials[name] or ideas[name])[0]
        runs.append(
            {
                "run": name,
                "model_id": scope["model_id"],
                "device": scope["device"],
                "dtype": scope["dtype"],
                "existing_kernels": existing_kernels(trials[name]),
                "development_tasks": development_tasks(trials[name], ideas[name]),
            }
        )
    result = {
        "schema_version": 1,
        "model_id": model_id,
        "reports": str(reports),
        "runs": runs,
    }
    if details:
        result["evaluations"] = rows
    return result
