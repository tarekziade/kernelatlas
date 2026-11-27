"""Summarize independent process repeats of one evaluated substitution."""

import argparse
import json
from pathlib import Path
from statistics import median


def summarize(files):
    trials = []
    files = [path.resolve() for path in files]
    if len(files) != len(set(files)):
        raise ValueError("Repeated paths would count the same trial twice")
    contract = None
    for file in files:
        data = json.loads(file.read_text())
        signature = (
            data["pattern"],
            data["implementation"],
            data["occurrences"],
            data["logits_tolerance"],
        )
        if contract is not None and signature != contract:
            raise ValueError("Trials refer to different candidates or sites")
        contract = signature
        times = data["prefill_ms"]
        savings = []
        if times:
            savings = [
                b - c
                for b, c in zip(times["baseline"], times["candidate"], strict=True)
            ]
        trials.append(
            {
                "file": file.name,
                "correctness": data["correctness"],
                "max_abs_logits_error": max(data["max_abs_logits_error_by_step"]),
                "samples_ms": times,
                "faster_pairs": sum(x > 0 for x in savings),
                "median_paired_savings_ms": median(savings) if savings else None,
            }
        )
    medians = [r["median_paired_savings_ms"] for r in trials if r["samples_ms"]]
    return {
        "implementation": contract[1],
        "trials": trials,
        "median_of_process_medians_ms": median(medians) if medians else None,
        "range_of_process_medians_ms": [min(medians), max(medians)]
        if medians
        else None,
        "positive_process_medians": sum(x > 0 for x in medians),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+")
    print(json.dumps(summarize(parser.parse_args().results), indent=2))
