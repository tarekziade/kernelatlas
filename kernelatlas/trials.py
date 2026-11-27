"""Persistent local trials for a trace-backed optimization opportunity."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from statistics import median
import subprocess
import sys
from tempfile import TemporaryDirectory

from .discovery import GATE, analyze_run, gate_sites

REFERENCE = '''"""Replace the body with your self-contained inference implementation."""
import torch


def candidate(gate, up):
    return torch.nn.functional.silu(gate) * up
'''
POLICY = {
    "version": 1,
    "objective": "median paired saving in prompt plus one cached-token model latency",
    "correctness": "exact gate outputs and last-position logits on tested prompts and cached steps",
    "warmup_blocks": 6,
    "measured_blocks": 24,
    "minimum_fraction": 0.01,
    "required_wins": 18,
    "noise_floor": "median absolute paired unchanged-versus-incumbent difference",
    "phase_guard": "reject a repeatable regression beyond the floor in either phase",
}


def classify(rows):
    """Require a paired effect larger than both practical tolerance and A/A noise."""
    by_phase = {}
    for phase in ("prefill", "decode", "total"):
        base = [r["incumbent"][phase] for r in rows]
        noise = median(abs(r["incumbent"][phase] - r["control"][phase]) for r in rows)
        floor = max(noise, POLICY["minimum_fraction"] * median(base))
        deltas = [r["incumbent"][phase] - r["candidate"][phase] for r in rows]
        by_phase[phase] = {
            "incumbent_ms": median(base),
            "candidate_ms": median(r["candidate"][phase] for r in rows),
            "noise_ms": noise,
            "floor_ms": floor,
            "median_paired_saving_ms": median(deltas),
            "wins_above_floor": sum(x > floor for x in deltas),
            "losses_above_floor": sum(x < -floor for x in deltas),
        }
    threshold = POLICY["required_wins"]
    if any(p["losses_above_floor"] >= threshold for p in by_phase.values()):
        verdict = "slower"
    elif by_phase["total"]["wins_above_floor"] >= threshold:
        verdict = "faster"
    else:
        verdict = "inconclusive"
    return verdict, by_phase


def _read(path):
    return json.loads(path.read_text())


def _write(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prepare(folder, root):
    report = _read(folder / "report.json")
    versions = {"report": _digest(folder / "report.json")}
    if report.get("prefill_dataflow"):
        versions["dataflow"] = _digest(folder / report["prefill_dataflow"])
    policy = POLICY
    if "workload" in report:
        policy = dict(
            POLICY,
            version=2,
            objective="median paired saving over the complete fixed-length greedy trajectory",
            timing="host clock synchronized after every selected token",
        )
    manifest = root / "task.json"
    if manifest.exists():
        task = _read(manifest)
        if task["profile_sha256"] != versions or task["policy"] != policy:
            raise ValueError(
                "Saved profile or trial policy changed. Use a fresh profile directory."
            )
        return task
    sites = gate_sites(report, analyze_run(folder))
    if not sites:
        raise ValueError(
            "Trial v1 requires a pretrained dense BF16 MPS Llama or Qwen3 profile with an observed SiLU/multiply pair."
        )
    from .gate_workload import prepare

    with TemporaryDirectory(dir=root) as scratch:
        staging = Path(scratch)
        task = prepare(report, sites, staging)
        task.update(
            opportunity=GATE,
            policy=policy,
            profile_sha256=versions,
            template="reference.py",
        )
        (staging / "reference.py").write_text(REFERENCE)
        for file in staging.iterdir():
            file.replace(root / file.name)
        _write(manifest, task)
    return task


def _history(attempts, task_digest):
    history = []
    for path in sorted(attempts.iterdir()):
        if not path.is_dir() or not path.name.isdecimal():
            continue
        evidence = path / "result.json"
        if not evidence.exists():
            # The lock outlives healthy workers. A previous coordinator was interrupted.
            req = _read(path / "request.json")
            _write(
                evidence,
                {
                    **req,
                    "verdict": "error",
                    "reason": "Trial interrupted before completion",
                    "promoted": False,
                },
            )
        record = _read(evidence)
        if record.get("task_sha256", task_digest) != task_digest:
            raise ValueError(
                "Trial workload differs from saved results. Use a fresh profile directory."
            )
        history.append(record)
    return history


def _status(root, task, history, winner):
    best = winner["id"] if winner else "eager"
    configuration = (
        {"implementation": "original model"}
        if winner is None
        else {
            "implementation": winner["candidate"],
            "sha256": winner["sha256"],
            "sites": task["sites"],
            "trace": f"trials/{best}/winner.dataflow.json",
            "remaining_pairs": winner["remaining_pairs"],
        }
    )
    return {
        "task": task,
        "best": best,
        "configuration": configuration,
        "directory": str(root),
        "history": [
            {
                k: r[k]
                for k in (
                    "id",
                    "sha256",
                    "incumbent",
                    "verdict",
                    "promoted",
                    "evidence",
                    "candidate",
                    "reason",
                    "failure",
                    "workload",
                    "per_cached_token",
                    "timing",
                )
                if k in r
            }
            | ({"total_timing": r["timing"]["total"]} if "timing" in r else {})
            for r in history
        ],
        "next_action": "Submit a self-contained candidate(gate, up) for the observed sites.",
    }


def run_trial(folder, candidate=None, *, replay=None, repeat=False):
    """Prepare, inspect or evaluate a local run. Paths in results are relative to optimization/."""
    import fcntl

    if (replay is not None and (candidate is not None or repeat)) or (
        repeat and candidate is None
    ):
        raise ValueError("Use a candidate with optional repeat, or replay one trial ID")
    folder = Path(folder).resolve()
    if not (folder / "report.json").is_file():
        raise ValueError(f"Missing profile report under {folder}")
    root = folder / "optimization"
    root.mkdir(exist_ok=True)
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        task = _prepare(folder, root)
        task_digest = _digest(root / "task.json")
        attempts = root / "trials"
        attempts.mkdir(exist_ok=True)
        history = _history(attempts, task_digest)
        winner = next((r for r in reversed(history) if r["promoted"]), None)
        if candidate is None and replay is None:
            return _status(root, task, history, winner)
        best = winner["id"] if winner else "eager"
        incumbent = best
        if replay is not None:
            record = next((r for r in history if r["id"] == replay), None)
            if record is None:
                raise ValueError(f"Unknown trial {replay}")
            incumbent = record["incumbent"]
            candidate = attempts / replay / "candidate.py"
            if _digest(candidate) != record["sha256"]:
                raise ValueError("Archived candidate changed since the original trial.")
        source = Path(candidate).read_bytes()
        fingerprint = hashlib.sha256(source).hexdigest()
        if not repeat and replay is None:
            for r in reversed(history):
                if (
                    r["sha256"] == fingerprint
                    and (r["incumbent"] == incumbent or r["id"] == incumbent)
                    and r["verdict"] != "error"
                ):
                    return {**r, "cached": True, "best": best}
        ident = f"{len(history) + 1:06d}"
        trial = attempts / ident
        req = {
            "id": ident,
            "task_sha256": task_digest,
            "opportunity": GATE,
            "sha256": fingerprint,
            "incumbent": incumbent,
            "replay_of": replay,
            "evidence": f"trials/{ident}/result.json",
            "candidate": f"trials/{ident}/candidate.py",
            "replay_command": ["kernelatlas", "trial", str(folder), "--replay", ident],
        }
        with TemporaryDirectory(dir=root) as scratch:
            staging = Path(scratch) / ident
            staging.mkdir()
            (staging / "candidate.py").write_bytes(source)
            _write(staging / "request.json", req)
            staging.rename(trial)
        with (trial / "worker.log").open("w") as log:
            proc = subprocess.run(
                [sys.executable, "-m", "kernelatlas.gate_workload", str(root), ident],
                stdout=log,
                stderr=subprocess.STDOUT,
                pass_fds=(lock.fileno(),),
            )
        if proc.returncode == 0 and (trial / "measurement.json").is_file():
            measured = _read(trial / "measurement.json")
        else:
            measured = {
                "verdict": "error",
                "reason": f"Worker exited with status {proc.returncode}. See trials/{ident}/worker.log.",
            }
        if (
            winner
            and incumbent == best
            and winner["sha256"] == fingerprint
            and measured["verdict"] in ("faster", "slower")
        ):
            measured.update(
                verdict="inconclusive",
                reason="Identical implementation to the current best; observed difference is run variability.",
            )
        promoted = measured["verdict"] == "faster" and replay is None
        result = {
            **req,
            **measured,
            "promoted": promoted,
            "best": ident if promoted else best,
        }
        _write(trial / "result.json", result)
        return result


@contextmanager
def best_model(folder):
    """Load the pinned eager model and apply the best validated adapter for this run."""
    from .gate_workload import installed, load_candidate, load_model

    status = run_trial(folder)
    model, modules = load_model(status["task"])
    fn = None
    if status["best"] != "eager":
        config = status["configuration"]
        fn = load_candidate(
            Path(status["directory"]) / config["implementation"], config["sha256"]
        )
    with installed(modules, fn):
        yield model
