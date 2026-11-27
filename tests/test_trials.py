import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from kernelatlas.trials import POLICY, classify, gate_sites, run_trial


class DecisionTests(unittest.TestCase):
    def blocks(self, candidate, control=(10, 5)):
        def sample(pair):
            return {"prefill": pair[0], "decode": pair[1], "total": sum(pair)}

        return [
            {
                "incumbent": sample((10, 5)),
                "control": sample(control),
                "candidate": sample(candidate),
            }
            for _ in range(POLICY["measured_blocks"])
        ]

    def test_noise_margin_and_phase_guard(self):
        self.assertEqual(classify(self.blocks((9, 5)))[0], "faster")
        self.assertEqual(classify(self.blocks((10, 5)))[0], "inconclusive")
        self.assertEqual(classify(self.blocks((9.8, 5), (10.5, 5)))[0], "inconclusive")
        self.assertEqual(classify(self.blocks((8, 5.5)))[0], "slower")
        # Only seventeen clearly faster blocks cannot promote a candidate.
        self.assertEqual(
            classify(self.blocks((9, 5))[:17] + self.blocks((10, 5))[17:])[0],
            "inconclusive",
        )

    def test_gate_requires_trace_and_supported_workload(self):
        report = {
            "weight_source": "pretrained",
            "revision": "a" * 40,
            "dtype": "torch.bfloat16",
            "device": "mps:0",
            "precision": {"weights": {"mode": "none"}},
            "config": {"model_type": "qwen3", "hidden_act": "silu"},
        }
        discovery = {
            "candidates": [
                {
                    "operators": ["aten.silu.default", "aten.mul.Tensor"],
                    "observations": [{"consumer_scope": "layer.mlp"}],
                }
            ]
        }
        self.assertEqual(gate_sites(report, discovery), ["layer.mlp"])
        self.assertFalse(gate_sites(dict(report, dtype="torch.float32"), discovery))
        self.assertFalse(gate_sites(report, {"candidates": []}))


class HistoryTests(unittest.TestCase):
    def setUp(self):
        scratch = TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.folder = Path(scratch.name)
        (self.folder / "report.json").write_text("{}")
        self.root = self.folder / "optimization"
        self.root.mkdir()
        self.task = {"sites": ["mlp"]}
        (self.root / "task.json").write_text(json.dumps(self.task))
        prep = patch("kernelatlas.trials._prepare", return_value=self.task)
        prep.start()
        self.addCleanup(prep.stop)
        self.file = self.folder / "gate.py"
        self.file.write_text("# fixture")
        self.verdict = "faster"

        def worker(args, **kwargs):
            trial = self.root / "trials" / args[-1]
            (trial / "measurement.json").write_text(
                json.dumps({"verdict": self.verdict, "remaining_pairs": []})
            )
            return SimpleNamespace(returncode=0)

        proc = patch("kernelatlas.trials.subprocess.run", side_effect=worker)
        self.worker = proc.start()
        self.addCleanup(proc.stop)

    def test_promote_resume_replay_and_fingerprint(self):
        won = run_trial(self.folder, self.file)
        self.assertTrue(won["promoted"])
        self.assertTrue(run_trial(self.folder, self.file)["cached"])
        self.assertEqual(self.worker.call_count, 1)
        self.file.write_text("# another fixture")
        self.verdict = "incorrect"
        failed = run_trial(self.folder, self.file)
        self.assertEqual(failed["incumbent"], won["id"])
        self.assertEqual(
            run_trial(self.folder)["configuration"]["implementation"], won["candidate"]
        )
        self.verdict = "faster"
        replay = run_trial(self.folder, replay=failed["id"])
        self.assertFalse(replay["promoted"])
        self.assertEqual(run_trial(self.folder)["best"], won["id"])
        (self.root / failed["candidate"]).write_text("# changed")
        with self.assertRaisesRegex(ValueError, "fingerprint|Archived"):
            run_trial(self.folder, replay=failed["id"])

    def test_worker_failure_is_recorded_and_retryable(self):
        self.worker.side_effect = None
        self.worker.return_value = SimpleNamespace(returncode=-5)
        failure = run_trial(self.folder, self.file)
        self.assertEqual((failure["verdict"], failure["best"]), ("error", "eager"))
        retry = run_trial(self.folder, self.file)
        self.assertNotEqual(failure["id"], retry["id"])
        self.assertIn("worker.log", retry["reason"])

    def test_interrupted_history_and_task_mismatch(self):
        self.worker.side_effect = KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            run_trial(self.folder, self.file)
        status = run_trial(self.folder)
        self.assertEqual(status["history"][0]["verdict"], "error")
        (self.root / "task.json").write_text('{"sites":["different"]}')
        with self.assertRaisesRegex(ValueError, "workload differs"):
            run_trial(self.folder)
