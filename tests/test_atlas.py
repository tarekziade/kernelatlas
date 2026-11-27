import json
from pathlib import Path
import unittest

from kernelatlas import Atlas


class MeasuredEvidenceTests(unittest.TestCase):
    def test_hub_dispatch_and_correctness_are_independent(self):
        root = Path(__file__).resolve().parents[1] / "evidence"
        atlas = Atlas.from_reports(root)
        rows = atlas.evaluations(torch_version="2.15.0", prompt_length=128)
        self.assertEqual(len(rows), 2)
        hub = next(
            row for row in rows if row["candidate"] == "kernels-community/mlx-rmsnorm"
        )
        self.assertEqual(hub["integration"], "confirmed")
        self.assertEqual(hub["correctness"], "rejected")
        self.assertIsNone(hub["median_paired_prefill_savings_ms"])
        metal = next(row for row in rows if row["candidate"] == "rmsnorm_tail.metal")
        source = json.loads((root / metal["evaluation"]).read_text())
        self.assertEqual(
            metal["median_paired_decode_savings_ms"],
            source["metal_tail"]["timing"]["decode_one_token"][
                "median_paired_saving_ms"
            ],
        )

    def test_runtime_filter_preserves_original_trials(self):
        root = Path(__file__).resolve().parents[1] / "evidence"
        atlas = Atlas.from_reports(root)
        self.assertEqual(len(atlas.evaluations(torch_version="2.14.1")), 6)
        self.assertEqual(atlas.data["architectures"], ["llama", "qwen3"])

    def test_hub_lookup_excludes_other_weights_in_same_family(self):
        atlas = Atlas.from_reports(Path(__file__).resolve().parents[1] / "evidence")
        model = "HuggingFaceTB/SmolLM2-135M-Instruct"
        rows = atlas.evaluations(model_id=model)
        self.assertEqual({r["run"] for r in rows}, {"smollm2-01"})
        self.assertLess(len(rows), len(atlas.evaluations(family="llama")))
        self.assertEqual(len(atlas.evaluations(model_id=model, prompt_length=128)), 2)
        self.assertEqual(atlas.evaluations(model_id=model, dtype="torch.bfloat16"), [])
        self.assertEqual(atlas.evaluations(model_id=model, weight_mode="int4"), [])
        self.assertEqual(atlas.evaluations(model_id=model, weight_mode="none"), rows)
        self.assertEqual(atlas.evaluations(model_id="random seed 0"), [])
        self.assertEqual(atlas.evaluations(model_id="unknown/checkpoint"), [])
