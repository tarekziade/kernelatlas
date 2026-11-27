import io
import os
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from kernelatlas import Atlas
from kernelatlas.terminal import show_evaluations


class GuidanceTests(unittest.TestCase):
    def render(self, *, details=False, **filters):
        atlas = Atlas.from_reports(Path(__file__).resolve().parents[1] / "evidence")
        model = "HuggingFaceTB/SmolLM2-135M-Instruct"
        filters.update(model_id=model)
        output = io.StringIO()
        with patch.dict(os.environ, COLUMNS="36"), redirect_stdout(output):
            show_evaluations(
                atlas.evaluations(**filters),
                model,
                "evidence",
                opportunities=atlas.opportunities(**filters),
                details=details,
            )
        text = output.getvalue()
        self.assertLessEqual(max(map(len, text.splitlines())), 36)
        return " ".join(text.split())

    def test_guidance_and_mlp_proposal(self):
        text = self.render()
        self.assertIn("Existing kernels to use", text)
        self.assertIn("Kernels to create", text)
        self.assertIn("30 MLP blocks", text)
        self.assertIn("decode 0.054 ms slower", text)
        self.assertNotIn("hub-integration-128.json", text)

    def test_filtered_evidence(self):
        text = self.render(details=True, torch_version="2.15.0", prompt_length=128)
        self.assertNotIn("30 MLP blocks", text)
        self.assertIn("hub-integration-128.json", text.replace(" ", ""))
        self.assertIn("full Metal RMSNorm", text)


class ProfileSummaryTests(unittest.TestCase):
    def setUp(self):
        self.lead = {
            "layer": "RMSNorm",
            "repo_id": "org/norm",
            "symbol": "RMSNorm",
            "example_status": "failed_trial",
            "failure_summary": "The earlier output check failed.",
            "sites": {"original_forward": ["norm"], "already_bound": [], "unknown": []},
            "mappings": [{"device": "mps"}],
        }
        self.discovery = {
            "candidates": [
                {
                    "operators": ["aten.silu.default", "aten.mul.Tensor"],
                    "occurrences": 28,
                    "total_logical_intermediate_bytes": 11010048,
                    "intermediate": {"shape": [1, 32, 6144]},
                }
            ],
            "hub_search": {
                "status": "searched",
                "layers": [self.lead],
                "catalog_size": 1,
                "limit_reached": False,
                "target_driver": "metal",
                "queries": [],
            },
        }
        self.baseline = {
            "weight_source": "pretrained",
            "precision": {"dtype": "torch.bfloat16", "weights": {"mode": "none"}},
            "device": "mps:0",
            "hardware": "Apple M5",
            "torch": "test",
            "report": "runs/space in path/report.json",
            "config_source": {"revision": "abc"},
            "prompt_tokens": 32,
            "batch_size": 1,
            "prefill_ms": 50,
            "decode_ms": 35,
            "discovery": self.discovery,
        }

    def render(self, width=80, **options):
        stream = io.StringIO()
        with patch.dict(os.environ, COLUMNS=str(width)), redirect_stdout(stream):
            show_evaluations(
                [], "org/model", "evidence", baseline=self.baseline, **options
            )
        return stream.getvalue()

    def test_verdict_precedes_timing_and_technical_details_are_opt_in(self):
        import json

        text = self.render()
        self.assertLess(text.index("Keep the current RMSNorm"), text.index("50.0 ms"))
        for fragment in (
            "earlier output check failed",
            "SiLU activation then multiply",
            "10.5 MiB",
            "32-token prompt",
            "35.0 ms",
        ):
            self.assertIn(fragment, text)
        self.assertNotIn("aten.", text)
        self.assertNotIn("Catalog", text)
        self.assertIn("aten.mul.Tensor", self.render(details=True))
        self.assertEqual(
            json.loads(self.render(output_format="json"))["baseline"], self.baseline
        )

    def test_copyable_example_and_narrow_layout(self):
        import shlex
        import sys

        self.lead["example_status"] = "picked_for_trial"
        self.discovery["hub_search"]["example"] = "try_kernels.py"
        text = self.render(width=32)
        self.assertIn("Test org/norm", text)
        self.assertIn("28 calls", " ".join(text.split()))
        self.assertIn(
            shlex.join([sys.executable, "runs/space in path/try_kernels.py"]),
            text.splitlines(),
        )
        self.assertNotIn("passed", text)

    def test_missing_search_and_disabled_discovery(self):
        self.discovery["hub_search"] = {"status": "offline"}
        self.discovery["candidates"] = []
        self.assertIn("offline", self.render())
        self.assertNotIn("search found no replacement", self.render())
        del self.baseline["discovery"]
        self.assertIn("discovery was disabled", self.render())
