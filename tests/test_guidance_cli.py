import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


class GuidanceOutputTests(unittest.TestCase):
    root = Path(__file__).resolve().parents[1]
    model = "HuggingFaceTB/SmolLM2-135M-Instruct"

    def show(self, *args, width="80"):
        return subprocess.check_output(
            [sys.executable, "-m", "kernelatlas.cli", "atlas", "show", *args],
            cwd=self.root,
            env=dict(os.environ, COLUMNS=width),
            text=True,
        )

    def test_saved_json_is_stable_and_scoped(self):
        args = ("--model-id", self.model, "--prompt-length", "128", "--format", "json")
        output = self.show(*args)
        self.assertEqual(output, self.show(*args, width="32"))
        self.assertNotIn("\x1b", output)
        doc = json.loads(output)
        self.assertEqual(doc["schema_version"], 1)
        (run,) = doc["runs"]
        choices = {r["candidate"]: r for r in run["existing_kernels"]}
        metal = choices["rmsnorm_tail.metal"]
        ast.parse(metal["python"])
        self.assertIn(
            metal["python"],
            self.show("--model-id", self.model, "--prompt-length", "128"),
        )
        self.assertEqual(metal["action"], "try_prefill")
        self.assertLess(metal["measurements"][0]["median_paired_decode_savings_ms"], 0)
        self.assertEqual(
            choices["kernels-community/mlx-rmsnorm"]["action"], "keep_eager"
        )
        self.assertEqual(
            len(json.loads(self.show(*args, "--details"))["evaluations"]), 2
        )
        self.assertTrue(
            all(t["measured_savings_ms"] is None for t in run["development_tasks"])
        )

    def test_show_keeps_build_ideas_alongside_hub_decisions(self):
        args = (
            str(self.root / "evidence"),
            "--model-id",
            "Qwen/Qwen3-0.6B",
            "--prompt-length",
            "32",
        )
        output = self.show(*args)
        self.assertIn("Which kernel could I build?", output)
        self.assertIn("Metal fused SiLU and multiply", output)
        doc = json.loads(self.show(*args, "--format", "json"))["saved_profile"]
        discovery = doc.get("discovery", doc)
        gate = next(
            t
            for t in discovery["development_tasks"]
            if "SiLU and multiply" in t["title"]
        )
        self.assertEqual(gate["shape"], [1, 32, 3072])
        self.assertEqual(gate["status"], "hypothesis")
        self.assertIn("Searched 651 public repositories for 8 operator pairs", output)
        self.assertIn("16 repositories matched", output)
        self.assertIn("Qwen3 MLPs compiled with Inductor", output)
        self.assertIn("Keep use_kernels=False", output)
        self.assertIn("automatic (use_kernels=True) used 113", output)
        self.assertIn("PyTorch still runs GPU kernels", output)
        self.assertIn("exceeded the logits tolerance", output)
        self.assertIn("0.01 + 0.016 × |native logit|", output)
        self.assertIn("113/113 Metal launches used the model’s eps=1e-06", output)
        self.assertIn("allowance near zero is a KernelAtlas heuristic", output)
        self.assertIn("silu_and_mul", output)
        self.assertEqual(discovery["hub_summary"]["matching_repositories"], 16)
        self.assertEqual(gate["hub_leads"][0]["symbols"], ["silu_and_mul"])
        self.assertEqual(gate["hub_leads"][0]["compatibility"], "unverified")

    def test_archived_comparison_keeps_original_tolerance(self):
        from contextlib import redirect_stdout
        from io import StringIO
        from kernelatlas.recipes import show_comparison

        archive = self.root / "evidence/hub-loading/qwen3-06b-m5.json"
        with redirect_stdout(StringIO()) as out:
            show_comparison(json.loads(archive.read_text()))
        self.assertIn("0.02 + 0.016 × |candidate logit|", out.getvalue())

    def test_empty_query(self):
        self.assertEqual(
            json.loads(self.show("--model-id", "missing/model", "--format", "json"))[
                "runs"
            ],
            [],
        )
