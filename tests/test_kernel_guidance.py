"""Synthetic contracts for comparing Hub configs and reopening build ideas."""

import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kernelatlas.discovery import build_ideas, latest_profile
from kernelatlas.recipes import loading_options


class BuildIdeasTests(unittest.TestCase):
    def test_saved_profile_filters_and_concrete_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for timestamp, model, length in (
                ("1", "org/test", 32),
                ("2", "org/test", 64),
                ("3", "org/other", 32),
            ):
                folder = root / timestamp
                folder.mkdir()
                (folder / "graph.json").write_text("{}")
                (folder / "report.json").write_text(
                    json.dumps(
                        {
                            "model_id": model,
                            "timestamp": timestamp,
                            "torch": "test",
                            "prefill_dataflow": "graph.json",
                            "config": {"model_type": "qwen3"},
                            "phases": {"prefill": {"cache_length": length}},
                        }
                    )
                )
            self.assertEqual(latest_profile(root, "org/test"), root / "2")
            self.assertEqual(
                latest_profile(root, "org/test", prompt_length=32), root / "1"
            )
            self.assertIsNone(latest_profile(root, "org/missing"))
            candidate = {
                "operators": ["aten.silu.default", "aten.mul.Tensor"],
                "intermediate": {"shape": [1, 32, 16], "dtype": "torch.bfloat16"},
                "observations": [
                    {"consumer_scope": "mlp", "producer": 1, "consumer": 2}
                ],
                "total_logical_intermediate_bytes": 2048,
                "occurrences": 2,
            }
            tasks = build_ideas(
                root,
                {
                    "device": "mps:0",
                    "candidates": [candidate, candidate],
                    "evidence": "synthetic",
                    "trial": {"command": ["kernelatlas", "trial", tmp]},
                },
            )
            self.assertEqual(len(tasks), 1)
            self.assertIn("SiLU", tasks[0]["title"])
            self.assertEqual(tasks[0]["nodes"], [1, 2])
            self.assertIsNone(tasks[0]["measured_speedup"])


class LoadingTests(unittest.TestCase):
    def test_selective_uses_only_inspected_inference_settings(self):
        try:
            import transformers.integrations.hub_kernels  # noqa: F401
        except ImportError:
            self.skipTest("Transformers unavailable")
        registry = {"layers": {"RMSNorm": 2}, "flash_attention": False}
        declaration = dict(
            layer="RMSNorm",
            layer_name="Norm",
            repo_id="kernels-community/test",
            version=1,
            device="mps",
            mode="Mode.INFERENCE",
        )

        def inspect(spec, device):
            return {**spec, "revision": "abc", "local_target": True}

        with (
            patch(
                "kernelatlas.recipes.mapping_declarations",
                return_value=[
                    declaration,
                    {
                        **declaration,
                        "layer_name": "Compiled",
                        "mode": "Mode.TORCH_COMPILE",
                    },
                ],
            ),
            patch("kernelatlas.recipes.inspect_adapter", side_effect=inspect),
        ):
            args = ("org/test", "revision", "mps", "bfloat16", registry)
            options = loading_options(*args)
            auto, selective = options["configs"]
            self.assertNotIn("KernelConfig", auto["python"])
            self.assertFalse(selective["inherit_mapping"])
            self.assertEqual(
                selective["kernel_mapping"]["RMSNorm"][0], "kernels-community/test:Norm"
            )
            ast.parse(selective["python"])
            rejected = loading_options(
                *args, excluded={("kernels-community/test", "abc")}
            )
            self.assertEqual([c["name"] for c in rejected["configs"]], ["automatic"])
