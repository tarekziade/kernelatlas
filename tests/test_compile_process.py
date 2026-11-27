import ast
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from kernelatlas.compilation import candidate_context, compare_compilation, recipe_for
from kernelatlas.discovery import analyze_run


class CompilationTests(unittest.TestCase):
    def trial(self, folder, results):
        def worker(cmd, **kwargs):
            mode = cmd[-1]
            record = results.get(mode)
            if record is None:
                return SimpleNamespace(returncode=1)
            (folder / "compile-trials" / f"{mode}.json").write_text(json.dumps(record))
            return SimpleNamespace(returncode=0)

        with patch("kernelatlas.compilation.subprocess.run", side_effect=worker):
            return compare_compilation(folder, progress=lambda _: None)

    def test_failed_worker_and_rejected_logits_keep_eager(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            result = self.trial(
                folder, {"mlp": {"verdict": "failed_logits", "step": 0}}
            )
            self.assertEqual(result["selected"], "eager")
            self.assertEqual(result["trials"][0]["verdict"], "error")
            self.assertFalse((folder / "compile-trials" / "apply_to_model.py").exists())

    def test_selection_survives_offline_reopen(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            recipe = {
                "mode": "mlp",
                "paths": ["layers.0.mlp"],
                "options": {"emulate_precision_casts": True},
            }
            result = self.trial(
                folder,
                {
                    "mlp": {
                        "recipe": recipe,
                        "verdict": "faster",
                        "timing": {"total": {"median_paired_saving_ms": 1.2}},
                    }
                },
            )
            self.assertEqual(result["recipe"], recipe)
            ast.parse((folder / "compile-trials" / "apply_to_model.py").read_text())
            (folder / "report.json").write_text(json.dumps({"compilation": result}))
            self.assertEqual(analyze_run(folder)["compilation"], result)

    def test_hub_scope_distinguishes_eager_from_compiled(self):
        from kernelatlas.loading_trial import scope

        report = {"dtype": "torch.bfloat16", "phases": {"prefill": {"cache_length": 32}}}
        options = {"configs": [], "adapters": {}}
        with patch("kernelatlas.loading_trial.version", return_value="test"):
            eager = scope(report, options)
            recipe = {"mode": "whole_model", "paths": [""], "options": {}}
            compiled = scope(report, {**options, "compile_recipe": recipe})
        self.assertNotIn("compile_recipe", eager)
        self.assertEqual(compiled["compile_recipe"], recipe)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
    def test_regional_comparison_restores_eager_on_exception(self):
        from torch import nn

        model = nn.Module()
        model.mlp = nn.Linear(2, 2)
        original = model.mlp
        with patch.object(nn.Module, "compile") as compile_module:
            install = candidate_context(model, recipe_for(model, "mlp"))
            compile_module.assert_called_once()
            with self.assertRaises(RuntimeError), install() as compiled:
                self.assertIs(compiled.mlp.weight, original.weight)
                self.assertIsNot(compiled.mlp, original)
                raise RuntimeError
        self.assertIs(model.mlp, original)
