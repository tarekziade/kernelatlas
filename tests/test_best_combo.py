import ast
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from kernelatlas.example import write_best_combo


class BestComboTests(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.report = {
            "weight_source": "pretrained",
            "revision": "abcd",
            "dtype": "torch.bfloat16",
            "config_source": {"model_id": "org/model", "revision": "abcd"},
            "device": "cuda:0",
            "attention": "sdpa",
            "torch": "test",
            "transformers": "test",
            "phases": {"prefill": {"cache_length": 32}},
            "precision": {
                "weights": {"mode": "none"},
                "requested_dtype": "bf16",
                "requested_quantization": "auto",
            },
            "compilation": {
                "selected": "whole_model",
                "python": 'model.compile(backend="inductor")',
            },
        }

    def script(self, options=None):
        folder = Path(self.folder.name)
        name = write_best_combo(folder, self.report, options)
        text = (folder / name).read_text()
        ast.parse(text)
        return text

    def test_unmeasured_and_failed_hub_configs_retain_native_compile(self):
        rejected = {"name": "automatic", "verdict": "incorrect"}
        text = self.script({"comparison": {"results": [rejected]}})
        self.assertIn("use_kernels=False", text)
        self.assertIn('model.compile(backend="inductor")', text)
        self.assertIn("attn_implementation='sdpa'", text)
        self.assertIn("range(2)", text)

    def test_only_the_best_tested_configuration_is_exported(self):
        def win(name, saving):
            return {
                "name": name,
                "verdict": "faster",
                "timing": {"total": {"median_paired_saving_ms": saving}},
            }

        configs = [
            {
                "name": "attention_only",
                "use_kernels": False,
                "attn_implementation": "org/attention@sha",
            },
            {
                "name": "selective",
                "use_kernels": True,
                "kernel_mapping": {"RMSNorm": ["org/norm:Norm", {"revision": "pin"}]},
            },
        ]
        options = {
            "configs": configs,
            "comparison": {"results": [win("attention_only", 1), win("selective", 2)]},
        }
        text = self.script(json.loads(json.dumps(options)))
        self.assertIn("org/norm:Norm", text)
        self.assertIn("inherit_mapping=False", text)
        self.assertNotIn("org/attention", text)
        self.assertIn("revision='abcd'", text)

    def test_proxy_reuses_profile_loader_and_saved_workload(self):
        self.report.update(
            weight_source="random",
            workload={
                "prompt_ids": [5, 2],
                "max_new_tokens": 3,
                "forward_kwargs": {"logits_to_keep": 1},
            },
        )
        text = self.script()
        self.assertIn("random_weights=True", text)
        self.assertNotIn("from_pretrained", text)
        self.assertIn("torch.tensor([[5, 2]]", text)
        self.assertIn("logits_to_keep=1", text)
        self.assertIn("range(3)", text)
