import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


class RandomWeightProfileTests(unittest.TestCase):
    def test_prefill_and_decode_with_only_a_config(self):
        if not all(importlib.util.find_spec(pkg) for pkg in ("torch", "transformers")):
            self.skipTest("Profiling extras are optional")
        from kernelatlas import profiling

        if not profiling.torch.backends.mps.is_available():
            self.skipTest("Requires MPS")
        with TemporaryDirectory() as temp:
            snapshot = Path(temp) / ("a" * 40)
            snapshot.mkdir()
            config = dict(
                model_type="llama",
                vocab_size=128,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=4,
            )
            (snapshot / "config.json").write_text(json.dumps(config))
            with (
                patch.object(
                    profiling, "snapshot_download", return_value=str(snapshot)
                ) as download,
                patch.object(
                    profiling.AutoModelForCausalLM,
                    "from_pretrained",
                    side_effect=AssertionError("Attempted checkpoint load"),
                ),
            ):
                out, report = profiling.profile_hub_model(
                    "example/model",
                    revision="test",
                    random_weights=True,
                    out=Path(temp) / "profile",
                )
            download.assert_called_once_with(
                "example/model", revision="test", allow_patterns=["config.json"]
            )
            self.assertEqual(
                report["config_source"],
                {"model_id": "example/model", "revision": snapshot.name},
            )
            from kernelatlas.discovery import analyze_run

            decode = analyze_run(out)["decode"]
            self.assertEqual(
                {a["slot"] for a in decode["retained_cache_appends"]},
                {"keys", "values"},
            )
            self.assertNotIn("cache_trial", decode)
            self.assertEqual(report["weight_source"], "random")
            self.assertIsNone(report["model_id"])
            self.assertIsNone(report["revision"])
            self.assertEqual(
                [r["cache_length"] for r in report["phases"].values()], [32, 33]
            )
            self.assertTrue((out / "prefill.trace.json").is_file())
            self.assertTrue((out / report["prefill_dataflow"]).is_file())

    def test_default_inkling_mps_uses_text_config_without_loading_checkpoint(self):
        from types import SimpleNamespace

        if not all(importlib.util.find_spec(pkg) for pkg in ("torch", "transformers")):
            self.skipTest("Profiling extras are optional")
        from kernelatlas import profiling

        class TextConfig(SimpleNamespace):
            def to_dict(self):
                return vars(self).copy()

            @classmethod
            def from_dict(cls, values):
                return cls(**values)

        text = TextConfig(
            vocab_size=201024,
            unpadded_vocab_size=200058,
            bos_token_id=None,
            eos_token_id=200006,
            num_hidden_layers=24,
            n_routed_experts=128,
            n_layers_per_attn_res_block=6,
            loop_layer_index_start=6,
            loop_layer_index_end=18,
            layer_types=(["hybrid_sliding"] * 5 + ["hybrid"]) * 4,
        )
        model_id = "thinkingmachines/Inkling-2-Mini"
        with (
            patch.object(
                profiling.torch.backends.mps, "is_available", return_value=True
            ),
            patch.object(
                profiling.torch.backends.mps, "is_macos_or_newer", return_value=True
            ),
            patch.object(
                profiling, "snapshot_download", return_value="/cache/commit"
            ) as download,
            patch.object(
                profiling.AutoConfig,
                "from_pretrained",
                return_value=SimpleNamespace(text_config=text),
            ),
            patch.object(
                profiling.AutoModelForCausalLM,
                "from_pretrained",
                side_effect=AssertionError("Attempted checkpoint load"),
            ),
            patch.object(profiling.AutoModelForCausalLM, "from_config") as build,
            patch.object(profiling, "profile_model") as profile,
        ):
            profiling.profile_hub_model(model_id, revision="test", out="unused")
        download.assert_called_once_with(
            model_id, revision="test", allow_patterns=["config.json"]
        )
        config = build.call_args.args[0]
        self.assertEqual(config.num_hidden_layers, 6)
        self.assertEqual(config.n_routed_experts, 16)
        self.assertEqual(config.vocab_size, 256)
        self.assertEqual(build.call_args.kwargs["dtype"], profiling.torch.bfloat16)
        workload = profile.call_args.kwargs
        self.assertEqual(workload["proxy"]["original"]["num_hidden_layers"], 24)
        self.assertEqual(
            workload["config_source"], {"model_id": model_id, "revision": "commit"}
        )
        self.assertIsNone(workload["model_id"])
        self.assertIsNone(workload["revision"])
