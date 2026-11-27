import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from kernelatlas.generation import decode_distribution


class DistributionTests(unittest.TestCase):
    def test_cached_tokens_only(self):
        self.assertEqual(
            decode_distribution([{"cached_step_ms": []}]),
            {"count": 0, "median_ms": None, "p90_ms": None},
        )
        self.assertEqual(
            decode_distribution(
                [{"cached_step_ms": [1, 2]}, {"cached_step_ms": [3, 4]}]
            ),
            {"count": 4, "median_ms": 2.5, "p90_ms": 4},
        )


@unittest.skipUnless(
    all(importlib.util.find_spec(p) for p in ("torch", "transformers")),
    "requires profile extras",
)
class GenerationTests(unittest.TestCase):
    def test_matches_transformers_greedy_generation(self):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM
        from kernelatlas.generation import trajectory

        torch.manual_seed(0)
        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=24,
                hidden_size=16,
                intermediate_size=24,
                num_attention_heads=2,
                num_key_value_heads=2,
                num_hidden_layers=1,
                eos_token_id=None,
            )
        ).eval()
        prompt = torch.tensor([[3, 4, 5]])
        with torch.inference_mode():
            for count in (1, 4):
                with self.subTest(count=count):
                    steps = []
                    logits = trajectory(
                        model,
                        prompt,
                        count - 1,
                        "past_key_values",
                        {"logits_to_keep": 1},
                        on_step=steps.append,
                    )
                    expected = model.generate(
                        prompt, max_new_tokens=count, do_sample=False
                    )
                    self.assertTrue(
                        torch.equal(
                            torch.cat([x.argmax(-1) for x in logits], dim=1),
                            expected[:, 3:],
                        )
                    )
                    self.assertEqual(steps, list(range(count)))
                    timed = trajectory(
                        model,
                        prompt,
                        count - 1,
                        "past_key_values",
                        {},
                        synchronize=lambda: None,
                    )
                    self.assertEqual(len(timed["cached_step_ms"]), count - 1)
                    self.assertAlmostEqual(
                        timed["total"], timed["prefill"] + timed["decode"]
                    )

    def test_migration_is_charged_to_first_cached_step(self):
        import torch
        from types import SimpleNamespace
        from kernelatlas.generation import trajectory

        default, reserved = object(), object()
        clock = 0
        visited = []

        def model(input_ids, use_cache, cache=None):
            nonlocal clock
            visited.append(cache)
            clock += 1_000_000
            return SimpleNamespace(
                cache=default if cache is None else cache, logits=torch.zeros(1, 1, 2)
            )

        def migrate(cache):
            nonlocal clock
            self.assertIs(cache, default)
            clock += 5_000_000
            return reserved

        with patch("kernelatlas.generation.perf_counter_ns", side_effect=lambda: clock):
            for count in (0, 1, 3):
                visited.clear()
                timed = trajectory(
                    model,
                    torch.tensor([[0]]),
                    count,
                    "cache",
                    {},
                    synchronize=lambda: None,
                    migrate_cache=migrate,
                )
                expected = [6.0] + [1.0] * (count - 1) if count else []
                self.assertEqual(visited, [None] + [reserved] * count)
                self.assertEqual(timed["cached_step_ms"], expected)
                self.assertEqual(timed["prefill"], 1.0)
                self.assertEqual(timed["total"], 1.0 + sum(expected))

    def test_migrate_full_cache_preserves_each_layer_and_next_position(self):
        import torch
        from transformers import DynamicCache, LlamaConfig
        from kernelatlas.cache_trial import reserve_for_decode

        config = LlamaConfig(
            num_hidden_layers=2,
            hidden_size=16,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
        source = DynamicCache(config=config)
        for index in range(2):
            keys = torch.arange(48).float().reshape(1, 2, 3, 8) + index
            source.update(keys, -keys, index)
        reserved = reserve_for_decode(source, config=config, capacity=5)
        for index, (old, new) in enumerate(
            zip(source.layers, reserved.layers, strict=True)
        ):
            self.assertEqual(int(new.get_seq_length()), 3)
            self.assertTrue(torch.equal(new.keys[:, :, :3], old.keys))
            self.assertTrue(torch.equal(new.values[:, :, :3], old.values))
            addition = torch.full_like(old.keys[:, :, :1], 7)
            ptr = new.keys.untyped_storage().data_ptr()
            reserved.update(addition, -addition, index)
            self.assertEqual(new.keys.untyped_storage().data_ptr(), ptr)
            self.assertEqual(int(new.get_seq_length()), 4)
            self.assertTrue(torch.equal(new.keys[:, :, 3:4], addition))
            self.assertTrue(torch.equal(new.values[:, :, 3:4], -addition))
            self.assertEqual(old.get_seq_length(), 3)

    def test_changed_decode_budget_invalidates_saved_session(self):
        from kernelatlas.trials import _prepare, _digest, _read, _write

        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            root = folder / "optimization"
            root.mkdir()
            report = folder / "report.json"
            workload = {"prompt_ids": [1, 2], "cached_tokens": 2, "max_new_tokens": 3}
            _write(report, {"workload": workload})
            with (
                patch("kernelatlas.trials.analyze_run", return_value={}),
                patch("kernelatlas.trials.gate_sites", return_value=["mlp"]),
                patch(
                    "kernelatlas.gate_workload.prepare",
                    return_value={"workload": workload},
                ),
            ):
                task = _prepare(folder, root)
            self.assertEqual(task["policy"]["version"], 2)
            self.assertEqual(task["profile_sha256"]["report"], _digest(report))
            updated = _read(report)
            updated["workload"].update(cached_tokens=3, max_new_tokens=4)
            _write(report, updated)
            with self.assertRaisesRegex(ValueError, "fresh profile directory"):
                _prepare(folder, root)

    def test_lengths_and_prefill_only_profile(self):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM
        from kernelatlas.profiling import profile_model

        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=16,
                hidden_size=16,
                intermediate_size=24,
                num_attention_heads=2,
                num_key_value_heads=2,
                num_hidden_layers=1,
                max_position_embeddings=32,
            )
        ).eval()
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "report"
            for prompt, budget in ((0, 2), (2, -1), (30, 4)):
                with self.assertRaises(ValueError):
                    profile_model(
                        model, "cpu", out, prompt_length=prompt, max_new_tokens=budget
                    )
                self.assertFalse(out.exists())
            if torch.backends.mps.is_available():
                result = profile_model(
                    model.to("mps"), "mps", out, prompt_length=17, max_new_tokens=1
                )
                self.assertEqual(
                    result["workload"]["prompt_ids"], list(range(16)) + [0]
                )
                self.assertEqual(set(result["phases"]), {"prefill"})
                self.assertNotIn("decode_dataflow", result)
                self.assertEqual(result["generation"]["per_cached_token"]["count"], 0)
                self.assertEqual(result["phases"]["prefill"]["cache_length"], 17)
