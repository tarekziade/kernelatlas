import importlib.util
import tempfile
import unittest

from kernelatlas.discovery import analyze_run, find_candidates


class DiscoveryTests(unittest.TestCase):
    def test_legacy_report_requires_new_capture(self):
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / "report.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "kernelatlas profile MODEL_ID"):
                analyze_run(temp)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Requires profiling extra")
    def test_connected_edges_and_rejections(self):
        import torch
        from kernelatlas.dataflow import capture

        class Model(torch.nn.Module):
            def forward(self, x, use_cache):
                results = []
                for i in range(2):
                    a = torch.sin(x)
                    unrelated = torch.exp(x)
                    results.extend([unrelated, torch.relu(a)])
                    # Fanout, a returned intermediate, a view and mutation.
                    b = torch.cos(x)
                    results.extend([torch.relu(b), torch.neg(b)])
                    c = torch.tanh(x)
                    results.extend([c, torch.relu(c)])
                    d = x.view_as(x)
                    results.append(torch.relu(d))
                    e = x.clone().neg_()
                    results.append(torch.relu(e))
                return results

        model = Model()
        x = torch.arange(6, dtype=torch.float32)
        graph = capture(model, x)
        candidates = find_candidates(graph)
        self.assertEqual(
            [c["operators"] for c in candidates],
            [["aten.sin.default", "aten.relu.default"]],
        )
        candidate = candidates[0]
        self.assertEqual(candidate["occurrences"], 2)
        self.assertEqual(candidate["total_logical_intermediate_bytes"], 48)
        self.assertTrue(
            all(o["consumer"] > o["producer"] + 1 for o in candidate["observations"])
        )
        self.assertFalse(model._forward_hooks)
        self.assertFalse(model._forward_pre_hooks)
        self.assertIsNone(candidate["measured_savings_ms"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Requires profiling extra")
    def test_cache_append_requires_prior_tensor_and_retained_result(self):
        from types import SimpleNamespace
        import torch
        from kernelatlas.dataflow import capture

        class Model(torch.nn.Module):
            def forward(self, x, cache, use_cache):
                layer = cache.layers[0]
                layer.keys = torch.cat([layer.keys, x], dim=2)
                # Same operation and shape, but the cache retains a different tensor.
                other = torch.cat([layer.values, x], dim=2)
                layer.values = other.clone()
                return SimpleNamespace(cache=cache, logits=x)

        previous = torch.zeros(1, 2, 4, 3)
        cache = SimpleNamespace(
            layers=[SimpleNamespace(keys=previous, values=previous.clone())]
        )
        graph = capture(
            Model(), torch.ones(1, 2, 1, 3), cache_argument="cache", cache=cache
        )
        self.assertEqual(len(graph["retained_cache_appends"]), 1)
        entry = graph["retained_cache_appends"][0]
        self.assertEqual(
            (entry["slot"], entry["layer"], entry["previous_shape"]),
            ("keys", 0, list(previous.shape)),
        )
        self.assertIn((entry["node"], 0), graph["returned"])

    def test_cache_trial_history_requires_both_fingerprints(self):
        from pathlib import Path
        import json
        from kernelatlas.discovery import cache_results
        from kernelatlas.trials import _digest

        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            graph, report = folder / "graph.json", folder / "report.json"
            graph.write_text("{}")
            report.write_text("{}")
            directory = folder / "optimization" / "cache-trials"
            directory.mkdir(parents=True)
            record = {
                "report_sha256": _digest(report),
                "graph_sha256": _digest(graph),
                "verdict": "inconclusive",
                "completed_at": "now",
            }
            (directory / "one.json").write_text(json.dumps(record))
            self.assertEqual(
                cache_results(folder, report, graph)[0]["verdict"], "inconclusive"
            )
            (directory / "two.json").write_text(
                json.dumps(dict(record, reserve_at="decode", completed_at="then"))
            )
            self.assertEqual(
                [r["reserve_at"] for r in cache_results(folder, report, graph)],
                ["prefill", "decode"],
            )
            graph.write_text("[]")
            self.assertEqual(cache_results(folder, report, graph), [])
