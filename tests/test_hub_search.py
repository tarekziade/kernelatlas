from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kernelatlas import hub
from kernelatlas.example import write_example
from kernelatlas.integrations import mapping_declarations


class HubSearchTests(unittest.TestCase):
    def setUp(self):
        self.discovery = {
            "device": "mps:0",
            "torch": "2.15.0",
            "candidates": [
                {
                    "operators": ["aten.linear.default", "aten.gelu.default"],
                    "observations": [
                        {"producer_scope": "proj", "consumer_scope": "act"}
                    ],
                }
            ],
            "transformers_integrations": {
                "sites": {
                    "act": {
                        "layer": "GeLU",
                        "binding": "original_forward",
                        "bound_repo": None,
                        "bound_symbol": None,
                    }
                },
                "declarations": [
                    {
                        "layer": "GeLU",
                        "device": "mps",
                        "mode": "Mode.INFERENCE",
                        "source": "fixture:1",
                        "repo_id": "kernels-community/example",
                        "layer_name": "Gelu",
                        "version": 1,
                    }
                ],
            },
        }

    def test_real_binding_separated_from_catalog_match(self):
        repo = "kernels-community/example"
        catalog = [
            {
                "id": repo,
                "tags": [],
                "supportedDriverFamilies": ["metal"],
                "sha": "a" * 40,
            }
        ]
        self.discovery["transformers_integrations"]["sites"]["act"].update(
            binding="hub_bound", bound_repo=repo, bound_symbol="Gelu"
        )
        with (
            patch.dict("os.environ", {"HF_HUB_OFFLINE": "0"}),
            patch.object(
                hub, "_json", side_effect=[catalog, {"functions": [{"name": "gelu"}]}]
            ),
        ):
            result = hub.search(self.discovery)
        (hit,) = result["queries"][0]["hits"]
        self.assertEqual(hit["catalog_match"]["kind"], "mapping")
        self.assertEqual(
            hit["observed_layer_bindings"],
            [{"symbol": "Gelu", "binding": "already_bound", "scope_count": 1}],
        )
        self.assertEqual(hit["symbols"]["matches"][0]["kind"], "component")
        self.assertEqual(
            result["layers"][0]["example_status"], "no_observed_original_sites"
        )
        self.assertIsNone(write_example(None, {}, result))

    def test_summary_deduplicates_repos_and_filters_pair_evidence(self):
        tasks = [{"operators": ["left", "right"]}, {"operators": ["first", "second"]}]
        candidate = {
            "repo_id": "org/mixed",
            "advertised_drivers": ["metal"],
            "url": "repo",
            "catalog_match": {"kind": "component"},
            "symbols": {"matches": [{"kind": "component", "name": "left"}]},
        }
        remote = {
            **candidate,
            "repo_id": "org/cuda",
            "advertised_drivers": ["cuda"],
            "catalog_match": {"kind": "pair"},
        }
        snapshot = {
            "status": "searched",
            "target_driver": "metal",
            "catalog_size": 3,
            "limit_reached": True,
            "queries": [
                {"operators": tasks[0]["operators"], "hits": [candidate, remote]},
                {
                    "operators": tasks[1]["operators"],
                    "hits": [{**candidate, "catalog_match": {"kind": "pair"}}],
                },
            ],
        }
        summary = hub.summarize_search(snapshot, tasks)
        self.assertEqual(summary["matching_repositories"], 2)
        self.assertEqual(summary["target_repositories"], ["org/mixed"])
        self.assertTrue(summary["limit_reached"])
        self.assertEqual(tasks[0]["hub_leads"], [])
        self.assertEqual(tasks[1]["hub_leads"][0]["basis"], "catalog_metadata")
        skipped = hub.summarize_search(
            {"status": "offline", "target_driver": "metal"}, []
        )
        self.assertNotIn("matching_repositories", skipped)

    def test_offline_and_unavailable(self):
        with (
            patch.dict("os.environ", {"HF_HUB_OFFLINE": "1"}),
            patch.object(hub, "_json") as fetch,
        ):
            offline = hub.search(self.discovery)
            self.assertEqual(offline["status"], "offline")
            self.assertEqual(offline["layers"][0]["sites"]["original_forward"], ["act"])
            fetch.assert_not_called()
        with (
            patch.dict("os.environ", {"HF_HUB_OFFLINE": "0"}),
            patch.object(hub, "_json", side_effect=TimeoutError("timeout")),
        ):
            self.assertEqual(hub.search(self.discovery)["status"], "unavailable")

    def test_literal_declarations_preserve_mode_and_device(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "mapping.py"
            p.write_text(
                '_KERNEL_MAPPING = {"GeLU": {"cuda": {Mode.TRAINING: LayerRepository(repo_id="a/b", layer_name="Gelu", version=3)}}}'
            )
            rows = mapping_declarations(p, {"GeLU"})
        self.assertEqual(rows[0]["version"], 3)
        self.assertEqual(rows[0]["device"], "cuda")
        self.discovery["transformers_integrations"]["declarations"] = rows
        self.assertFalse(hub.layer_leads(self.discovery))

    def test_all_picked_sites_written_other_bindings_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            layer = hub.layer_leads(self.discovery)[0]
            layer.update(example_status="picked_for_trial", pinned_revision="a" * 40)
            layer["sites"]["already_bound"].append("previously_selected")
            second = dict(
                layer,
                symbol="Other",
                layer="Other",
                sites={"original_forward": ["other_site"]},
            )
            report = {
                "config_source": {"model_id": "test/model", "revision": "b" * 40},
                "weight_source": "random",
                "device": "mps",
                "precision": {"dtype": "torch.float16", "weights": {"mode": "int4"}},
            }
            path = write_example(folder, report, {"layers": [layer, second]}, True)
            source = (folder / path).read_text()
            compile(source, path, "exec")
            self.assertIn("getattr(torch, 'float16')", source)
            self.assertIn("pack_for_profile(model, 'int4', DEVICE, 32, None)", source)
            self.assertIn("other_site", source)
            self.assertNotIn("previously_selected", source)
            self.assertIn("metal_rmsnorm_prefill(model)", source)
            report["workload"] = {
                "prompt_ids": list(range(64)),
                "forward_kwargs": {"logits_to_keep": 1},
            }
            path = write_example(folder, report, {"layers": [layer]}, False)
            scoped = (folder / path).read_text()
            compile(scoped, path, "exec")
            self.assertIn("torch.arange(64,", scoped)
            self.assertIn("FORWARD_KWARGS = {'logits_to_keep': 1}", scoped)
            self.assertIn("64-token prefill", scoped)
