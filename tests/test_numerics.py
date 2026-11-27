import importlib.util
import unittest

from kernelatlas.numerics import logits_close, model_logit_policy


class LogitPolicyTests(unittest.TestCase):
    def test_scoped_tolerances(self):
        self.assertEqual(
            model_logit_policy("torch.bfloat16"),
            {
                "atol": 0.01,
                "rtol": 0.016,
                "relative_to": "native",
            },
        )
        self.assertEqual(model_logit_policy("float16")["atol"], 0.001)
        self.assertEqual(model_logit_policy("float32")["rtol"], 1e-4)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
    def test_near_zero_and_asymmetric_relative_term(self):
        import torch

        policy = model_logit_policy(torch.bfloat16)
        native = torch.tensor([0.0, 0.0, 1.0, 1.0])
        candidate = torch.tensor([0.009, 0.015, 1.025, 1.02625])
        self.assertEqual(
            logits_close(candidate, native, policy).tolist(), [True, False, True, False]
        )
        self.assertTrue(logits_close(native, candidate, policy)[-1])
        for a, b, ok in zip(candidate, native, logits_close(candidate, native, policy)):
            if ok:
                torch.testing.assert_close(
                    a, b, atol=policy["atol"], rtol=policy["rtol"]
                )
            else:
                with self.assertRaises(AssertionError):
                    torch.testing.assert_close(
                        a, b, atol=policy["atol"], rtol=policy["rtol"]
                    )
