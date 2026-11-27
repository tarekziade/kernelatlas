import importlib.util
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("torch"), "Requires profiling extras")
class PrecisionTests(unittest.TestCase):
    def test_auto_dtype_fallback_and_overrides(self):
        import torch
        from kernelatlas.precision import resolve_dtype

        with patch.object(torch.cuda, "is_bf16_supported", return_value=False):
            self.assertEqual(resolve_dtype("cuda", "auto"), torch.float16)
            with self.assertRaisesRegex(ValueError, "BF16 is unavailable"):
                resolve_dtype("cuda", "bf16")
        with patch.object(torch.backends.mps, "is_macos_or_newer", return_value=True):
            self.assertEqual(resolve_dtype("mps", "auto"), torch.bfloat16)
            self.assertEqual(resolve_dtype("mps", "fp32"), torch.float32)

    def test_packed_weights_compute_reference_on_mps(self):
        import torch
        from torch import nn
        from kernelatlas.precision import quantize_linears

        if not torch.backends.mps.is_available():
            self.skipTest("Requires MPS")
        torch.manual_seed(42)
        for mode in ("int4", "int8"):
            with self.subTest(mode=mode):
                # Exactly representable weights expose incorrect nibble order, scales,
                # offsets, constant groups, and zero rows independently of quantization loss.
                codes = torch.randint(0, 16, (32, 64)).float()
                codes[:, ::32] = 0
                codes[:, 1::32] = 15
                if mode == "int8":
                    codes = codes * 8 + 7
                codes[0] = 0
                codes[1] = 7
                model = nn.Module()
                model.proj = nn.Linear(64, 32)
                model.proj.weight = nn.Parameter(codes)
                model.embed = nn.Embedding(32, 64)
                model.tied = nn.Linear(64, 32, bias=False)
                model.tied.weight = model.embed.weight
                model.unaligned = nn.Linear(31, 9)
                model.direct = nn.Linear(64, 32)
                model.to("mps")
                x = torch.randn(3, 4, 64, device="mps").transpose(0, 1)
                ref = model.proj(x)

                def exercise():
                    model.proj(x)
                    model.direct(x.to(model.direct.weight.dtype))

                weights = quantize_linears(model, mode, exercise=exercise)
                torch.testing.assert_close(model.proj(x), ref, rtol=1e-4, atol=1e-3)
                self.assertEqual(weights["converted_linears"], ["proj"])
                self.assertEqual(
                    set(weights["skipped_linears"]), {"tied", "unaligned", "direct"}
                )
                self.assertEqual(
                    model.proj.packed.nbytes,
                    codes.numel() // (2 if mode == "int4" else 1),
                )
                self.assertEqual(weights["packed_weight_elements"], codes.numel())
