"""Check the cached official Metal RMSNorm build against the eager FP32 contract."""

import importlib.metadata
import json
from datetime import datetime, timezone

from huggingface_hub import snapshot_download
from pathlib import Path
import kernels
import torch

from evaluate_rmsnorm import reference


@torch.inference_mode()
def probe():
    kernel_id = "kernels-community/mlx-rmsnorm"
    revision = "d5793d964608e8e1ee2c4de42e83341b72e5f3dc"
    snapshot = Path(
        snapshot_download(
            kernel_id,
            revision=revision,
            repo_type="kernel",
            allow_patterns=["build/torch215-metal-aarch64-darwin/*"],
            local_files_only=True,
        )
    )
    kernel = kernels.get_local_kernel(snapshot, backend="metal")
    torch.manual_seed(0)
    rows = []
    for width in (16, 64, 576):
        w = torch.randn(width, device="mps")
        for length in (1, 32):
            for step in (1, 2):
                x = torch.randn(2, length, width * step, device="mps")[..., ::step]
                expected = reference(x, w, 1e-5)
                row = {"shape": list(x.shape), "stride": list(x.stride())}
                try:
                    actual = kernel.rmsnorm_forward(x, w, 1e-5)
                    torch.mps.synchronize()
                    row.update(
                        passed=torch.allclose(actual, expected, rtol=1e-5, atol=1e-6),
                        max_abs_error=(expected - actual).abs().max().item(),
                    )
                except (RuntimeError, ValueError) as error:
                    row.update(passed=False, error=str(error))
                rows.append(row)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kernel_id": kernel_id,
        "kernel_revision": revision,
        "torch": torch.__version__,
        "kernels": importlib.metadata.version("kernels"),
        "backend": "metal",
        "dtype": "float32",
        "eps": 1e-5,
        "seed": 0,
        "rtol": 1e-5,
        "atol": 1e-6,
        "cases": rows,
    }


if __name__ == "__main__":
    print(json.dumps(probe(), indent=2))
