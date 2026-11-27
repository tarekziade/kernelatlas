"""End-to-end model regression policy for logits relative to the native output."""


def model_logit_policy(dtype):
    rtol, atol = {
        "bfloat16": (0.016, 0.01),
        "float16": (0.001, 0.001),
    }.get(str(dtype).removeprefix("torch."), (1e-4, 1e-5))
    return {"rtol": rtol, "atol": atol, "relative_to": "native"}


def logits_close(candidate, native, policy):
    import torch

    return torch.isclose(candidate, native, rtol=policy["rtol"], atol=policy["atol"])
