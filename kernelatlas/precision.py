"""Floating-point inference precision and optional packed MPS linear weights."""

from collections import Counter

import torch
from torch import nn


def resolve_dtype(device, requested):
    requested = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}.get(
        requested, requested
    )
    choices = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    if requested != "auto" and requested not in choices:
        raise ValueError(f"Unknown dtype {requested}")
    if requested in ("fp16", "fp32"):
        return choices[requested]
    if device == "mps":
        supports_bf16 = torch.backends.mps.is_macos_or_newer(14, 0)
    else:
        supports_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    if requested == "bf16" and not supports_bf16:
        raise ValueError(f"BF16 is unavailable on {device}. Try --dtype fp16.")
    return torch.bfloat16 if supports_bf16 else torch.float16


class PackedLinear(nn.Module):
    """Inference-only MPS matmul with per-row INT8 or groupwise INT4 weights."""

    def __init__(self, source, mode):
        super().__init__()
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.bias = source.bias
        weight = source.weight.detach().float()
        dtype = source.weight.dtype
        if mode == "int8":
            scale = weight.abs().amax(dim=1).div(127)
            divisor = torch.where(scale == 0, 1.0, scale)
            packed = (weight / divisor[:, None]).round().clamp(-127, 127).to(torch.int8)
            params = scale.to(dtype)
        else:
            groups = weight.reshape(self.out_features, -1, 32)
            low, high = groups.amin(dim=-1), groups.amax(dim=-1)
            scale = (high - low) / 15
            divisor = torch.where(scale == 0, 1.0, scale)
            values = (
                ((groups - low[..., None]) / divisor[..., None]).round().clamp(0, 15)
            )
            values = values.reshape_as(weight).to(torch.uint8)
            pairs = (values[:, ::2] << 4) | values[:, 1::2]
            packed = torch._convert_weight_to_int4pack(pairs, 2)
            params = (
                torch.stack((scale, low + 8 * scale), dim=-1)
                .transpose(0, 1)
                .contiguous()
                .to(dtype)
            )
        self.register_buffer("packed", packed)
        self.register_buffer("params", params)

    def forward(self, x):
        flat = x.reshape(-1, self.in_features).contiguous()
        params = self.params.to(x.dtype)
        if self.packed.dtype == torch.int8:
            out = torch._weight_int8pack_mm(flat, self.packed, params)
        else:
            out = torch._weight_int4pack_mm(flat, self.packed, 32, params)
        out = out.reshape(*x.shape[:-1], self.out_features)
        return out if self.bias is None else out + self.bias


class _ObservedLinear(nn.Linear):
    """Delegate the dry run and record any parent that accesses layer.weight."""

    def __init__(self, layer):
        nn.Module.__init__(self)
        self.training = layer.training
        self.layer = layer
        self.weight_read = False

    def forward(self, *args, **kwargs):
        return self.layer(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "weight":
                self.weight_read = True
            return getattr(super().__getattr__("layer"), name)


def _external_weights(model, names, exercise):
    probes = {}
    try:
        for name in names:
            probe = _ObservedLinear(model.get_submodule(name))
            probes[name] = probe
            model.set_submodule(name, probe)
        exercise()
        return {name for name, probe in probes.items() if probe.weight_read}
    finally:
        for name, probe in probes.items():
            model.set_submodule(name, probe.layer)


def quantize_linears(model, mode, *, exercise=None):
    """Pack eligible unshared dense Linear weights on MPS. Return coverage."""
    if mode not in ("int4", "int8"):
        raise ValueError(f"Unknown weight quantization {mode}")
    counts = Counter(id(p) for _, p in model.named_parameters(remove_duplicate=False))
    converted, skipped = [], {}
    model_elements = sum(p.numel() for p in model.parameters())
    packed_elements = 0
    linears = [
        name for name, layer in model.named_modules() if isinstance(layer, nn.Linear)
    ]
    plain = [name for name in linears if type(model.get_submodule(name)) is nn.Linear]
    external = (
        _external_weights(model, plain, exercise) if exercise is not None else set()
    )
    for name in linears:
        layer = model.get_submodule(name)
        if type(layer) is not nn.Linear:
            skipped[name] = "custom linear implementation"
        elif name in external:
            skipped[name] = "weight accessed outside Linear.forward"
        elif counts[id(layer.weight)] > 1:
            skipped[name] = "shared weight"
        elif layer.in_features % 32 or layer.out_features % 32:
            skipped[name] = "requires input and output widths divisible by 32"
        elif layer.weight.device.type != "mps" or not layer.weight.is_floating_point():
            raise ValueError(
                "Weight-only INT4/INT8 conversion requires floating-point weights on MPS."
            )
        else:
            model.set_submodule(name, PackedLinear(layer, mode).eval())
            packed_elements += layer.weight.numel()
            converted.append(name)
    if not converted:
        raise ValueError(
            f"No eligible MPS Linear weights for {mode} conversion. Try omitting --quantization."
        )
    return {
        "mode": mode,
        "method": "pytorch_mps_weight_only",
        "group_size": 32 if mode == "int4" else None,
        "original_parameter_elements": model_elements,
        "packed_weight_elements": packed_elements,
        "converted_linears": converted,
        "skipped_linears": skipped,
    }
