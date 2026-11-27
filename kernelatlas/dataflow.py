"""Capture eager ATen tensor dependencies without retaining intermediate tensors."""

from contextlib import ExitStack
from weakref import ref

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves


class DataflowTrace(TorchDispatchMode):
    def __init__(self, cached=None):
        super().__init__()
        self.producers = {}
        self.nodes = []
        self.scopes = []
        self.cached = cached or {}
        self.appends = {}

    def inputs(self, value):
        edges = []
        for tensor in tree_leaves(value):
            if isinstance(tensor, torch.Tensor):
                entry = self.producers.get(id(tensor))
                if entry is not None and entry[0]() is tensor:
                    edges.append(entry[1])
        return edges

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        edges = self.inputs((args, kwargs))
        result = func(*args, **kwargs)
        schema = func._schema
        node_id = len(self.nodes)
        outputs = []
        for tensor in tree_leaves(result):
            if not isinstance(tensor, torch.Tensor):
                continue
            self.producers[id(tensor)] = (ref(tensor), (node_id, len(outputs)))
            outputs.append(
                {
                    "shape": list(tensor.shape),
                    "stride": list(tensor.stride())
                    if tensor.layout == torch.strided
                    else None,
                    "dtype": str(tensor.dtype),
                    "device": str(tensor.device),
                    "logical_bytes": tensor.numel() * tensor.element_size(),
                    "floating": tensor.is_floating_point(),
                }
            )
        if self.cached and func == torch.ops.aten.cat.default:
            parts = args[0]
            dim = kwargs.get("dim", args[1] if len(args) > 1 else 0)
            if len(parts) == 2 and id(parts[0]) in self.cached and parts[0].ndim == 4:
                layer, slot, original = self.cached[id(parts[0])]
                if (
                    original() is parts[0]
                    and dim in (-2, 2)
                    and parts[1].ndim == 4
                    and parts[1].shape[-2] == 1
                ):
                    self.appends[node_id] = {
                        "layer": layer,
                        "slot": slot,
                        "node": node_id,
                        "previous_shape": list(parts[0].shape),
                        "added_shape": list(parts[1].shape),
                    }
        self.nodes.append(
            {
                "op": str(func),
                "scope": self.scopes[-1] if self.scopes else "",
                "pure_tensor_result": not schema.is_mutable
                and all(r.alias_info is None for r in schema.returns)
                and torch.Tag.nondeterministic_seeded not in func.tags,
                "pointwise": torch.Tag.pointwise in func.tags,
                "inputs": edges,
                "outputs": outputs,
            }
        )
        return result


def cache_tensors(cache):
    for index, layer in enumerate(getattr(cache, "layers", ())):
        for slot in ("keys", "values"):
            value = getattr(layer, slot, None)
            if isinstance(value, torch.Tensor):
                yield index, slot, value


def capture(model, prompt, *, cache_argument=None, **forward_kwargs):
    from .integrations import binding, integrations

    cached = {
        id(tensor): (layer, slot, ref(tensor))
        for layer, slot, tensor in cache_tensors(forward_kwargs.get(cache_argument))
    }
    trace = DataflowTrace(cached)
    sites = {}
    with ExitStack() as hooks:
        for name, module in model.named_modules():

            def enter(mod, args, scope=name):
                trace.scopes.append(scope)
                observed = binding(mod)
                if observed:
                    sites[scope] = observed

            def leave(mod, args, output):
                trace.scopes.pop()

            hooks.callback(module.register_forward_pre_hook(enter).remove)
            hooks.callback(module.register_forward_hook(leave, always_call=True).remove)
        with trace:
            result = model(prompt, use_cache=True, **forward_kwargs)
    returned = trace.inputs(result)
    retained_appends = []
    if cache_argument:
        for layer, slot, tensor in cache_tensors(getattr(result, cache_argument, None)):
            for edge in trace.inputs(tensor):
                if edge not in returned:
                    returned.append(edge)
                append = trace.appends.get(edge[0])
                if append and (append["layer"], append["slot"]) == (layer, slot):
                    retained_appends.append(append)
    return {
        **({"retained_cache_appends": retained_appends} if cache_argument else {}),
        "schema_version": 1,
        "nodes": trace.nodes,
        "returned": returned,
        "integrations": integrations(sites),
    }
