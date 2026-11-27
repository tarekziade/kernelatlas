"""Read literal integration declarations from the installed Transformers sources."""

import ast
from functools import lru_cache
import inspect
from pathlib import Path


@lru_cache(maxsize=128)
def _declaration(cls):
    if not cls.__module__.startswith("transformers."):
        return None
    try:
        lines, first = inspect.getsourcelines(cls)
        node = ast.parse("".join(lines)).body[0]
    except (OSError, TypeError, SyntaxError):
        return None
    for dec in getattr(node, "decorator_list", []):
        if (
            isinstance(dec, ast.Call)
            and isinstance(dec.func, ast.Name)
            and dec.func.id == "use_kernel_forward_from_hub"
            and dec.args
            and isinstance(dec.args[0], ast.Constant)
            and isinstance(dec.args[0].value, str)
        ):
            return dec.args[0].value, f"{cls.__module__}:{first + dec.lineno - 1}"
    return None


def binding(module):
    cls = type(module)
    decl = _declaration(cls)
    repo = vars(module).get("_kernel_repo")
    if not (decl or repo):
        return None
    forward = getattr(module.forward, "__func__", module.forward)
    repo_id = getattr(repo, "_repo_id", None)
    # A missing marker alone cannot distinguish eager from an unknown wrapper.
    original = (
        getattr(forward, "__module__", None) == cls.__module__
        and getattr(forward, "__qualname__", None) == f"{cls.__qualname__}.forward"
        and forward is cls.__dict__.get("forward")
    )
    return {
        "layer": decl[0] if decl else getattr(repo, "layer_name", None),
        "source": decl[1] if decl else None,
        "forward": f"{getattr(forward, '__module__', '')}.{getattr(forward, '__qualname__', '')}",
        "binding": "hub_bound"
        if repo_id
        else "original_forward"
        if original and repo is None
        else "unknown",
        "bound_repo": repo_id,
        "bound_symbol": getattr(repo, "layer_name", None),
    }


def mapping_declarations(path, layers):
    """Keep explicit device and mode restrictions, ignoring dynamic mapping expressions."""
    entries = []
    for node in ast.walk(ast.parse(path.read_text())):
        if not (
            isinstance(node, (ast.Assign, ast.AnnAssign))
            and isinstance(node.value, ast.Dict)
        ):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(
            isinstance(t, ast.Name) and t.id == "_KERNEL_MAPPING" for t in targets
        ):
            continue
        for key, devices in zip(node.value.keys, node.value.values):
            if (
                not isinstance(key, ast.Constant)
                or key.value not in layers
                or not isinstance(devices, ast.Dict)
            ):
                continue
            for device, modes in zip(devices.keys, devices.values):
                if not isinstance(device, ast.Constant) or not isinstance(
                    device.value, str
                ):
                    continue
                cases = (
                    zip(modes.keys, modes.values)
                    if isinstance(modes, ast.Dict)
                    else [(None, modes)]
                )
                for mode, repo in cases:
                    if (
                        not isinstance(repo, ast.Call)
                        or ast.unparse(repo.func) != "LayerRepository"
                    ):
                        continue
                    values = {
                        k.arg: k.value.value
                        for k in repo.keywords
                        if isinstance(k.value, ast.Constant)
                    }
                    if "repo_id" in values and "layer_name" in values:
                        entries.append(
                            {
                                "layer": key.value,
                                "device": device.value,
                                "mode": ast.unparse(mode) if mode else None,
                                "source": f"transformers/integrations/hub_kernels.py:{repo.lineno}",
                                **values,
                            }
                        )
    return entries


def integrations(sites):
    if not sites:
        return {"sites": {}, "declarations": []}
    from transformers.integrations import hub_kernels

    layers = {s["layer"] for s in sites.values()}
    try:
        declarations = mapping_declarations(Path(hub_kernels.__file__), layers)
    except (TypeError, OSError, SyntaxError):
        declarations = []
    return {"sites": sites, "declarations": declarations}
