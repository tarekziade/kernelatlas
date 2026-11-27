from importlib.metadata import PackageNotFoundError, version

from .atlas import Atlas
from .trials import best_model, run_trial

try:
    __version__ = version("kernelatlas")
except PackageNotFoundError:
    __version__ = "0+unknown"


def profile(model_id, *, compile=True, **kwargs):
    """Capture a Hub workload and compare Inductor recipes at its original cache and precision."""
    from .profiling import profile_hub_model, _quiet
    from .compilation import compare_compilation
    from .trials import _write

    out, report = profile_hub_model(model_id, **kwargs)
    report["compilation"] = (
        compare_compilation(out, progress=kwargs.get("progress", _quiet))
        if compile
        else {"selected": "eager", "recipe": None, "trials": [], "disabled": True}
    )
    _write(out / "report.json", report)
    return out, report


__all__ = ["profile", "Atlas", "best_model", "run_trial"]
