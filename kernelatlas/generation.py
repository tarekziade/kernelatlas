"""Fixed-length greedy model timing, including token selection and cache updates."""

from math import ceil
from statistics import median
from time import perf_counter_ns


def decode_distribution(samples):
    values = sorted(value for row in samples for value in row["cached_step_ms"])
    if values:
        return {
            "count": len(values),
            "median_ms": median(values),
            "p90_ms": values[ceil(0.9 * len(values)) - 1],
        }
    return {"count": 0, "median_ms": None, "p90_ms": None}


def decode_step_savings(blocks):
    """Median paired milliseconds saved on the first step and the sum of later steps."""
    first, later = [], []
    for block in blocks:
        baseline = block["incumbent"]["cached_step_ms"]
        candidate = block["candidate"]["cached_step_ms"]
        first.append(baseline[0] - candidate[0])
        if len(baseline) > 1:
            later.append(sum(baseline[1:]) - sum(candidate[1:]))
    return {
        "first_cached_step_ms": median(first),
        "remaining_cached_steps_ms": median(later) if later else None,
    }


def trajectory(
    model,
    ids,
    cached_tokens,
    cache_key,
    kwargs,
    *,
    synchronize=None,
    on_step=None,
    cache_factory=None,
    migrate_cache=None,
):
    """A fresh cache per trajectory. Clocked mode synchronizes after each token."""
    if cache_factory is not None and migrate_cache is not None:
        raise ValueError("Choose one cache initialization point")
    candidate_cache = None
    inputs = {"input_ids": ids}
    samples = []
    logits = []
    if synchronize:
        synchronize()
        start = previous = perf_counter_ns()
    if cache_factory is not None:
        candidate_cache = inputs[cache_key] = cache_factory()
    for step in range(cached_tokens + 1):
        if on_step:
            on_step(step)
        if step == 1 and migrate_cache is not None:
            candidate_cache = inputs[cache_key] = migrate_cache(inputs[cache_key])
        output = model(**inputs, use_cache=True, **kwargs)
        last = output.logits[:, -1:]
        token = last[:, 0].argmax(-1, keepdim=True)
        if step < cached_tokens:
            inputs = {"input_ids": token, cache_key: getattr(output, cache_key)}
        if synchronize:
            synchronize()
            now = perf_counter_ns()
            samples.append((now - previous) / 1e6)
            previous = now
        else:
            logits.append(last)
    if (
        candidate_cache is not None
        and getattr(output, cache_key) is not candidate_cache
    ):
        raise ValueError("Model did not retain the supplied cache")
    if synchronize:
        return {
            "prefill": samples[0],
            "decode": sum(samples[1:]),
            "total": (now - start) / 1e6,
            "cached_step_ms": samples[1:],
        }
    return tuple(logits)
