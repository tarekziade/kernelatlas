"""Config-only BF16 text workloads verified on Apple Silicon."""

REVISIONS = {
    "Qwen/Qwen3.8-27B": "37561816e66b5b9153b1b64c10968c18c1e6f4a0",
    "thinkingmachines/Inkling-2-Mini": "c9d028302cd0f035fdf8e7e91f089a4ce94d2195",
}


def text_proxy(model_id, original):
    """Build the measured decoder shape from the checkpoint's text config."""
    settings = dict(vocab_size=256, bos_token_id=1, eos_token_id=2)
    if model_id == "Qwen/Qwen3.8-27B":
        indices = [0, 1, 2, 3]
    elif model_id == "thinkingmachines/Inkling-2-Mini":
        # One sliding/full pair before, within and after the three-pass core.
        indices = [0, 5, 6, 11, 18, 23]
        settings.update(
            unpadded_vocab_size=256,
            n_routed_experts=16,
            loop_layer_index_start=2,
            loop_layer_index_end=4,
            n_layers_per_attn_res_block=2,
        )
    else:
        raise ValueError(f"No reduced text workload for {model_id}")
    if len(original.layer_types) <= max(indices):
        raise ValueError(
            f"Revision of {model_id} has too few layers for the reduced text workload"
        )
    settings.update(
        num_hidden_layers=len(indices),
        layer_types=[original.layer_types[i] for i in indices],
    )
    config = original.to_dict()
    # The Inkling checkpoint uses a nested legacy representation on load.
    config.pop("loop_config", None)
    config.update(settings)
    text = type(original).from_dict(config)
    for key, value in settings.items():
        if getattr(text, key) != value:
            raise ValueError(f"Unexpected effective configuration for {key}")
    return text, {
        "scope": "random BF16 text decoder only, reduced layers and vocabulary",
        "reference_layer_indices": indices,
        "original": {key: getattr(original, key) for key in settings},
        "effective": settings,
    }
