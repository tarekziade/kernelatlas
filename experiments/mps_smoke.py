"""Exercise real Transformers prefill and cached decode with local random weights."""

import torch
import transformers
from transformers import LlamaConfig, LlamaForCausalLM


@torch.inference_mode()
def main():
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    model = LlamaForCausalLM(config).eval().to("mps")
    prompt = torch.arange(32, device="mps")[None, :]
    prefill = model(prompt, use_cache=True)
    cache = prefill.past_key_values
    prompt_length = cache.get_seq_length()
    next_token = prefill.logits[:, -1].argmax(-1, keepdim=True)
    decode = model(next_token, past_key_values=cache, use_cache=True)
    torch.mps.synchronize()
    assert prefill.logits.shape == (1, 32, 128)
    assert decode.logits.shape == (1, 1, 128)
    assert prompt_length == 32 and decode.past_key_values.get_seq_length() == 33
    assert decode.logits.device.type == "mps" and torch.isfinite(decode.logits).all()
    print(f"torch {torch.__version__}, transformers {transformers.__version__}")
    print(f"{decode.logits.device}: prefill 32 tokens, cached decode 1 token")


if __name__ == "__main__":
    main()
