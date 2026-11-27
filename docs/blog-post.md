# KernelAtlas ~ chasing the best kernel combo with Transformers

When I worked on [making Qwen3.8-Flash-Next run on my
laptop](https://huggingface.co/blog/moe-expert-streaming), I spent a lot of time
trying to find the fastest combination of kernels.

I asked my AI agent to search the Hub, try a few combos and pick the fastest.
Some ran faster but changed the model's output. My agent ended up writing custom
Metal kernels, including one for Gated DeltaNet.

A lot of what the agent did was searching and running trials. A script could do
that part just as well. So I wondered whether a deterministic script could take
a model, find a good kernel combo and suggest which kernel to write next. By
deterministic I mean it runs without an LLM. The idea reminds me a bit of
Optimum, but I only wanted to know which kernels to use on this GPU, and which
ones were missing.

If you haven't dealt with kernels before: a kernel is a small program the GPU
runs to do a piece of the model's work, like multiplying two matrices. Sometimes
you can save time by doing two operations in one kernel.

Transformers keeps a mapping of registered operations to kernels for each
device. Set `use_kernels=True` and it picks from that mapping. You can also pass
a `KernelConfig` to choose the ones you want, and select attention separately
with `attn_implementation`.

I wrote [KernelAtlas](https://github.com/tarekziade/kernelatlas) to be that
script. It profiles the model, checks which kernels Transformers actually calls
and searches the Hub for candidates. For each combo it tests, it checks the
logits against native Transformers before timing the whole model. "Best" is the
fastest combo that passed on that workload.

I left kernel writing to the agent. KernelAtlas gives it the tensor shapes, the
reference behavior and a trial it can rerun, and the agent can bring back a
Python file and see how it does.

## Qwen3-0.6B on my M5

Here's what I ran.

```sh
kernelatlas profile Qwen/Qwen3-0.6B --device mps
```

One prompt, 32 tokens, pretrained BF16 weights and one cached step. The Hub
search found 16 leads across 651 repos for eight pairs of operations in the
trace. Two of those repos advertised Metal support.

Automatic and selective loading both picked the same RMSNorm,
`kernels-community/mlx-rmsnorm`, at 113 sites. Qwen still picked the same next
token, but about 1.77 million of the 4.86 million prefill logits fell outside
tolerance. The attention-only and combined configs failed too, so KernelAtlas
skipped timing all four.

## The RMSNorm that drifted

When I showed this to Arthur (it's always good to have an expert around!), he
suspected RMSNorm's epsilon first. But all 113 Metal calls got Qwen's value of
`1e-6`. As a control, I gave the first layer's input norm `1e-5` instead, and
28,902 of its 32,768 values changed.

I then replayed the norms on identical inputs. The first difference I could
reproduce was one BF16 value in the second decoder layer's `q_norm`. Switching
just RMSNorm back to native made all 4.86 million prefill logits match exactly.

I still don't know which arithmetic step caused the difference, or whether it
affects answer quality. [Epsilon
experiment](https://github.com/tarekziade/kernelatlas/blob/main/evidence/qwen3-06b-rmsnorm-epsilon-m5.json)

## What I ended up with

The best combo uses no Hub kernel: `use_kernels=False`, native attention (SDPA),
and the 28 MLPs compiled with Inductor. That saved a median 4.54 ms over eager
across 24 paired measurements of the prompt plus cached step. The run writes a
`best_combo.py` I can use directly. [Measured
configurations](https://github.com/tarekziade/kernelatlas/blob/main/evidence/hub-loading/qwen3-06b-m5-parity.json)

## SiLU and multiply

Those MLPs are also where Qwen applies SiLU and then multiplies the result by
another tensor. That happens 28 times per forward pass. The Hub already has a
Metal function called `silu_and_mul` in `kernels-community/activation`. Next I
want to check whether it can take Qwen's two separate tensors. If I have to pack
them first, does it still save time? I can replay the real inputs, check the
logits and time the model with packing included.

Then I'd like to run the same prompt on NVIDIA with `--device cuda` and see
which kernels are worth using there.

In the meantime, ask your agent to give it a try!

```sh
pip install kernelatlas
```
