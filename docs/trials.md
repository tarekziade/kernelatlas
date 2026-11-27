# Local optimization trials

Use the profile directory as the durable session identifier. Both the CLI and `kernelatlas.run_trial(folder, candidate=None, *, replay=None, repeat=False)` support preparing a session, reading its history and running candidates. Eligible profiles come from pretrained dense BF16 Llama and Qwen3 causal models on MPS. The adapter checks the trace against the installed Transformers MLP class and every observed gate site.

## Candidate contract

Start with the generated `optimization/reference.py` or the included Metal example. Define `candidate(gate, up)` in one self-contained file. Each call receives two equal-shape, contiguous BF16 tensors on MPS. Return a fresh tensor equal to `torch.nn.functional.silu(gate) * up`, with the same shape, device and dtype. Preserve both inputs. Include any shader source in the file so the archived copy can reproduce the implementation. Setup such as shader compilation can happen at import. A trial runs local Python in a separate process.

The first inspect or submit command prepares `optimization/task.json` and CPU copies of the real gate inputs, second projection inputs and reference output at one site for prefill and decode. Use `torch.load(path, weights_only=True)` to inspect a fixture. The task records its site names, source, tensor shapes, strides, model revision, attention implementation, runtime, prompt, objective and decision thresholds. Existing profiles can be prepared directly.

## Validation and timing

Each trial reloads the pinned model. Generation-shaped forwards pass `logits_to_keep=1`. With either length flag, the profile measures fixed-length greedy generation. Defaults for the unspecified flag are 32 prompt tokens and two generated tokens. Prompt IDs are sequential modulo vocabulary size. The first token comes from prefill, so a budget of one skips decode and 32 generated tokens use 31 cached forwards. All trajectories start with a fresh cache and run to the requested length even if EOS is selected. Profiles without either flag keep the original full-logit forward measurements and their 32-token plus one-cached-forward trial.

The trial pins both lengths, IDs, model revision, runtime and policy in `optimization/task.json`. Changing either length requires a fresh profile directory with its own incumbent. Three validation prompts (the timed prompt, seeded IDs of the same length and up to seven tokens from the timed prompt) each run the whole trajectory. The diagnostic pass checks the candidate outputs, input preservation and invocation counts at every site. It also compares last-position logits at every generated position to the original model. The first failure stops timing and saves the input tensors or prompt and logits under the trial directory.

Timing uses the sequential prompt. Six warmup blocks precede 24 measured blocks. Each block runs the current best twice and the candidate once. All six orderings cycle evenly. The fixed-length policy synchronizes after every greedy selection, including the last. Per-token median and nearest-rank p90 pool the cached steps across measured trajectories. Timing includes token selection and model execution. Tokenization, text output and generation processors run outside this benchmark. The diagnostic traces cover prefill and the first cached forward. Raw timings cover every step. Legacy sessions retain their original synchronization and token selection boundaries. Timing excludes setup, instrumentation and candidate installation. The objective is the sum of both model phases. This measures steady-state model latency on this workload.

For each phase and the sum, the required margin is the larger of 1% of current-best median time and the median absolute difference between the two unchanged measurements. At least 18 of 24 paired differences must exceed the margin in a consistent direction. These are engineering defaults for this experiment. Raw paired samples are available to evaluate the decision.

| Verdict | Action |
| --- | --- |
| `incorrect` | Inspect the first failing site, phase and fixture; correct the candidate and resubmit. |
| `error` | Read the reason and worker log for execution, import or environment errors. |
| `slower` | Retain the incumbent following a repeatable regression in either phase or the total. |
| `inconclusive` | Retain the incumbent. Use `--repeat` for a fresh measurement if warranted. |
| `faster` | A new submission with a repeatable overall gain and no repeatable phase regression becomes the incumbent. Its generation-shaped prefill dataflow is captured again. |

## Resume or replay

Read `kernelatlas trial "$RUN" --format json` for the best configuration, archived candidate sources, previous outcomes and evidence paths relative to `$RUN/optimization`. Trial directories include a request, a copy of the candidate, its SHA-256, the worker log and the result. The winner trace records observed PyTorch operations; custom Metal dispatch timing remains unavailable. This version optimizes one gate across all eligible sites. The next candidate replaces that same gate.

An identical source against the same incumbent reuses its completed result. `--repeat` measures against today's incumbent. `--replay ID` reruns the archived candidate against its original incumbent and preserves the current selection. Measurements can vary. Inspect and submit commands serialize changes to a local session and incomplete runs are recorded on resume. Evaluated negative outcomes are returned as data; CLI invocation errors return a nonzero exit code.

To use the selected model:

```python
import torch
from kernelatlas import best_model

with torch.inference_mode(), best_model(run_directory) as model:
    tokens = model.generate(input_ids, max_new_tokens=2, do_sample=False)
```

Provide token IDs on the model's device. The saved performance evidence applies to the pinned workload in the task.

## Static cache trial

A profile with at least one cached step captures tensor dependencies in the first cached forward. KernelAtlas records an append when the prior layer key or value is the first input to sequence-axis concatenation with one new position and the returned cache retains the result in that slot. The byte count is the logical size of those outputs. Cache layouts that hide their tensors can still provide operation pairs.

For a pinned pretrained dense BF16 MPS generation profile with observed appends, run `kernelatlas trial "$RUN" --static-cache`. Each trajectory constructs a fresh Transformers StaticCache sized to the prompt plus the cached steps. The comparison uses the same three correctness prompts, phase regression checks and paired timing policy as the gate trial. Allocation and mask changes count toward the candidate’s time. Each invocation saves a new measurement in `optimization/cache-trials/`. Offline discovery shows the history matching the saved report and decode graph. Gate trials keep their own incumbent.

Use `kernelatlas trial "$RUN" --decode-static-cache` to leave the prompt on the default cache and reserve capacity at the first cached step. It copies each initialized full-attention layer’s keys and values into a fresh Transformers StaticCache. The first cached step and whole-generation timings include allocation, copying, and the forward itself. The raw blocks preserve individual step times. Reports distinguish the two reservation points and compare each with the unmodified default.


## Hub loading comparisons

Hub loading and compiler trials compare every returned logit with the native model and require the same greedy token at each tested step. For BF16, the elementwise check is `torch.isclose(candidate, native, atol=0.01, rtol=0.016)`. A native logit of zero allows an absolute difference of 0.01. At magnitude 1 the allowance is 0.026, and at magnitude 10 it is 0.17.

The relative term follows PyTorch’s BF16 testing default. The absolute floor is a proposed end-to-end regression threshold in logit units. Its relationship to generation quality or perplexity has not been calibrated. RMSNorm’s epsilon is a separate model parameter used inside the normalization. New comparison records store which side supplies the reference magnitude. Archived Hub and compiler trials without that field used 0.02 near zero for BF16 and scaled against the candidate. The local gate-kernel trial retains its exact checks. [PyTorch testing tolerances](https://docs.pytorch.org/docs/stable/testing.html#torch.testing.assert_close)
