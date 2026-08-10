---
name: audio-pipeline-e2e
description: Author the end-to-end audio pipeline for a multi-component TTS / voice-cloning / music-generation model after its components have been brought up individually — the audio analogue of the image-gen pipeline template. Writes pipeline.py in tt-forge-models chaining the learned nn.Modules on Tenstorrent (speaker/conditioning encoders, autoregressive audio-code loop over a single-compile static-KV-cache step, vocoder) while keeping tokenizer, mel front-end, chunking and sampling on the host, plus the tt-xla e2e test and a WAV-emitting demo. Use after model-bringup-scaffold-audio archetype 6 components pass, or when the user asks for "a real text-to-speech run", an audio demo, or an e2e audio pipeline on device.
allowed-tools: Bash Read Write Edit Grep Glob
---

# audio-pipeline-e2e — end-to-end audio on device

Component PCC is not a bringup for a generative audio model. The deliverable is
a real text → waveform run on Tenstorrent hardware that produces a listenable
WAV.

## Invocation
`/audio-pipeline-e2e <family> [--max-tokens N] [--arch <arch>]`

## Prerequisites

- `third_party/tt_forge_models/<family>/pytorch/loader.py` exposes one variant
  per learned `nn.Module` (from `model-bringup-scaffold-audio`, archetype 6).
- `src/model.py` holds the component wrappers, including the cached decode step.
- Each component passes its single-device PCC test, or is xfailed with a
  recorded OOM.

If components have not been brought up, stop and run
`model-bringup-scaffold-audio` first — chaining components that have never
passed individually gives you a failure you cannot localise.

## Why a pipeline module, not a test

`pipeline.py` lives in **tt-forge-models**, next to the loader, and is imported
by the tt-xla e2e test, the demo script and any benchmark entry. All model-shaping
code belongs upstream; the tt-xla side stays a thin driver. This mirrors the
image-gen pipeline convention in `add-benchmark-model`.

## The split: what runs where

| Host (CPU) | Device (TT) |
|---|---|
| tokenizer / text normalisation | speaker encoder |
| resampling, mel / STFT front-end, chunking | conditioning encoder + perceiver |
| logits processors + sampling, RNG | AR audio-code decode step |
| loop control, stop-token check | full-sequence latents forward |
| WAV writing | vocoder |

Fixed DSP and orchestration gain nothing from the device and would make graphs
dynamic. Everything learned runs on TT.

## Structure

```python
class <Family>Config:
    def __init__(self, text=..., language=..., speaker_wav=None,
                 max_audio_tokens=None, seed=0): ...

class <Family>Pipeline:
    def setup(self):  # build model once, wrap components, compile(backend="tt")
    def run(self) -> torch.Tensor:  # text -> waveform

def save_wav(wav, filepath, sample_rate): ...
def run_<family>_pipeline(output_path=..., **cfg) -> torch.Tensor: ...
```

`setup()` builds the full model once via the loader's own builder
(`self._loader._build_<model>()`), wraps each component with the **same**
wrapper classes the component tests use, and calls `.compile(backend="tt")` on
each. Reusing the wrappers is the point: the pipeline exercises the graphs that
were PCC-verified, not lookalikes.

## Memory discipline — move components on and off

Components do not all fit resident at once. Move each to the device for its
stage and back to CPU immediately after:

```python
self.speaker_encoder = self.speaker_encoder.to(device)
speaker_embedding = self.speaker_encoder(mel.to(device)).to("cpu")
self.speaker_encoder = self.speaker_encoder.to("cpu")
```

This is the same CPU→TT→CPU eviction discipline the image-gen pipelines use.
Skipping it turns a working pipeline into an OOM at the vocoder stage.

## The autoregressive loop

The single largest correctness/perf risk. Requirements:

1. **One graph, compiled once.** Constant input shapes every step: one token id,
   one position, a fixed `(1, max_cache_len)` mask, a static KV cache.
   Varying shapes means a recompile per token and a pipeline that never finishes.
2. **Cache built on CPU, moved manually.** Construct `StaticCache` +
   `early_initialization` on CPU, then move `keys` / `values` /
   `cumulative_length` to the device (tt-xla#1645).
3. **Sampling on the host, matching the reference.** Use HF's own processors in
   `generate()`'s order so the output distribution matches the reference
   implementation:

```python
processors = LogitsProcessorList([
    RepetitionPenaltyLogitsProcessor(penalty=REF_REPETITION_PENALTY),
    TemperatureLogitsWarper(REF_TEMPERATURE),
    TopKLogitsWarper(REF_TOP_K),
    TopPLogitsWarper(REF_TOP_P),
])
rng = torch.Generator().manual_seed(int(self.config.seed))
```

   Take the sampling parameters from the reference implementation's defaults and
   name them as constants with a comment pointing at the source. The running
   token sequence drives the repetition penalty, exactly as HF does it.
4. **Seeded.** A `seed` in the config so runs are reproducible despite sampling.
5. **Bounded.** `max_audio_tokens` capped at the model's own maximum; log
   progress every N tokens (`DECODE_LOG_INTERVAL`) — these loops run for
   hundreds of steps and a silent pipeline looks hung.
6. **Prefill then decode.** Prefill `[prefix, START]` in one call, then one token
   per step with `position = step`; break on the stop token.

## Faithfulness to the reference path

Mirror the reference implementation's preprocessing exactly, including the parts
that look like details:

- chunking (`gpt_cond_len` / `gpt_cond_chunk_len`) and the mean over chunk
  embeddings;
- the minimum-chunk-length skip (chunks under ~0.33 s are dropped);
- mel parameters and the model's own `mel_stats`;
- output shape fixups (e.g. `unsqueeze(-1)` so a speaker embedding matches the
  reference's `[1, 512, 1]`).

Name each of these as a module constant with a comment naming the reference
function. When output quality is off, this list is the first thing to re-check —
before suspecting the compiler.

## Deliverables

1. **`pipeline.py`** in `third_party/tt_forge_models/<family>/pytorch/`.
2. **tt-xla e2e test** under `tests/torch/models/<family>/` that runs the
   pipeline and asserts the waveform is finite, non-silent, and of the expected
   duration. Set `EXCLUDE_MODEL` on the runner YAML entries for these variants.
3. **Demo script** that writes a WAV via the stdlib `wave` writer (no FFmpeg
   dependency) and logs the path + sample count.
4. **Decode-loop test** exercising the cached step directly — it is the piece
   most likely to regress, and it is cheap to test in isolation.

Optionally, a benchmark entry — hand off to `add-benchmark-model` (audio goes
through the multimodal harness; `test_gemma4_12b_audio` is the only existing
audio entry and is the template).

## Verification

Before declaring done:

- The WAV is audible speech/music, not noise. PCC alone does not establish this.
- The decode loop compiled **once** — check the log for a single compile of the
  step graph.
- The run is reproducible: same seed → same token count and same waveform.
- Set `torch_xla.set_custom_compile_options({"optimization_level": 0})` for
  first-light, then raise it deliberately and re-verify.

Report the token count, wall-clock, output duration and sample rate. If quality
is degraded relative to a CPU run of the same pipeline, localise it by swapping
one stage back to CPU at a time — the per-component tests already tell you each
stage's PCC in isolation, so a discrepancy is in the chaining, the host-side
preprocessing, or the sampling stack.
