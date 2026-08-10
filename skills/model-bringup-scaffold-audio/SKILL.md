---
name: model-bringup-scaffold-audio
description: VALIDATE-stage scaffold variant for audio and speech models — TTS / vocoders / ASR / speech-translation / audio-classification / audio-LLMs / music generation. Where model-bringup-scaffold assumes one traceable forward fed by synthetic tensors, audio models break on three axes — inputs are real waveforms behind a feature extractor (sample rate, mel front-end, dataset/codec deps), many TTS models have no single traceable forward (tokenizer + AR code LM + vocoder), and the output is a spectrogram or waveform rather than logits. This skill classifies the model into one of six audio archetypes, picks the loader shape for that archetype (single forward, submodule slice, precomputed-embeds, or per-component variants), builds real audio inputs from the shared cached assets, and registers the right unpack_forward_output handler. Use whenever the target is an audio/speech model — ModelTask AUDIO_ASR / AUDIO_CLS / MM_TTS / MM_AUDIO_TTT, or anything whose processor is a FeatureExtractor over a waveform.
allowed-tools: Bash Read Write Edit Grep Glob AskUserQuestion
---

# Model Bringup — Audio / Speech Scaffold

You are the **audio-specialized** scaffold stage of the model bringup pipeline.
Use this skill *instead of* `model-bringup-scaffold` when the target is an
audio or speech model.

## Invocation
`/model-bringup-scaffold-audio <hf_repo_or_model_key> [--arch <arch>] [--archetype <name>]`

## Why this exists

The default scaffold writes one loader whose `load_inputs()` returns synthetic
tensors matching a `forward()` signature. Audio models break that in three
places:

1. **Inputs are not synthesizable.** A random `(1, 80, 3000)` tensor compiles
   fine and tells you nothing — mel bins carry structure, and a TTS/ASR model
   fed noise produces garbage that hides real PCC regressions. Audio inputs come
   from a real clip through a real feature extractor, and that pulls in sample
   rates, `datasets`, `torchaudio`, `soundfile`/`torchcodec`, and `ffmpeg` — the
   single largest source of `FAILED_FE_COMPILATION` on this family (see
   `references/audio_failure_catalog.md`).
2. **Many TTS models have no single traceable forward.** XTTS-v2, Pocket-TTS and
   the SpeechT5 stack are *pipelines in a trench coat*: tokenizer → conditioning
   encoder → autoregressive audio-code LM → vocoder. `load_model()` returning
   "the model" produces a graph that either does not trace at all or traces the
   `generate()` loop. These need per-component variants
   (`references/tts_component_decomposition.md`).
3. **The output is not logits.** TTS returns `Seq2SeqSpectrogramOutput`
   (`.spectrogram`), vocoders return a raw waveform tensor, ASR returns
   `Seq2SeqLMOutput`. `unpack_forward_output` must be told, or the training path
   fails with "No handler for class ... exists in `unpack_forward_output`".

Read `references/audio_model_inventory.md` first — 11 audio families are already
brought up, and the archetype you are scaffolding almost certainly has a
landed precedent to copy rather than invent.

---

## Step 0 — Classify into an archetype

Decide before writing anything. Ask: *what does one on-device forward consume
and produce?*

| # | Archetype | Signature | Loader shape | Reference model |
|---|-----------|-----------|--------------|-----------------|
| 1 | **Encoder-decoder ASR / speech-seq2seq** | `input_features` + `decoder_input_ids` → logits | single forward, thin wrapper | `whisper/pytorch` |
| 2 | **Audio classifier / CTC encoder** | waveform → logits | single forward, no wrapper | `wav2vec2/audio_classification/jax` |
| 3 | **TTS acoustic model** | text ids + `decoder_input_values` → spectrogram | single forward + spectrogram handler | `speecht5/pytorch` |
| 4 | **Vocoder** | spectrogram → waveform | single forward, synthetic input **is** legitimate | `vocoder_speecht5/pytorch` |
| 5 | **Audio-conditioned LLM** | audio tower + text merge → LM | precompute embeds on host, ship the LM | `voxtral/pytorch`, `llasa/causal_lm` |
| 6 | **Multi-component TTS / music stack** | no single traceable forward | one variant per learned `nn.Module` | `xtts_v2/pytorch`, `pocket_tts/pytorch` |

Rules for picking:

- If `AutoModel` loads it and `forward()` takes tensors end-to-end → 1–4.
- If the HF forward merges audio embeddings into text embeddings with
  `masked_scatter` / boolean-mask indexing → **5**, always. That op lowers to a
  dynamic-shape `set_dimension_size` the TT/Shardy compiler rejects. Do not try
  to repair it on-device; precompute host-side.
- If the only public entry point is `generate()` / `synthesize()` / `tts()`, or
  the package is not `transformers` (coqui-tts, `pocket_tts`, XCodec2) → **6**.
- Music generation (MusicGen, ACE-Step) is 1 or 6 depending on whether a single
  conditional-generation forward exists. MusicGen has one (`musicgen_small`);
  diffusion-based music DiTs do not.

If the classification is genuinely ambiguous after inspecting the forward
signature, use `AskUserQuestion` with the two candidate archetypes and what each
implies for scope — do not guess between 1 and 6, the cost difference is a day.

Record the archetype in `state.json` as `audio_archetype`.

---

## Step 1 — Inspect before writing

```bash
python -c "
from transformers import AutoConfig, AutoProcessor
import inspect
cfg = AutoConfig.from_pretrained('<hf_id>')
print(type(cfg), cfg)
p = AutoProcessor.from_pretrained('<hf_id>')
print(type(p), type(getattr(p, 'feature_extractor', None)))
print(getattr(getattr(p,'feature_extractor',None), 'sampling_rate', None))
"
```

Record: model class, forward signature, **sampling rate**, mel bins /
`num_mel_bins`, whether a vocoder is a separate checkpoint, and the output
dataclass name. The sampling rate is not decoration — feeding 22.05 kHz audio to
a 16 kHz front-end silently halves PCC and looks like a compiler bug.

For archetype 6, also list every learned submodule:

```bash
python -c "
import torch
m = ...  # build the model the package's own way
for n, mod in m.named_children():
    print(n, type(mod).__name__, sum(p.numel() for p in mod.parameters()))
"
```

---

## Step 2 — Choose the audio asset

Never download inside `forward()`, and never leave the input as
`torch.randn`. In order of preference:

1. **The shared cached tensor asset** —
   `get_file("test_files/pytorch/whisper/1272-128104-0000.pt")` (LibriSpeech,
   16 kHz). Already used by `whisper/pytorch` and `xtts_v2`; no `datasets`,
   `ffmpeg` or codec dependency. **Default choice.**
2. **`hf-internal-testing/librispeech_asr_dummy`** via `datasets`. Requires
   `datasets` in `requirements.txt`, `ffmpeg` in `system-requirements.txt`, and
   `datasets.config.AUDIO_BACKEND = "soundfile"` to dodge torchcodec.
3. **A pinned public URL** through `get_file(url)` — only when the model needs a
   specific clip (e.g. the Voxtral multi-clip conversation).

Resample explicitly with `torchaudio.functional.resample` when the asset rate
differs from the model rate; state both rates as named constants in the loader.
Full recipes and the codec-dependency escape hatches:
`references/audio_input_recipes.md`.

For archetype 4 (vocoders) a synthetic `torch.randn(1, T, n_mels)` spectrogram
**is** the correct input — the vocoder is a fixed function of its input and the
graph is what is under test. Say so in the docstring, as
`vocoder_speecht5` does.

---

## Step 3 — Write the loader

Common to every archetype:

- SPDX header; module docstring that names the archetype and, for 5 and 6,
  **why** the graph was reshaped (that docstring is the design record — see
  `voxtral/pytorch/loader.py` for the standard).
- `ModelVariant(StrEnum)`, `_VARIANTS`, `DEFAULT_VARIANT`.
- `_get_model_info()` with the right `ModelTask`:
  `AUDIO_ASR` (transcription / speech translation), `AUDIO_CLS`
  (classification / CTC), `MM_TTS` (text-to-speech **and** music generation),
  `MM_CONDITIONAL_GENERATION` (audio-conditioned LLM), `MM_AUDIO_TTT`.
- `load_model(*, dtype_override=None, **kwargs)` — propagate `dtype_override`
  into **both** the model and the processor kwargs, and merge `**kwargs` last.
- `load_inputs(dtype_override=None)` returning a **dict** keyed by the real
  forward parameter names.

Archetype-specific:

**1 — ASR seq2seq.** Return `{"input_features": ..., "decoder_input_ids": ...}`.
Build `decoder_input_ids` from `config.decoder_start_token_id`, not a literal.
If variants disagree on the model class (`WhisperModel` vs
`WhisperForConditionalGeneration`) branch inside `load_model` and keep a
`self.feature_extractor` / `self.processor` pair — copy `whisper/pytorch`.

**3 — TTS acoustic.** Return
`{"input_ids", "attention_mask", "decoder_input_values"}` where
`decoder_input_values = torch.zeros((1, 1, config.num_mel_bins), dtype=dtype_override)`.
Passing `dtype=dtype_override` is mandatory — omitting it is the exact bf16
mismatch fixed in `speecht5` commit `96a6d9f487`.

**5 — Audio-conditioned LLM.** In `load_inputs`, under `@torch.no_grad()`:
run the text embedding, run the audio tower, merge with a **static**
`index_copy` at the audio-token positions (numerically identical to
`masked_scatter`, but static), and return
`{"inputs_embeds", "attention_mask"}`. With `inputs_embeds` supplied and
`input_features` omitted, HF's forward skips the dynamic merge and the
on-device graph is just the LM. Add `load_shard_spec()` + `get_mesh_config()`
now — these models are usually weight-bound and will be promoted.
If the model is a plain causal LM in TTS clothing (Llasa), skip the merge
entirely, build the prompt with `apply_chat_template`, and pad to a static
length with `pad_inputs` so shapes are fixed.

**6 — Multi-component.** See `references/tts_component_decomposition.md`. One
`ModelVariant` per learned `nn.Module`; a thin tensor-in/tensor-out
`nn.Module` wrapper per component under `src/model.py`; every component fed
the **real** tensor it receives end-to-end, derived deterministically from the
reference clip and memoized on the loader. No synthetic intermediates — a
component fed synthetic latents PCCs perfectly and proves nothing about the
chain.

---

## Step 4 — Register the output handler

If one CPU forward returns a dataclass not in
`third_party/tt_forge_models/training_utils.py`'s `_HANDLER_REGISTRY`, add it:

```python
_register_attr("Seq2SeqSpectrogramOutput", "spectrogram")
```

Register the *audio-meaningful* field: `spectrogram` for TTS acoustic models,
`logits` for ASR/CTC/classification. If the output is an ambiguous tuple/list
(common in archetype 6), do **not** register globally — override
`unpack_forward_output` on the loader instead.

Skipping this is what produces `FAILED_FE_COMPILATION` /
"no unpack_forward_output handler" in the training config; that pattern has its
own triage skill (`triage-unpack-forward-output`).

---

## Step 5 — Declare dependencies

Next to `loader.py`:

- `requirements.txt` — Python deps (`coqui-tts`, `pocket_tts`, `datasets`,
  `torchaudio==<pin>+cpu` with the CPU extra-index URL).
- `system-requirements.txt` — system packages, `ffmpeg` above all.
- `requirements.nodeps.txt` — packages that must install with `--no-deps`
  (audio packages routinely pin a CUDA torch and would clobber the env).

`RequirementsManager` installs these per-test and rolls back afterwards, so an
audio package's torch pin does not leak into the rest of the suite. Pin
`torchaudio` to the CPU wheel — the default wheel drags in CUDA and breaks the
host.

---

## Step 6 — CPU sanity before hardware

Run one CPU forward per variant and record the output shape, dtype and a finite
check. For archetype 6 run the **whole chain** on CPU once and keep the
intermediates — those become the golden tensors each component is compared
against, and the chain is what proves the decomposition is faithful.

For TTS, also write a WAV from the CPU output and listen to / inspect it. A
spectrogram with PCC 0.99 against a broken reference is still a broken model;
`references/audio_input_recipes.md` has the stdlib `wave` writer that avoids an
FFmpeg dependency.

If CPU sanity fails, stop — the loader is broken. Transition to ESCALATED with
`loader_cpu_sanity_failed:<reason>`. Do not spend hardware time on it.

---

## Step 7 — Test wiring

Default: the generic runner discovers the loader; add the YAML entry at
CONFIG_UPDATE. Write a **hand-written test** under
`tests/torch/models/<family>/` (and set `status: EXCLUDE_MODEL` on the runner
entries) when:

- variants need different wrapper `forward` signatures (Whisper), or
- variants need different `required_pcc` / dtype overrides (Whisper large-v3),
  or
- the model is archetype 6 and each component needs its own PCC assertion.

Hand-written wrappers must accept **keyword** arguments matching the loader's
input dict keys. `TorchModelTester._get_forward_method_kwargs` unpacks any
`Mapping` into kwargs, so `forward(self, *inputs)` fails with
"got an unexpected keyword argument 'input_features'" — the exact bug fixed in
tt-xla `8b8ba4180`. Write
`forward(self, input_features, decoder_input_ids, attention_mask=None)`.

---

## Step 8 — Exit contract

Same as the other scaffold variants. PASSED requires:

- `third_party/tt_forge_models/<family>/<task>/pytorch/loader.py` + `__init__.py`
- `requirements.txt` / `system-requirements.txt` where needed
- `src/model.py` wrappers for archetype 6
- output handler registered (or overridden)
- CPU sanity log with real (not synthetic) audio for archetypes 1–3, 5, 6
- `state.json` initialised with `audio_archetype`

On failure → ESCALATED with the reason. Then hand back to the orchestrator for
OVERVIEW → FIRST_RUN.

For archetype 6, the natural follow-up after all components pass is an
end-to-end audio pipeline — hand off to `audio-pipeline-e2e`.

---

## References

- `references/audio_model_inventory.md` — every audio family already in
  tt-forge-models, its archetype, its current tt-xla status, and what it is
  worth copying from.
- `references/audio_input_recipes.md` — audio assets, sample rates, mel
  front-ends, dependency escape hatches, WAV writing.
- `references/audio_failure_catalog.md` — observed failure → root cause → fix,
  drawn from landed commits. Check here **before** diagnosing from scratch.
- `references/tts_component_decomposition.md` — how to split a non-traceable
  TTS stack into per-component variants, including the static-KV-cache
  autoregressive decode step.
