# Audio / speech model inventory (tt-forge-models × tt-xla)

Snapshot of every audio family already brought up. Use it to pick the closest
precedent before scaffolding anything new. Status column is the tt-xla runner
config (`tests/runner/test_config/...`) as of the tt-forge-models checkout that
carries XTTS-v2 / Llasa / Voxtral.

## By archetype

### 1 — Encoder-decoder ASR / speech-seq2seq

| Family | Loader | Task | tt-xla status |
|---|---|---|---|
| `whisper/pytorch` | 7 variants (Tiny→Large_v3_Turbo) | `AUDIO_ASR` | `EXCLUDE_MODEL` in the runner — **hand-written test** at `tests/torch/models/whisper/`. Large xfails (DRAM OOM, tt-xla#1886). Large_v3 / Large_v3_Turbo run at lowered PCC (0.5 / 0.7) because their HF configs declare `torch_dtype=float16`. |
| `whisper/audio_classification/jax` | Base / Medium / Large_v3, EasyDeL | `AUDIO_CLS` | Base+Medium `EXPECTED_PASSING` with `INCORRECT_RESULT` bringup status (PCC 0.668 / 0.734); Large_v3 skipped, 6.7 GB DRAM buffer. |
| `whisper/audio_classification/onnx` | — | `AUDIO_CLS` | ONNX path. |
| `seamless_m4t/pytorch`, `seamless_m4t_v2/pytorch` | Large | `AUDIO_ASR` | Training entry only, `NOT_SUPPORTED_SKIP` / `FAILED_FE_COMPILATION` — `ModuleNotFoundError: Could not import module 'SeamlessM4TModel'` behind `RuntimeError: function '_has_torch_function' already has a docstring`. **No inference entry exists — this is an open gap.** Loader returns `full_model.text_decoder`, i.e. an archetype-1 submodule slice. |

**Copy from Whisper:** per-variant model-class branching, `feature_extractor` vs
`processor` split, `decoder_start_token_id`-derived decoder ids, the
cached-`.pt` audio asset, and the per-variant PCC/dtype table in the test.

### 2 — Audio classifier / CTC encoder

| Family | Loader | Task | tt-xla status |
|---|---|---|---|
| `wav2vec2/audio_classification/jax` | Large_Lv_60, `FlaxWav2Vec2ForCTC` | `AUDIO_CLS` | `NOT_SUPPORTED_SKIP` — "Flax model class removed in transformers 5.x". The whole Flax audio path is dead on transformers ≥5; new work should be Torch. |

Historical wav2vec2 Torch loaders (ASR / feature-extraction / nonverbal
classification) live in git history and are the source of the **dtype-cast**
lessons in the failure catalog: Conv1d has no bf16 kernel, so those loaders
force float32 and cast inputs with `cast_input_to_type`.

### 3 — TTS acoustic model

| Family | Loader | Task | tt-xla status |
|---|---|---|---|
| `speecht5/pytorch` | `Tts` (`microsoft/speecht5_tts`) | `MM_TTS` | Training: `KNOWN_FAILURE_XFAIL` / `FAILED_TTMLIR_COMPILATION` — `'ttir.sum' op Reduce dimensions are not unique`. **No inference entry — open gap.** |
| `musicgen_small/pytorch` | `facebook/musicgen-small` | `MM_TTS` | Inference `EXPECTED_PASSING`; training `EXPECTED_PASSING` with `assert_pcc: false` + `notimeout`. |

`speecht5` is the canonical shape: text ids + `attention_mask` +
zeroed `decoder_input_values` of `(1, 1, num_mel_bins)`, output
`Seq2SeqSpectrogramOutput` unpacked on `.spectrogram`.

`musicgen_small` is the canonical shape for the *conditional generation* twist:
`decoder_input_ids` is `pad_token_id` repeated over
`batch × decoder.num_codebooks`, and `max_new_tokens=1` is passed as an input so
the graph is one step, not a generate loop.

### 4 — Vocoder

| Family | Loader | Task | tt-xla status |
|---|---|---|---|
| `vocoder_speecht5/pytorch` | `hifigan` (`microsoft/speecht5_hifigan`) | `MM_TTS` | Inference `EXPECTED_PASSING`. |

The one place a synthetic input is correct: `torch.randn(1, 512, 80)`
(batch, frames, mel bins). Cleanest audio loader in the repo — start here when
learning the file shape.

### 5 — Audio-conditioned LLM

| Family | Loader | Task | tt-xla status |
|---|---|---|---|
| `voxtral/pytorch` | `Voxtral-Mini-3B-2507` | `MM_CONDITIONAL_GENERATION` | Loader landed (tt-forge-models #778); no tt-xla config entry yet. |
| `llasa/causal_lm/pytorch` | `Llasa-8B` | `MM_TTS` | Loader landed (#785); no tt-xla config entry yet. A benchmark run of a 1B Llasa variant recorded prefill PCC 0.951 / first-decode 0.933 against a 0.94 gate on p150 with `bfp_bf8 + opt=2 + trace`. |

Voxtral is the reference for the **dynamic-merge escape**: the HF forward does
`inputs_embeds.masked_scatter(...)`, which lowers to a dynamic-shape
`set_dimension_size` the compiler rejects, so the loader precomputes
`inputs_embeds` host-side with a static `index_copy` and ships only the LM.
It also carries a `transformers` 5.5.x chat-template monkeypatch —
`_get_template_variables(None)` raises for mistral_common-backed processors.

Llasa is the reference for **TTS that is really a causal LM**: expanded vocab
(text + XCodec2 speech tokens), TTS chat prompt, `pad_inputs` to a static
length, Megatron shard spec included for the promotion path. The XCodec2 decoder
is deliberately out of the graph.

### 6 — Multi-component TTS / music stack

| Family | Loader | Task | tt-xla status |
|---|---|---|---|
| `xtts_v2/pytorch` | 6 component variants + `pipeline.py` | `MM_TTS` | Loader + e2e pipeline landed (tt-forge-models #784); no tt-xla config entry yet. |
| `pocket_tts/pytorch` | `Pocket_tts` | `MM_TTS` | `NOT_SUPPORTED_SKIP`, `FAILED_RUNTIME`, `assert_pcc: false` — "Aborted", tt-mlir#6786. PCC stabilised around 0.78 only after zeroing FlowLM noise and seeding RNGs. |

XTTS-v2 is the full worked example — read
`tts_component_decomposition.md`. Its six variants are `speaker_encoder`,
`conditioning`, `gpt_prefill`, `gpt_decode`, `gpt_latents`, `hifigan_decoder`.

Pocket-TTS is the reference for **monkeypatching a third-party TTS package**:
`src/model.py` replaces `TTSModel.forward` to tap latents out of an internal
`queue.Queue`, adds `post_process()` to return the already-decoded audio, and
overrides `FlowLMModel.device` to return the full device string (`xla:0`, not
`xla`). `copy_state` defaults to `False` because `copy.deepcopy` under
`torch.compile` conflicts with the global `TorchFunctionMode`.

## Adjacent, not audio-only

- `gemma4` — multimodal with an audio modality; `load_audio_inputs()` feeds the
  shared multimodal benchmark harness (`tests/benchmark/test_multimodal.py::test_gemma4_12b_audio`).
  The only audio entry in the benchmark suite; use it as the template when
  wiring an audio model into nightly perf via `add-benchmark-model`.
- `minicpm_o_2_6`, `stereo` — omni / non-audio despite the name; check the task
  enum before assuming.

## Open gaps worth picking up

1. **SpeechT5 TTS inference** — loader exists, vocoder passes, but there is no
   `speecht5/pytorch-Tts-single_device-inference` entry. Lowest-effort landable
   audio bringup available.
2. **SeamlessM4T / v2 inference** — same: loaders exist, only a broken training
   entry is configured.
3. **XTTS-v2, Llasa, Voxtral tt-xla wiring** — loaders landed upstream; the
   tt-xla config entries and (for XTTS) component/e2e tests are the follow-up.
4. **Torch replacements for the dead Flax audio path** — `wav2vec2` JAX is
   permanently skipped on transformers ≥5.
