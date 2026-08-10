# Add audio / speech model bringup skills to tt-foundry

## Problem description

tt-foundry's bringup FSM had two VALIDATE-stage scaffold variants beyond the
default — `model-bringup-scaffold-github` (GitHub-hosted sources) and
`model-bringup-scaffold-pipeline` (multi-component `DiffusionPipeline`). Audio
and speech models fell through to the default scaffold, which does not fit them
on three independent axes:

1. **Inputs are not synthesizable.** The default scaffold generates synthetic
   tensors matching a `forward()` signature. A random `(1, 80, 3000)` mel
   compiles fine and proves nothing — an ASR or TTS model fed noise still
   produces finite outputs with a respectable PCC, hiding real regressions.
   Real audio inputs pull in sample rates, `datasets`, `torchaudio`,
   `soundfile`/`torchcodec` and `ffmpeg`, which is the single largest source of
   `FAILED_FE_COMPILATION` on this family.
2. **Many TTS stacks have no single traceable forward.** XTTS-v2, Pocket-TTS and
   the SpeechT5 stack are tokenizer → conditioning encoder → autoregressive
   audio-code LM → vocoder. `load_model()` returning "the model" either fails to
   trace or traces the `generate()` loop and recompiles per token.
3. **The output is not logits.** TTS returns `Seq2SeqSpectrogramOutput`
   (`.spectrogram`), vocoders return a raw waveform, ASR returns
   `Seq2SeqLMOutput`. Without a registered handler the training path fails with
   "No handler for class ... exists in `unpack_forward_output`".

The result was that audio bringups repeatedly re-derived the same fixes. The
existing corpus already contains the answers — 11 audio families across
tt-forge-models and tt-xla, with the failure modes recorded in landed commits
and configured YAML entries — but none of it was captured as reusable guidance.

## What's changed

Two new skills plus the routing to reach them.

### `skills/model-bringup-scaffold-audio/` (new)

VALIDATE-stage variant for audio/speech models. Classifies the target into one
of six archetypes and picks the loader shape for each:

| # | Archetype | Loader shape | Reference |
|---|---|---|---|
| 1 | Encoder-decoder ASR / speech-seq2seq | single forward + thin wrapper | `whisper/pytorch` |
| 2 | Audio classifier / CTC encoder | single forward | `wav2vec2/audio_classification` |
| 3 | TTS acoustic model | single forward + spectrogram handler | `speecht5/pytorch` |
| 4 | Vocoder | single forward (synthetic input **is** valid here) | `vocoder_speecht5/pytorch` |
| 5 | Audio-conditioned LLM | precompute embeds host-side, ship the LM | `voxtral`, `llasa` |
| 6 | Multi-component TTS / music stack | one variant per learned `nn.Module` | `xtts_v2`, `pocket_tts` |

Then eight steps: inspect (recording the sampling rate as a first-class fact),
choose the audio asset, write the loader per archetype, register the output
handler, declare `requirements.txt` / `system-requirements.txt` /
`requirements.nodeps.txt`, CPU sanity with real audio, test wiring, exit
contract.

Four references:

- **`audio_model_inventory.md`** — every audio family already brought up, its
  archetype, its current tt-xla status, and what is worth copying from it.
- **`audio_input_recipes.md`** — the cached LibriSpeech `.pt` asset as the
  default (no `datasets`/`ffmpeg`/codec dependency), the dataset path with
  `datasets.config.AUDIO_BACKEND = "soundfile"`, pinned public clips, when
  synthetic input is legitimate, host-side mel front-ends, dependency hygiene
  (CPU torchaudio wheels, licence gates), and an FFmpeg-free stdlib WAV writer.
- **`audio_failure_catalog.md`** — observed failure → root cause → fix, grouped
  as dependency / dtype / graph-shape / harness / capacity / numerics. Every row
  traces to a landed commit or a configured YAML entry.
- **`tts_component_decomposition.md`** — the XTTS-v2 worked example: one variant
  per learned module, real memoized intermediates, and the static-KV-cache
  decode step that compiles once and is reused every token.

### `skills/audio-pipeline-e2e/` (new)

The follow-up for archetype-6 families, and the audio analogue of the image-gen
pipeline template. Component PCC is not a bringup for a generative audio model —
the deliverable is a real text → waveform run that produces a listenable WAV.
Covers the host/device split (learned modules on TT; tokenizer, mel front-end,
chunking and sampling on the host), CPU→TT→CPU component eviction, the
autoregressive loop requirements (one graph compiled once, cache built on CPU
then moved manually, HF logits processors host-side on a seeded generator),
faithfulness to the reference preprocessing, the four deliverables, and the
verification gate.

### Wiring (modified)

- `skills/model-bringup/SKILL.md` — audio row in the VALIDATE routing table,
  explicit detection criteria, routing precedence relative to the pipeline row,
  and the archetype-6 → `audio-pipeline-e2e` follow-up.
- `skills/model-bringup-scaffold/SKILL.md` — delegation callout alongside the
  existing pipeline one.
- `README.md` — skills-tree entries, flow diagram, auxiliary-skills list.

## Sources

Everything is grounded in the existing corpus rather than invented:

- **tt-forge-models loaders** — `whisper`, `wav2vec2`, `speecht5`,
  `vocoder_speecht5`, `musicgen_small`, `seamless_m4t`/`_v2`, `voxtral`,
  `llasa`, `xtts_v2`, `pocket_tts`.
- **tt-xla** — runner YAML statuses across the torch/jax inference and training
  configs, the hand-written Whisper test, and the Gemma4 audio benchmark entry.
- **Landed commits** — the XTTS-v2 bringup and its review history (component PCC
  over synthetic codes was explicitly rejected as not a bringup), the whisper
  E2E fix (keyword-arg wrapper signature + the `cast_bool_cumsum_to_int32`
  backend pass), the SpeechT5 spectrogram-handler registration and dtype fix,
  and the Pocket-TTS deepcopy-under-compile and device-string fixes.

## Findings surfaced during the analysis

Recorded in `audio_model_inventory.md` as open gaps, not addressed here:

1. **SpeechT5 TTS has no inference config entry.** The loader exists and the
   HiFi-GAN vocoder passes, but only a broken training entry is configured
   (`KNOWN_FAILURE_XFAIL`, `'ttir.sum' op Reduce dimensions are not unique`).
   This is the cheapest landable audio bringup currently available.
2. **SeamlessM4T / v2 likewise** — loaders exist, only a `NOT_SUPPORTED_SKIP`
   training entry is configured.
3. **XTTS-v2, Llasa and Voxtral loaders landed upstream but are unwired in
   tt-xla** — the config entries, component tests and (for XTTS) the e2e test are
   the natural follow-up.
4. **The Flax audio path is dead on transformers ≥5** — `wav2vec2` JAX is
   permanently `NOT_SUPPORTED_SKIP`; replacements should be Torch.

## Checklist

- [x] No code — `SKILL.md` + reference docs only, per the tt-foundry invariant
- [x] No changes to consuming repos (tt-xla / tt-forge-models are read-only here)
- [x] New skills reachable from the orchestrator's VALIDATE routing table
- [x] README index, flow diagram and auxiliary list updated
