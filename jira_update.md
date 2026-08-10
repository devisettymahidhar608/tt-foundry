# Audio / speech model bringup skills — tt-foundry

## Description

Audio and speech models had no dedicated path through the tt-foundry bringup
FSM. They fell through to the default `model-bringup-scaffold`, which assumes
one traceable forward fed by synthetic tensors — an assumption audio breaks on
three independent axes: inputs are real waveforms behind a feature extractor
(sample rate, mel front-end, dataset/codec dependencies), many TTS stacks have
no single traceable forward (tokenizer → conditioning encoder → autoregressive
audio-code LM → vocoder), and the output is a spectrogram or waveform rather
than logits. The practical consequence was that each audio bringup re-derived
the same dependency, dtype and dynamic-shape fixes from scratch.

Analysed all audio/speech models across tt-forge-models and tt-xla — 11 families
(whisper, wav2vec2, speecht5, vocoder_speecht5, musicgen_small, seamless_m4t /
v2, voxtral, llasa, xtts_v2, pocket_tts) plus the gemma4 audio modality — along
with their runner YAML statuses and the commits that brought them up. Captured
the result as two new skills with four reference documents, wired into the
orchestrator's VALIDATE routing.

The work also surfaced four open gaps in the audio coverage itself, recorded in
the inventory reference: SpeechT5 TTS and SeamlessM4T have loaders but no
inference config entry at all (only broken training entries); the XTTS-v2, Llasa
and Voxtral loaders landed upstream but are unwired in tt-xla; and the Flax audio
path is permanently dead on transformers ≥5.

---

Date: Aug 10, 2026

Updates:

Analysed all audio/speech models across tt-forge-models and tt-xla — 11 families plus the gemma4 audio modality — covering loader shapes, runner YAML statuses, and the bringup commits behind each
Authored `model-bringup-scaffold-audio` — VALIDATE-stage scaffold variant that classifies an audio target into one of six archetypes (ASR seq2seq / audio classifier / TTS acoustic / vocoder / audio-conditioned LLM / multi-component TTS stack) and picks the loader shape per archetype
Added four references: audio model inventory (archetype + current status + what to copy per family), audio input recipes (cached assets, sample rates, codec escape hatches, host-side mel front-ends, FFmpeg-free WAV writer), a failure catalog of ~25 observed failure → root cause → fix rows each traced to a landed commit or configured YAML entry, and the XTTS-v2 component-decomposition case study including the single-compile static-KV-cache decode step
Authored `audio-pipeline-e2e` — end-to-end audio pipeline skill (learned modules on TT, tokenizer / mel front-end / sampling on the host, CPU→TT→CPU eviction, seeded reference-matching sampling stack) for the real text→waveform run after archetype-6 components pass
Wired both into the bringup FSM: audio row and detection criteria in the orchestrator's VALIDATE routing table, delegation callout in the default scaffold, README tree / flow diagram / auxiliary list

Next Steps:

Land SpeechT5 TTS and SeamlessM4T inference config entries (loaders already exist, only broken training entries are configured — cheapest landable audio bringups available), then wire the XTTS-v2 / Llasa / Voxtral loaders into tt-xla

Challenges / Blockers:

NIL
