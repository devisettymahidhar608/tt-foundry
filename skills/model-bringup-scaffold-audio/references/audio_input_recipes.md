# Audio input recipes

How to get a real waveform into a loader without dragging a codec stack onto the
bringup host. Every snippet below is drawn from a landed loader.

## Rule 0 — the sampling rate is part of the contract

Every audio model declares a rate. Feeding the wrong one does not error; it
silently halves PCC and reads like a compiler bug. Record both rates as named
constants and resample explicitly.

```python
XTTS_SR = 22050            # model native rate
SPEAKER_ENCODER_SR = 16000 # a *different* submodule's rate in the same model
```

A single model can have more than one — XTTS-v2's GPT conditioning path is
22.05 kHz while its speaker encoder is 16 kHz.

## Recipe A — the cached tensor asset (default)

No `datasets`, no `ffmpeg`, no codec. This is what `whisper/pytorch` and
`xtts_v2` use.

```python
from ...tools.utils import get_file

sample = torch.load(
    get_file("test_files/pytorch/whisper/1272-128104-0000.pt"), weights_only=False
)
sample_audio = sample["audio"]["array"]          # numpy float array
sr = int(sample["audio"].get("sampling_rate", 16000))
```

`get_file` resolves local paths, the shared cache, and URLs (hashing the URL
into the cache filename), so the same call works for a pinned public clip.

Resample when the model rate differs:

```python
import numpy as np, torchaudio
audio = torch.tensor(np.asarray(sample["audio"]["array"], dtype="float32")).unsqueeze(0)
if sr != MODEL_SR:
    audio = torchaudio.functional.resample(audio, sr, MODEL_SR)
```

## Recipe B — the LibriSpeech dummy dataset

Needed when the model's processor wants the dataset's exact framing.

```python
import datasets
datasets.config.AUDIO_BACKEND = "soundfile"   # do this BEFORE load_dataset

from datasets import load_dataset
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
sample = ds[0]["audio"]
inputs = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt")
```

Costs you, in `requirements.txt` / `system-requirements.txt`:

```
datasets
```
```
ffmpeg
```

The `AUDIO_BACKEND = "soundfile"` line is not optional — without it `datasets`
reaches for torchcodec, which fails with
`RuntimeError: Could not load libtorchcodec` on bringup hosts. That is the
recorded cause of `whisper/pytorch-Large_v3-single_device-training` being
`NOT_SUPPORTED_SKIP`.

## Recipe C — pinned public clips

For models whose prompt format needs specific audio (e.g. a multi-clip
conversation), pass URLs straight through the processor's chat template:

```python
_DEFAULT_CONVERSATION = [{
    "role": "user",
    "content": [
        {"type": "audio", "path": "https://huggingface.co/datasets/hf-internal-testing/"
                                  "dummy-audio-samples/resolve/main/mary_had_lamb.mp3"},
        {"type": "text",  "text": "What is referenced?"},
    ],
}]
inputs = processor.apply_chat_template(_DEFAULT_CONVERSATION)
```

Prefer `hf-internal-testing` assets — they are small, permanently hosted, and
already cached on CI.

## Recipe D — synthetic, and when it is right

Legitimate only when the input is itself a model *output* whose distribution
does not affect the graph — i.e. vocoders:

```python
# (batch, frames, mel_bins) — SpeechT5 HiFi-GAN
spectrogram = torch.randn(1, 512, 80)
if dtype_override is not None:
    spectrogram = spectrogram.to(dtype_override)
```

Say so in the docstring. For every other archetype, synthetic input hides real
regressions: an ASR model fed noise still produces finite logits with a
respectable PCC.

For multi-component stacks (archetype 6), synthetic intermediates are
**forbidden**: derive each component's input from the reference clip through the
real upstream components, memoized on the loader:

```python
def _gpt_cond_latent(self):
    if "gpt_cond_latent" not in self._cache:
        with torch.no_grad():
            self._cache["gpt_cond_latent"] = self._xtts.get_gpt_cond_latents(...)
    return self._cache["gpt_cond_latent"]
```

Determinism matters: use `do_sample=False, num_beams=1` when a component's input
comes from a `generate()` call, so the captured tensors are reproducible.

## Mel front-ends stay on the host

The mel/STFT front-end is fixed DSP, not learned. Compute it on CPU in
`load_inputs` and feed the learned trunk the mel:

```python
with torch.no_grad():
    mel_spec = model.speaker_encoder.torch_spec(audio_16k)
return {"mel_spec": mel_spec}
```

If the module computes its own spectrogram inside `forward`, disable that path
in the wrapper (`self.speaker_encoder.use_torch_spec = False`) so the compiled
graph is the trunk only. This keeps the graph static and keeps FFT ops off the
device.

Reproduce the reference front-end's parameters *exactly* — for XTTS's
conditioning path that means `n_fft=2048, hop_length=256, win_length=1024,
power=2, normalized=False, f_min=0, f_max=8000, n_mels=80` and the model's own
`mel_stats`. Getting one of these wrong is indistinguishable from a numerics bug
downstream.

## Dependency hygiene

Next to `loader.py`:

| File | Contents | Why |
|---|---|---|
| `requirements.txt` | `coqui-tts`, `pocket_tts`, `datasets`, `torchaudio==<ver>+cpu` with `--extra-index-url https://download.pytorch.org/whl/cpu` | `RequirementsManager` installs per-test and rolls back |
| `system-requirements.txt` | `ffmpeg` | system package, installed separately |
| `requirements.nodeps.txt` | packages whose transitive pins would clobber torch | installed with `--no-deps` |

Always pin the **CPU** torchaudio wheel. The default wheel pulls a CUDA torch
and replaces the host's torch build.

Gated weights: set the acceptance env var in the loader rather than asking the
operator to remember it —

```python
os.environ.setdefault("COQUI_TOS_AGREED", "1")   # XTTS-v2 CPML gate
```

## Writing a WAV without FFmpeg

`torchaudio.save` needs FFmpeg/torchcodec. The stdlib does not:

```python
import wave, torch

def save_wav(wav, filepath="out.wav", sample_rate=24000):
    audio = torch.clamp(wav.detach().cpu().float().reshape(-1), -1.0, 1.0)
    pcm = (audio * 32767.0).round().to(torch.int16).numpy()
    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return filepath
```

Use this in demos and in the CPU-sanity step. A listenable WAV is the only check
that catches "PCC 0.99 against a reference that was already wrong".

## Batching audio inputs

Feature-extractor outputs are `(batch, mel_bins, frames)` or `(batch, samples)`;
replicate along dim 0 with `repeat_interleave`, and keep integer tensors integer:

```python
for key in inputs:
    inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)
```

MusicGen shows the awkward case — its natural batch is 2 (two prompts), so
odd target batch sizes need a remainder repeat of the first example. Only
cast **floating** inputs with `dtype_override`; casting ids to bf16 breaks
embedding lookups. `cast_input_to_type` in `tools/utils.py` does this correctly.
