# Decomposing a non-traceable TTS stack (archetype 6)

The worked example is XTTS-v2 (`xtts_v2/pytorch`, tt-forge-models #784). Read
this before splitting any TTS/music model that has no single traceable forward.

## When decomposition is required

- The public entry point is `generate()` / `synthesize()` / `tts()`, not
  `forward()`.
- The package is not `transformers` (coqui-tts, `pocket_tts`, XCodec2, …).
- The stack is tokenizer → conditioning encoder → autoregressive audio-code LM
  → vocoder, with host-side DSP between stages.

Symptom if you skip it: the traced graph either fails to trace, or traces the
generate loop and recompiles every step.

## The rule that decides everything

**One variant per learned `nn.Module`. Fixed DSP and orchestration stay on the
host.**

Mel/STFT front-ends, tokenizers, resampling, chunking, logits processing and
sampling are not learned — they are deterministic functions the compiler gains
nothing from. Keeping them on the host makes each device graph static and
single-purpose.

XTTS-v2's six variants:

| Variant | Input → output | Notes |
|---|---|---|
| `speaker_encoder` | mel `(b, 64, T)` → embedding `(b, 512)` | mel computed host-side; `use_torch_spec = False` so the wrapper runs only the trunk |
| `conditioning` | reference mel `(b, 80, s)` → cond latents `(b, 1024, 32)` | conditioning encoder + perceiver |
| `gpt_prefill` | `[prefix, START]` → first-step audio logits | `use_cache=False` |
| `gpt_decode` | one token + static KV cache → next audio logits | the graph the AR loop reuses |
| `gpt_latents` | text + audio codes → GPT latents | full-sequence forward |
| `hifigan_decoder` | latents + speaker emb → 24 kHz waveform | |

## History worth knowing

XTTS-v2 was **first** brought up as a single monkeypatched `Xtts.forward` over a
fixed synthetic `gpt_codes` tensor, PCC 1.0 on CPU. It was rejected in review:
*"We should strive to have a text-to-speech run. Just having a valid PCC for the
model isn't enough as end results are not reproducible."* Component PCC over
synthetic intermediates is not a bringup. Plan for the per-component +
end-to-end shape from the start.

## Layout

```
<family>/pytorch/
├── __init__.py          # re-exports ModelLoader, ModelVariant
├── loader.py            # variants, real-input derivation, memoization
├── pipeline.py          # e2e chain (see the audio-pipeline-e2e skill)
├── requirements.txt
└── src/
    ├── __init__.py
    └── model.py         # thin nn.Module wrappers + import-order shims
```

## `src/model.py` — wrappers

Each wrapper is tensor-in / tensor-out, holds references to the real submodules,
and does nothing else:

```python
class ConditioningWrapper(nn.Module):
    def __init__(self, xtts):
        super().__init__()
        self.conditioning_encoder = xtts.gpt.conditioning_encoder
        self.conditioning_perceiver = xtts.gpt.conditioning_perceiver

    def forward(self, cond_mel):
        conds = self.conditioning_encoder(cond_mel)                    # (b, 1024, s)
        return self.conditioning_perceiver(conds.permute(0, 2, 1)).transpose(1, 2)
```

Rules:

- Register any constant the graph needs as a **buffer**
  (`self.register_buffer("prefix_emb", prefix_emb)`) so it moves with
  `.to(device)`. Plain attributes silently stay on CPU and produce a device
  mismatch mid-graph.
- Put import-order shims at the **top of the module**, idempotently — they must
  run before the third-party package is imported anywhere:

```python
import transformers.pytorch_utils as _pu
if not hasattr(_pu, "isin_mps_friendly"):
    _pu.isin_mps_friendly = lambda e, t: torch.isin(e, t)
```

- Keep the forward **pure**. No cache objects constructed inside `forward`, no
  in-place mutation of inputs. Both break repeated CPU/TT comparison runs and
  trip device-kwarg tensor creation on TT.

## The autoregressive decode step

This is the hard part. The goal: **compile once, reuse every token.**

Build the cache skeleton on CPU in `__init__`, outside the graph:

```python
self._cache = StaticCache(config=gpt.gpt.config, max_cache_len=max_cache_len)
```

Pass the cache contents in as tensors and rebuild from clones each call:

```python
def forward(self, audio_ids, positions, attention_mask, cache_len, cache_keys, cache_values):
    cache = self._load_cache(cache_len, cache_keys, cache_values)
    emb = self.mel_embedding(audio_ids)
    emb = emb + self.mel_pos_embedding.emb(positions).unsqueeze(0)
    ...
```

Points that are easy to get wrong:

- `cache_len` (the prefill's `cumulative_length`) is an **input tensor**, not a
  Python int — otherwise the forward creates a device-placed tensor internally.
- Clone `layer.cumulative_length`; `StaticLayer.update()` does an in-place
  `add_` on it.
- Set `layer.is_initialized = True` plus the dtype/device/shape metadata so
  `update()` skips lazy allocation and writes at `cumulative_length`.
- Some GPT2-derived stacks have a null `wpe` — positions come from a separate
  `mel_pos_embedding` and there is no `cache_position`. Check before assuming
  HF's default position handling.
- Build + `early_initialization` on CPU, then move `keys` / `values` /
  `cumulative_length` to the device manually (a trace/fusion issue, tt-xla#1645).
- The attention mask is a constant-shape `(1, max_cache_len)` with ones in the
  written slots. Constant shape is what keeps it to one compile.

## Real inputs, memoized

Every component is fed the tensor it would actually receive end-to-end, derived
from the reference clip through the real upstream path, cached on the loader so
a variant only pays for what it needs:

```python
def _gpt_codes(self):
    if "gpt_codes" not in self._cache:
        with torch.no_grad():
            self._cache["gpt_codes"] = self._xtts.gpt.generate(
                cond_latents=self._gpt_cond_latent(),
                text_inputs=self._text_tokens(),
                do_sample=False, num_beams=1, num_return_sequences=1,
            )
    return self._cache["gpt_codes"]
```

`do_sample=False, num_beams=1` is what makes the captured codes reproducible.
Build the full model once (`_build_xtts`) and share it across variants.

## Prefill/decode state capture

`gpt_decode` needs a realistic prefilled cache. Run the prefill on CPU inside
the loader, then hand the decode variant both the inputs and the cache
metadata:

1. `compute_embeddings(cond_latent, text_tokens)` → prefix embedding, length `P`.
2. Size the static cache `P + 2` (prefill writes `P + 1`, decode writes one more).
3. Run the prefill, argmax the first audio token.
4. Return `{audio_ids, positions, attention_mask, cache_len, cache_keys, cache_values}`
   with keys/values stacked over layers as `[n_layer, b, n_head, L, head_dim]`.

## Testing

One standalone pytest per component under `tests/torch/models/<family>/`, each
with `ComparisonConfig(required_pcc=0.99)` against the CPU golden. Do not route
component PCC through the runner YAML — set `EXCLUDE_MODEL` there. Components
that exceed device DRAM xfail with the exact OOM message plus a tracking issue;
that is a capacity fact, and one passing component alongside OOM-xfailed
siblings is still a landable bringup.

Then the end-to-end run: hand off to the `audio-pipeline-e2e` skill.
