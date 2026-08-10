# Audio bringup failure catalog

Observed failure → root cause → fix. Every row is drawn from a landed commit or
a configured YAML entry. Check here before diagnosing an audio failure from
scratch — the audio family repeats a small number of failures constantly.

## A. Dependency / environment (`FAILED_FE_COMPILATION`)

These dominate audio bringup. They are not compiler bugs and must not be
diagnosed as such.

| Symptom | Root cause | Fix |
|---|---|---|
| `RuntimeError: Could not load libtorchcodec` | `datasets` defaults to the torchcodec audio backend | `datasets.config.AUDIO_BACKEND = "soundfile"` before `load_dataset`; add `ffmpeg` to `system-requirements.txt`. (`whisper/pytorch-Large_v3-single_device-training`) |
| `ModuleNotFoundError: Could not import module 'SeamlessM4TModel'` preceded by `RuntimeError: function '_has_torch_function' already has a docstring` | a per-model requirements install replaced torch, so the C-extension docstring is registered twice on re-import | pin the **CPU** torch/torchaudio wheels; move the offending package to `requirements.nodeps.txt`. (`seamless_m4t/pytorch-Large-single_device-training`) |
| `ImportError` on an audio package that exists | package installed but purged from `sys.modules` by the requirements rollback | let `RequirementsManager` own it — do not import audio packages at module scope in the loader; import inside `load_model` / `load_inputs` |
| `Dataset '<x>' is a gated dataset on the Hub` | gated asset | switch to Recipe A/C in `audio_input_recipes.md` |
| weights refuse to download | licence gate (XTTS-v2 is CPML) | `os.environ.setdefault("COQUI_TOS_AGREED", "1")` inside the loader |

## B. dtype

| Symptom | Root cause | Fix |
|---|---|---|
| `mixed dtype (CPU)` / BFloat16-vs-Float mismatch in the first conv | model cast to bf16 but inputs left float32 (or vice versa) | cast floating inputs with `cast_input_to_type(x, dtype_override)`; never cast integer ids |
| `Conv1d` unsupported for bf16 | tt-metal has no bf16 Conv1d kernel on this path | load the model in **float32** and let the compiler pick device precision (the wav2vec2 feature-extraction fix) |
| TTS decoder input dtype mismatch | `torch.zeros((1, 1, num_mel_bins))` created without `dtype` | `torch.zeros((1, 1, config.num_mel_bins), dtype=dtype_override)` (speecht5 `96a6d9f487`) |
| PCC collapses from >0.99 to ~0.53 on a large variant | the HF config declares `torch_dtype: float16` (Whisper large-v3 / large-v3-turbo changed in transformers ≥5) | pass `dtype_override=torch.float32` for those variants and set a measured `required_pcc` per variant — record the measurement, do not tune to pass |
| whole-model bf16 cast produces mixed-dtype errors on a multi-component stack | submodules do not cast uniformly; some internal tensors stay fp32 | keep the model in float32 and say so in the docstring (XTTS-v2 ignores `dtype_override` deliberately) |

## C. Graph shape / dynamic ops

| Symptom | Root cause | Fix |
|---|---|---|
| dynamic-shape `set_dimension_size`; Shardy/TT rejects the graph | HF's audio-LLM forward merges audio into text embeddings with `masked_scatter` | precompute `inputs_embeds` on the host with a static `index_copy` at the audio-token positions, pass `inputs_embeds` and omit `input_features` (Voxtral) |
| `Unsupported data format for add_int` on a cumsum | `aten.cumsum` over a **bool** tensor (attention-mask construction); tt-metal's accumulation kernel takes Int32/UInt32/UInt16 only | the `cast_bool_cumsum_to_int32` backend pass in `python_package/tt_torch/backend/passes.py` (tt-xla `8b8ba4180`). If it recurs on a new path, extend the pass — don't patch the model |
| graph recompiles every decode step | per-step-varying shapes in the AR loop | one `GptCachedStep`-style graph over a `StaticCache`, constant-shaped inputs, positions/masks passed as tensors (see `tts_component_decomposition.md`) |
| device-kwarg tensor creation fails inside a compiled forward | cache/mask objects built **inside** `forward` | build the cache skeleton on CPU in `__init__`; `forward` only swaps in cloned tensors |
| `error: 'ttir.sum' op Reduce dimensions are not unique` | duplicated reduce dims in the TTS training graph | open/track a tt-mlir issue; currently `KNOWN_FAILURE_XFAIL` for `speecht5/pytorch-Tts-single_device-training` |

## D. Harness / plumbing

| Symptom | Root cause | Fix |
|---|---|---|
| `TypeError: <X>Wrapper.forward() got an unexpected keyword argument 'input_features'` | loader returns a dict; `TorchModelTester._get_forward_method_kwargs` unpacks Mappings into **kwargs**; the wrapper declared `forward(self, *inputs)` | declare named parameters matching the dict keys, optional ones defaulted: `forward(self, input_features, decoder_input_ids, attention_mask=None)` (tt-xla `8b8ba4180`) |
| `No handler for class Seq2SeqSpectrogramOutput exists in unpack_forward_output` | TTS output dataclass not registered | `_register_attr("Seq2SeqSpectrogramOutput", "spectrogram")` in `training_utils.py`; ambiguous tuple outputs → override `unpack_forward_output` on the loader instead |
| `copy.deepcopy` fails / misbehaves under `torch.compile` | deepcopying model state conflicts with the global `TorchFunctionMode` | thread a `copy_state=False` default through the patched forward (Pocket-TTS) |
| device resolves to `xla` instead of `xla:0` | a package's `device` property returns `next(self.parameters()).device.type` | override the property to return `str(next(self.parameters()).device)` (Pocket-TTS `FlowLMModel.device`) |
| `isin_mps_friendly` missing on import | dropped in transformers ≥5; coqui-tts' tortoise import chain needs it | shim it in `src/model.py` **before** any package import, idempotently |
| `Can't compile non template nodes` from `apply_chat_template` | `_get_template_variables(None)` for mistral_common-backed processors with no Jinja template | idempotent monkeypatch returning `frozenset()` for `None` (Voxtral `_patch_chat_template_guard`) |
| `Qwen2TSForCausalLM.forward() got an unexpected keyword argument 'cache_position'` | audio/TS model forward signature predates transformers 5.2 | loader-side signature adaptation, or pin — track upstream |

## E. Capacity

| Symptom | Reading |
|---|---|
| `Not enough space to allocate 6710886400 B DRAM buffer across 12 banks` (whisper-large / whisper-large-v3 JAX) | weight-bound. A capacity fact, not a bug — do not re-run to confirm. Record with the exact message + issue link, then take the `classify-oom` → `write-promotion` → `model-bringup-multichip` path |
| a TTS component OOMs while siblings pass | expected for archetype 6 — per-component sizing, not per-model. One passing component plus OOM-xfailed siblings is a complete, landable bringup |

## F. Numerics specific to audio

- **Stochastic generation destroys reproducibility.** Any component whose input
  comes from sampling must be captured with `do_sample=False, num_beams=1`.
  Where the reference path is genuinely stochastic (XTTS inference:
  temperature 0.75 / top-k 50 / top-p 0.85 / repetition penalty 10.0), apply
  HF's own logits processors host-side on a seeded generator so the run is
  reproducible and matches `generate()` ordering.
- **Injected noise.** Flow-matching TTS (Pocket-TTS) samples noise inside the
  model; PCC is meaningless until it is zeroed (`nn.init.normal_` /
  `trunc_normal_`) and all RNGs are seeded in a dynamo-safe way. That is a
  **debug** configuration — say so, and do not land the PCC number as if it were
  the model's.
- **PCC is necessary, not sufficient.** For TTS, always write a WAV and check it
  is audible speech. A per-component PCC of 0.99 against a wrong reference is
  still wrong; the XTTS review explicitly required a real text→speech run rather
  than component PCC alone.
- **Record measured PCC, never tune to pass.** When a threshold must be lowered,
  state the measured values from at least two runs, say whether the gap is
  device-side or a bf16 floor, and leave a note not to lower it further.
