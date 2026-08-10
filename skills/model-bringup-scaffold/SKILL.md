---
name: model-bringup-scaffold
description: Scaffold and validation stage of the model bringup pipeline. Validates that a model_key has a loader and is importable. If the loader is missing, creates loader.py and package __init__.py files following the pattern of existing models. Initializes state.json. Invoked by the model-bringup orchestrator at the VALIDATE stage.
allowed-tools: Bash Read Write Edit Grep Glob
---

# Model Bringup — Scaffold & Validation

You are the **scaffold and validation** stage of the model bringup pipeline.

> **Pipeline models — delegate.** If the HF target resolves to a
> `DiffusionPipeline` (i.e. its `model_index.json` has `_class_name` ending in
> `Pipeline` and lists multiple component sub-folders: `unet`/`transformer`,
> `vae`, `text_encoder`, …), do **not** scaffold here. A single loader cannot
> serve a multi-component pipeline cleanly. Hand off to
> `model-bringup-scaffold-pipeline`, which scaffolds one variant per
> component, captures real per-component I/O shapes, and emits
> shard specs / `TT_VISIBLE_DEVICES` planning.

> **Audio / speech models — delegate.** If the target is an audio or speech
> model (`ModelTask` in `AUDIO_ASR` / `AUDIO_CLS` / `MM_TTS` / `MM_AUDIO_TTT`,
> a processor with a `feature_extractor.sampling_rate`, a config with
> `num_mel_bins` / `num_codebooks`, or a card describing TTS / ASR / voice
> cloning / speech translation / vocoding / music generation), do **not**
> scaffold here. Synthetic tensors are not a valid input for audio, many TTS
> stacks have no single traceable forward, and the output is a spectrogram or
> waveform rather than logits. Hand off to `model-bringup-scaffold-audio`.

## Invocation
`/model-bringup-scaffold <model_key> [--arch <arch>] [--custom-test]`

- `--custom-test` — opt-in flag. When set (or auto-detected via the
  heuristic in Step 4c), scaffold writes a per-model test file in addition
  to relying on the generic `tests/runner/test_models.py` discovery. Use
  for models where the generic runner cannot drive the loader cleanly
  (multi-modal inputs, non-tensor outputs, or any case where the model
  needs assertions on one specific output field).

## Responsibility
Validate that the model_key is ready to run before any test execution begins.
If any file is missing, **create it** following the conventions of existing models.
Initialize the bringup state directory and state.json.

---

## Step 1 — Parse model_key

The model_key may be in one of two formats:

**Format A — structured key** (preferred):
```
<family>/<framework>-<variant>-<parallelism>-<run_mode>
```
Example: `ltx2/pytorch-Fast-single_device-inference`
- `family` = `ltx2`
- `framework` = `pytorch`
- `variant` = `Fast`
- `parallelism` = `single_device`
- `run_mode` = `inference`

**Format B — HuggingFace model ID** (e.g. `google/bert_uncased_L-2_H-128_A-2`):
- Detect: Part 1 does not match `<framework>-<variant>-<parallelism>-<run_mode>`
- Normalize: derive `family` from the model name (e.g. `bert_tiny`), set `framework=pytorch`, `parallelism=single_device`, `run_mode=inference`, derive `variant` from the repo name
- Synthesize the structured key and continue

---

## Step 2 — Locate or create the loader

Check for `third_party/tt_forge_models/<family>/pytorch/loader.py`.

**If loader exists:** proceed to Step 3.

**If loader is absent:** create the full model directory structure (see "Creating a new loader" below), then proceed to Step 3.

---

### Creating a new loader

#### 2a. Find a reference model
Find the most similar existing model to use as a template:
- For text/NLP models: look at `bert/`, `gpt_neo/`, `llama/`
- For image models: look at `resnet/`, `clip/`
- For video/diffusion: look at `flux/`, `ltx2/`
- Use `find third_party/tt_forge_models -name loader.py | head -5` to browse

Read the reference `loader.py` in full before writing anything.

#### 2b. Inspect the HuggingFace model
Use `python -c` to inspect the model:
```python
from transformers import AutoModel, AutoConfig
config = AutoConfig.from_pretrained("<hf_model_id>")
print(config)
model = AutoModel.from_pretrained("<hf_model_id>")
print(type(model))
# Print forward signature
import inspect
print(inspect.signature(model.forward))
```
Record: model class, input names, output structure, hidden dim, num layers.

#### 2c. Create directory structure
```
third_party/tt_forge_models/<family>/
├── __init__.py
└── pytorch/
    ├── __init__.py          ← re-exports ModelLoader, ModelVariant
    └── loader.py            ← model loader
```

No `tests/` subdirectory is needed. `tests/runner/test_models.py` discovers models
automatically via the dynamic loader from `loader.py` alone.

#### 2d. Write `loader.py`
Follow the reference model's structure exactly:
- `ModelVariant(StrEnum)` — one entry per checkpoint; name the variant after the HF repo slug (e.g. `L2_H128_A2`)
- `ModelConfig` or `LLMModelConfig` — set `pretrained_model_name`
- `ModelLoader(ForgeModel)`:
  - `_get_model_info()` — use appropriate `ModelTask` (e.g. `NLP_MASKED_LM`, `MM_VIDEO_TTT`)
  - `load_model()` — load from HuggingFace with `torch_dtype=torch.bfloat16`
  - `load_inputs()` — generate synthetic inputs matching the model's `forward()` signature
  - `unpack_forward_output()` — extract tensor from dataclass/tuple/dict output
- SPDX header at top

#### 2e. Write `pytorch/__init__.py`
```python
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from .loader import ModelLoader, ModelVariant
```

---

## Step 3 — Validate imports

Run:
```bash
python -c "from third_party.tt_forge_models.<family>.pytorch import ModelLoader, ModelVariant; print('OK')"
```

If this fails, fix the import error before proceeding.

---

## Step 4 — Validate discoverability via collect

Confirm the model is visible to `tests/runner/test_models.py`:
```bash
pytest -q --collect-only tests/runner/test_models.py 2>&1 \
  | grep "test_all_models_torch\[<family>/pytorch-"
```

If no lines appear, the loader failed to import during collection — check the import error in the collect output and fix it before proceeding.

---

## Step 4b — Pre-flight weight_fit (per-arch DRAM gate)

Before any pytest run, estimate parameter count and write
`.claude/bringup/<safe_key>/weight_fit.json`. Full schema:
`model-bringup-multichip/references/weight_fit_schema.md`. DRAM table:
`model-bringup-multichip/references/dram_budget_torch_tp.md`.

**Helper script** (preferred when param count is known):
```bash
python model-bringup-multichip/scripts/compute_weight_fit.py \
  --model-key "<model_key>" \
  --num-params <N> \
  --hf-repo "<org/repo>" \
  --param-source loader \
  --output .claude/bringup/<safe_key>/weight_fit.json
```
For pipeline models, add `--component <name>` and `--test-path <pytest node>`.

**Do not hard-reject** large models or route to multichip here — single-device
bringup always proceeds on eligible arch(es); multichip is promotion-only
after HW proves weight-bound on every eligible arch.

### Parameter estimate (order of preference)

1. **Live count** from loader (`sum(p.numel())`) when cheap.
2. **Config-only** (`AutoConfig.from_pretrained`).
3. **Name heuristic** (same regex table as `failure_summary`).

### Per-arch DRAM (single chip)

```python
DRAM_BYTES = {"n150": 12 * 2**30, "p150": 32 * 2**30}  # Wormhole / Blackhole
BUDGET_FRAC = 0.85
weight_bytes_fp32 = num_params * 4
weight_bytes_bf16 = num_params * 2
```

For each arch in `n150`, `p150`:

- `budget_bytes = int(BUDGET_FRAC * DRAM_BYTES[arch])`
- `fits_fp32 = weight_bytes_fp32 <= budget_bytes`
- `fits_bf16 = weight_bytes_bf16 <= budget_bytes`
- `weight_predicted = not fits_fp32`
- `eligible_single_chip[arch] = fits_fp32 or fits_bf16`

`eligible_archs` = list of arches with `eligible_single_chip` true.
`p150_only` = eligible is only `["p150"]`.

### Persist

1. Write `weight_fit.json` (monolithic: one object; see schema).
2. Record **source dtype** from loader `DTYPE`, HF config, or inference script
   → `details.source_dtype` in `state.json` (see `dtype_ladder.md`).
3. Mirror summary in `state.json`:
   - `details.param_estimate`, `details.weight_fit_path`
   - `details.eligible_archs`, `details.weight_predicted_any`

### Orchestrator hints (print to terminal)

```
[scaffold] weight_fit: eligible_archs=[n150, p150]  (or [p150] only)
  n150: fits_fp32=... fits_bf16=... weight_predicted=...
  p150: fits_fp32=... fits_bf16=... weight_predicted=...
  Next: /model-bringup <key> --archs n150,p150   # runs BOTH when both eligible
```

If **no** arch is eligible even at bf16, set `details.all_arches_ineligible=true`
and warn — orchestrator may still run one probe or suggest early promotion after
user confirms.

### Optional shard hint (non-blocking)

If `weight_predicted` on any arch, still record a **hint-only**
`state.details.shard_plan_hint` for escalation (mesh suggestion) — do **not**
block VALIDATE and do **not** skip single-chip runs.

---

## Step 4c — Optional: custom per-model test file

The generic `tests/runner/test_models.py` discovery handles the vast
majority of models. **Skip this step** unless one of the following is true:

1. The user passed `--custom-test`.
2. **Auto-detect trigger** — any of:
   - `load_inputs()` returns a dict whose keys do not match the
     `model.forward` signature exactly (post-`unpack` keys, kwargs-only
     args, etc.).
   - The loader exposes `load_model(component=...)` (pipeline-style
     multi-component model — the generic runner only sees the default
     component).
   - `unpack_forward_output` returns a non-tensor that the runner cannot
     coerce (e.g. a list of tensors with different shapes).

If triggered, scaffold writes
`third_party/tt_forge_models/<family>/pytorch/tests/test_<family>.py`
following the template below (and creates `tests/__init__.py` if absent).

Template:
```python
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import torch_xla.core.xla_model as xm

from third_party.tt_forge_models.<family>.pytorch import ModelLoader, ModelVariant


@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category="model_test",
    model_name="<family>",
    run_mode="inference",
    parallelism="single_device",
    bringup_status="<inferred BringupStatus, default INCOMPLETE>",
)
def test_<family>_inference_single_device():
    loader = ModelLoader()
    model = loader.load_model().eval()
    inputs = loader.load_inputs()

    device = xm.xla_device()
    model = model.to(device)
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    out = model(**inputs) if isinstance(inputs, dict) else model(*inputs)
    tensor = loader.unpack_forward_output(out) if hasattr(loader, "unpack_forward_output") else out
    xm.mark_step()

    assert torch.is_tensor(tensor)
    assert torch.isfinite(tensor.cpu()).all(), "non-finite output on device"
```

Notes:
- Leave the assertion body intentionally light — the runner's PCC and
  bringup_status reporting do the heavy lifting elsewhere.
- Mark `bringup_status=INCOMPLETE` initially; `model-bringup-config-update`
  flips it at the end of the pipeline.
- If `--custom-test` was passed but no auto-trigger fired, still write the
  file. Do not overwrite an existing test_<family>.py — if one is already
  present, log "custom test already exists" and continue.

Record in state.json:
```json
"details": { "custom_test_path": "third_party/tt_forge_models/<family>/pytorch/tests/test_<family>.py" | null }
```

## Step 5 — Initialize bringup state

Create `.claude/bringup/<safe_model_key>/` with subdirectories `logs/` and `patches/`.

Write `state.json`:
```json
{
  "model_key": "<model_key>",
  "arch": "<arch>",
  "stage": "validate",
  "iteration": 0,
  "history": [],
  "applied_patches": [],
  "failure_reasons": [],
  "created_at": <unix_timestamp>,
  "updated_at": <unix_timestamp>
}
```

Append history entry: `{ "stage": "validate", "result": "passed", "details": { "loader_path": "third_party/tt_forge_models/<family>/pytorch/loader.py", "loader_created": true|false } }`.

---

## Step 6 — Write to bringup_steps.txt

Timing capture: at the very start of this skill's execution (Step 1), capture
`_step_start_ts=$(date +%s)` and `_step_start_iso=$(date -Iseconds)`. After
all scaffold work (validation, loader synth, state init) finishes, capture
`_step_end_ts=$(date +%s)` and `_step_end_iso=$(date -Iseconds)`. These
values are emitted in the section below.

Append a section to `.claude/bringup/<safe_key>/bringup_steps.txt`.
If the file does not exist yet (first stage), write the header block first.
The orchestrator has already recorded `pipeline_start_iso` in `state.json`;
read it and emit it as the `Start time` line:
```
================================================================================
MODEL BRINGUP LOG
================================================================================
Model Key  : <model_key>
Arch       : <arch>
Date       : <YYYY-MM-DD>
Start time : <pipeline_start_iso from state.json>
================================================================================
```

Then append:
```
--------------------------------------------------------------------------------
STEP 1 — Parse & Scaffold (model-bringup-scaffold)
--------------------------------------------------------------------------------
Start    : <_step_start_iso>
End      : <_step_end_iso>
Elapsed  : <_step_end_ts - _step_start_ts>s

Input model_key : <original model_key>
Format detected : A (structured) | B (HuggingFace model ID)
Normalized to   : family=<family>  variant=<variant>  parallelism=<p>  run_mode=<r>

Loader path     : third_party/tt_forge_models/<family>/pytorch/loader.py
Loader created  : yes | no

[If created:]
  Reference model : <reference loader path>
  Model class     : <class name>
  HF model ID     : <id>
  Input signature : <key input fields>
  Files written   :
    - third_party/tt_forge_models/<family>/__init__.py
    - third_party/tt_forge_models/<family>/pytorch/__init__.py
    - third_party/tt_forge_models/<family>/pytorch/loader.py

Import validation  : python -c "from ... import ModelLoader, ModelVariant" → OK | FAILED
Collect validation : pytest --collect-only tests/runner/test_models.py | grep '<family>/pytorch-' → <N> test(s) found | NONE FOUND

Weight fit         : weight_fit.json → eligible_archs=[...] per n150(12GiB)/p150(32GiB)
Shard plan hint    : <optional mesh hint if weight_predicted> | n/a

Custom test file   : <path or 'none — generic runner suffices'>

SCAFFOLD RESULT: PASSED | FAILED
```

## Step 7 — Output

On success:
```
[scaffold] PASSED
  loader:         third_party/tt_forge_models/<family>/pytorch/loader.py
  collect check:  <N> test(s) visible in tests/runner/test_models.py
  loader_created: yes | no
```

Only exit with failure (and let the orchestrator escalate) if:
- The HF model ID is unreachable / does not exist
- The model inspection (`AutoModel.from_pretrained`) fails with an unrecoverable error
- The import validation still fails after creating the loader (syntax/logic error)
- The size gate (Step 4b) rejects the model as too large for single-device
