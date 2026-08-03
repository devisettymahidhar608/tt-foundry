---
name: model-bringup-scaffold-pipeline
description: Scaffold per-component loaders AND single-device component tests for pipeline models (Stable Diffusion family, video-generation DiTs, text-to-image MMDiTs, any multi-stage HF DiffusionPipeline). Where model-bringup-scaffold writes one loader that returns one model, this skill enumerates the pipeline's components, captures each component's real I/O shapes/dtypes from one CPU forward, then writes (a) a loader where load_model() returns the requested *component* wrapped to a clean tensors-only forward and load_inputs() builds matched synthetic tensors, and (b) one standalone pytest per component under tests/torch/models/<family>/. It brings each component up on a SINGLE device first — components that fit pass, components that exceed device DRAM are marked xfail with the exact OOM message and a tracking issue — and emits tensor-parallel shard specs for the follow-up multi-chip bringup. Use this skill whenever the HF target is a DiffusionPipeline or any multi-component pipeline — not the default model-bringup-scaffold.
allowed-tools: Bash Read Write Edit Grep Glob AskUserQuestion
---

# Model Bringup — Pipeline Scaffold

You are the **pipeline-specialized** scaffold stage of the model bringup pipeline.
Use this skill *instead of* `model-bringup-scaffold` when the HuggingFace
target is a multi-component pipeline (anything that loads via
`DiffusionPipeline.from_pretrained(...)` and exposes `.unet` / `.transformer` /
`.vae` / `.text_encoder` etc.).

## Invocation
`/model-bringup-scaffold-pipeline <hf_repo_or_model_key> [--arch <arch>] [--device <n150|p150|auto>]`

## Why this exists

The default scaffold skill assumes one model → one loader → one test. Pipeline
models break that assumption in three places:

1. **`load_model()` returning the whole pipeline** forces test scripts to dig
   `pipe.unet`, `pipe.vae`, etc. out themselves. All model-shaping code must
   live in `tt-forge-models`, not in the runner.
2. **`load_inputs()` for a multi-component pipeline is ambiguous** — inputs to
   the UNet/transformer, VAE, and text encoder are different shapes and
   dtypes, and several components take non-tensor structural args. A single
   `load_inputs()` cannot serve all of them.
3. **Device sizing is per-component, not per-pipeline.** A 28 B aggregate
   pipeline never fits one device, but its individual components might. Bring
   each component up *independently* on a single device; only the oversized
   ones need a multi-chip fabric.

This skill scaffolds a loader where each component is a *variant*, with its
own `load_model()` slice (wrapped to a tensors-only `forward`) and its own
`load_inputs()` builder generated from real captured shapes/dtypes — not
guessed — plus one standalone pytest per component.

---

## Guiding principle — single-device first

**Do not start with multi-chip.** Even when the aggregate pipeline is far too
big for one device, bring each component up on a single device first. This is
the convention proven by the Playground v2.5 and Qwen-Image component-test PRs:

- The component that fits (usually the VAE, ~0.1–0.3 B) **passes** on one
  device — a real, mergeable signal.
- The components that exceed device DRAM (text encoder, UNet/transformer)
  **xfail deterministically with an OOM** — captured with the exact
  `TT_FATAL` message and a tracking issue.

A single-device component-test PR with one pass + a couple of OOM xfails is a
complete, landable bringup. Tensor-parallel multi-chip is a **follow-up**, not
a prerequisite — but you still emit `weight_fit.json` and shard specs now
(Steps 6–7) so the follow-up has everything it needs.

OOM here is a **capacity fact, not a bug**: if a component's weights exceed the
device's DRAM bank capacity, it OOMs every time. Do not re-run to "confirm" the
xfail, and do not route it to the runtime debugger — it is a sizing result.

---

## Step 0 — Confirm target is a pipeline

Before doing any work, verify the HF id resolves to a pipeline class:

```python
import json, huggingface_hub
p = huggingface_hub.hf_hub_download("<hf_repo>", "model_index.json")
idx = json.load(open(p))
print(idx.get("_class_name"))
print({k: v for k, v in idx.items() if not k.startswith("_")})
```

- If `_class_name` ends in `Pipeline` and `model_index.json` lists multiple
  sub-folders (`text_encoder`, `unet`/`transformer`, `vae`, `scheduler`, …)
  → proceed.
- Otherwise → **abort** with `result=wrong_skill, hint="use model-bringup-scaffold"`.

Do not fall through to single-component scaffolding here — wrong-target
detection should be loud, not silent.

---

## Step 1 — Parse model_key / hf_repo

Accept either:
- A bare HF repo id (`Qwen/Qwen-Image`, `playgroundai/playground-v2.5-1024px-aesthetic`)
- A structured `model_key` (`playground_v2_5/pytorch-Aesthetic_1024px-single_device-inference`)

Derive `family` from the repo slug (lowercased, non-alnum → `_`), e.g.
`Qwen-Image` → `qwen_image`. Truncate to a reasonable length (≤ 40 chars).
Prefer the short, human-recognizable family name (`qwen_image`, not
`qwen_qwen_image`).

Persist the chosen `family`, `variant_prefix` (the human-readable variant
stem before the per-component suffix), and `hf_repo` in
`.claude/bringup/<safe_key>/scaffold_pipeline.json` so later steps don't
re-derive.

---

## Step 2 — Pull architecture + parameter counts (NO weights yet)

Goal: enumerate the pipeline's components and their parameter counts *before*
downloading multi-GB weights. Use HF `config.json` per subfolder where
possible.

```python
from huggingface_hub import hf_hub_download
import json
idx = json.load(open(hf_hub_download("<hf_repo>", "model_index.json")))

components = {}              # name -> {class_name, params_estimate, source_file}
for name, spec in idx.items():
    if name.startswith("_") or not isinstance(spec, list): continue
    module_name, class_name = spec   # e.g. ["diffusers", "QwenImageTransformer2DModel"]
    try:
        cfg = json.load(open(hf_hub_download("<hf_repo>", f"{name}/config.json")))
    except Exception:
        cfg = {}
    components[name] = {"module": module_name, "class": class_name, "config": cfg}
```

For each component, compute an approximate parameter count:

- **UNet / transformer (DiT/MMDiT)**: prefer a `num_parameters` field if the
  config carries one; otherwise scale from `hidden_size`/`num_attention_heads`
  × `num_layers`. A 60-block MMDiT at hidden 3072+ is ~20 B — size it before
  you plan, because it dictates the chip count.
- **VAE**: usually ~0.1–0.3 B; coarse scaling from `block_out_channels` /
  `layers_per_block` is fine.
- **Text encoder(s)**: `AutoConfig.from_pretrained("<repo>", subfolder="text_encoder")`
  → `hidden_size² × num_hidden_layers × 12`. Note: a text encoder can itself
  be a large VLM (Qwen-Image's is Qwen2.5-VL 7B) — do not assume "text encoder"
  means "small."
- **Scheduler / safety_checker / feature_extractor / tokenizer**: skip — no
  parameters to compile.

Fallback: name-based heuristic from `failure_summary` (`\d+B`/`\d+M` regex,
size words). The fallback applies *per component*, not to the aggregate.

Save the enumeration as a table in `scaffold_pipeline.json` (`components`,
`total_params`, `denoising_steps_default`).

---

## Step 3 — Locate each component's source file via class name

For every entry in the components table, resolve `<module>.<class>` to a real
file path on disk so step 4 can patch in logging:

```python
import importlib, inspect
mod = importlib.import_module(module_name)
cls = getattr(mod, class_name)
src = inspect.getsourcefile(cls)
```

Record `source_file` in `scaffold_pipeline.json`. If `inspect.getsourcefile`
returns `None`, fall back to `mod.__file__` and grep for `class <class_name>(`.
Schedulers / tokenizers / feature extractors are intentionally skipped.

---

## Step 4 — Insert logging instrumentation (offline capture only)

Goal: capture the **real** input/output tensor shapes and dtypes for every
component on one forward pass. This is what step 5's `load_inputs()` builders
consume.

For huge pipelines, prefer a **per-component monkey-patch** over copying source
files: wrap each component's `forward` (and the VAE's `decode`) on the loaded
class object to log args before delegating, e.g.

```python
def _probe(orig, tag):
    def wrapped(self, *a, **kw):
        named = {**{f"arg{i}": v for i, v in enumerate(a)}, **kw}
        shapes = {k: tuple(v.shape) for k, v in named.items() if hasattr(v, "shape")}
        dtypes = {k: str(v.dtype) for k, v in named.items() if hasattr(v, "dtype")}
        nonten = {k: v for k, v in named.items() if not hasattr(v, "shape")}
        _BR_LOG.info("FWD_IN %s shapes=%s dtypes=%s nontensor=%s", tag, shapes, dtypes, nonten)
        out = orig(self, *a, **kw)
        return out
    return wrapped
```

**Capture the non-tensor structural args too** (`img_shapes`, `txt_seq_lens`,
`guidance`, `added_cond_kwargs` dict fields, `return_dict`). These are the args
the wrapper in Step 8 must *pin* — if you only log tensor shapes you will not
know what to hardcode and the wrapper's `forward` will be wrong.

**Critical:** never mutate the installed `diffusers` / `transformers` source
tree — that contaminates every other test on the machine. Monkey-patch on the
loaded class objects (auto-reverts when the process exits), or use a
`PYTHONPATH` overlay you delete afterward. Persist the strategy in
`scaffold_pipeline.json` under `capture.strategy`.

---

## Step 5 — Run one tiny pipeline pass and harvest

Run the pipeline **once** with minimal inputs (lowest resolution, batch=1,
`num_inference_steps=2`) on CPU, logging to
`.claude/bringup/<safe_key>/capture.log`:

```bash
python - <<'PY'
import logging, torch
from diffusers import DiffusionPipeline
logging.basicConfig(level=logging.INFO)
pipe = DiffusionPipeline.from_pretrained("<hf_repo>", torch_dtype=torch.bfloat16)
pipe.to("cpu")  # we only need shapes; do not waste a TT card on this
# ... install monkey-patches from Step 4 here ...
out = pipe("a small red cube on a wooden table",
           num_inference_steps=2, height=256, width=256)
PY
```

**Slow-VAE escape hatch.** The VAE *decode* on a full-res latent can take many
minutes on CPU and you do not need it to finish — once the text-encoder and
transformer/UNet `FWD_IN` lines are logged, the run can bail. The VAE's input
shape is fully determined by the latent channel count and the
height/width/8 (and temporal dim for video/3D VAEs), so derive it analytically
rather than waiting for the decode. Record in the log which components were
captured live vs. derived (Qwen-Image's `capture_v2` bailed before the slow
decode and derived the VAE latent shape `[1,16,1,32,32]` from config).

Parse `capture.log` into a per-component **I/O spec** dict and record:
- tensor inputs (name, shape, dtype),
- **non-tensor inputs** (the pinned structural args, with their literal values),
- outputs (shape, dtype),
- `denoise_steps_observed`,
- whether each component is called once per pipeline call or once per step,
- **`provenance` per field** — `observed` or `authored` (see Hard rules).

`observed` means the value was read out of a real call: a `FWD_IN` line, a
tokenizer, the scheduler, or an upstream component's output. Anything else is
`authored` — including analytically derived fields (the VAE latent shape above)
and any field with no upstream producer in your capture chain.

Each `authored` field is an open question. Resolve it against its real producer
before writing the loader:

- **Batch dim** — read the pipeline's call site. CFG (`guidance_scale > 1`)
  concatenates `[uncond, cond]` before the denoise loop, so the DiT runs at
  **batch 2** while text encoders and VAE stay at 1.
- **`timesteps` / `sigmas`** — read them off the scheduler; a float here
  compiles a different graph than production:
  ```python
  sch = SchedulerClass.from_pretrained(repo, subfolder="scheduler")
  sch.set_timesteps(50); sch.timesteps.dtype   # e.g. torch.int64
  ```
- **`input_ids` / masks** — run the real tokenizer on a real prompt.

An unresolvable field may still ship, but stays tagged `authored` and is named
as a guess in `model_overview.md`. Do not call a spec "captured" unless every
field is `observed`.

Save `spec` to `scaffold_pipeline.json` under `io_spec`, and also dump the raw
parsed values to `io_spec_raw.json` for auditability.

---

## Step 6 — Per-component weight_fit (n150 + p150 single-chip)

For **each** component in the Step 2 table, compute `weight_fit` using the
same rules as `model-bringup-scaffold` Step 4b and
`model-bringup-multichip/references/dram_budget_torch_tp.md`:

- **n150** = 12 GiB, **p150** = 32 GiB per device, **85%** weight budget.
- `activation_class`: `video` if pipeline class matches
  `Video|I2V|T2V|HunyuanVideo|LTX|Wan|CogVideo|Mochi`; else `image` for
  diffusion pipelines; `text_encoder` for TE subfolders; `vae` for vae.

Write **one** merged file:
`.claude/bringup/<safe_key>/weight_fit.json` with `components[]` (see
`model-bringup-multichip/references/weight_fit_schema.md`).

Per component record:
- `eligible_archs` — arches where `fits_fp32` or `fits_bf16`
- `p150_only` — true when only p150 is eligible (Janus Pro-7B pattern)
- `weight_predicted` per arch — hint for orchestrator, **not** a skip of single-chip
- **`parallelism_mode`** — **`single_device`** or **`tensor_parallel`** (per
  component, not one choice for the whole pipeline). Decide from weight_fit:
  - **`single_device`**: bf16 weights fit on one chip on at least one eligible
    arch → `@pytest.mark.single_device`; bringup runs
    `tests/torch/models/<family>/test_<role>.py` (unsharded path).
  - **`tensor_parallel`**: weight-bound on every eligible single-chip arch →
    scaffold a sharded test (e.g. `test_transformer_sharded`) and/or promote
    later via `/model-bringup-multichip`; `@pytest.mark.tensor_parallel` on the
    sharded node (Krea/Mochi/Wan pattern).
- **`single_device_only`**: true when `parallelism_mode == single_device` and
  the component must **never** be the primary TP promotion target (encoders, VAE).
  On a multichip mesh for a sibling DiT, replicate only (`load_shard_spec` → `None`).
- **`test_path`** — exact pytest node for `model-bringup-run` (mandatory).

**Do not** pick one parallelism mode for the entire pipeline — text_encoder may
be `single_device` while transformer is `tensor_parallel` on the same family.

Multichip TP is **promotion-only** for components with
`parallelism_mode: tensor_parallel` (or after single-chip exhaust + `promotion.json`).

Optional `promotion_hint` per component in `scaffold_pipeline.json` (mesh
chip count **estimate only** for humans) — must not auto-run multichip.

Copy `eligible_archs` summary into `scaffold_pipeline.json` under
`components[].weight_fit`.

---

## Step 7 — Device target & shard-spec planning (for the multi-chip follow-up)

Even though Step 9 tests on a single device first, compute and record the
multi-chip plan now so the follow-up bringup is ready.

Ask the user (via `AskUserQuestion`) which target device (n150 ~7 B/chip,
p150 ~12 B/chip, or `auto`). Then compute the **chip count** per component:

```python
def needed_chips(params, device, is_video_gen):
    derate = 2.0 if (is_video_gen and device == "n150") else (1.5 if is_video_gen else 1.0)
    effective = params * derate
    cap = {"n150": 7_000_000_000, "p150": 12_000_000_000}[device]
    import math
    return max(1, math.ceil(effective / cap))
```

`is_video_gen` is True if the pipeline class name contains
`Video|I2V|T2V|HunyuanVideo|LTX|Wan|CogVideo|Mochi`.

Apply the **uniform-chip-count rule** for the *fabric demo*: if any component
needs N chips, the pipeline-wide `chip_count` is that max (this is the **mesh /
runtime chip count** for SPMD), and you emit shard specs for every component.
Record `device`, `chip_count`, and `shard_specs` in `scaffold_pipeline.json`.
Shard-spec rules:
- `params < per_device_cap` → `data_parallel`.
- `params ≥ per_device_cap` → `tensor_parallel`, mesh `[1, chip_count]`.

**`TT_VISIBLE_DEVICES` is NOT `chip_count`.** Do not set `0..N-1` from
`chip_count` alone. At run time, **`probe_host.py`** reads **tt-smi -ls**
resettable **board** count (`visible_board_count`) — e.g. n300 llmbox with
8 runtime chips → `0,1,2,3` (4 boards), not `0..7`. See `host_device_probe.md`.
Record only a placeholder note in scaffold JSON if needed; the orchestrator
and `model-bringup-run-torch-tp` always export from `host_probe.json`.

If the largest component does not fit at `chip_count = 8`, note
`block_reason="exceeds host fabric capacity"` for the multi-chip path — but
still scaffold the single-device tests (the OOM xfail is itself the record).

---

## Step 8 — Write the per-component loader (direct load + wrapper)

Directory layout (one variant *per component*):

```
third_party/tt_forge_models/<family>/
├── __init__.py
└── pytorch/
    ├── __init__.py          # re-exports ModelLoader, ModelVariant
    └── loader.py
```

`loader.py` must:

1. **Embed the captured I/O spec** as a Python literal near the top
   (`_COMPONENT_IO_SPEC = {...}`) plus the shape constants, so the loader is
   self-contained and reproducible without re-running capture. Embed
   `SHARD_SPECS` from Step 7. Do **not** hardcode `TT_VISIBLE_DEVICES` in the
   loader — resolve at run time from `probe_host.py` / tt-smi (see Step 7).

2. **Never load the whole pipeline.** Fetch each component *directly* via its
   own class + `subfolder=` — loading a 28 B `DiffusionPipeline` just to pull
   one component will OOM host RAM:
   ```python
   # GOOD — direct, one component into RAM:
   from diffusers import QwenImageTransformer2DModel
   transformer = QwenImageTransformer2DModel.from_pretrained(
       repo, subfolder="transformer", torch_dtype=dtype)

   # BAD — pulls every component into RAM first:
   # pipe = DiffusionPipeline.from_pretrained(repo); m = pipe.transformer
   ```

3. **Wrap each component in a thin `nn.Module`** that exposes a *tensors-only*
   `forward`. This is where you (a) pin the non-tensor structural args captured
   in Step 5, (b) select the right callable (some components decode, not
   forward), and (c) unwrap complex output objects to a bare tensor so the
   runner's `run_graph_test` needs no `unpack_forward_output`:
   ```python
   class _QwenImageTransformerWrapper(torch.nn.Module):
       def __init__(self, transformer):
           super().__init__(); self.transformer = transformer
       def forward(self, hidden_states, timestep,
                   encoder_hidden_states, encoder_hidden_states_mask):
           out = self.transformer(
               hidden_states=hidden_states, timestep=timestep, guidance=None,
               encoder_hidden_states=encoder_hidden_states,
               encoder_hidden_states_mask=encoder_hidden_states_mask,
               img_shapes=TR_IMG_SHAPES, txt_seq_lens=TR_TXT_SEQ_LENS,  # PINNED
               return_dict=False)
           return out[0]

   class _QwenTextEncoderWrapper(torch.nn.Module):     # VLM text encoder
       def forward(self, input_ids, attention_mask):
           out = self.text_encoder(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   output_hidden_states=True)
           return out.hidden_states[-1]                # unwrap to a tensor

   class _QwenVAEDecoderWrapper(torch.nn.Module):      # decode, not forward
       def forward(self, latent):
           return self.vae.decode(latent, return_dict=False)[0]
   ```
   The wrapper is the contract: the test feeds only tensors, the runner traces
   a clean graph, and all the pipeline-specific plumbing stays in the loader.

4. **`ModelVariant(StrEnum)`** with one entry per component
   (`<Prefix>_TextEncoder`, `<Prefix>_Transformer`, `<Prefix>_Vae`). Keep them
   stable — the test files import them by name. `_VARIANTS` maps each to a
   `ModelConfig(pretrained_model_name=<repo>)` plus `subfolder` (the HF
   `model_index.json` key). `DEFAULT_VARIANT` should be the headline component
   (the transformer/UNet).

5. **`load_model(dtype_override=...)`** branches on `self._variant`, does the
   direct `from_pretrained(..., subfolder=...)`, calls `.eval()`, and returns
   the wrapped module.

6. **`load_inputs(dtype_override=..., batch_size=<observed call-site batch — do
   not assume 1>)`** synthesizes tensors at the captured shapes with
   `torch.randn` (float dtypes) / `torch.randint` (int dtypes like `input_ids`,
   masks, timesteps). Return them **as a list in the wrapper's `forward` arg
   order**. Do **not** call the pipeline to derive inputs — that breaks
   component isolation.

   Build every field at its captured dtype, not the model dtype: `timesteps`
   captured as `int64` needs `torch.full(..., dtype=torch.int64)`, never
   `torch.ones(b, dtype=dtype)`.

Run `model-bringup-overview`-style sanity: import the loader, resolve each
variant, and confirm `load_inputs()` shapes match `_COMPONENT_IO_SPEC`. This
proves nothing for `authored` fields — those match themselves. Report them as
unverified.

---

## Step 9 — Write standalone single-device component tests

This is the current convention (Playground v2.5, SDXL-Lightning, Qwen-Image,
Krea, HunyuanImage): one pytest **file per component** under
`tests/torch/models/<family>/`, each driving `run_graph_test`. This is **not**
the YAML-runner path — pipeline component bringups use test files.

```
tests/torch/models/<family>/
├── __init__.py                 # SPDX header only
├── test_text_encoder.py
├── test_transformer.py   (or test_unet.py; may add test_transformer_sharded)
└── test_vae_decoder.py
```

Each test file follows this shape (add markers per `weight_fit.json`):

```python
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""<Model> — <Class> (<component>) component test."""

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.evaluators import ComparisonConfig, PccConfig

from third_party.tt_forge_models.<family>.pytorch import ModelLoader, ModelVariant


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.single_device   # or @pytest.mark.tensor_parallel for sharded node
def test_<component>():
    xr.set_device_type("TT")
    torch.manual_seed(42)

    loader = ModelLoader(ModelVariant.<COMPONENT>)
    model = loader.load_model(dtype_override=torch.float32)
    inputs = loader.load_inputs(dtype_override=torch.float32)

    run_graph_test(
        model,
        inputs,
        framework=Framework.TORCH,
        comparison_config=ComparisonConfig(pcc=PccConfig(required_pcc=0.99)),
    )
```

Additional rules:
1. **Test path** — store the exact node in `weight_fit.json` → `test_path`;
   `model-bringup-run` executes that node (not runner YAML collect).
2. **PCC 0.99** — enforce in the test via `ComparisonConfig`, not runner YAML.
3. **Arch guards** — runtime skip from `weight_fit` (`p150_only`); see
   `arch_eligibility.md`. Do **not** use `@pytest.mark.n150` / `@pytest.mark.p150`.
4. **Do not add** `test_config_inference_single_device.yaml` entries for pipeline
   component tests unless the family also has a monolithic `test_models.py` key.
5. **Host routing** — before HW, `probe_host.py` with `--parallelism-mode` from
   `weight_fit.json`. **`TT_VISIBLE_DEVICES`** from tt-smi boards
   (`visible_board_count`); mesh from **`runtime_chip_count`**. On n300 llmbox:
   `0,1,2,3` for 8 chips — never derive VIS from chip count. See
   `host_device_probe.md`. Install tt-smi if missing; `tt-smi -r` resets boards.

Notes:
- Tests run **fp32** (`dtype_override=torch.float32`) even though the loader
  default is bf16 — `run_graph_test` compares against an fp32 CPU golden.
- `inputs` is the **list** `load_inputs` returns, in the wrapper's arg order.
- Confirm the files **collect** (`pytest --collect-only`) before running on HW.

Then **run each component on a single device** (delegated to
`model-bringup-run`, or directly with a generous timeout — large weight
downloads + compile can take 10–70 min). Set `IRD_LF_CACHE` for IRD-backed
models. Per outcome:

- **Passes** → leave the plain `@nightly` + `@model_test` markers.
- **OOMs** (the big components, by design) → add `@pytest.mark.xfail` *above*
  the existing markers, with the **exact** `TT_FATAL` message and a tracking
  issue:
  ```python
  @pytest.mark.xfail(
      reason="Out of Memory: Not enough space to allocate 150994944 B DRAM "
             "buffer across 12 banks ... (allocated: 1056285728 B, free: "
             "15536064 B) — 20B QwenImageTransformer2DModel does not fit a "
             "single device; needs multi-chip tensor-parallel — "
             "https://github.com/tenstorrent/tt-xla/issues/NNNN")
  ```
  Draft the issue body (component, class, params, full error, repro, the
  tensor-parallel resolution) into
  `.claude/bringup/<safe_key>/issue_<component>_oom.md`. Filing the issue and
  replacing `NNNN` is an outward-facing action — leave it to the operator
  unless told otherwise; until then a clearly-marked `issues/TBD-<COMPONENT>`
  placeholder is acceptable in the committed reason.
- **Other failures** (PCC drop, compile error) → route through
  `model-bringup-diagnose` / `runtime-failure-debugger`, not a blanket xfail.

The **tensor-parallel YAML entries** (Step 7 shard specs registered in
`test_config_inference_tensor_parallel.yaml` as `NOT_SUPPORTED_SKIP`) remain
the record for the multi-chip follow-up. They coexist with these single-device
test files; do not delete them.

Run `pre-commit` on the new test files (black collapses imports, isort orders
them, copyright header is required) before handing off.

This skill does *not* execute tests — `model-bringup-run` runs `test_path` with
PCC enforced by the test's `ComparisonConfig`.

---

## Step 10 — State, log, and output

`state.json` gets `details.scaffold_variant: "pipeline"`, plus
`components_scaffolded`, `denoise_steps_observed`, `device`, `chip_count`,
`weight_fit_path`, and `io_spec_path`. The `stage` advances to
`SCAFFOLD_DONE` and the per-component single-device results
(`passed` / `xfail_oom`) are recorded in `history`.

Append a `STEP 1P — Pipeline Scaffold` block to `bringup_steps.txt` capturing:
HF repo, family, pipeline class, total params, per-component table (class /
params / source), capture pass + which components were live vs derived, device
+ chip-count + shard specs, loader path, variants written, **test files
written**, and the per-component single-device result (PASSED / xfail-OOM).

Terminal output (one screen) should list: hf repo, components, total params,
device, chip_count, loader path, test files, and per-component single-device
result, then `next stage: file OOM issues + open PR` (or `model-bringup-run`
if any component is still pending a HW run).

---

## Hard rules

- **Tag every captured field with its provenance.** In `io_spec`, mark each
  input `observed` (read from a real call — by the pipeline, scheduler, or
  tokenizer) or `authored` (a value you invented because nothing upstream
  produced it). A loader must not ship an `authored` field: resolve it against
  its real producer first, or state plainly in the overview that it is a guess.
  Never validate `load_inputs()` against a spec field you authored — that
  checks a guess against itself. A default copied from a template is not a
  decision: when a model-specific fact contradicts a template default, the fact
  wins.
- **Single-device first.** Bring each component up on one device before any
  multi-chip work. A PR with one passing component + OOM-xfailed big components
  is complete and landable.
- **OOM is a capacity fact, not a bug.** Component-too-big-for-device OOMs
  deterministically — xfail it with the exact message + issue; do not re-run to
  "confirm" and do not send it to the runtime debugger.
- **Never load the whole pipeline for a large model.** Fetch each component
  directly via `Class.from_pretrained(repo, subfolder=...)`. Pulling a 20–30 B
  `DiffusionPipeline` into host RAM to grab one component will OOM the host.
- **Wrap every component to a tensors-only `forward`.** Pin the captured
  non-tensor structural args inside the wrapper, select the right callable
  (`decode` vs `forward`), and unwrap output objects to a bare tensor. If a
  test file has to pass non-tensor args or unpack the output, the wrapper is
  wrong.
- **Tests are standalone files, not YAML-runner entries.** Pipeline component
  bringups live in `tests/torch/models/<family>/test_<component>.py` and call
  `run_graph_test`. Tests run fp32. The tensor-parallel YAML entries are a
  separate, coexisting record for the multi-chip follow-up.
- **No source-package mutation.** Step 4 instrumentation is a monkey-patch or
  a deletable overlay — never edit the installed `diffusers`/`transformers`.
- **Capture once, on CPU, and bail past the slow VAE decode.** Derive the VAE
  latent shape analytically rather than waiting minutes for a CPU decode.
- **Synthetic inputs only.** `load_inputs()` builds tensors at captured
  shape/dtype; it never re-runs the pipeline.
- **Bail early on wrong target.** If `model_index.json` does not declare a
  `Pipeline`, return `result=wrong_skill` for `model-bringup-scaffold`.
- **Boards vs chips for `TT_VISIBLE_DEVICES`.** Never set `TT_VISIBLE_DEVICES`
  from `runtime_chip_count` / mesh size. Use **`tt-smi -ls`** via
  `probe_host.py` (`visible_board_count`). Mesh uses runtime chips; VIS uses
  board IDs only.
- **Uniform chip count on multichip promotion only.** Single-chip bringup uses
  one device per `single_device` component. When `/model-bringup-multichip`
  promotes a weight-bound component, siblings may **replicate** on that mesh
  (`load_shard_spec` → `None`) but keep `single_device` tests.

---

## Cross-reference

- `model-bringup-scaffold` — sibling skill for single-component models;
  this skill replaces it for pipelines.
- `model-bringup-overview` — golden capture / CPU sanity for each component.
- `model-bringup-run` / `model-bringup-run-torch-tp` — execute component tests.
- `model-bringup-multichip` — promotion-only TP follow-up.
- `model-bringup-multichip/references/host_device_probe.md` — tt-smi routing.
- `model-bringup-diagnose` / `runtime-failure-debugger` — for non-OOM
  failures (PCC, compile).
- `failure_summary` — name-based param heuristic table (step-2 fallback).
