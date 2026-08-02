# Component test patterns

Applies to **`tests/torch/models/<family>/`** — one compile target per file, not
full pipeline `generate()`.

## Runner YAML vs component tests

| Test style | Location | CI selection | PCC |
|------------|----------|--------------|-----|
| **Monolithic** (`test_models.py`) | `tests/runner/test_config/...yaml` | Runner collect + YAML `required_pcc` | YAML + test |
| **Pipeline component** | `tests/torch/models/<family>/` | **`@pytest.mark.nightly`** + `model_test` + `single_device` or `tensor_parallel` | **`ComparisonConfig(required_pcc=0.99)` in the test** — not runner YAML |

Do **not** register **pipeline component** tests — any pytest node under
`tests/torch/models/<family>/` — in `test_config_inference_single_device.yaml`
or `test_config_inference_tensor_parallel.yaml`. That runner path is for
**monolithic** `test_all_models_torch[...]` keys only. Component bringups use
standalone test files and nightly CI filters on markers (see `pytest.ini`:
`nightly`, `model_test`, `single_device`, `tensor_parallel`).

## Per-component single vs multichip (pipeline)

Each pipeline component gets its own decision in `weight_fit.json`:

| Field | Values | Meaning |
|-------|--------|---------|
| `parallelism_mode` | `single_device` \| `tensor_parallel` | Which test node and pytest marker to use |
| `single_device_only` | bool | Never promote this component to TP; replicate on mesh if sibling shards |

Examples on one family:
- `text_encoder`: `single_device`, `single_device_only: true`
- `vae`: `single_device`, `single_device_only: true`
- `transformer`: `tensor_parallel` if weight-bound on all single-chip arches; else `single_device` until promotion

Scaffold **both** test nodes when unsure (e.g. `test_transformer` skipped/xfail OOM +
`test_transformer_sharded` for TP path) — match Krea Realtime / Wan 2.2 patterns.

## Principles

1. **One compile target per test file** — e.g. DiT/transformer step, VAE decode,
   text-encoder encode, not end-to-end multi-minute pipeline loops.
2. **Loaders in tt-forge-models** — tests import `ModelLoader(variant=...)` or
   `ModelLoader(subfolder=...)`; avoid reaching into `pipe.unet` in the test body.
3. **No `@pytest.mark.n150` / `@pytest.mark.p150`** on component tests — use
   runtime skip when `p150_only` (see `arch_eligibility.md`).
4. **Fair CPU/Forge compare** — clone mutable state (KV cache, buffers) before
   CPU golden if `run_graph_test` mutates in-place.
5. **PCC 0.99** — always set in test via `ComparisonConfig` or fixture `pcc=0.99`;
   bringup is not done until `model-bringup-run` passes that test node.

## Standard component roles

Map pipeline submodules to these **roles** when scaffolding `weight_fit.json`
and choosing reference loaders:

| Role | Typical loader variant / subfolder | Example test file |
|------|-----------------------------------|---------------------|
| **transformer** / **dit** / **unet** | `Transformer`, `UNet`, `subfolder="transformer"` | `test_transformer.py`, `test_unet.py`, `test_wan_dit.py` |
| **vae_decoder** | `Vae`, `VAE`, `subfolder="vae"` | `test_vae_decoder.py` |
| **vae_encoder** | encoder-only wrapper | `test_vae_encoder.py` |
| **text_encoder** | T5/CLIP/UMT5 first encoder | `test_text_encoder.py` |
| **text_encoder_2** … **text_encoder_N** | second+ text stacks | `test_text_encoder_2.py`, … |
| **vision_language_encoder** | VLM conditioning stack | (glm_image loader — add test when present) |
| **image_token_step** / **mmgpt** | autoregressive image-token path | Janus `test_image_token_step.py` |
| **gen_img_embed** / **gen_vision_decode** | decode-side helpers | Janus embed/decode components |
| **llm_backbone** | causal LM inside a pipeline | HiDream `shard_llama_specs` path |
| **moe / sparse_mlp** | MoE FFN blocks | `test_deepseek_v4_tp.py`, `test_glm4_moe.py` |

## Families under `tests/torch/models/` (representative)

| Family directory | Components (test files) | Loader / shard reference |
|------------------|-------------------------|--------------------------|
| `mochi/` | vae_decoder (`test_mochi_vae.py`) | `.../mochi/pytorch/` |
| `hunyuan_video/` | transformer, vae_decoder, text_encoder(s) | `.../hunyuan_video/pytorch/` |
| `HiDream_I1/` | transformer, vae_decoder, text_encoder … | `.../hidream_i1/pytorch/` |
| `krea_realtime/` | transformer, vae_decoder, text_encoder | `.../krea_realtime_video/pytorch/` |
| `wan2_2/` | wan_dit, vae_decoder, vae_encoder, umt5_text_encoder | sharded + `required_pcc=0.99` |
| `janus_pro/` | image_token_step, gen_img_embed, gen_vision_decode | `nightly` + `model_test` + `single_device` |
| `playground_v2_5/`, `sdxl_lightning/` | unet, vae, text_encoder | diffusion UNet-style |

List each `components[].test_path` in `weight_fit.json` pointing at the pytest node,
not `test_all_models_torch[...]`.

## Scaffold template (single_device)

```python
@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.single_device
def test_vae_decoder():
    xr.set_device_type("TT")
    loader = ModelLoader(ModelVariant.VAE_DECODER)
    model = loader.load_model(dtype_override=torch.bfloat16)
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)
    run_graph_test(
        model,
        inputs,
        framework=Framework.TORCH,
        comparison_config=ComparisonConfig(pcc=PccConfig(required_pcc=0.99)),
    )
```

## Scaffold template (tensor_parallel / sharded)

```python
@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.tensor_parallel
def test_transformer_sharded():
    xr.use_spmd()
    # CONVERT_SHLO_TO_SHARDY=1 as needed
    _run(sharded=True)  # get_mesh + load_shard_spec
```

## Per-component arch eligibility

Store in **`weight_fit.json`** per component (not pytest arch markers):

```json
{
  "name": "transformer",
  "parallelism_mode": "tensor_parallel",
  "eligible_archs": ["p150"],
  "p150_only": true,
  "test_path": "tests/torch/models/z_image/test_transformer.py::test_transformer_sharded"
}
```

Runtime skip when wormhole and `p150_only`:

```python
from tests.runner.test_utils import get_xla_device_arch

if get_xla_device_arch() == "wormhole" and component["p150_only"]:
    pytest.skip("component requires p150 (Blackhole)")
```

## model-bringup-run mapping

Always set `weight_fit.json` → `components[].test_path` to the node matching
`parallelism_mode`.

**Host-aware runs** (`host_device_probe.md`):

Probe with `--parallelism-mode` from the active component before pytest:

```bash
MODE=$(jq -r '.components[] | select(.name=="<component>") | .parallelism_mode' weight_fit.json)
python .../probe_host.py --parallelism-mode "$MODE" [--expected-mesh-chips N] -o host_probe.json
```

| Mode | Host requirement | n300 llmbox (8 chips, 4 boards) |
|------|------------------|----------------------------------|
| `single_device` | at least one chip | **Run** — unsharded, on whatever mesh it compiles to; no `TT_VISIBLE_DEVICES` |
| `tensor_parallel` | `runtime_chip_count>=2`, mesh ∈ `valid_tp_degrees` | **Run** — `TT_VISIBLE_DEVICES=0,1,2,3` from tt-smi; mesh (1,8) |

- **n150/p150 bringup:** unsharded; runs on whatever mesh it compiles to, so host
  chip count is not a gate. Do **not** set `TT_VISIBLE_DEVICES`. Arch from silicon
  (blackhole → `p150`, wormhole → `n150`).
- **TP bringup:** `TT_VISIBLE_DEVICES` from **tt-smi** board IDs; mesh from **runtime**
  chip count. Install tt-smi if missing; `tt-smi -r` to reset boards after hangs.
- On skip: print `component_skip_reason` and ask user to **change machine**.

## Multichip promotion

- Update **test file** markers / `bringup_status`, not runner YAML.
- Sharded tests: `xr.use_spmd()`, `CONVERT_SHLO_TO_SHARDY=1`, `get_mesh` +
  `load_shard_spec` when needed.
- Record mesh in `scaffold_multichip.json`.
