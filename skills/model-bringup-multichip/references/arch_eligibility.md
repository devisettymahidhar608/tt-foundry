# Arch eligibility — n150 and p150 single-chip

## Host probe (current machine)

Before any HW run, use **`host_device_probe.md`** and `scripts/probe_host.py`.

- **n150/p150 bringup (`single_device`):** runs on **any** host with at least one
  chip — the component uses whatever mesh its graph compiles to, so host chip count
  does not gate it. **Do not set `TT_VISIBLE_DEVICES`.** Arch comes from the observed
  silicon: blackhole → `p150`, wormhole → `n150`.
- **Multichip TP (`tensor_parallel`):** `runtime_chip_count >= 2`; `TT_VISIBLE_DEVICES` from
  **tt-smi** resettable boards; mesh from runtime chip count.
- **Connected boards:** known via **`tt-smi -ls` only** — not from runtime device count.
- Install tt-smi if missing: `git clone https://github.com/tenstorrent/tt-smi && cd tt-smi && pip install .`
- Reset after bad state: `tt-smi -r`

## Component vs host matrix

| Host | `runtime_chip_count` | `single_device` (unsharded) | `tensor_parallel` |
|------|----------------------|------------------------------------|-------------------|
| Dedicated n150/p150 | 1 | **Run** | **Skip** — need multichip host |
| lb-blackhole / qb2 | 4 | **Run** as `p150` | **Run** if mesh ∈ {2,4}; `TT_VISIBLE_DEVICES` from tt-smi |
| n300 llmbox | 8 | **Run** as `n150` | **Run** if mesh ∈ {2,4,8}; e.g. `TT_VISIBLE_DEVICES=0,1,2,3` |
| galaxy-wh-6u | 32 | **Run** as `n150` | **Run** if mesh ∈ {2,4,8,32}; meshes `(4,8)` / `(8,4)` — see `host_device_probe.md` |

So a whole pipeline is brought up on one machine: the weight-bound component runs
N-way TP, and every other component runs `single_device` **unsharded** on the same
box, on whatever mesh it compiles to. Record the mesh the passing run actually
used — do not assume it.

Orchestrator checks `weight_fit.json` → `parallelism_mode` and probe → `can_run_component`.
On mismatch: `host_skip`, print `component_skip_reason`, ask user to change machine.

## Defaults

- **`--archs n150,p150`** when `weight_fit.json` → `eligible_archs` contains both.
- Run **full single-chip ladder on each** eligible arch in order: **n150**, then **p150**.
- **`--arch p150`** when scaffold marks **p150-only** (e.g. a component that only fits
  Blackhole DRAM — see `weight_fit.json` → `p150_only`).

## weight_fit.json fields

```json
{
  "component": "transformer",
  "weight_bytes_fp32": 40000000000,
  "weight_bytes_bf16": 20000000000,
  "eligible_archs": ["n150", "p150"],
  "per_arch": {
    "n150": { "dram_gib": 12, "fits_fp32": false, "fits_bf16": true, "weight_predicted": true },
    "p150": { "dram_gib": 32, "fits_fp32": true, "fits_bf16": true, "weight_predicted": false }
  },
  "supported_archs": [],
  "p150_only": false
}
```

After HW bringup, `supported_archs` = arches that **PASSED** (subset of eligible).

## Promotion gate

Promote to multichip only when:

1. Every arch in `eligible_archs` completed the ladder (dtype → activation repair → bf16 if needed), and  
2. Each ended **weight_bound**, and  
3. **No** arch PASSED.

If **any** arch PASSED → CONFIG_UPDATE `supported_archs: [n150]` and/or `[p150]` — **no multichip**.

## Runner tests vs component tests

### `tests/runner/` (collect via `test_all_models_torch`)

Arch selection is driven by **YAML config**, not per-test arch markers:

- Single-device:
  `tests/runner/test_config/torch/test_config_inference_single_device.yaml`
- Tensor parallel:
  `tests/runner/test_config/torch/test_config_inference_tensor_parallel.yaml`
- LLM variants:
  `tests/runner/test_config/torch_llm/test_config_inference_*.yaml`

Set **`supported_archs`** on each model entry to the machines that passed bringup
(e.g. `[n150, p150]` or `[p150]` only). The runner resolves `TT_XLA_ARCH` and
filters entries against the current host.

Use **`arch_overrides`** when one logical `model_key` needs different status or
arch lists per machine (see `tests/runner/validate_test_config.py`).

**Do not** add `@pytest.mark.n150` / `@pytest.mark.p150` on runner-collected tests.

### `tests/torch/models/` (component tests)

Component tests are **not** registered in runner YAML. Default: **no arch marker**
— the test runs on whatever device is present (`TT_XLA_ARCH` / current silicon).

Add a **skip guard** only when a component genuinely cannot run on the current
device class. Use `get_xla_device_arch()` from `tests/runner/test_utils.py`:

```python
from tests.runner.test_utils import get_xla_device_arch

def test_transformer():
    arch = get_xla_device_arch()
    if arch == "wormhole" and _component_is_p150_only():
        pytest.skip("transformer requires Blackhole (p150) — see weight_fit.json")
    ...
```

Map device class → bringup arch when documenting skips:

| `get_xla_device_arch()` | Single-chip bringup arch |
|-------------------------|--------------------------|
| `wormhole`              | `n150`                   |
| `blackhole`             | `p150`                   |

Record per-component eligibility in **`weight_fit.json`** (`eligible_archs`,
`p150_only`, `parallelism_mode`). Nightly CI uses **`@pytest.mark.nightly`** +
`model_test` + `single_device` or `tensor_parallel` — not runner YAML.

Bringup orchestrator **must** probe before run — pytest skip guards are a safety net,
not a substitute for `host_skip` on wrong SSH session.

## CONFIG_UPDATE

**Pipeline components:** update `bringup_status` in `tests/torch/models/<family>/`
and keep `required_pcc=0.99` in the test's `ComparisonConfig`.

**Monolithic `test_models.py` keys only:**

```yaml
<model_key>:
  status: EXPECTED_PASSING
  supported_archs: [n150, p150]
```

Use `arch_overrides` when one logical model_key needs different status per arch.
