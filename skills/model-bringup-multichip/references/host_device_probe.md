# Host device probe — chips vs boards (TT_VISIBLE_DEVICES)

**Two counts — do not conflate them.**

| Field | API / source | Example (n300 llmbox) | Example (galaxy-wh-6u) | Used for |
|-------|----------------|----------------------|------------------------|----------|
| **`runtime_chip_count`** | `xr.global_runtime_device_count()` | **8** | **32** | SPMD mesh, `get_mesh_config`, shard across **chips** |
| **`visible_board_count`** | **`tt-smi -ls`** resettable boards | **4** | **from tt-smi** (do not assume) | **`TT_VISIBLE_DEVICES`** board IDs |
| **`chips_per_board`** | `runtime_chip_count / visible_board_count` | **2** | **from probe** | Sanity / documentation |

On n300 llmbox: **4 boards × 2 chips/board = 8 runtime chips**.  
`TT_VISIBLE_DEVICES` enumerates **boards** (0–3), **not** chips (0–7).

**Parser note:** `tt-smi -ls` has two tables — "All available boards" (8 UMD chip
rows on n300) and "Boards that can be reset" (4 board rows). The probe uses the
**resettable section only** for `TT_VISIBLE_DEVICES`. Merging both tables produced
`visible_board_count=12` (Krea bringup, Jun 2026) — fixed in `probe_host.py`.

**Connected board count comes from tt-smi only** — runtime cannot tell you valid
`TT_VISIBLE_DEVICES` indices.

---

## tt-smi (install, list, reset)

Install when probe reports `tt-smi not found on PATH`:

```bash
git clone https://github.com/tenstorrent/tt-smi
cd tt-smi
pip install .
```

```bash
tt-smi -ls    # "Boards that can be reset" → valid TT_VISIBLE_DEVICES board IDs
tt-smi -r     # reset board(s) — use after hang / bad device state before bringup
```

If tt-smi is missing, **skip all HW runs** and print the install steps above.

---

## Probe command

```bash
python <skills>/model-bringup-multichip/scripts/probe_host.py \
  --min-multichip-devices 2 \
  --expected-mesh-chips 8 \
  --parallelism-mode tensor_parallel \
  -o .claude/bringup/<safe_key>/host_probe.json
```

For a **single_device** component:

```bash
python .../probe_host.py --parallelism-mode single_device -o host_probe.json
```

---

## APIs (what to use when)

| Need | Use |
|------|-----|
| Mesh / SPMD size | `torch_xla.runtime.global_runtime_device_count()` → **`runtime_chip_count`** |
| `TT_VISIBLE_DEVICES` | **`visible_board_count`** from **`tt-smi -ls`** (resettable rows) |
| Silicon family | `global_runtime_device_attributes()[0]["device_arch"]` → wormhole / blackhole |
| Lab inventory | `ird list-machines` (boards installed, not XLA indices) |

**ird `NUM BOARDS`** ≠ **`runtime_chip_count`** ≠ **`visible_board_count`**.

---

## Component routing (single vs multichip)

Before **any** pytest, read `weight_fit.json` → active component → `parallelism_mode`.
Then check `host_probe.json` (re-run probe if stale):

| `parallelism_mode` | Requires | n300 llmbox (4 boards, 8 chips) | galaxy-wh-6u (32 chips) |
|--------------------|----------|-----------------------------------|-------------------------|
| **`single_device`** | `runtime_chip_count >= 1` | **OK** — unsharded | **OK** — same |
| **`tensor_parallel`** | `runtime_chip_count >= 2` + mesh ∈ `valid_tp_degrees` | **OK** mesh 8 (or 2/4); `TT_VISIBLE_DEVICES` from tt-smi | **OK** mesh **32** (or 2/4/8); `TT_VISIBLE_DEVICES` from tt-smi |

**Valid multichip TP degrees:** powers of 2 dividing `runtime_chip_count` (probe field
**`valid_tp_degrees`**). Bringup promotion ladder uses **`{2, 4, 8, 32}`** (ignores 16).

| Host | `runtime_chip_count` | Typical `valid_tp_degrees` | Promotion mesh examples |
|------|----------------------|----------------------------|-------------------------|
| lb-blackhole / qb2 | **4** | `[2, 4]` | `(1,4)`, `(2,2)` |
| n300-llmbox | **8** | `[2, 4, 8]` | `(1,8)`, `(2,4)` |
| galaxy-wh-6u | **32** | `[2, 4, 8, 32]` | `(4,8)`, `(8,4)` — see `architecture_shard_templates.md` |

Use **`can_run_component`** and **`component_skip_reason`** from probe when
`--parallelism-mode` is passed.

### Skip message (orchestrator must print and stop — no pytest)

```
[host_skip] Component "<name>" requires <parallelism_mode> but this host cannot run it.
  runtime_chip_count=<N>  visible_board_count=<B>
  recommended TT_VISIBLE_DEVICES=<multichip_bringup value>
  Reason: <component_skip_reason | single_chip_skip_reason | multichip_skip_reason>
  Action: <set TT_XLA_ARCH so the silicon can be attributed | switch to a multichip host with N chips> and re-run.
```

---

## Export env (multichip)

### n300-llmbox (8 chips)

```bash
CHIPS=$(jq -r '.runtime_chip_count' host_probe.json)
VIS=$(jq -r '.recommended_tt_visible_devices.multichip_bringup' host_probe.json)

export TT_XLA_ARCH=n300-llmbox
export TT_VISIBLE_DEVICES="$VIS"    # boards from tt-smi — e.g. 0,1,2,3 (NOT 0..7)
export TT_XLA_SPMD=1
export CONVERT_SHLO_TO_SHARDY=1
```

Loader `get_mesh_config(num_devices)` uses **`runtime_chip_count`** (8).  
Only **`TT_VISIBLE_DEVICES`** uses board indices from tt-smi.

### galaxy-wh-6u (32 chips)

```bash
CHIPS=$(jq -r '.runtime_chip_count' host_probe.json)   # expect 32
VIS=$(jq -r '.recommended_tt_visible_devices.multichip_bringup' host_probe.json)

export TT_XLA_ARCH=galaxy-wh-6u
export TT_VISIBLE_DEVICES="$VIS"    # board IDs from tt-smi — NEVER hardcode 0..31
export TT_XLA_SPMD=1
export CONVERT_SHLO_TO_SHARDY=1
```

Probe with `--expected-mesh-chips 32` when scaffold/promotion targets Galaxy.

**Mesh / shard specs at 32** (family-dependent — copy from nearest loader):

| Pattern | Typical mesh @ 32 | Reference |
|---------|-------------------|-----------|
| Megatron 1D (DiT / pipeline) | `(8, 4)` | `architecture_shard_templates.md` Pattern A |
| FSDP-style LLM (70B+) | `(4, 8)` | `llama/causal_lm/pytorch/loader.py` `num_devices==32` |
| MoE (DeepSeek V3.2, GPT-OSS) | `(4, 8)` | `deepseek/deepseek_v3_2_exp/`, `gpt_oss/pytorch/loader.py` |

See also `dram_budget_torch_tp.md` and `pytorch_multichip_tp.md` (>24 GiB/device → Galaxy).

---

## Skip rules

| Check | Skip if |
|-------|---------|
| `parallelism_mode=single_device` | `can_run_single_chip_bringup` false |
| `parallelism_mode=tensor_parallel` | `can_run_multichip_bringup` false |
| Expected mesh not in `valid_tp_degrees` | e.g. scaffold wants 6-way on 8-chip host |
| `TT_VISIBLE_DEVICES` max index ≥ `visible_board_count` | env lists chip IDs 4–7 when boards 0–3 |
| tt-smi missing | cannot confirm board count — skip HW, show install steps |

Print `component_skip_reason` / `board_vs_chip_note` to user.

---

## n300 llmbox summary

```
runtime_chip_count     = 8     → mesh (1, 8) or (2, 4)     valid_tp_degrees: [2, 4, 8]
visible_board_count    = 4     → TT_VISIBLE_DEVICES=0,1,2,3  (tt-smi -ls)
single_device          = YES   → run unsharded; recorded as n150
```

---

## lb-blackhole / qb2 summary (4 chips)

```
runtime_chip_count     = 4     → mesh (1, 4) or (2, 2)     valid_tp_degrees: [2, 4]
visible_board_count    = from tt-smi (often 4 board IDs — confirm with probe)
single_device          = YES   → run unsharded; recorded as p150
TT_XLA_ARCH            = lb-blackhole or qb2-blackhole
```

Use `--expected-mesh-chips 4` when promotion/scaffold targets 4-way TP.

---

## galaxy-wh-6u summary (32 chips)

```
runtime_chip_count     = 32    → mesh (4, 8) or (8, 4)     valid_tp_degrees: [2, 4, 8, 32]
visible_board_count    = from tt-smi ONLY — do not derive from runtime_chip_count
TT_VISIBLE_DEVICES     = probe recommended_tt_visible_devices.multichip_bringup
single_device          = YES   → run unsharded; recorded as n150
TT_XLA_ARCH            = galaxy-wh-6u
```

**Board vs chip on Galaxy:** same rule as llmbox — **`global_runtime_device_count()`**
returns **32 chips** for mesh/SPMD; **`tt-smi -ls`** resettable rows define valid
**`TT_VISIBLE_DEVICES`** board indices. Never set `TT_VISIBLE_DEVICES=0,1,...,31`
unless tt-smi confirms that many resettable boards.

**When to use Galaxy bringup:** `promotion.json` / `write_promotion.py` suggests
**`suggested_chip_count: 32`** when per-chip weight budget exceeds llmbox capacity
(see `dram_budget_torch_tp.md` chip ladder `{1,2,4,8,32}`).

Probe example:

```bash
python .../probe_host.py \
  --parallelism-mode tensor_parallel \
  --expected-mesh-chips 32 \
  -o host_probe.json
```

After hang or stale device state: **`tt-smi -r`** before re-run.

**Partial llmbox sessions:** some hosts expose fewer resettable boards (e.g.
`visible_board_count=2` → `TT_VISIBLE_DEVICES=0,1`, `runtime_chip_count=4`).
Probe is authoritative — do not assume full 4-board / 8-chip unless tt-smi shows it.
Re-run probe after `tt-smi -r` if chip count changed.

---

## Single-chip (unchanged)

Requires **`runtime_chip_count == 1`**. No fabric pinning via `TT_VISIBLE_DEVICES=0`.
