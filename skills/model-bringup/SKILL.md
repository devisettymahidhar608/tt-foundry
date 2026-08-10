---
name: model-bringup
description: E2E model bringup pipeline orchestrator for Tenstorrent hardware. Drives the full VALIDATE → FIRST_RUN → DIAGNOSE → REPAIR → VERIFY → CONFIG_UPDATE FSM for bringing up a new model in tt-forge-models. Use when the user says "bringup <model>", invokes /model-bringup, or wants to run the full bringup pipeline on a model.
allowed-tools: Bash Read Write Edit Grep Glob Task Agent
---

# Model Bringup Pipeline — Orchestrator

You are the E2E Model Bringup Pipeline orchestrator for Tenstorrent hardware.

## Invocation
`/model-bringup [model_key] [--mode auto|bringup|retriage] [--arch <arch>] [--archs <a>,<b>,...] [--resume]`

Examples:
- `/model-bringup ltx2/pytorch-Fast-single_device-inference --arch n150`
- `/model-bringup ltx2/pytorch-Fast-single_device-inference --archs n150,p150` (multi-arch verify in finalize)
- `/model-bringup perceiverio_vision/pytorch-Vision_Perceiver_Conv-single_device-inference --mode retriage`
- `/model-bringup github:open-mmlab/mmdetection3d@v1.4.0 --arch n150` (GitHub-hosted; routes to scaffold-github)
- `/model-bringup https://github.com/foo/bar --arch n150` (full URL form also accepted)
- `/model-bringup` (no key → list smallest XFAIL candidates and exit)

## Argument Parsing
Parse `$ARGUMENTS`:
- `model_key` (optional): first positional argument.
  - If omitted, see **No-key behavior** below.
- `--mode` (optional, default `bringup`):
  - `bringup` — the existing new-model FSM (VALIDATE → FIRST_RUN → …).
  - `retriage` — the XFAIL re-triage FSM (see **XFAIL Re-triage Mode** below).
    Use this when the entry already exists in the YAML with status
    `KNOWN_FAILURE_XFAIL` and you want to check whether it still reproduces.
- `--arch` (optional): force a **single** arch for the whole bringup (debug or
  p150-only components). Equivalent to `--archs <arch>` with one entry.
- `--archs` (optional): comma-separated single-chip arches, default derived
  from `weight_fit.json` after VALIDATE — typically `n150,p150` when both
  eligible. See **Per-arch single-chip loop** below.
- `--resume` (optional flag): resume from existing state.json instead of starting fresh

If both `--arch` and `--archs` are given, `--archs` wins.

## No-key behavior

If `model_key` is omitted, do **not** run the FSM. Instead:
1. Invoke the `failure_summary` skill internally to generate the digest
   (target arch from `--arch`).
2. Print the **Quick-pick: 3 smallest models** section directly to the
   terminal so the user can copy a candidate key for the next invocation.
3. Print a one-line hint:
   ```
   Run: /model-bringup <model_key> --mode retriage  to re-verify an XFAIL,
        /model-bringup <model_key>                  to bring up a new model.
   ```
4. Exit. Do not create any bringup state.

## State Location
All state lives at `.claude/bringup/<model_key_with_slashes_replaced_by_double_underscore>/state.json`.

## Startup
1. If `--resume` is set and `state.json` exists, load it and resume from the recorded stage.
   Keep the existing `pipeline_start_ts` and `pipeline_start_iso` from state — we want
   total elapsed to include the original run's wall clock, not just the resumed leg.
2. Otherwise, if `state.json` exists, ask the user whether to resume or restart.
3. If no state exists, create a new one by invoking the `model-bringup-scaffold` skill
   (bringup mode only — retriage mode skips scaffold; see below).
4. **Record pipeline start time** before invoking the first stage:
   ```bash
   _pipeline_start_ts=$(date +%s)
   _pipeline_start_iso=$(date -Iseconds)
   ```
   Persist both fields in `state.json` as `pipeline_start_ts` (unix epoch seconds)
   and `pipeline_start_iso` (ISO 8601). Sub-skills read these for the header and
   the final-summary computation.

## XFAIL Re-triage Mode (`--mode retriage`)

This mode is for entries that already exist in
`tests/runner/test_config/torch/test_config_inference_single_device.yaml`
with status `KNOWN_FAILURE_XFAIL`. The goal is to determine whether the
recorded failure still reproduces against current code.

### Entry gate
Before doing anything else, verify the entry's current status in the YAML
(top-level `status` OR `arch_overrides.<arch>.status`). If it is **not**
`KNOWN_FAILURE_XFAIL`:
- Print `[bringup] --mode retriage requires a KNOWN_FAILURE_XFAIL entry; <key> is currently <status>.`
- Suggest dropping `--mode retriage` to use the standard bringup flow.
- Exit.

### Skip scaffold
The loader already exists (the entry is in the YAML, so it has a node id).
Do **not** invoke `model-bringup-scaffold`. Initialize a minimal `state.json`
at `.claude/bringup/<safe_key>/state.json` with `mode: "retriage"` and an
empty `history` array.

### Single delegated step
Invoke the `model_issue_pick` skill with `<model_key> --arch <arch>`. It will
run the pytest with `--runxfail` in the background, classify the log, and
return a verdict. The verdicts and routing:

| Verdict from model_issue_pick | Next action |
|---|---|
| `now_passing`           | Invoke `model-bringup-config-update` with `result=PASSED`. Promote `KNOWN_FAILURE_XFAIL` → `EXPECTED_PASSING`; drop `reason`. Done. |
| `now_incorrect_result`  | Invoke `model-bringup-config-update` with `result=PASSED` AND pass through the recorded PCC so config-update can add `assert_pcc: false` + a lowered `required_pcc`. Done. |
| `xfail_same`            | The recorded failure still reproduces. **Fall into the standard pipeline at DIAGNOSE** with the new run.log so the orchestrator can attempt a fix via REPAIR → VERIFY. The first iteration counts as iteration 1. |
| `xfail_changed`         | The failure is real but different from the YAML's recorded reason. **Fall into the standard pipeline at DIAGNOSE** with the new run.log, same as `xfail_same`. The first iteration counts as iteration 1. |
| `timeout`               | No YAML change. Append a `retriage` history entry with `result=timeout`. Exit. (Consistent with the rule that automated timeouts are not evidence.) |
| `runner_error`          | No YAML change. Surface the error and exit ESCALATED. |

The `xfail_same` vs `xfail_changed` distinction is informational only — it
controls the **final** YAML write at CONFIG_UPDATE time, not the routing:

- If `xfail_same` and the pipeline ends in PASSED → config-update promotes to
  `EXPECTED_PASSING` and drops `reason` (same as `now_passing` path).
- If `xfail_changed` and the pipeline ends in ESCALATED → config-update keeps
  status as `KNOWN_FAILURE_XFAIL` but rewrites `reason` to reflect the new
  failure (the recorded reason is stale).
- If `xfail_same` and the pipeline ends in ESCALATED → leave the YAML
  untouched; the existing reason is still accurate.

Stash the verdict (`xfail_same` | `xfail_changed`) in
`state.retriage_verdict` so CONFIG_UPDATE can read it when the loop ends.

### Logging
Open `.claude/bringup/<safe_key>/bringup_steps.txt` with the same header
block as standard bringup (including `Start time`), but tag the mode:
```
================================================================================
MODEL BRINGUP LOG (mode: retriage)
================================================================================
Model Key  : <model_key>
Arch       : <arch>
Date       : <YYYY-MM-DD>
Start time : <pipeline_start_iso>
================================================================================
```
Append one step section for the `model_issue_pick` invocation, then a step
section per fall-through stage if `xfail_changed` routed into DIAGNOSE.
Every step section follows the standard timing template (`Start:` /
`End:` / `Elapsed:` at the top of the section).

## Host probe (mandatory after VALIDATE; re-run before FIRST_RUN and VERIFY)

Probe the **current SSH session** with `probe_host.py` — see `host_device_probe.md`.

```bash
python <skills>/model-bringup-multichip/scripts/probe_host.py \
  --arch-queue <planned,n150,p150> [--p150-only] \
  -o .claude/bringup/<safe_key>/host_probe.json
```

Persist `state.details.host_probe`. Use **`runtime_chip_count`** for mesh; **`visible_board_count`**
from tt-smi for `TT_VISIBLE_DEVICES` (not ird board count).

**Hard rules:**

| Bringup | Requires | On wrong host |
|---------|----------|---------------|
| **`/model-bringup`** (n150/p150) | `can_run_single_chip_bringup` → **`runtime_chip_count == 1`** | **Skip** all HW; no pytest; no `TT_VISIBLE_DEVICES=0` on fabric |
| **`/model-bringup-multichip`** | `can_run_multichip_bringup` → **`runtime_chip_count >= 2`** + tt-smi boards | **Skip** TP; tell user to change host or fix env |

**Pipeline component gate** (read `weight_fit.json` → active `parallelism_mode`):

```bash
MODE=$(jq -r '.components[] | select(.name=="<component>") | .parallelism_mode' weight_fit.json)
MESH=$(jq -r '.chip_count // 0' scaffold_multichip.json 2>/dev/null || echo 0)
python .../probe_host.py --parallelism-mode "$MODE" --expected-mesh-chips "$MESH" -o host_probe.json
CAN=$(jq -r '.can_run_component // empty' host_probe.json)
if [ "$CAN" = "false" ]; then
  # Print component_skip_reason; stage=host_skip; STOP — user changes machine
fi
```

| `parallelism_mode` | Example on n300 llmbox (8 chips, 4 boards) |
|--------------------|---------------------------------------------|
| `single_device` | **Run** — unsharded, on whatever mesh it compiles to |
| `tensor_parallel` | Run if mesh ∈ `valid_tp_degrees` (2/4/8); `TT_VISIBLE_DEVICES=0,1,2,3` |

Re-probe before **VERIFY** after repair. If skip → `stage: host_skip`, print
`component_skip_reason`, `single_chip_skip_reason`, or `multichip_skip_reason`.

## Per-arch single-chip loop (bringup mode)

After VALIDATE **and host probe**, read `.claude/bringup/<safe_key>/weight_fit.json`:

1. `arch_queue` = `$ARCHS` from CLI if set, else `eligible_archs` from weight_fit
   (order: **`n150` then `p150`** when both present).
2. If `p150_only` on active component, `arch_queue = [p150]`.
3. **Filter** with `probe_host.py` → use `runnable_arch_queue`; record `skipped_archs`.
   If `runnable_arch_queue` is empty → `host_skip` (no pytest on this machine).
4. Persist `state.arch_queue`, `state.arch_results = {}`.

For each `current_arch` in `arch_queue`:

- Set `state.current_arch = current_arch`, `TT_XLA_ARCH=current_arch` for runs.
- Run **OVERVIEW** once globally (first arch only); then per-arch **FIRST_RUN fp32**.
- On fail → invoke **`model-bringup-classify-oom`** with `--arch current_arch`.
- Map class → repair: `activation` → `reduce_resolution` / `enable_vae_tiling` /
  `enable_compile_flags`; then `dtype_bf16_activations` on **same arch**.
- `arch_insufficient` on n150 → record in `arch_results.n150`, **continue** to p150
  (not promotion).
- On **PASSED** for this arch → `arch_results[current_arch] = passed`; continue to
  **next** arch in queue (verify **both** when both eligible).
- On **weight_bound** for this arch (after bf16 ladder) →
  `arch_results[current_arch] = weight_bound`.

After the arch queue:

- If **any** arch `passed` → CONFIG_UPDATE with `supported_archs` = passing arches
  → FINALIZE → **done** (no multichip).
- If **all** arches `weight_bound` (none passed) → invoke
  **`model-bringup-write-promotion`** (writes `promotion.json` via
  `model-bringup-multichip/scripts/write_promotion.py`), set
  `state.stage = promotion_pending`, print:
  ```
  Single-chip exhausted (weight-bound on: <list>). Next:
  /model-bringup-multichip <model_key> --from-promotion
  ```
  Do **not** auto-invoke multichip in the same turn.

Dtype policy: see `model-bringup-multichip/references/dtype_ladder.md` — match
source dtype (bf16 when HF/loader/script uses bf16); fp32-first only when justified.

## FSM Loop
Run the following loop (max 5 iterations total across repair cycles) **within each
arch** in `arch_queue`.
In `--mode retriage`, skip the loop entirely for `now_passing`,
`now_incorrect_result`, `timeout`, and `runner_error` verdicts. For both
`xfail_same` and `xfail_changed` verdicts, enter the loop at **DIAGNOSE**
with the run.log produced by `model_issue_pick`.

```
VALIDATE  →  OVERVIEW  →  FIRST_RUN
                  ↓ cpu_sanity fail → ESCALATE (loader bug; do not run on HW)
                              ↓ pass → CONFIG_UPDATE → FINALIZE → PASSED
                              ↓ timeout (300s) → MANUAL-RUN PAUSE  (handled inside model-bringup-run)
                                                     ↓ user provides log, tail = passed → CONFIG_UPDATE → FINALIZE → PASSED
                                                     ↓ user provides log, tail = failed → DIAGNOSE
                                                     ↓ user replies "skip" or log inconclusive → CONFIG_UPDATE(TIMEOUT) → STOP
                              ↓ fail → DIAGNOSE
                                          ↓ low confidence (iter >= 2) → ESCALATE
                                          ↓ runtime_debug → REPAIR (delegate to runtime-failure-debugger)
                                                              ↓ debug_report.md → PAUSE for human review
                                                              ↓ user fix → VERIFY (sanity gate first)
                                          ↓ → REPAIR
                                                ↓ blocked → ESCALATE
                                                ↓ → VERIFY
                                                      ↓ sanity gate (runtime_debug only)
                                                          ↓ sanity fails → DIAGNOSE (skip full model)
                                                          ↓ sanity passes → run full model
                                                      ↓ pass → CONFIG_UPDATE → FINALIZE → PASSED
                                                      ↓ fail → DIAGNOSE (next iteration)
                                                      ↓ no progress → ESCALATE

FINALIZE (G6+G7) — runs only after a successful CONFIG_UPDATE(--apply):
  for each remaining arch in --archs:
      model-bringup-run → pass → continue
                       → fail → STOP (partial; YAML keeps first arch only)
  pre-commit on changed files
  create local branch bringup/<safe_key> if still on main
  draft PR body
```

### Stage Execution

**VALIDATE**: Pick the scaffold variant based on the model_key:

| model_key shape                                             | Skill invoked                       |
|--------------------------------------------------------------|--------------------------------------|
| `github:<org>/<repo>` / `https://github.com/...` / `git+https://...` | `model-bringup-scaffold-github`    |
| HF id whose `model_index.json` is a `DiffusionPipeline`     | `model-bringup-scaffold-pipeline`   |
| Audio / speech model (see below)                            | `model-bringup-scaffold-audio`      |
| Anything else (structured key or plain HF id)               | `model-bringup-scaffold`            |

All four scaffold variants share the same exit contract: PASSED writes
a loader at `third_party/tt_forge_models/<family>/pytorch/loader.py` and
initialises `state.json`. The orchestrator does not need to special-case
the variant after VALIDATE. On failure → ESCALATED.

**GitHub sources are always submoduled.** `model-bringup-scaffold-github`
adds the upstream repo to `tt-forge-models` as a git submodule at
`third_party/<repo>/` (repo name verbatim) and never copies source into
the tree. The extra artifacts it produces — the `.gitmodules` entry and
the gitlink — must be staged by FINALIZE along with the loader.

Routing precedence: if the user explicitly passed a variant flag (e.g.
hypothetical future `--scaffold github`), respect it; otherwise pick by
the table above. Audio is checked **before** the default row and **after**
the pipeline row (an audio DiffusionPipeline is still a pipeline).

**Audio detection (`model-bringup-scaffold-audio`):** route here when any of
these hold — the model's `ModelTask` is `AUDIO_ASR`, `AUDIO_CLS`, `MM_TTS` or
`MM_AUDIO_TTT`; the processor exposes a `feature_extractor` with a
`sampling_rate`; the config declares `num_mel_bins` / `num_codebooks`; or the
model card describes TTS, ASR, voice cloning, speech translation, vocoding or
music generation. Audio models break the default scaffold's assumptions on
inputs (real waveforms behind a feature extractor), graph shape (many TTS
models have no single traceable forward) and outputs (spectrograms/waveforms,
not logits). That skill classifies into six audio archetypes and picks the
loader shape per archetype.

Archetype-6 (multi-component TTS) families follow the same per-component
single-device discipline as pipeline families, and their natural follow-up
after CONFIG_UPDATE is `audio-pipeline-e2e` for the real text→waveform run.

**Pipeline families (`model-bringup-scaffold-pipeline`):** bring up **each
component separately** using `weight_fit.json` → `components[].test_path`
(under `tests/torch/models/<family>/`). Each component has its own
`parallelism_mode` (`single_device` vs `tensor_parallel`). Do **not** use
runner YAML or `test_all_models_torch[...]` for component PCC — tests must
use `ComparisonConfig(required_pcc=0.99)` and `model-bringup-run` must
execute that node. See `component_test_patterns.md`.

**OVERVIEW**: Invoke `model-bringup-overview` skill with the model_key.
This runs the loader on CPU with random inputs and writes:
- `model_overview.md` — the model card (family, params, modality, signature)
- `cpu_sanity.log` — forward result on CPU
- `golden.pt` (+ `golden_meta.json`) — captured output for downstream PCC

If the CPU sanity check fails, the loader is broken — do **not** continue
to FIRST_RUN. Transition directly to ESCALATED with
`failure_reason="loader_cpu_sanity_failed:<reason>"`. The hardware run
would have produced the same failure at higher cost, and the diagnosis
would be misleading because it would point at TT runtime instead of the
real (loader) cause.

If `weight_fit.json` shows `weight_predicted` on any arch, print a one-line
hint (non-blocking): see `eligible_archs` and `model-bringup-multichip/references/dram_budget_torch_tp.md`.

**FIRST_RUN / VERIFY**: Invoke `model-bringup-run` with model_key and
`state.current_arch` (from per-arch loop).

For **VERIFY** specifically: if the previous repair stage was `runtime_debug`
and recorded a `sanity_test_path` in state, **gate the full-model run on the
sanity test passing first**. Sanity tests are single-op and complete in
seconds, so this avoids spending several minutes on a full-model rerun when
the candidate fix did not actually resolve the failing op.

Sanity-gate procedure (runtime_debug only):
1. Look up `sanity_test_path` from the most recent `repair` history entry.
2. Run `pytest -svv <sanity_test_path> 2>&1 | tee
   .claude/bringup/<safe_key>/logs/iter_<N>_sanity.log`.
3. If the sanity exits **non-zero** (fails or errors) → do **not** invoke the
   full model run. Save the sanity log path to state and transition to
   DIAGNOSE for the next iteration. Append a note to `failure_reasons`
   indicating the sanity gate did not pass (e.g. `sanity_failed:<exit_code>`).
4. If the sanity exits **zero** (passes) → proceed to invoke
   `model-bringup-run` for the full model as below.

For all other repair strategies, skip the sanity gate and invoke
`model-bringup-run` directly.

Then, regardless of how VERIFY arrives at the full-model invocation:
- On pass → transition to CONFIG_UPDATE.
- On timeout → the run skill pauses internally and asks the user for a manual
  longer-budget run. The orchestrator should not transition until the run skill
  returns a final verdict. The verdict can be:
    - `passed` (manual log shows pytest pass) → CONFIG_UPDATE.
    - `failed` (manual log shows pytest fail / traceback) → DIAGNOSE, with
      `details.source: "manual_run"` on the history entry so DIAGNOSE knows the
      log came from a longer-budget run.
    - `timeout` (user replied "skip" or manual log was still inconclusive) →
      CONFIG_UPDATE with result=TIMEOUT, then STOP.
- On fail → save log path to state, transition to DIAGNOSE.

**DIAGNOSE**: For DRAM/OOM failures, invoke **`model-bringup-classify-oom`** first,
then `model-bringup-diagnose` with the log (pass classification JSON into diagnosis
context). Route repair per classification — not multichip from REPAIR.
- If confidence is `low` and iteration >= 2 → ESCALATED.
- If diagnose sets `escalation_skill: "runtime-failure-debugger"` (i.e.
  `suggested_repair_strategy: "runtime_debug"`) → REPAIR with that strategy.
  This path is taken when standard pattern-matching is insufficient or when a
  prior cheap strategy did not resolve the same root cause.
- Otherwise → REPAIR.

**REPAIR**: Invoke `model-bringup-repair` skill with diagnosis and model_key.
- If strategy is `runtime_debug`, repair delegates to the
  `runtime-failure-debugger` skill. Important characteristics of this path:
  - **Long-running**: the debugger does an architecture-print pass, several
    bisect runs (each a full pytest), a block-sanity run, a Phase 3B
    minimal-sanity bisect (more pytest runs), a Phase 4 codegen + TTNN run,
    and a Phase 5 tt-metal run. Budget tens of minutes to a few hours and do
    not enforce the FIRST_RUN/VERIFY 5-minute timeout on it.
  - **Human-input gate**: Phase 5 needs `tt_metal_machine`, `tt_metal_repo`,
    and `tt_metal_branch`. The orchestrator does not have these from
    bringup state. Before invoking, prompt the user for them (or accept
    "skip Phase 5" — the debugger still produces a useful report up to
    Phase 4). Pass the values through to the repair stage so it can
    pre-fill the debugger's Phase 0.
  - **No automated patch**: the deliverable is `debug_report.md` at
    `<tt_xla_repo>/claude_logs_<model_name>/debug_report.md` plus the block
    and minimal sanity files under `tests/torch/ops/<model_name>/`. Treat
    the repair result as `requires_human_review: true`: pause, surface the
    report and sanity paths, and wait for the user to either supply a fix
    (continue to VERIFY) or escalate.
  - **Cleanup check**: after the debugger returns, run `git status -s`. If
    the debugger left transient edits in model files (its own rules say it
    must revert), flag this in the orchestrator's pause message so the user
    can clean up before re-running.
- If blocked → ESCALATED.
- If requires_human_review → pause and show the generated patch/instructions, wait for user confirmation before continuing.
- On proceed → increment iteration, transition to VERIFY.

**CONFIG_UPDATE**: Invoke `model-bringup-config-update` skill with model_key and result.

For PASSED results, run CONFIG_UPDATE in two passes:
1. Dry-run (no `--apply`) — generates `config_update_proposed.md`. Show
   the diff to the user.
2. `--apply` — flips YAML + bringup_status marker once the user confirms
   (or the orchestrator was invoked with autonomous approval).

Only after the `--apply` pass succeeds, transition to **FINALIZE**.

**FINALIZE** (runs only after PASSED + CONFIG_UPDATE applied): Invoke
`model-bringup-finalize` skill with the model_key and `--archs <list>`.
The skill:
- Re-verifies on every arch in `--archs` after the first (skipping the
  one we already passed on).
- Runs pre-commit on the changed files.
- Creates `bringup/<safe_key>` local branch if the user is still on
  `main`.
- Drafts `pr_body.md`.

Branching on FINALIZE result:
- `passed` → terminal PASSED. Print the FINALIZE output block. Done.
- `partial` (one arch failed) → record `state.failed_archs`, do **not**
  re-enter the FSM automatically. Surface the failing arch so the user
  can choose to re-invoke `/model-bringup <key> --arch <failed_arch>`
  separately.
- `dry_run` (no `--apply`) → propose changes and stop. The user re-invokes
  with `--apply` later.

If `--archs` was not passed (single-arch run), FINALIZE still runs but
skips Step 1 (multi-arch verify) and goes straight to pre-commit + branch
+ PR body.

**ESCALATE**: Generate `escalation_report.md` (see below), invoke `model-bringup-config-update` skill with result=ESCALATED.

## Escalation Conditions
Escalate immediately when any of the following is true:
- Iteration count reaches 5 with no PASSED result
- Diagnosis confidence is `low` after iteration 2
- Repair is `blocked`
- The same failure_reason repeats across two consecutive iterations (no progress)
- Scaffold/validate fails

## Escalation Report
Write `.claude/bringup/<model_key>/escalation_report.md` containing:
- **Provenance block** at the top:
  ```
  tt-xla SHA       : <short sha of tt-xla HEAD>
  tt-foundry SHA   : <short sha if submodule present, else 'not a submodule'>
  Generated        : <YYYY-MM-DD HH:MM>
  Source skill     : model-bringup (orchestrator)
  Mode             : bringup | retriage
  ```
- model_key, arch, total iterations
- Each iteration: stage, diagnosis (with `source: json_report|stdout_fallback`
  and `last_stage` from the run's `._bringup_stage.txt` marker),
  repair attempted, links to `iter_<N>_run.log` and `iter_<N>_result.json`
- Final failure category, confidence, and last compilation stage reached
  (`FE_COMPILATION` / `TTMLIR_COMPILATION` / `RUNTIME_EXECUTION` / `unknown`)
- Recommended next human action

## Progress Display
After each stage transition, print a one-line status:
`[model_key] stage=<STAGE> iteration=<N> → <result>`

If the last run recorded `state.details.last_stage` (set by
`model-bringup-run` from `._bringup_stage.txt`) and the stage is not
`unknown`, append it: `... → <result> last_stage=<value>`. This makes
the compilation-stage trail visible in the orchestrator output without
having to open the log.

## Bringup Steps Log
Maintain `.claude/bringup/<safe_key>/bringup_steps.txt` throughout the pipeline run.
Append one section per stage as each completes (do not write the whole file at the end).
The log is the human-readable audit trail and must survive partial runs.

Each section follows this template:
```
--------------------------------------------------------------------------------
STEP <N> — <Stage Name> (<skill name>)
--------------------------------------------------------------------------------
<key facts: what was done, what was decided, what was found>
<any commands run and their one-line result>
<files created or modified>
<stage result: PASSED | FAILED | TIMEOUT | ESCALATED>
```

Open the file with a header block when the pipeline starts:
```
================================================================================
MODEL BRINGUP LOG
================================================================================
Model Key  : <model_key>
Arch       : <arch>
Date       : <YYYY-MM-DD>
Start time : <pipeline_start_iso>
================================================================================
```

Each step section MUST include a timing block. Sub-skills capture
`_step_start_ts=$(date +%s); _step_start_iso=$(date -Iseconds)` before doing
their work and `_step_end_ts=$(date +%s); _step_end_iso=$(date -Iseconds)`
after, then write the following lines as the first content of the section:
```
Start    : <_step_start_iso>
End      : <_step_end_iso>
Elapsed  : <_step_end_ts - _step_start_ts>s
```

Close the file with a summary block when the pipeline ends. The final
summary is written by `model-bringup-config-update`, which reads
`pipeline_start_ts` from `state.json` and computes the total elapsed:
```bash
_pipeline_end_ts=$(date +%s)
_pipeline_end_iso=$(date -Iseconds)
_total_elapsed_s=$((_pipeline_end_ts - $(jq -r '.pipeline_start_ts' state.json)))
# Format as Hh Mm Ss for readability when over 60s
```

```
================================================================================
FINAL RESULT
================================================================================
<✓|✗> <model_key> — <PASSED|ESCALATED|TIMEOUT> after <N> repair iteration(s)
  Loader created  : yes | no
  Applied patches : <list or 'none'>
  Last stage      : <FE_COMPILATION | TTMLIR_COMPILATION | RUNTIME_EXECUTION | unknown>
  Start time      : <pipeline_start_iso>
  End time        : <_pipeline_end_iso>
  Total elapsed   : <Xh Ym Zs>  (<total_elapsed_seconds>s)
  YAML entry      : <key added to YAML or 'none'>
================================================================================
```

## Terminal Output
On PASSED:
```
✓ <model_key> — PASSED after <N> iteration(s)
  Archs verified : <comma list>
  Applied patches: <list or 'none'>
  Overview       : .claude/bringup/<safe_key>/model_overview.md
  Golden         : .claude/bringup/<safe_key>/golden.pt
  Steps log      : .claude/bringup/<safe_key>/bringup_steps.txt
  Branch         : bringup/<safe_key> (local only, not pushed)
  PR body draft  : .claude/bringup/<safe_key>/pr_body.md

Next:
  git push -u origin bringup/<safe_key>
  gh pr create --title "Bringup: <family>/<variant>" --body-file .claude/bringup/<safe_key>/pr_body.md
```

On PARTIAL (some archs passed, some failed in FINALIZE):
```
⚠ <model_key> — PARTIAL
  Passing archs : <list>
  Failing arch  : <arch>
  YAML supported_archs left as <first arch> only.
  Re-invoke: /model-bringup <model_key> --arch <failed_arch>
```
On ESCALATED:
```
✗ <model_key> — ESCALATED
  Reason: <last failure_reason>
  Last stage: <FE_COMPILATION | TTMLIR_COMPILATION | RUNTIME_EXECUTION | unknown>
  Report: .claude/bringup/<model_key>/escalation_report.md
  Steps log: .claude/bringup/<safe_key>/bringup_steps.txt
```
