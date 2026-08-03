---
name: model-bringup-run
description: Test execution stage of the model bringup pipeline. Runs pytest for a given model_key with a 5-minute hard timeout, captures output to a log file, and records result (passed/failed/timeout) in state.json. Invoked by the model-bringup orchestrator at the FIRST_RUN and VERIFY stages.
allowed-tools: Bash Read Write
---

# Model Bringup — Test Execution

You are the **test execution** stage of the model bringup pipeline.

## Invocation
`/model-bringup-run <model_key> [--arch <n150|p150>] [--iteration <N>] [--timeout <seconds>] [--dtype fp32|bf16]`

The orchestrator sets **`state.current_arch`** before each invoke. If present,
`--arch` must match it (or omit `--arch` and read from state).

## Host gate (run before every pytest — FIRST_RUN and VERIFY)

Re-run or read `.claude/bringup/<safe_key>/host_probe.json` (`probe_host.py` if stale).
See `host_device_probe.md`.

**Single-device components only** (`parallelism_mode: single_device` in `weight_fit.json`):

```bash
python .../probe_host.py --parallelism-mode single_device -o host_probe.json
CAN=$(jq -r '.can_run_component // .can_run_single_chip_bringup' host_probe.json)
if [ "$CAN" != "true" ]; then
  # Abort: print jq -r '.component_skip_reason // .single_chip_skip_reason'
  # Only fails when no TT device is visible at all
fi
export TT_XLA_ARCH=$ARCH
# Do NOT set TT_VISIBLE_DEVICES — the unsharded graph picks its own mesh
```

A single-device component runs **unsharded** on whatever mesh its graph compiles
to, so the host's chip count does not gate it and **`TT_VISIBLE_DEVICES` need not
be set**. Run it on the box you have; if it passes, record the mesh the run
actually compiled to rather than assuming one.

This is what lets a pipeline be brought up on one machine: the weight-bound
component runs tensor-parallel across the box, and every component that fits one
chip runs single-device on that same box — unsharded, no second host.

The arch comes from the observed silicon, never the host name, so a Blackhole box
records `p150` (`filter_arch_queue` drops `n150`).

**Multichip** (`tensor_parallel`) uses `model-bringup-run-torch-tp` (separate host gate).

## Per-arch, dtype, and test path (orchestrator contract)

1. **Architecture:** `ARCH=$(jq -r '.current_arch // empty' state.json)` or `--arch`.
   Valid single-chip values: `n150`, `p150`. Export `TT_XLA_ARCH=$ARCH` when in arch loop;
   on multichip-only sessions use probe `inferred_bringup_arch` only for pinned smoke tests.
2. **Dtype ladder:** Read `details.source_dtype` from scaffold / loader / HF config
   (see `dtype_ladder.md`). Default **`fp32`** when source is fp32-safe. Use **`bf16`**
   when loader `DTYPE`, HF `torch_dtype`, or inference script is bf16-only. Override
   with `--dtype bf16` or `state.details.dtype_ladder[$ARCH] == "bf16"` after activation repair.
   Record in history: `"dtype": "fp32" | "bf16"`.
3. **Log naming (per arch):** use arch-prefixed files so n150 and p150 runs
   do not overwrite each other:
   - `logs/iter_<arch>_<N>_run.log` (e.g. `iter_n150_0_run.log`)
   - `logs/iter_<arch>_<N>_result.json`
4. **Pipeline component tests (required path):** read
   `weight_fit.json` → `components[].test_path` for the active component
   (must be under `tests/torch/models/<family>/`). Run that pytest node —
   **never** rely on `test_config_inference_single_device.yaml` for PCC;
   the test must use `ComparisonConfig(pcc=PccConfig(required_pcc=0.99))`.
   ```bash
   TEST_PATH=$(jq -r '.components[] | select(.name=="<component>") | .test_path' weight_fit.json)
   TT_XLA_ARCH=$ARCH timeout $TIMEOUT_S python -m pytest -svv "$TEST_PATH" ...
   ```
   A PASSED result means PCC **was** evaluated at 0.99 (or whatever the test sets).
5. **Monolithic / runner collect:** if no `test_path` under `tests/torch/models/`,
   use `test_all_models_torch[model_key]` collect (grep pattern below).
6. **Do not** set multichip env here (`TT_VISIBLE_DEVICES`, SPMD) — that is
   `model-bringup-run-torch-tp` on the `tensor_parallel` test node.

See `model-bringup-multichip/references/dtype_ladder.md`.

## Responsibility
Run the pytest test for the given model_key and capture the full output to a log file.

## Timeout Policy

**Adaptive budget** based on rough param-count estimate from the model name.
The orchestrator may override the chosen budget via `--timeout <seconds>`;
otherwise the table below applies. First regex match wins, case-insensitive,
fallback 900s.

| Pattern (regex on model_key)            | Budget |
|------------------------------------------|--------|
| `\b\d+(?:\.\d+)?\s*B(?:[_\-]\|$\|\b)` (N B params) | 1800s (orchestrator may promote to multichip if weight-bound) |
| `bi_lstm\|hippynn\|mobilenetv3\|\bnano\b\|\d+M` | 300s |
| `\btiny\b\|\bsmall\b`                     | 600s |
| `\bbase\b\|resnet50\|resnet101`           | 900s |
| `\blarge\b\|vgg16\|perceiver`             | 1200s |
| (fallback)                               | 900s  |

If the chosen budget is exceeded, kill the process and treat the result as
`TIMEOUT`. Record the chosen budget in `state.json` under
`details.timeout_seconds` so escalation reports show what was actually
allowed.

## Steps

### 1. Discover test node IDs via collect

Parse the `model_key` to build a grep pattern for `tests/runner/test_models.py`:

- **Full structured key** (`ltx2/pytorch-Fast-single_device-inference`): match exact bracket content
  ```bash
  grep 'test_all_models_torch\[ltx2/pytorch-Fast-single_device-inference\]'
  ```
- **Family-only key** (`ltx2`): match all single-device inference variants
  ```bash
  grep 'test_all_models_torch\[ltx2/pytorch-.*-single_device-inference\]'
  ```

Run collect and filter:
```bash
pytest -q --collect-only tests/runner/test_models.py 2>&1 \
  | grep 'test_all_models_torch\[<pattern>\]'
```

Collect the full `tests/runner/test_models.py::test_all_models_torch[...]` lines as `test_node_ids`.

If `test_node_ids` is empty, **fail immediately** — do not proceed to run:
```
[run] FAILED — no test node found for <model_key> in tests/runner/test_models.py
  Check: third_party/tt_forge_models/<family>/pytorch/loader.py exists and imports cleanly.
  Verify: python -c "from third_party.tt_forge_models.<family>.pytorch import ModelLoader"
```

### 2. Run pytest with adaptive timeout + JSON report

Resolve `TIMEOUT_S` from the table in **Timeout Policy** above (or the
explicit `--timeout` arg). Then:

```bash
mkdir -p .claude/bringup/<safe_key>/logs
LOG_SUFFIX="iter_${ARCH}_${N}"   # e.g. iter_n150_0 — see Per-arch section
ENABLE_BRINGUP_STAGE_LOGGING=1 TT_XLA_ARCH=$ARCH timeout $TIMEOUT_S python -m pytest <test_node_ids> \
  -svv --tb=long -p no:cacheprovider \
  --json-report --json-report-file=.claude/bringup/<safe_key>/logs/${LOG_SUFFIX}_result.json \
  2>&1 | tee .claude/bringup/<safe_key>/logs/${LOG_SUFFIX}_run.log
exit_code=${PIPESTATUS[0]}
```

`ENABLE_BRINGUP_STAGE_LOGGING=1` makes the C++ pipeline write the current
compilation stage to `._bringup_stage.txt` at the tt-xla repo root. The
file is updated as the pipeline progresses through
`FE_COMPILATION_START` → `TTMLIR_COMPILATION_START` → `RUNTIME_EXECUTION_START`,
so on failure its final content tells us **which stage the model died in**.
Step 4 reads it and records `details.last_stage` so DIAGNOSE can route
patterns correctly (e.g. `missing_op@FE_COMPILATION` is a StableHLO emission
gap, while `missing_op@TTMLIR_COMPILATION` is a tt-mlir lowering gap — very
different repair strategies). conftest.py owns lifecycle of the file;
this skill must **not** delete it.

Flag rationale (same as model_issue_pick):
- **`-s`**: no stdout capture → live progress (downloads, compile traces,
  tqdm) reaches the log. Silent-hang failures are otherwise undiagnosable.
- **`-vv`**: full assertion diffs, unambiguous test summary.
- **`--tb=long`**: full traceback with source context — DIAGNOSE needs this
  to extract a useful root-cause snippet.
- **`--json-report-file`**: machine-readable result. Prefer reading this
  over grepping the log; the log is for human review.

Exit-code rules:
- `exit_code == 124` → `result = "timeout"`.
- `exit_code == 0`   → `result = "passed"`.
- otherwise          → `result = "failed"`.

Do not suppress any output — the full stdout+stderr must be captured.

### 3. Handle TIMEOUT result

If `exit_code == 124`, do **NOT** immediately mark as UNKNOWN. The 300s budget
is often too short for first-run weight downloads, HF cache misses, or initial
JIT/XLA compilation, and the actual pass/fail signal usually appears within a
few extra minutes. **Pause the pipeline and ask the user to run the test
manually with a longer budget**, then resume with their log.

#### 3a. Print the manual-run prompt and stop the turn

Print exactly this block, then **end the assistant turn without further tool
calls** so the user can reply:

```
[run] iter=<N>  TIMEOUT at 300s — pipeline paused, manual run required.

Please run the test yourself with a longer budget and share the log:

  ENABLE_BRINGUP_STAGE_LOGGING=1 TT_XLA_ARCH=<arch> timeout 1800 python -m pytest \
    '<test_node_id>' \
    -svv --tb=long -p no:cacheprovider \
    --json-report --json-report-file=/tmp/<safe_key>_manual.json \
    2>&1 | tee /tmp/<safe_key>_manual.log

Then reply with one of:
  - the path to the log file (e.g. /tmp/<safe_key>_manual.log), or
  - the pasted log content, or
  - "skip" to fall through to the original TIMEOUT/UNKNOWN escalation.
```

Do not proceed past this point until the user has replied.

#### 3b. Process the user's reply

When the user replies, branch on what they provided:

- **Log path** (file exists, e.g. `/tmp/<safe_key>_manual.log`):
  Copy it to `.claude/bringup/<safe_key>/logs/iter_<N>_manual.log` and inspect
  the tail.

- **Pasted log content**:
  Write it to `.claude/bringup/<safe_key>/logs/iter_<N>_manual.log` and inspect
  the tail.

- **"skip"** (or any clear opt-out):
  Fall through to the original TIMEOUT escalation in **3c** below.

Inspect the tail of the manual log and classify:

| Tail signature | Recorded `result` | Next stage |
|---|---|---|
| `=== <N> passed in <T>s ===` | `passed` | continue to Step 4, then orchestrator → CONFIG_UPDATE(PASSED) |
| `=== <N> failed in <T>s ===` or traceback / `1 error` | `failed` | continue to Step 4 with the manual log path; orchestrator → DIAGNOSE |
| Truncated / still timed out at the larger budget / no clear verdict | (treat as still TIMEOUT) | fall through to **3c** |

When recording a passed/failed result from a manual log, use the manual log
path in `details.log` and add `details.source: "manual_run"` plus the budget
the user used (`details.manual_timeout_seconds`). Re-run Step 4 (read
`._bringup_stage.txt`) right after the user's reply — their manual run
overwrote the marker, so it now reflects the longer-budget verdict.
If `._bringup_stage.txt` is missing or pre-dates the manual run, record
`details.last_stage: "unknown"` rather than carrying the stale value
from the auto-run.

#### 3c. Fallback: original TIMEOUT/UNKNOWN escalation

Only reached if the user replied "skip" or the manual log was still
inconclusive.

- Append to `history`:
  ```json
  {
    "stage": "first_run",
    "timestamp": <now>,
    "result": "timeout",
    "details": {
      "log": "logs/iter_<N>_run.log",
      "returncode": 124,
      "duration_seconds": 300,
      "failure_reason": "Test exceeded 300s wall-clock limit and manual run did not produce a verdict. Marking UNKNOWN (bringup not attempted)."
    }
  }
  ```
- Set `state.stage = "escalated"` in state.json.
- Call `model-bringup-config-update` skill with `result=TIMEOUT`.
- **Stop the FSM — do not proceed to DIAGNOSE.**

Output:
```
[run] iter=<N>  result=TIMEOUT  duration=300s
  Marking as UNKNOWN (bringup not attempted) — exceeded 5-minute execution limit
  and manual run did not produce a verdict.
  log: .claude/bringup/<safe_key>/logs/iter_<N>_run.log
```

### 4. Read the compilation-stage marker

Before reading the JSON report, capture the last compilation stage the
pipeline reached. This is independent of pass/fail/timeout — useful for
PASSED runs too as a "what was actually exercised" stamp.

```bash
stage_file=._bringup_stage.txt   # at tt-xla repo root
if [ -f "$stage_file" ]; then
    last_stage_raw=$(tr -d '[:space:]' < "$stage_file")
else
    last_stage_raw=""
fi
```

Map the raw marker to a canonical value for state.json:

| `._bringup_stage.txt` content | `details.last_stage` |
|---|---|
| `FE_COMPILATION_START`       | `FE_COMPILATION`     |
| `TTMLIR_COMPILATION_START`   | `TTMLIR_COMPILATION` |
| `RUNTIME_EXECUTION_START`    | `RUNTIME_EXECUTION`  |
| empty / file missing / other | `unknown`            |

`unknown` is expected for crashes that happen before the C++ pipeline
gets a chance to write the file (e.g. import errors, loader exceptions in
Python). DIAGNOSE treats `unknown` as "no stage signal, fall back to
stdout patterns alone."

### 5. Record result in state.json (non-timeout)

Read the JSON report at `logs/iter_<N>_result.json` (written by
`pytest-json-report`). Use it to populate richer fields than the stdout
log can carry:

- `tests[0].outcome` → cross-check against exit code.
- `tests[0].metadata.tags.bringup_status` → already-classified outcome
  from the runner (PASSED / INCORRECT_RESULT / FAILED_*).
- `tests[0].metadata.tags.pcc` and `tags.pcc_threshold` → for VERIFY
  acceptance and for `now_incorrect_result` PCC guardrails downstream.
- `tests[0].metadata.tags.failing_reason.description` → canonical reason
  string (preferred over regexing stdout).

If the JSON file is missing (e.g. the test crashed before pytest could
write it), fall back to stdout summary and record `details.json_report:
"missing"`.

Append to `history` (include **`arch`** and **`dtype`** on every entry):
```json
{
  "stage": "first_run",
  "timestamp": <now>,
  "arch": "<n150|p150>",
  "result": "passed" | "failed",
  "details": {
    "dtype": "fp32" | "bf16",
    "log": "logs/iter_<arch>_<N>_run.log",
    "json_report": "logs/iter_<N>_result.json",
    "returncode": <int>,
    "duration_seconds": <float>,
    "timeout_seconds": <budget that was applied>,
    "bringup_status": "<from json tags, if present>",
    "pcc": <from json tags, if present>,
    "pcc_threshold": <from json tags, if present>,
    "failing_reason": "<from json tags, if present>",
    "last_stage": "FE_COMPILATION | TTMLIR_COMPILATION | RUNTIME_EXECUTION | unknown"
  }
}
```

Also mirror `last_stage` into `state.details.last_stage` (top-level
shortcut for downstream skills that don't want to walk history).

### 6. Write to bringup_steps.txt

Timing capture: `_step_start_ts`/`_step_start_iso` at the very start of this
skill's execution (before collect); `_step_end_ts`/`_step_end_iso` after the
pytest run completes and logs are flushed. `Elapsed` is the full step
wall-clock (includes collect + pytest + log post-processing). `Duration`
below is the pytest sub-process only — keep both because the difference is
useful for diagnosing slow collect or large-log handling.

Append to `.claude/bringup/<safe_key>/bringup_steps.txt`:
```
--------------------------------------------------------------------------------
STEP <N> — First Run / Verify (model-bringup-run, iteration <N>)
--------------------------------------------------------------------------------
Start    : <_step_start_iso>
End      : <_step_end_iso>
Elapsed  : <_step_end_ts - _step_start_ts>s

Collect : pytest -q --collect-only tests/runner/test_models.py | grep '<family>/pytorch-...'
Node IDs: <list of discovered test node IDs>
Budget  : <TIMEOUT_S>s (auto from name heuristic | --timeout override)
Command : TT_XLA_ARCH=<arch> timeout $TIMEOUT_S python -m pytest <test_node_ids> -svv --tb=long --json-report --json-report-file=...
Log     : .claude/bringup/<safe_key>/logs/iter_<N>_run.log
JSON    : .claude/bringup/<safe_key>/logs/iter_<N>_result.json
Duration: <Xs>  (pytest only)
Exit    : <exit_code>
Stage   : <FE_COMPILATION | TTMLIR_COMPILATION | RUNTIME_EXECUTION | unknown>

Test result summary:
  <last 3 lines of pytest output, e.g. "1 passed" or "1 failed">

RUN RESULT: PASSED | FAILED | TIMEOUT
[If TIMEOUT:]  Marking as UNKNOWN — exceeded 5-minute execution limit.
[If FAILED:]   Passing log to DIAGNOSE stage (last_stage=<value>).
```

### 7. Output (non-timeout)
```
[run] iter=<N>  result=<PASSED|FAILED>  duration=<Xs>  stage=<last_stage>
  log: .claude/bringup/<safe_key>/logs/iter_<N>_run.log
```

Return the log path so the orchestrator can pass it to `model-bringup-diagnose`.
