---
name: model-bringup-finalize
description: Post-PASSED finalize stage of the model bringup pipeline. Runs (a) multi-arch re-verify on remaining target archs, (b) pre-commit on the changed files, (c) branch + PR-body draft so the user can land the work without manual hygiene. Invoked by the model-bringup orchestrator after a successful CONFIG_UPDATE (--apply) on the first arch.
allowed-tools: Bash Read Write Edit Grep
---

# Model Bringup — Finalize (multi-arch verify + branch hygiene)

You are the **finalize** stage of the model bringup pipeline. Runs only
when the upstream FSM has ended in PASSED on at least the first target
arch. Has two responsibilities:

1. **Multi-arch verify (G7)** — if the orchestrator was invoked with more
   than one target arch, re-run FIRST_RUN on the remaining archs using the
   same loader. CONFIG_UPDATE is only flipped to `EXPECTED_PASSING`
   *globally* once every arch passes.
2. **Branch + pre-commit hygiene (G6)** — create a working branch (if the
   user is still on `main`), run pre-commit against the changed files, and
   draft a PR body the user can paste into `gh pr create`.

This skill never opens a PR or pushes a branch by itself — it stops at
"branch created locally + PR body drafted". The user pushes when they're
ready.

## Invocation
`/model-bringup-finalize <model_key> --archs <arch>[,<arch>...] [--apply]`

- `--archs` is comma-separated and includes the **already-passed** arch as
  the first entry, so the skill knows what to skip.
- `--apply` controls writes: without it, the skill produces a dry-run
  report at `.claude/bringup/<safe_key>/finalize_proposed.md` and exits.
  With it, branch creation and pre-commit auto-fixes are applied.

---

## Step 1 — Multi-arch verify

For each arch in `--archs` **after** the first (which already passed):

1. Invoke `model-bringup-run` skill with `<model_key> --arch <arch>
   --iteration <N+1>`. Use the same timeout policy as FIRST_RUN.
2. Classify per arch:
   - `passed` → record in state, move to next arch.
   - `failed` / `timeout` → stop here. Do **not** roll back the first arch.
     Surface the failing arch and route back into the parent orchestrator
     at DIAGNOSE for that arch only.
3. After all archs pass, the YAML config for this model should list every
   passing arch under `supported_archs:`. Re-invoke
   `model-bringup-config-update` with `--apply --archs <all-passing>` so
   it can rewrite the list in one shot.

If one arch fails and the rest pass, the YAML stays at the original
`supported_archs: [<first arch>]` and the orchestrator records the failed
arch under `state.failed_archs` for the escalation report.

Append to state.json:
```json
"history": [
  ...,
  { "stage": "finalize_multiarch", "arch": "<arch>", "result": "passed" | "failed", "iteration": <N> }
]
```

---

## Step 2 — Pre-commit on changed files

Detect changed files (relative to `main`):
```bash
git diff --name-only main...HEAD
git diff --name-only            # working tree, in case scaffold/repair did not commit
```

Concatenate and dedupe. Skip files outside the repo (just in case).

**GitHub-sourced models (`details.scaffold_variant == "github"`):** the
loader and its submodule live inside `tt-forge-models`, so the commands
above will not list them. Collect those separately:
```bash
git -C third_party/tt_forge_models diff --name-only
git -C third_party/tt_forge_models status --short .gitmodules
```
Both the `.gitmodules` entry and the `third_party/<repo>` gitlink must
appear as pending changes in `tt-forge-models`. If the loader is present
but the submodule is not registered, stop and report it — the change is
incomplete and would break a fresh clone. Never resolve this by copying
the upstream source into the tree.

If `--apply`:
```bash
pre-commit run --files <space-separated changed file list> 2>&1 \
  | tee .claude/bringup/<safe_key>/pre_commit.log
```

If `pre-commit` rewrote any files (clang-format, black, etc.), record the
list in `state.json` under `details.precommit_autofixed`. Do **not** stage
or commit — leave it for the user. If pre-commit exited non-zero with no
auto-fixable diff (e.g. an actual lint error), surface the failing hook
in the proposed report and stop before branch creation.

If not `--apply`:
- Write the command into `finalize_proposed.md` so the user can run it.

---

## Step 3 — Branch creation

Branch policy:
- Name: `bringup/<safe_key>` (slashes-to-double-underscore form).
- Base: `main`.
- Only create the branch when we are still on `main` *and* there are
  uncommitted edits.

If `--apply`:
```bash
current=$(git rev-parse --abbrev-ref HEAD)
if [ "$current" = "main" ]; then
  git checkout -b bringup/<safe_key>
fi
```

If the current branch is already non-main, do **not** create a new branch;
the user is on a branch of their own choosing. Just record the branch
name in state and move on.

Do **not** commit. Do **not** push. The user owns those actions —
`user_bringup_prefs` calls out branch/push as a confirm-before-act step.

---

## Step 4 — PR body draft

Write `.claude/bringup/<safe_key>/pr_body.md`. Pull facts from state.json
+ `model_overview.md`:

```
# Bringup: <family>/<variant>

## What
Adds a single-device-inference path for `<family>` (`<HF model ID>`).

## Why
<one-liner copied from model_overview.md modality/task line>

## Highlights
- Loader      : third_party/tt_forge_models/<family>/pytorch/loader.py
- Source       : <HF id> | submodule `third_party/<repo>` @ `<short_sha>` (github)
- Params      : <X> B (source: <loader|config|name_heuristic>)
- Modality    : <text|vision|...>
- Archs       : <comma-list of all passing archs>
- PCC         : <measured> (threshold <required>) | n/a (golden ref)
- Patches     : <list from applied_patches, or 'none — loader-only'>

## Test
<test node id> — passes on <arch1> [, <arch2>, ...]

## Bringup audit trail
- Steps log    : .claude/bringup/<safe_key>/bringup_steps.txt
- Overview     : .claude/bringup/<safe_key>/model_overview.md
- Golden       : .claude/bringup/<safe_key>/golden.pt
- Total iters  : <N>

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

If state has `failed_archs` non-empty, prepend a `> ⚠️ Partial bringup —
failed on <arch>` callout so the user does not silently open a PR claiming
all-arch coverage.

---

## Step 5 — Output

On `--apply` success:
```
[finalize] PASSED
  archs verified : <comma list>
  branch         : bringup/<safe_key> (local only, not pushed)
  pre-commit     : OK (auto-fixed: <N> file(s)) | NO CHANGES
  PR body draft  : .claude/bringup/<safe_key>/pr_body.md

Next:
  git push -u origin bringup/<safe_key>
  gh pr create --title "Bringup: <family>/<variant>" --body-file .claude/bringup/<safe_key>/pr_body.md
```

On dry-run:
```
[finalize] DRY RUN
  proposed     : .claude/bringup/<safe_key>/finalize_proposed.md
  next         : re-invoke with --apply to create branch + run pre-commit
```

On multi-arch partial failure:
```
[finalize] PARTIAL — <arch> failed
  passing archs : <list>
  failing arch  : <arch>  log: <path>
  YAML untouched; recommended action: re-enter pipeline with --arch <arch>
```

---

## Bringup steps log

Append to `.claude/bringup/<safe_key>/bringup_steps.txt`:
```
--------------------------------------------------------------------------------
STEP <N> — Finalize (model-bringup-finalize)
--------------------------------------------------------------------------------
Mode             : dry-run | apply
Archs target     : <comma list>
Archs verified   : <comma list>
Failed archs     : <comma list or 'none'>
Pre-commit       : <OK | autofixed N files | failed hook=<name>>
Branch           : bringup/<safe_key> (created | already on <branch> | n/a)
PR body draft    : .claude/bringup/<safe_key>/pr_body.md

FINALIZE RESULT: PASSED | PARTIAL | DRY_RUN
```
