---
name: model-bringup-scaffold-github
description: Scaffold a tt-forge-models loader for a model whose source lives on GitHub (no HuggingFace mirror). Always adds the upstream repo to tt-forge-models as a git submodule at third_party/<repo>/ — never copies or ports the source into the tree. Writes loader.py + __init__.py, validates import + collect, and hands off to the standard OVERVIEW / FIRST_RUN / ... FSM. Used by the model-bringup orchestrator when the model_key is a github short form, a git+ URL, or a full github.com URL.
allowed-tools: Bash Read Write Edit Grep Glob
---

# Model Bringup — GitHub Scaffold

You are the **GitHub variant** of the scaffold stage. Use when the model
lives on GitHub with no HuggingFace mirror. Same role as
`model-bringup-scaffold`, different inputs and a different way of laying
the source on disk.

**Vendoring rule (no exceptions):** the upstream GitHub repo is added to
`tt-forge-models` as a **git submodule** at `third_party/<repo>/`, where
`<repo>` is the GitHub repository name verbatim (the last path segment of
the URL, minus a trailing `.git`) — not the derived `family` slug. Never
copy, port, or paste upstream source files into `tt-forge-models`. If the
model code is not importable as a submodule, that is a loader problem to
solve with `sys.path` / import shims — not a reason to vendor a copy.

After this skill exits PASSED, downstream stages (`model-bringup-overview`,
`-run`, `-diagnose`, `-repair`, `-finalize`) run unchanged — they only
require that `ModelLoader().load_model()` returns an `nn.Module`.

## Invocation
`/model-bringup-scaffold-github <model_key> [--arch <arch>] [--entry <module.path:ClassName>] [--ref <git_ref>] [--custom-test]`

- `model_key` (required) — one of:
  - `github:<org>/<repo>` — short form. Optional `@<ref>` selects a SHA,
    tag, or branch, e.g. `github:open-mmlab/mmdetection3d@v1.4.0`.
  - `https://github.com/<org>/<repo>[.git]` — full URL.
  - `git+https://...` — also accepted (pip-style).

  All three normalize to `<repo_url>` + optional `<ref>`. `family` is
  derived from the repo name (lower-snake-case, strip `pytorch_`,
  `_pytorch`, `-model`, etc.).

- `--entry "<module.path:ClassName>"` — entry point (e.g.
  `mmdet3d.models.detectors:FCOS3D`). Required because we cannot
  `AutoModel.from_pretrained` a class out of GitHub. If omitted, the
  skill asks.
- `--ref <git_ref>` — overrides the ref parsed from `@<ref>`.
- `--custom-test` — same meaning as in `model-bringup-scaffold`.

---

## Step 0 — Normalize the model_key

Parse one of the three forms. Set:
- `repo_url` (e.g. `https://github.com/open-mmlab/mmdetection3d`)
- `repo` — the GitHub repository name **verbatim**: last path segment of
  the URL with a trailing `.git` stripped, case and punctuation
  preserved (`mmdetection3d`, `YOLOv9`, `segment-anything-2`). This is
  the submodule directory name; do **not** snake-case or otherwise
  rewrite it.
- `ref` (SHA / tag / branch; default `HEAD` resolved to a specific SHA)
- `family` (snake-case, derived from repo name; user-overridable via the
  ask below). `family` names the loader directory only — it never names
  the submodule.

Construct the structured model_key the orchestrator already understands:
```
<family>/pytorch-<variant>-single_device-inference
```
where `variant` defaults to the repo's `<ref>` short SHA (8 chars) or the
ref name if it was a tag. The user can override this via the variant
question below.

---

## Step 1 — Ask the user (only when args missing)

Use `AskUserQuestion` for genuine forks. Skip the question if the user
already passed the flag.

There is **no vendor-mode question** — GitHub sources are always added as
a submodule at `third_party/<repo>/`. Do not offer the user a copy/port
alternative.

**Q1 — Entry class** (skip if `--entry` was passed):

> What is the Python entry point of the model? Format:
> `module.path:ClassName` — e.g. `mmdet3d.models.detectors:FCOS3D`.

**Q2 — Variant slug** (only if the orchestrator did not pass one):

> Variant slug for `ModelVariant`? (Defaults to the short SHA of the
> selected ref.)

Record the answers in state.json under `details.scaffold_github`:
```json
{
  "repo_url":   "<url>",
  "repo":       "<repo name verbatim>",
  "ref":        "<sha or tag>",
  "mode":       "submodule",
  "submodule_path": "third_party/<repo>",
  "entry":      "<module:Class>",
  "family":     "<family>",
  "variant":    "<variant_slug>"
}
```

---

## Step 2 — Add the source as a submodule

This is the **only** vendoring path. Target:
`third_party/tt_forge_models/third_party/<repo>/` — i.e. `third_party/<repo>`
relative to the `tt-forge-models` repo root, with `<repo>` the verbatim
GitHub repository name from Step 0.

```bash
git -C third_party/tt_forge_models submodule add --depth 1 "<repo_url>" third_party/<repo>
git -C third_party/tt_forge_models/third_party/<repo> fetch --depth 1 origin "<ref>"
git -C third_party/tt_forge_models/third_party/<repo> checkout "<ref>"
git -C third_party/tt_forge_models submodule update --init --recursive
```

If `third_party/tt_forge_models/third_party/` does not exist yet, create
it and add a `__init__.py` so it is importable as a Python package.

`git submodule add` writes/updates `third_party/tt_forge_models/.gitmodules`
and stages the gitlink. Verify both are staged before moving on — a
submodule that exists only as an untracked directory will not survive a
fresh clone:

```bash
git -C third_party/tt_forge_models status --short .gitmodules "third_party/<repo>"
grep -A2 "path = third_party/<repo>" third_party/tt_forge_models/.gitmodules
```

The `.gitmodules` entry must record `path = third_party/<repo>` and the
upstream `url`. If the name already exists (re-running the skill for the
same repo), do **not** add a second entry — reuse the existing submodule
and just move it to the requested `<ref>`.

**If the repo is huge** (> 1 GiB checked out), keep the submodule and
narrow the checkout rather than abandoning it:

```bash
git -C third_party/tt_forge_models/third_party/<repo> sparse-checkout init --cone
git -C third_party/tt_forge_models/third_party/<repo> sparse-checkout set <model-code dirs>
```

Note the sparse paths in the steps log. Cloning or copying the source
into the tree is not an accepted fallback under any size.

Pin the ref. Write `third_party/tt_forge_models/third_party/<repo>/.bringup_ref`:
```
ref:    <full sha>
ref_kind: <sha|tag|branch>
fetched: <YYYY-MM-DD>
url:    <repo_url>
```

### Import path

When `<repo>` is a valid Python identifier:
```
from third_party.tt_forge_models.third_party.<repo>.<module.path> import <ClassName>
```

Many GitHub repo names are **not** valid identifiers (`segment-anything-2`,
`YOLOv9`, names starting with a digit). Do not rename the directory to
work around this — keep the verbatim name and have the loader put the
submodule root on `sys.path`, then import by the repo's own top-level
package name:

```python
import os, sys
_REPO_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "third_party", "<repo>",
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from <module.path> import <ClassName>
```

Use the same `sys.path` shim whenever the repo has its own `setup.py` /
`pyproject.toml` and its model code is not importable directly from the
source tree. Do **not** `pip install -e` automatically — that mutates the
venv state of the user's whole environment, which is out of scope for
scaffold.

### Do not create a `src/` copy

Do not create `third_party/tt_forge_models/<family>/pytorch/src/`, do not
copy upstream `.py` files into the loader directory, and do not write a
`SRC_VENDORED_FROM.txt` provenance file — the submodule gitlink plus
`.bringup_ref` *is* the provenance. The `<family>/pytorch/` directory
holds only `loader.py`, `__init__.py`, and (optionally) a custom test.

If an existing family under `tt-forge-models` still carries a ported
`src/` tree from before this rule, leave it alone — migrating it is out
of scope for scaffold. Note it in the steps log so it can be converted
deliberately.

---

## Step 3 — Write loader.py

Same template skeleton as `model-bringup-scaffold` step 2d, with three
differences:

1. `ModelVariant` — populated with the user's `<variant_slug>` (default
   short-SHA).
2. `ModelConfig.pretrained_model_name` is **not** set (there is no HF
   id). Instead set a new field `source_repo` = `<repo_url>@<ref>`.
3. `load_model()` body imports the entry class **from the submodule**
   (dotted `third_party.tt_forge_models.third_party.<repo>.…`, or the
   `sys.path` shim from Step 2 when `<repo>` is not a valid identifier)
   and instantiates it from a config-only path (random weights — per
   `user-bringup-prefs`). It must never import from a local `src/`
   package:

```python
def load_model(self, dtype_override=None):
    # Import from the third_party/<repo> submodule — see Step 2
    {imports}

    # No HF download. Construct from default config / random weights.
    {ctor_invocation}     # filled by the skill: usually MyModel(**default_cfg)
    if dtype_override is not None:
        model = model.to(dtype_override)
    return model.eval()
```

The skill fills `{ctor_invocation}` by inspecting the class signature:
- If `MyClass.__init__` has no required positional args → `model = MyClass()`.
- If it has required args, prompt the user once for a minimal config
  dict (`AskUserQuestion`, free-form), wrap it in `model =
  MyClass(**ctor_kwargs)`, and persist `ctor_kwargs` in state.

`load_inputs()` body: synthesize tensors matching the class's `forward`
signature, same heuristic as `model-bringup-scaffold` (use
`inspect.signature(model.forward)`, default shapes from annotations
where present, else `(1, 3, 224, 224)` for image-like, `(1, 128)` for
text-like).

`unpack_forward_output()` body: by default return the input unchanged
(model returns a tensor). If the user provided forward output shape
during the ask, generate the unpack from there.

Also write `pytorch/__init__.py` re-exporting `ModelLoader, ModelVariant`
exactly as `model-bringup-scaffold` does.

---

## Step 4 — Validate imports + discovery

Same as `model-bringup-scaffold` Step 3 + Step 4:

```bash
python -c "from third_party.tt_forge_models.<family>.pytorch import ModelLoader, ModelVariant; print('OK')"
pytest -q --collect-only tests/runner/test_models.py 2>&1 \
  | grep "test_all_models_torch\[<family>/pytorch-"
```

Also confirm the submodule is registered, not just present on disk:

```bash
git -C third_party/tt_forge_models submodule status third_party/<repo>
```

If the loader import fails because the repo expects `sys.path` munging or
an installed package, edit loader.py to inject the right path and retry
**once**. If it still fails, exit with `result=failed`,
`failure_reason=<import error one-liner>` so the orchestrator escalates
to a human rather than thrashing. Copying the failing modules into the
loader directory is **not** an acceptable fix — escalate instead.

---

## Step 4b — Pre-flight size gate + shard plan

Inherit the same gate logic as `model-bringup-scaffold` Step 4b. Param
estimate sources, in order:

1. Live count from the loader (load random-weights model, sum
   `p.numel()`).
2. Name-based heuristic on `<repo_url>` and `<entry_class>`.
   (`AutoConfig` is unavailable for GitHub-only models.)

Same gate thresholds (14 B warn, 30 B reject) and same shard plan emit.

---

## Step 4c — Optional custom test file

Same heuristics as `model-bringup-scaffold` Step 4c. GitHub-sourced
models more often need a custom test (non-standard input ctors are
common) — default `--custom-test` to true whenever `load_inputs()` needed
a hand-written ctor, unless the user explicitly disabled it.

---

## Step 5 — Initialize bringup state

Same shape as `model-bringup-scaffold` Step 5, plus persist the GitHub
provenance fields under `details.scaffold_github`:

```json
{
  "stage": "validate",
  "result": "passed",
  "details": {
    "loader_path":      "third_party/tt_forge_models/<family>/pytorch/loader.py",
    "loader_created":   true,
    "scaffold_variant": "github",
    "scaffold_github": {
      "repo_url": "<url>",
      "repo":     "<repo name verbatim>",
      "ref":      "<sha>",
      "mode":     "submodule",
      "entry":    "<module:Class>",
      "src_path": "third_party/tt_forge_models/third_party/<repo>/",
      "submodule_path": "third_party/<repo>",
      "sparse_checkout": ["<dir>", "..."]
    }
  }
}
```

---

## Step 6 — Bringup steps log

Append to `.claude/bringup/<safe_key>/bringup_steps.txt`:
```
--------------------------------------------------------------------------------
STEP 1 — Parse & Scaffold (model-bringup-scaffold-github)
--------------------------------------------------------------------------------
Input model_key : <original>
Repo URL        : <url>
Ref             : <sha> (<sha|tag|branch>)
Family          : <family>
Variant         : <variant_slug>
Entry           : <module:Class>

Vendor mode     : submodule (always)
Submodule path  : third_party/<repo>   (in tt-forge-models)
.gitmodules     : entry added | reused existing
Sparse checkout : <dirs or 'full'>
Provenance file : third_party/<repo>/.bringup_ref

Loader created  : yes
Files written   :
  - third_party/tt_forge_models/.gitmodules            (submodule entry)
  - third_party/tt_forge_models/third_party/<repo>     (gitlink @ <short_sha>)
  - third_party/tt_forge_models/<family>/__init__.py
  - third_party/tt_forge_models/<family>/pytorch/__init__.py
  - third_party/tt_forge_models/<family>/pytorch/loader.py

Import validation  : python -c "from ... import ModelLoader, ModelVariant" → OK | FAILED
Collect validation : pytest --collect-only ... → <N> test(s) found | NONE FOUND

Size gate          : <X> B params (source: loader | name_heuristic) → proceed | warn | reject
Shard plan         : <mesh> / TT_VISIBLE_DEVICES=<list> | n/a

Custom test file   : <path or 'none — generic runner suffices'>

SCAFFOLD RESULT: PASSED | FAILED
```

---

## Step 7 — Output

On success:
```
[scaffold-github] PASSED
  repo            : <repo_url>@<short_sha>
  vendor          : submodule
  submodule       : third_party/<repo> (registered in .gitmodules)
  loader          : third_party/tt_forge_models/<family>/pytorch/loader.py
  collect check   : <N> test(s) visible in tests/runner/test_models.py
```

On failure: same escalation rules as `model-bringup-scaffold` (failed
import, blocked size gate). A repo that cannot be added as a submodule
(dead URL, unreachable ref, auth-gated) is `result=blocked` with
`block_reason="submodule add failed: <git error>"` — do not fall back to
copying the source.

---

## Notes for the orchestrator

- Same exit codes as `model-bringup-scaffold` — orchestrator does not
  need to special-case the variant.
- The OVERVIEW skill runs unchanged: it imports `ModelLoader`, runs
  `load_model()` + `load_inputs()` on CPU, and captures golden. Because
  the loader returns a random-init model, CPU sanity should be fast
  (single forward) — the same as the HF path.
- If the user later updates the source (new SHA, new `--ref`), they
  should re-invoke this skill with `--ref` to refresh provenance. The
  skill moves the existing submodule to the new ref and rewrites
  `.bringup_ref` — it does not add a second `.gitmodules` entry. The
  pipeline does not auto-detect upstream drift.
- FINALIZE / PR stages must stage `tt-forge-models`' `.gitmodules` and the
  new gitlink alongside the loader. A PR that adds `loader.py` but not the
  submodule entry will fail on a fresh clone.
