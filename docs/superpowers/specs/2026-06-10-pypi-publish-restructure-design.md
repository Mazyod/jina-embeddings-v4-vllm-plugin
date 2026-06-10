# PyPI Publishing + Root-as-Plugin Restructure — Design

**Date:** 2026-06-10
**Status:** Approved (design)
**Goal:** Restructure this repo so its root *is* the publishable `jina-v4-vllm-plugin` package
(mirroring the `puckling` / `lsp-python-types` reference repos), demote the vLLM/Modal/deploy/validation
harness into a still-runnable nested `research/` subproject, and add PyPI **Trusted Publishing** (OIDC,
no API tokens) via a GitHub Actions workflow copied from those reference repos.

## Why

The publishable artifact is the vLLM out-of-tree plugin (`jina-v4-vllm-plugin`). Today it is a *nested*
package at `src/jinav4_vllm/vllm_plugin/`, inside a repo whose **root** `pyproject.toml` is a different
package (`jinav4-vllm`, the Modal/study harness). A verbatim copy of the reference `publish.yml` would
version-bump and build the wrong package. The user maintains two other PyPI packages
(`puckling`, `lsp-python-types`) that are each a single package at repo root with an identical
Trusted-Publishing workflow; this repo should match that pattern so there is only one approach to
maintain.

## Decisions (locked)

- **Structure:** root becomes the plugin package; the harness is demoted to a nested directory.
- **Nested infra stays fully runnable** (Modal entry points, Makefile, validation tests rewired to the
  new location), as a self-contained uv subproject — NOT a frozen archive.
- **License:** MIT (matches `lsp-python-types`).
- **`requires-python = ">=3.12"`** (matches the reference repos; vLLM 0.22.0 supports 3.10–3.12, so
  3.12 is a safe host floor).
- **Nested dir name:** `research/` (trivially renameable to `tooling/`).
- **Author identity in `pyproject.toml`:** `Mazyod <860511+Mazyod@users.noreply.github.com>` — the
  identity already used for this public repo. `mazjaleel` was deliberately scrubbed from this repo, so
  we do NOT reintroduce the gmail identity here (even though `lsp-python-types` uses it).
- **Infra installs the plugin from the local repo-root package** (via `..` relative paths from
  `research/`), preserving local-edit validation; switching to `pip install jina-v4-vllm-plugin`
  (PyPI) is a documented post-publish simplification.
- **Sub-approach:** `research/` is a standalone uv subproject you `cd` into — NOT a uv workspace.
  Rationale: a `[tool.uv.workspace]` table on the root pyproject is a needless deviation from the
  reference repos, and root must stay "uv-runs-naturally-from-here = the plugin."

## Target layout

```
jina-embeddings-v4-vllm-plugin/         # repo root = THE published package
├── pyproject.toml      # name=jina-v4-vllm-plugin · hatchling · MIT · >=3.12 · entry point + metadata
├── README.md           # the plugin README (PyPI long-description) — was src/jinav4_vllm/vllm_plugin/README.md
├── LICENSE             # NEW: MIT text
├── uv.lock             # the plugin's own lock (minimal — no runtime deps)
├── Makefile            # slim: install / test / build
├── .github/workflows/
│   └── publish.yml     # mirror of puckling's publish.yml
├── src/
│   └── jina_v4_vllm_plugin/            # promoted from src/jinav4_vllm/vllm_plugin/jina_v4_vllm_plugin/
│       ├── __init__.py
│       ├── model.py
│       └── jina_image_chat_template.jinja
├── tests/
│   └── test_plugin_packaging.py        # rewritten for root layout + hatchling
└── research/                           # demoted infra — still runnable (standalone uv subproject)
    ├── pyproject.toml  # name=jinav4-vllm (modal/numpy/pillow/requests; dev: pytest)
    ├── uv.lock
    ├── Makefile        # the modal/validation targets
    ├── README.md       # the study/product overview — was the root README.md
    ├── src/jinav4_vllm/    # client, common, eval, modal_app, multivector, projector (internals unchanged)
    ├── tests/          # the 8 pure-logic tests
    ├── deploy/         # bake_checkpoint.py, DEPLOY.md, Dockerfile
    ├── docs/           # VALIDATION.md, COMPAT.md  (+ docs/superpowers/ planning docs move here too)
    ├── artifacts/  data/  reports/
```

### Test split
- **Root (plugin):** `tests/test_plugin_packaging.py` — rewritten (new path + hatchling, not setuptools).
- **`research/tests/` (infra):** `test_artifacts.py`, `test_client.py`, `test_eval_metrics.py`,
  `test_eval_report.py`, `test_imaging.py`, `test_multivector_core.py`, `test_probes.py`,
  `test_projector_merge.py` — these import `jinav4_vllm.{client,common,eval,multivector,projector}`
  and stay with the harness.

## Root package `pyproject.toml` (target)

```toml
[project]
name = "jina-v4-vllm-plugin"
version = "0.1.0"
description = "vLLM out-of-tree model: Jina Embeddings v4 multi-vector (token_embed) on Qwen2.5-VL"
readme = "README.md"
requires-python = ">=3.12"
license = "MIT"
authors = [{ name = "Mazyod", email = "860511+Mazyod@users.noreply.github.com" }]
keywords = ["vllm","jina","embeddings","multi-vector","colbert","late-interaction","multimodal","qwen2-5-vl","retrieval"]
classifiers = [
  "Development Status :: 4 - Beta",
  "Intended Audience :: Developers",
  "License :: OSI Approved :: MIT License",
  "Programming Language :: Python :: 3",
  "Programming Language :: Python :: 3.12",
  "Programming Language :: Python :: 3.13",
  "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

[project.entry-points."vllm.general_plugins"]
jina_v4 = "jina_v4_vllm_plugin:register"

[project.urls]
Homepage = "https://github.com/Mazyod/jina-embeddings-v4-vllm-plugin"
Repository = "https://github.com/Mazyod/jina-embeddings-v4-vllm-plugin"
Documentation = "https://github.com/Mazyod/jina-embeddings-v4-vllm-plugin#readme"

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/jina_v4_vllm_plugin"]
```

**Notes**
- **No runtime dependencies.** vLLM/torch are intentionally absent so `pip install --no-deps` into the
  official vLLM image does not re-resolve them. (Keep the explanatory comment from the current pyproject.)
- The `.jinja` chat template ships in the wheel automatically — hatchling includes all files under the
  package directory, so the old `[tool.setuptools.package-data]` block is dropped. Verified at build
  time (see Verification).
- `jina_v4_vllm_plugin/__init__.py` and `model.py` are **unchanged** — the package imports only
  stdlib / numpy / torch / vllm (no `jinav4_vllm.*`), so it promotes to a standalone root package
  cleanly.

## `.github/workflows/publish.yml` (mirror of puckling)

Copy puckling's workflow as-is (it is the cleaner of the two — main-branch guard + careful git config):

- `on: workflow_dispatch` with a `bump_type` choice input (`patch`/`minor`/`major`).
- `permissions: { contents: write, id-token: write }`.
- Steps: main-branch guard → `actions/checkout@v6` (fetch-depth 0) → `astral-sh/setup-uv@v8.1.0`
  (cache on `uv.lock`) → `uv python install 3.13` → `uv sync --all-extras` → configure git author
  (`github-actions[bot]`) → `uv version --bump ${bump_type}` → commit `pyproject.toml uv.lock` →
  tag `vX.Y.Z` → push main + tag → `uv build` → `pypa/gh-action-pypi-publish@release/v1` (OIDC, no
  token) → `gh release create vX.Y.Z dist/* --generate-notes`.

Because the root **is** the package, the workflow needs ZERO structural deviation from the reference:
clean `vX.Y.Z` tags, no `working-directory`, no tag disambiguation.

## PyPI Trusted Publisher (manual — performed by the user on pypi.org)

Add a **pending publisher** at pypi.org → Account → Publishing:

| Field | Value |
|---|---|
| PyPI Project Name | `jina-v4-vllm-plugin` |
| Owner | `Mazyod` |
| Repository name | `jina-embeddings-v4-vllm-plugin` |
| Workflow name | `publish.yml` |
| Environment name | *(blank — matches the other two repos)* |

(Optional) configure the same on test.pypi.org first for a dry run. Account must have 2FA enabled
(already required by PyPI).

## Keeping the infra runnable — required rewiring

1. **`research/src/jinav4_vllm/modal_app/app.py`**
   - Plugin install into `vllm_plugin_image`: the current
     `.add_local_dir("src/jinav4_vllm/vllm_plugin", remote_path="/opt/jina_plugin", copy=True)` must
     install the plugin from the **repo-root** package instead — add `../src/jina_v4_vllm_plugin`,
     `../pyproject.toml`, `../README.md` into the image and `pip install --no-deps` it. (Modal is
     invoked from `research/`, so `..` is the repo root.)
   - Chat template: replace the hardcoded
     `CHAT_TEMPLATE_SRC = "/root/jinav4_vllm/vllm_plugin/.../jina_image_chat_template.jinja"` with the
     installed plugin's `jina_v4_vllm_plugin.chat_template_path()` (the same mechanism `serve_c.py`
     already uses).
2. **`research/deploy/Dockerfile` + `research/deploy/DEPLOY.md`**
   - `COPY src/jinav4_vllm/vllm_plugin /opt/jina_plugin` → copy the root plugin (build context = repo
     root, `-f research/deploy/Dockerfile`), or, post-publish, simplify to
     `pip install jina-v4-vllm-plugin`. Update the build-context note in DEPLOY.md.
3. **`research/Makefile`** — keep the modal/validation targets (paths inside `research/` stay valid).
   Drop the `package` target (the plugin builds from root now).
4. **Root `Makefile`** — `install` (`uv sync`), `test` (`uv run pytest`), `build` (`uv build`).
5. **Doc path references** — `src/jinav4_vllm/...` mentions inside `research/docs/*`, `research/README.md`
   stay valid (relative to `research/`); plugin-path mentions update to the root package /
   `pip install jina-v4-vllm-plugin`.
6. **Root `README.md`** — the plugin README, with a top-of-file pointer to `research/` for the
   build/validate/bake/deploy story.

## Downstream

- **HF model card** (`Mazyod/jina-embeddings-v4-vllm-mv`): swap the install line from
  `pip install --no-deps "git+https://github.com/Mazyod/jina-embeddings-v4-vllm-plugin@main#subdirectory=src/jinav4_vllm/vllm_plugin"`
  → `pip install jina-v4-vllm-plugin` **after the first PyPI release** (push via `hf`).
- Any other docs referencing the `#subdirectory=...` install URL → update to the PyPI name.

## Verification

- **Pre-flight:** confirm `jina-v4-vllm-plugin` is available on PyPI (and TestPyPI) before tagging.
- **Build:** `uv build` at root → sdist + wheel. `unzip -l dist/*.whl` shows
  `jina_v4_vllm_plugin/{__init__.py,model.py,jina_image_chat_template.jinja}` and a dist-info
  `entry_points.txt` containing `jina_v4 = jina_v4_vllm_plugin:register`.
- **Metadata:** `uvx twine check dist/*` passes (README renders, license/classifiers valid).
- **Root tests:** `uv run pytest` (the rewritten packaging test) is green.
- **Infra tests:** `cd research && uv run pytest` → all 8 pure-logic tests green; plus an import smoke
  (`uv run python -c "import jinav4_vllm.client, jinav4_vllm.modal_app.app"`).
- **Dry run (optional):** a `workflow_dispatch` against TestPyPI once the test.pypi.org trusted
  publisher is configured.

## Out of scope

- Actually cutting the first `v0.1.0` release (a separate, user-triggered `workflow_dispatch`).
- Re-running the GPU validation harness (the artifacts are already verified; this is a layout move).
- The `../nlu` hf-server → vLLM `/pooling` migration (tracked separately).
