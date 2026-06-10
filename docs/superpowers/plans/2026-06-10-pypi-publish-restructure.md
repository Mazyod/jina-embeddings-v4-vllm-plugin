# PyPI Publishing + Root-as-Plugin Restructure — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the repo root the publishable `jina-v4-vllm-plugin` package (mirroring the `puckling`/`lsp-python-types` reference repos), demote the vLLM/Modal/deploy/validation harness into a still-runnable nested `research/` subproject, and add PyPI Trusted Publishing via a GitHub Actions workflow.

**Architecture:** Two self-contained uv projects in one repo. **Root** = the plugin (hatchling, MIT, `>=3.12`, no runtime deps, `src/jina_v4_vllm_plugin/`); `uv build`/`uv version`/`uv run pytest` run naturally at root. **`research/`** = the harness (`jinav4-vllm`, modal/numpy/pillow/requests), operated via `cd research && uv run modal ...`. The harness installs the plugin into its Modal/Docker images from the repo-root package (one level up).

**Tech Stack:** uv, hatchling, pytest, GitHub Actions + `pypa/gh-action-pypi-publish` (OIDC Trusted Publishing), Modal (harness only).

**Spec:** `docs/superpowers/specs/2026-06-10-pypi-publish-restructure-design.md`

**Repo root for all commands:** `/Users/mazyod/projects/clients/boubyan/jina-hf`

**Commit identity (MANDATORY — public repo, scrubbed of client identity):** every commit in this plan MUST use
`git -c user.name="Mazyod" -c user.email="860511+Mazyod@users.noreply.github.com" commit ...`. Never commit with the default `Boubyan AI Team` identity. Do NOT `git push` (the user controls releases).

---

### Task 1: Relocate the harness into `research/` (bulk move, keep it runnable)

Move everything that is NOT the plugin into `research/`, leaving only `docs/superpowers/` (planning), `.git`, `.gitignore` at root. The plugin is still nested at `research/src/jinav4_vllm/vllm_plugin/` after this task — it is promoted in Task 2. Repoint the root-anchored `.gitignore` rules so harness artifacts stay ignored under the new path.

**Files:**
- Create dir: `research/`, `research/docs/`
- Move: `src/ tests/ deploy/ artifacts/ data/ reports/ Makefile pyproject.toml README.md uv.lock` → `research/`
- Move: `docs/VALIDATION.md docs/COMPAT.md` → `research/docs/`
- Modify: `.gitignore`

- [ ] **Step 1: Move the harness directories/files**

```bash
cd /Users/mazyod/projects/clients/boubyan/jina-hf
rm -rf .venv research/dist 2>/dev/null || true
mkdir -p research/docs
mv src tests deploy artifacts data reports Makefile pyproject.toml README.md uv.lock research/
mv docs/VALIDATION.md docs/COMPAT.md research/docs/
```

- [ ] **Step 2: Repoint root-anchored `.gitignore` rules to `research/`**

Edit `.gitignore` — replace these three lines:

```
# build outputs (e.g. the plugin wheel: src/jinav4_vllm/vllm_plugin/dist)
```
→
```
# build outputs (the plugin wheel builds at the repo root: ./dist)
```

```
# generated parity outputs (curated findings live in docs/VALIDATION.md)
reports/parity.json
reports/parity.md
```
→
```
# generated parity outputs (curated findings live in research/docs/VALIDATION.md)
research/reports/parity.json
research/reports/parity.md
```

```
# local artifact copies pulled from the Modal volume
artifacts/*/
!artifacts/.gitkeep
```
→
```
# local artifact copies pulled from the Modal volume
research/artifacts/*/
!research/artifacts/.gitkeep
```

- [ ] **Step 3: Verify the harness still imports and tests pass from its new home**

```bash
cd /Users/mazyod/projects/clients/boubyan/jina-hf/research
uv sync
uv run pytest -q
```
Expected: exit 0, all tests pass (9 test modules incl. `test_plugin_packaging.py`, which still finds the plugin at `research/src/jinav4_vllm/vllm_plugin/`).

- [ ] **Step 4: Verify root has only planning docs + research left**

```bash
cd /Users/mazyod/projects/clients/boubyan/jina-hf
ls -A1 | grep -vE '^(\.git|\.gitignore|docs|research)$'
```
Expected: no output (nothing else at root). `ls docs` → `superpowers`.

- [ ] **Step 5: Commit**

```bash
cd /Users/mazyod/projects/clients/boubyan/jina-hf
git add -A
git -c user.name="Mazyod" -c user.email="860511+Mazyod@users.noreply.github.com" \
  commit -m "refactor: relocate the vLLM/Modal harness into research/

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git show --stat --oneline HEAD | head -40
```
Expected: git reports renames (R) for the moved files, preserving history.

---

### Task 2: Promote the plugin package to the repo root

Move the plugin package up to `src/jina_v4_vllm_plugin/`, move its README to the repo root (becomes the PyPI long-description in Task 3/5), drop the old subdir `pyproject.toml` (a fresh root one is authored in Task 3), and remove the now-stale `research/tests/test_plugin_packaging.py` (rewritten at root in Task 4). The plugin imports only stdlib/numpy/torch/vllm, so it is self-contained.

**Files:**
- Move: `research/src/jinav4_vllm/vllm_plugin/jina_v4_vllm_plugin/` → `src/jina_v4_vllm_plugin/`
- Move: `research/src/jinav4_vllm/vllm_plugin/README.md` → `README.md` (repo root)
- Delete: `research/src/jinav4_vllm/vllm_plugin/pyproject.toml`, then the emptied `vllm_plugin/` dir
- Delete: `research/tests/test_plugin_packaging.py`

- [ ] **Step 1: Move the package + README, drop the old subdir manifest**

```bash
cd /Users/mazyod/projects/clients/boubyan/jina-hf
mkdir -p src
mv research/src/jinav4_vllm/vllm_plugin/jina_v4_vllm_plugin src/jina_v4_vllm_plugin
mv research/src/jinav4_vllm/vllm_plugin/README.md README.md
rm research/src/jinav4_vllm/vllm_plugin/pyproject.toml
rmdir research/src/jinav4_vllm/vllm_plugin
git rm -q research/tests/test_plugin_packaging.py
```

- [ ] **Step 2: Verify the package landed with its chat template**

```bash
cd /Users/mazyod/projects/clients/boubyan/jina-hf
ls src/jina_v4_vllm_plugin
test -f src/jina_v4_vllm_plugin/jina_image_chat_template.jinja && echo "TEMPLATE OK"
test ! -e research/src/jinav4_vllm/vllm_plugin && echo "OLD SUBDIR GONE"
```
Expected: lists `__init__.py  jina_image_chat_template.jinja  model.py`, then `TEMPLATE OK` and `OLD SUBDIR GONE`.

- [ ] **Step 3: Verify the harness suite is still green (now without the packaging test)**

```bash
cd /Users/mazyod/projects/clients/boubyan/jina-hf/research
uv run pytest -q
```
Expected: exit 0, all pass (8 test modules — the packaging test is gone).

- [ ] **Step 4: Commit**

```bash
cd /Users/mazyod/projects/clients/boubyan/jina-hf
git add -A
git -c user.name="Mazyod" -c user.email="860511+Mazyod@users.noreply.github.com" \
  commit -m "refactor: promote the plugin to the repo root (src/jina_v4_vllm_plugin)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Author the root plugin `pyproject.toml`, `LICENSE`, lock + dynamic version

Create the root package manifest (hatchling, MIT, `>=3.12`, full metadata, entry point, NO runtime deps), the MIT `LICENSE`, make `__version__` metadata-derived (so `uv version --bump` stays authoritative), generate the lock, and prove the wheel builds with the `.jinja` and entry point inside.

**Files:**
- Create: `pyproject.toml` (repo root)
- Create: `LICENSE`
- Modify: `src/jina_v4_vllm_plugin/__init__.py`
- Create (generated): `uv.lock`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "jina-v4-vllm-plugin"
version = "0.1.0"
description = "vLLM out-of-tree model: Jina Embeddings v4 multi-vector (token_embed) on Qwen2.5-VL"
readme = "README.md"
requires-python = ">=3.12"
license = "MIT"
authors = [{ name = "Mazyod", email = "860511+Mazyod@users.noreply.github.com" }]
keywords = ["vllm", "jina", "embeddings", "multi-vector", "colbert", "late-interaction", "multimodal", "qwen2-5-vl", "retrieval"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
# vLLM is the runtime host (the official vllm/vllm-openai image provides it); intentionally NOT a
# hard dependency so `pip install --no-deps` into that image does not try to re-resolve vLLM/torch.
# Pin the host vLLM version separately — see research/docs/COMPAT.md.

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

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"
```

- [ ] **Step 2: Write `LICENSE` (MIT)**

```text
MIT License

Copyright (c) 2026 Mazyod

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 3: Make `__version__` metadata-derived** in `src/jina_v4_vllm_plugin/__init__.py`

Replace the top of the file (the docstring stays; change the imports + version block). Replace:

```python
from importlib import resources

__version__ = "0.1.0"
```
with:
```python
from importlib import resources
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("jina-v4-vllm-plugin")
except PackageNotFoundError:  # running from a source checkout that isn't installed
    __version__ = "0+unknown"
```
Leave `CHAT_TEMPLATE_FILE`, `chat_template_path()`, and `register()` unchanged.

- [ ] **Step 4: Generate the lock and build**

```bash
cd /Users/mazyod/projects/clients/boubyan/jina-hf
uv lock
uv build
```
Expected: `uv build` writes `dist/jina_v4_vllm_plugin-0.1.0-py3-none-any.whl` and `dist/jina_v4_vllm_plugin-0.1.0.tar.gz`.

- [ ] **Step 5: Verify the wheel ships the template + entry point**

```bash
cd /Users/mazyod/projects/clients/boubyan/jina-hf
uv run --no-project --with twine python - <<'PY'
import glob, zipfile
whl = glob.glob("dist/*.whl")[0]
z = zipfile.ZipFile(whl)
names = z.namelist()
assert "jina_v4_vllm_plugin/jina_image_chat_template.jinja" in names, names
assert "jina_v4_vllm_plugin/model.py" in names, names
ep = next(n for n in names if n.endswith("entry_points.txt"))
body = z.read(ep).decode()
assert "jina_v4 = jina_v4_vllm_plugin:register" in body, body
print("WHEEL OK:", whl)
print(body.strip())
PY
uvx twine check dist/*
```
Expected: `WHEEL OK: ...`, the printed entry-point section, and `twine check` → `PASSED` for both artifacts.

- [ ] **Step 6: Commit** (dist/ is gitignored)

```bash
cd /Users/mazyod/projects/clients/boubyan/jina-hf
git add pyproject.toml LICENSE uv.lock src/jina_v4_vllm_plugin/__init__.py
git -c user.name="Mazyod" -c user.email="860511+Mazyod@users.noreply.github.com" \
  commit -m "feat: root plugin package manifest + MIT license + dynamic version

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Rewrite the packaging contract test for the root layout

Recreate the CI-friendly (no vLLM import, no GPU) packaging test at the repo root. It asserts the entry point, the register target, the template presence + vision tokens, and the **no-runtime-deps invariant** (so `pip install --no-deps` stays valid). It no longer references setuptools `package-data` (hatchling bundles the package dir; Task 3 Step 5 already proved the `.jinja` lands in the wheel).

**Files:**
- Create: `tests/test_plugin_packaging.py`

- [ ] **Step 1: Write the test**

```python
"""Guard the plugin's packaging contract without importing vLLM (CI-friendly, no GPU).

These assert the wiring a stock `vllm serve` relies on: the entry point, the registered model
target, the chat template shipping with the package, and the deliberate no-runtime-deps invariant
that keeps `pip install --no-deps` working inside the official vLLM image.
"""
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "jina_v4_vllm_plugin"


def _pyproject():
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def test_entry_point_registers_plugin():
    eps = _pyproject()["project"]["entry-points"]["vllm.general_plugins"]
    assert eps == {"jina_v4": "jina_v4_vllm_plugin:register"}


def test_register_targets_model_class():
    src = (PKG / "__init__.py").read_text()
    assert "JinaV4MultiVector" in src
    assert "jina_v4_vllm_plugin.model:JinaV4MultiVectorModel" in src


def test_chat_template_ships_with_package():
    # hatchling bundles every file under the package directory; assert it is present at the
    # packaged path (the wheel-contents check lives in the build step / CI).
    assert (PKG / "jina_image_chat_template.jinja").exists()


def test_chat_template_has_vision_tokens():
    tmpl = (PKG / "jina_image_chat_template.jinja").read_text()
    for tok in ("<|vision_start|>", "<|image_pad|>", "<|vision_end|>"):
        assert tok in tmpl


def test_no_runtime_dependencies():
    # vLLM/torch must NOT be hard deps so `pip install --no-deps` does not re-resolve them.
    assert _pyproject()["project"].get("dependencies", []) == []
```

- [ ] **Step 2: Run the test**

```bash
cd /Users/mazyod/projects/clients/boubyan/jina-hf
uv run pytest -q
```
Expected: exit 0, `5 passed`.

- [ ] **Step 3: Commit**

```bash
cd /Users/mazyod/projects/clients/boubyan/jina-hf
git add tests/test_plugin_packaging.py
git -c user.name="Mazyod" -c user.email="860511+Mazyod@users.noreply.github.com" \
  commit -m "test: packaging contract for the root plugin layout

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Root `Makefile` + `README.md`, and `research/` README/Makefile fixups

Give the root a slim `install`/`test`/`build` Makefile; make the root README the plugin's PyPI-facing README with a pointer to `research/`; fix the harness README's plugin-location row; and drop the harness `package` target (the plugin builds from root now).

**Files:**
- Create: `Makefile` (repo root)
- Overwrite: `README.md` (repo root)
- Modify: `research/README.md` (the plugin-location table row)
- Modify: `research/Makefile` (remove the `package` target)

- [ ] **Step 1: Write the root `Makefile`**

```makefile
.PHONY: help install test build
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install:  ## sync the dev environment (uv)
	uv sync

test:  ## run the packaging contract tests (no GPU/vLLM)
	uv run pytest -q

build:  ## build the sdist + wheel into dist/
	uv build
```

- [ ] **Step 2: Overwrite the root `README.md`** (PyPI long-description)

```markdown
# jina-v4-vllm-plugin

vLLM out-of-tree model plugin that makes a **stock vLLM OpenAI server** serve
**Jina Embeddings v4 multi-vector** (128-dim/token, ColBERT-style late interaction) **multimodal**
(text + image) embeddings. With the plugin installed, the server's `/pooling` endpoint returns final
L2-normalized `[n,128]` per-token multivectors directly — no proxy, no client-side projection.

It registers a `JinaV4MultiVector` architecture (Qwen2.5-VL backbone + Jina's `multi_vector_projector`
applied in-engine, mirroring vLLM's in-tree ColQwen3/ColPali pattern) via a `vllm.general_plugins`
entry point, so it loads in every vLLM process including the v1 EngineCore worker.

## Install

```bash
pip install jina-v4-vllm-plugin        # from PyPI
# into an image that already provides vLLM (e.g. vllm/vllm-openai), skip re-resolving vLLM/torch:
pip install --no-deps jina-v4-vllm-plugin
```

`--no-deps` keeps pip from re-resolving vLLM/torch inside the official image. Pin the host vLLM
version the plugin was validated against — see `research/docs/COMPAT.md`.

## Use

```bash
vllm serve <jina-v4-checkpoint> \
  --runner pooling --pooler-config.task token_embed \
  --hf-overrides '{"architectures":["JinaV4MultiVector"]}' \
  --chat-template "$(python -c 'import jina_v4_vllm_plugin as p; print(p.chat_template_path())')"
```

The projector weights (`128×2048` + bias) are **not** in the vLLM checkpoint; the plugin loads them
at startup from `JINA_MV_PROJECTOR` (default `/artifacts/projector/retrieval.npz`), or from the
checkpoint itself if baked in. A ready-made baked, drop-in checkpoint is published at
[`Mazyod/jina-embeddings-v4-vllm-mv`](https://huggingface.co/Mazyod/jina-embeddings-v4-vllm-mv).

## Build & validation tooling

The Modal build/validate/bake/deploy harness that produced and verified the artifacts lives under
[`research/`](research/) (its own uv project): projector extraction, checkpoint baking, HF-vs-vLLM
parity, the deploy runbook, and the vLLM-version compatibility matrix
(`research/docs/COMPAT.md`, `research/deploy/DEPLOY.md`).

## Develop

```bash
make install   # uv sync
make test      # packaging contract tests (no GPU/vLLM)
make build     # sdist + wheel into dist/
```

Releases publish to PyPI via GitHub Actions Trusted Publishing (OIDC) — run the **Publish to PyPI**
workflow (`workflow_dispatch`, choose patch/minor/major).
```

- [ ] **Step 3: Fix the plugin-location row in `research/README.md`**

Replace this line:

```markdown
| **`src/jinav4_vllm/vllm_plugin/`** | **The plugin** — `JinaV4MultiVector` model + entry-point registration + image chat template. Builds as the `jina-v4-vllm-plugin` wheel. The maintained core. |
```
with:
```markdown
| **the repo root** (`../pyproject.toml`, `../src/jina_v4_vllm_plugin/`) | **The plugin** — `JinaV4MultiVector` model + entry-point registration + image chat template. Published as `jina-v4-vllm-plugin` (`pip install jina-v4-vllm-plugin`). The maintained core. |
```

- [ ] **Step 4: Remove the `package` target from `research/Makefile`**

Delete these two lines (and the blank line after them) from `research/Makefile`:

```makefile
package:  ## build the installable plugin wheel -> src/jinav4_vllm/vllm_plugin/dist
	cd src/jinav4_vllm/vllm_plugin && uv build
```
Also remove `package` from the `.PHONY:` list on the first line (change `... test package extract ...` to `... test extract ...`).

- [ ] **Step 5: Verify**

```bash
cd /Users/mazyod/projects/clients/boubyan/jina-hf
make test
make build
make help
cd research && make help && ! grep -q '^package:' Makefile && echo "PACKAGE TARGET REMOVED"
```
Expected: `make test` → `5 passed`; `make build` → wheel in `dist/`; root `make help` lists install/test/build; research `make help` no longer lists `package`; prints `PACKAGE TARGET REMOVED`.

- [ ] **Step 6: Commit**

```bash
cd /Users/mazyod/projects/clients/boubyan/jina-hf
git add Makefile README.md research/README.md research/Makefile
git -c user.name="Mazyod" -c user.email="860511+Mazyod@users.noreply.github.com" \
  commit -m "docs: root README/Makefile for the plugin; repoint harness README + drop package target

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Rewire the Modal harness to install the plugin from the repo root

The harness Modal image installed the plugin from the old subdir. Post-move, install it from the
repo-root package (one level up from `research/`), and source the chat template (needed by the bake
job, which does NOT install the plugin) from a file mounted out of the root package.

**Files:**
- Modify: `research/src/jinav4_vllm/modal_app/app.py`

- [ ] **Step 1: Update `CHAT_TEMPLATE_SRC`**

Replace:
```python
# Chat template now ships inside the plugin package; this is its path under the mounted source.
CHAT_TEMPLATE_SRC = "/root/jinav4_vllm/vllm_plugin/jina_v4_vllm_plugin/jina_image_chat_template.jinja"
```
with:
```python
# Chat template ships with the root plugin package (one level up from research/); _with_local mounts
# it at this stable path so the bake job can read it without installing the plugin.
CHAT_TEMPLATE_SRC = "/root/jina_image_chat_template.jinja"
```

- [ ] **Step 2: Mount the template in `_with_local`**

Replace:
```python
def _with_local(img):
    """Attach project source + probe images so containers can import jinav4_vllm and read probes."""
    return (
        img.add_local_dir("src/jinav4_vllm", remote_path="/root/jinav4_vllm")
           .add_local_dir("data/probes", remote_path="/root/data/probes")
    )
```
with:
```python
def _with_local(img):
    """Attach project source + probe images so containers can import jinav4_vllm and read probes.

    The image chat template ships with the root plugin package (one level up from research/); mount
    it at a stable path so the bake job can read it without installing the plugin.
    """
    return (
        img.add_local_dir("src/jinav4_vllm", remote_path="/root/jinav4_vllm")
           .add_local_dir("data/probes", remote_path="/root/data/probes")
           .add_local_file(
               "../src/jina_v4_vllm_plugin/jina_image_chat_template.jinja",
               remote_path="/root/jina_image_chat_template.jinja",
           )
    )
```

- [ ] **Step 3: Install the plugin from the root package in `vllm_plugin_image`**

Replace:
```python
vllm_plugin_image = _with_local(
    _vllm_base
    .add_local_dir("src/jinav4_vllm/vllm_plugin", remote_path="/opt/jina_plugin", copy=True)
    .run_commands(
        "python -m pip install --no-deps /opt/jina_plugin "
        "|| (python -m ensurepip && python -m pip install --no-deps /opt/jina_plugin)"
    )
)
```
with:
```python
# Install our out-of-tree model as a vLLM general plugin (entry point). The plugin is now the repo's
# ROOT package (one level up from research/); assemble its project layout under /opt/jina_plugin and
# pip-install it. Post-publish you can replace the three add_local_* + install with a single:
#   .run_commands("python -m pip install --no-deps jina-v4-vllm-plugin")
vllm_plugin_image = _with_local(
    _vllm_base
    .add_local_file("../pyproject.toml", remote_path="/opt/jina_plugin/pyproject.toml", copy=True)
    .add_local_file("../README.md", remote_path="/opt/jina_plugin/README.md", copy=True)
    .add_local_dir("../src/jina_v4_vllm_plugin",
                   remote_path="/opt/jina_plugin/src/jina_v4_vllm_plugin", copy=True)
    .run_commands(
        "python -m pip install --no-deps /opt/jina_plugin "
        "|| (python -m ensurepip && python -m pip install --no-deps /opt/jina_plugin)"
    )
)
```

**Note (risk):** this relies on Modal accepting `..` local paths in `add_local_file`/`add_local_dir`
(resolved relative to the `research/` CWD where `modal run` is invoked). If a future Modal version
rejects `..`, switch to the post-publish form in the comment (`pip install --no-deps jina-v4-vllm-plugin`).

- [ ] **Step 4: Verify the module imports and the image graph constructs (no network/GPU)**

```bash
cd /Users/mazyod/projects/clients/boubyan/jina-hf/research
uv run python -c "import jinav4_vllm.modal_app.app as a; print('CHAT_TEMPLATE_SRC =', a.CHAT_TEMPLATE_SRC); print('plugin image:', type(a.vllm_plugin_image).__name__)"
```
Expected: prints `CHAT_TEMPLATE_SRC = /root/jina_image_chat_template.jinja` and `plugin image: Image` with no exception. (The actual Modal image **build** is GPU/Modal-side and is smoke-tested out of band by the user — see Task 9 notes.)

- [ ] **Step 5: Commit**

```bash
cd /Users/mazyod/projects/clients/boubyan/jina-hf
git add research/src/jinav4_vllm/modal_app/app.py
git -c user.name="Mazyod" -c user.email="860511+Mazyod@users.noreply.github.com" \
  commit -m "refactor(research): install the plugin into the Modal image from the root package

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: Rewire the deploy Dockerfile + runbook to the root plugin

The production Dockerfile baked the plugin from the old subdir. Repoint it to the repo-root package
with the build context at the repo root, and update the runbook's plugin-location + build-context notes.

**Files:**
- Overwrite: `research/deploy/Dockerfile`
- Modify: `research/deploy/DEPLOY.md`

- [ ] **Step 1: Overwrite `research/deploy/Dockerfile`**

```dockerfile
# Jina v4 multi-vector on the OFFICIAL vLLM OpenAI server image (Variant C).
#
# Extends `vllm/vllm-openai` with the JinaV4MultiVector out-of-tree model plugin so the stock
# OpenAI server's /pooling endpoint returns final L2-normalized [n,128] multivectors for text+image.
#
# Pin VLLM_TAG to the version the plugin was validated against (the model class touches vLLM
# internals; re-validate on upgrades — see research/docs/COMPAT.md).
ARG VLLM_TAG=v0.22.0
FROM vllm/vllm-openai:${VLLM_TAG}

# 1) Install the plugin (registers JinaV4MultiVector via a vllm.general_plugins entry point, loaded in
#    every process incl. the v1 EngineCore worker). The plugin is the repo ROOT package, so build with
#    the build context = repo root:  docker build -f research/deploy/Dockerfile -t <img> .
#    Post-publish you can replace the COPY+install with:  RUN pip install --no-deps jina-v4-vllm-plugin
COPY pyproject.toml README.md /opt/jina_plugin/
COPY src/jina_v4_vllm_plugin /opt/jina_plugin/src/jina_v4_vllm_plugin
RUN python3 -m pip install --no-cache-dir --no-deps /opt/jina_plugin

# 2) Chat template (only needed for the "Mode A" run below; the baked checkpoint embeds its own).
#    It ships inside the installed package; keep a copy at a stable path for the Mode A example.
RUN cp /opt/jina_plugin/src/jina_v4_vllm_plugin/jina_image_chat_template.jinja /opt/jina_image_chat_template.jinja

# 3) (Mode A only) the projector, if you are NOT using the baked checkpoint. Uncomment + provide it:
# COPY research/artifacts/projector/retrieval.npz /opt/retrieval.npz
# ENV JINA_MV_PROJECTOR=/opt/retrieval.npz

# The base image's entrypoint is already `vllm serve`. Pass the model + flags at `docker run` time
# (see research/deploy/DEPLOY.md). Example (Mode B, fully drop-in baked checkpoint):
#   docker run --gpus all -p 8000:8000 <image> \
#     <your-baked-checkpoint-repo-or-path> \
#     --runner pooling --pooler-config.task token_embed
```

- [ ] **Step 2: Update the runbook plugin-location row in `research/deploy/DEPLOY.md`**

Replace:
```markdown
| Plugin package | `src/jinav4_vllm/vllm_plugin/` | registers `JinaV4MultiVector` (entry point) + model class |
```
with:
```markdown
| Plugin package | repo root (`src/jina_v4_vllm_plugin/`, published as `jina-v4-vllm-plugin`) | registers `JinaV4MultiVector` (entry point) + model class |
```

- [ ] **Step 3: Update the build-context note in `research/deploy/DEPLOY.md`**

Replace:
```markdown
# from the repo root (build context must include src/jinav4_vllm/vllm_plugin)
```
with:
```markdown
# from the repo root (build context = repo root):  docker build -f research/deploy/Dockerfile -t <img> .
```

- [ ] **Step 4: Verify no stale plugin-subdir path remains in deploy/**

```bash
cd /Users/mazyod/projects/clients/boubyan/jina-hf
! grep -rn 'src/jinav4_vllm/vllm_plugin' research/deploy && echo "DEPLOY CLEAN"
grep -n 'src/jina_v4_vllm_plugin' research/deploy/Dockerfile
```
Expected: prints `DEPLOY CLEAN`, then the two new `COPY`/`cp` lines referencing `src/jina_v4_vllm_plugin`.

- [ ] **Step 5: Commit**

```bash
cd /Users/mazyod/projects/clients/boubyan/jina-hf
git add research/deploy/Dockerfile research/deploy/DEPLOY.md
git -c user.name="Mazyod" -c user.email="860511+Mazyod@users.noreply.github.com" \
  commit -m "refactor(deploy): build the plugin from the repo-root package

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: Add the Trusted Publishing workflow (`publish.yml`)

Add the GitHub Actions release workflow, copied from the `puckling` reference repo (the cleaner of
the two — main-branch guard + careful git config). Because the root **is** the package, it needs zero
structural deviation. The user has already configured the PyPI Trusted Publisher (project
`jina-v4-vllm-plugin`, owner `Mazyod`, repo `jina-embeddings-v4-vllm-plugin`, workflow `publish.yml`,
no environment).

**Files:**
- Create: `.github/workflows/publish.yml`

- [ ] **Step 1: Write `.github/workflows/publish.yml`**

```yaml
name: Publish to PyPI

on:
  workflow_dispatch:
    inputs:
      bump_type:
        description: Version bump type
        required: true
        default: patch
        type: choice
        options:
          - patch
          - minor
          - major

permissions:
  contents: write
  id-token: write

jobs:
  publish:
    name: Build and publish
    runs-on: ubuntu-latest

    steps:
      - name: Check main branch
        if: github.ref_name != 'main'
        run: |
          echo "Publish workflow must be dispatched from main."
          exit 1

      - name: Check out repository
        uses: actions/checkout@v6
        with:
          fetch-depth: 0

      - name: Install uv
        uses: astral-sh/setup-uv@v8.1.0
        with:
          enable-cache: true
          cache-dependency-glob: uv.lock

      - name: Install Python
        run: uv python install 3.13

      - name: Install dependencies
        run: uv sync --all-extras --python 3.13

      - name: Configure Git author
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "41898282+github-actions[bot]@users.noreply.github.com"

      - name: Bump version
        id: version
        run: |
          uv version --bump "${{ inputs.bump_type }}"
          VERSION="$(uv version --short)"
          echo "version=${VERSION}" >> "${GITHUB_OUTPUT}"

      - name: Commit version bump
        run: |
          git add pyproject.toml uv.lock
          git commit -m "chore: release v${{ steps.version.outputs.version }}"

      - name: Tag release
        run: git tag "v${{ steps.version.outputs.version }}"

      - name: Push main and tag
        run: |
          git push origin HEAD:main
          git push origin "v${{ steps.version.outputs.version }}"

      - name: Build distribution
        run: uv build

      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1

      - name: Create GitHub Release
        env:
          GH_TOKEN: ${{ github.token }}
        run: |
          gh release create "v${{ steps.version.outputs.version }}" dist/* --title "v${{ steps.version.outputs.version }}" --generate-notes
```

- [ ] **Step 2: Verify the workflow is valid YAML and references the right bits**

```bash
cd /Users/mazyod/projects/clients/boubyan/jina-hf
uv run --no-project --with pyyaml python - <<'PY'
import yaml
d = yaml.safe_load(open(".github/workflows/publish.yml"))
assert d["jobs"]["publish"]["permissions"]["id-token"] == "write"
assert any(s.get("uses", "").startswith("pypa/gh-action-pypi-publish")
           for s in d["jobs"]["publish"]["steps"])
print("WORKFLOW OK")
PY
```
Expected: `WORKFLOW OK`.

- [ ] **Step 3: Commit**

```bash
cd /Users/mazyod/projects/clients/boubyan/jina-hf
git add .github/workflows/publish.yml
git -c user.name="Mazyod" -c user.email="860511+Mazyod@users.noreply.github.com" \
  commit -m "ci: PyPI Trusted Publishing workflow (workflow_dispatch + version bump)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 9: Stale-path sweep + full verification

Catch any lingering references to the old plugin-subdir path or `#subdirectory=` install URL, fix
them, then run the full verification matrix. Record the PyPI name availability and the two follow-ups
(HF model-card install line; the actual first release) that are intentionally out of scope.

**Files:**
- Modify: any file surfaced by the sweep (most likely `research/docs/*.md`)

- [ ] **Step 1: Sweep for stale references** (exclude the planning docs, which describe the old layout intentionally, and `.git`)

```bash
cd /Users/mazyod/projects/clients/boubyan/jina-hf
grep -rn -E 'src/jinav4_vllm/vllm_plugin|#subdirectory=src/jinav4_vllm/vllm_plugin' . \
  --exclude-dir=.git --exclude-dir=docs 2>/dev/null || echo "NO STALE REFERENCES"
```
Expected: ideally `NO STALE REFERENCES`. If any non-planning file appears (e.g. a doc that still
documents the git `#subdirectory` install), edit it to `pip install jina-v4-vllm-plugin` (or the
root path `src/jina_v4_vllm_plugin`), then re-run until clean. Do NOT edit files under `docs/`
(`docs/superpowers/*` records the migration history on purpose).

- [ ] **Step 2: Full verification matrix**

```bash
cd /Users/mazyod/projects/clients/boubyan/jina-hf
rm -rf dist
uv build
uvx twine check dist/*
uv run pytest -q
cd research
uv run pytest -q
uv run python -c "import jinav4_vllm.client, jinav4_vllm.modal_app.app; print('HARNESS IMPORTS OK')"
```
Expected: wheel+sdist build; `twine check` PASSED ×2; root `5 passed`; research all pass (8 modules); `HARNESS IMPORTS OK`.

- [ ] **Step 3: Record PyPI name availability** (not a gate; informational)

```bash
curl -s -o /dev/null -w "%{http_code}\n" https://pypi.org/pypi/jina-v4-vllm-plugin/json
```
Expected: `404` = name is free. `200` = already taken → stop and tell the user before any release.

- [ ] **Step 4: Commit any sweep fixes** (skip if the working tree is clean)

```bash
cd /Users/mazyod/projects/clients/boubyan/jina-hf
git add -A
git -c user.name="Mazyod" -c user.email="860511+Mazyod@users.noreply.github.com" \
  commit -m "docs: update stale plugin-subdir references after restructure

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>" || echo "nothing to commit"
```

**Out of scope (do NOT do here — user-triggered follow-ups):**
- **First release:** run the GitHub **Publish to PyPI** workflow (`workflow_dispatch`, pick a bump). This is the user's action.
- **HF model card** (`Mazyod/jina-embeddings-v4-vllm-mv`): after the first PyPI release, change the install line from `git+…#subdirectory=src/jinav4_vllm/vllm_plugin` → `pip install jina-v4-vllm-plugin` (push via `hf`).
- **Modal/Docker live smoke:** the `..`-path Modal image build and the Docker build are validated on the user's Modal/GPU/Docker environment, not in this restructure.

---

## Self-Review

**Spec coverage:** root-as-plugin layout (T1–T3) ✓; MIT license (T3) ✓; `>=3.12` (T3) ✓; hatchling + no-deps + template-in-wheel (T3, verified) ✓; dynamic version (T3) ✓; rewritten packaging test (T4) ✓; root Makefile/README + research README/Makefile (T5) ✓; infra stays runnable — Modal app (T6) + Dockerfile/runbook (T7) ✓; `publish.yml` mirror of puckling (T8) ✓; Trusted Publisher (user-done, documented) ✓; `.gitignore` repoint (T1) ✓; stale-path sweep + verification + name check (T9) ✓; HF card + first release marked out-of-scope ✓.

**Placeholder scan:** no TBD/TODO; every code/edit step shows full content or exact before→after strings; every verify step has an exact command + expected output.

**Type/string consistency:** package import name `jina_v4_vllm_plugin` and PyPI name `jina-v4-vllm-plugin` used consistently; `CHAT_TEMPLATE_SRC = /root/jina_image_chat_template.jinja` matches the `_with_local` mount path; `/opt/jina_plugin/src/jina_v4_vllm_plugin/...` path consistent between Dockerfile COPY and the `cp` line; the entry point `jina_v4 = jina_v4_vllm_plugin:register` matches across pyproject, the wheel check, and the test.
