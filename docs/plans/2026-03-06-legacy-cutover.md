# Legacy Cutover Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate README and notebooks to the new `src/` pipeline, then remove the legacy training code so the repository has one authoritative runtime path.

**Architecture:** The cutover keeps notebooks and docs as thin clients of the new CLI and `.pth` artifact format. After those visible surfaces are updated, the obsolete script-era training modules are deleted and a regression test enforces the absence of active legacy references.

**Tech Stack:** Markdown, Jupyter notebooks, Python 3.12, pytest

---

### Task 1: Add a legacy-reference regression test

**Files:**
- Create: `tests/test_no_legacy_runtime_refs.py`

**Step 1: Write the failing test**

```python
from pathlib import Path


def test_readme_and_runtime_docs_do_not_reference_legacy_modules():
    text = Path("README.md").read_text()
    assert "datasets.nsd" not in text
```

**Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_no_legacy_runtime_refs.py -v`
Expected: FAIL because README still references the legacy workflow.

**Step 3: Write minimal implementation**

Expand the test to check README and notebook JSON sources for active references to:
- `datasets.nsd`
- `datasets.nsd_utils`
- `engine.py`
- legacy `python main.py --subj ...`

**Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_no_legacy_runtime_refs.py -v`
Expected: PASS after docs/notebooks are updated.

**Step 5: Commit**

```bash
git add tests/test_no_legacy_runtime_refs.py
git commit -m "test: guard against legacy runtime references"
```

### Task 2: Rewrite README for the new authoritative workflow

**Files:**
- Modify: `README.md`

**Step 1: Replace old training commands**

Document:
- `python -m src.cli.inspect_data`
- `python -m src.cli.train`
- `python -m src.cli.evaluate`
- `python -m src.cli.predict`

**Step 2: Explain artifact layout**

Describe `last.pth`, `best.pth`, config/metrics files, and the paper config preset.

**Step 3: Run targeted verification**

Run: `uv run python -m pytest tests/test_no_legacy_runtime_refs.py -v`
Expected: fewer or no README-related failures.

**Step 4: Commit**

```bash
git add README.md
git commit -m "docs: migrate readme to rewritten pipeline"
```

### Task 3: Rewrite notebooks as thin clients

**Files:**
- Modify: `run_model.ipynb`
- Modify: `test_wrapper.ipynb`
- Modify: `visualize_results.ipynb`

**Step 1: Remove legacy imports and machine-specific paths**

Expected:
- no `datasets.nsd`
- no `datasets.nsd_utils`
- no `engine.py`
- no hard-coded `/engram/...` working-directory assumptions

**Step 2: Repoint workflows**

- `run_model.ipynb`: CLI launcher notebook
- `test_wrapper.ipynb`: wrapper load and prediction from `.pth`
- `visualize_results.ipynb`: load saved metrics/predictions from run directories

**Step 3: Run regression test**

Run: `uv run python -m pytest tests/test_no_legacy_runtime_refs.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add run_model.ipynb test_wrapper.ipynb visualize_results.ipynb
git commit -m "docs: port notebooks to new runtime path"
```

### Task 4: Remove obsolete training modules

**Files:**
- Delete: `datasets/__init__.py`
- Delete: `datasets/nsd.py`
- Delete: `datasets/nsd_utils.py`
- Delete: `models/__init__.py`
- Delete: `models/activations.py`
- Delete: `models/backbone.py`
- Delete: `models/brain_encoder.py`
- Delete: `models/clip.py`
- Delete: `models/custom_transformer.py`
- Delete: `models/dino.py`
- Delete: `models/position_encoding.py`
- Delete: `models/resnet.py`
- Delete: `models/transformer.py`
- Delete: `engine.py`

**Step 1: Confirm no active references remain**

Run: `rg -n "from models|import models|from datasets|import datasets|engine\\.py" -S .`
Expected: only archival or test-exempt references remain, or none.

**Step 2: Delete legacy files**

Use `apply_patch` deletions only.

**Step 3: Run full suite**

Run: `uv run python -m pytest tests -v`
Expected: PASS

**Step 4: Commit**

```bash
git add -A
git commit -m "refactor: remove legacy training code"
```
