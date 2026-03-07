import json
from pathlib import Path


LEGACY_TOKENS = (
    "datasets.nsd",
    "datasets.nsd_utils",
    "from engine import",
    "train_one_epoch",
    "python main.py --subj",
)


def _read_notebook(path: Path) -> str:
    notebook = json.loads(path.read_text())
    return "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))


def test_readme_and_notebooks_do_not_reference_legacy_runtime():
    readme = Path("README.md").read_text()
    notebooks = "\n".join(
        _read_notebook(Path(name))
        for name in ("run_model.ipynb", "test_wrapper.ipynb", "visualize_results.ipynb")
    )

    haystack = f"{readme}\n{notebooks}"
    for token in LEGACY_TOKENS:
        assert token not in haystack
