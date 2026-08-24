import json
from pathlib import Path


NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "persistence_time_validation_runner.ipynb"


def test_notebook_has_data_creation_and_persistence_cells():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    assert "create_full_ta_validation_folds()" in code
    assert "create_scaled_full_ta_validation_folds()" in code
    assert "run_persistence_baseline()" in code
