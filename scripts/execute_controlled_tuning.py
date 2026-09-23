"""Rerun the controlled supplement in its own notebook and fresh study directory."""
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "output/jupyter-notebook/custom_llm_tuning.ipynb"
notebook = nbformat.read(path, as_version=4)
for cell in notebook.cells:
    if cell.cell_type == "code":
        cell.outputs = []
        cell.execution_count = None


def save_progress(cell, cell_index, execute_reply=None):
    nbformat.write(notebook, path)
    print(f"Executed cell {cell_index + 1}/{len(notebook.cells)}", flush=True)


client = NotebookClient(notebook, timeout=3600, kernel_name="python3",
                        resources={"metadata": {"path": str(ROOT)}},
                        on_cell_executed=save_progress)
try:
    client.execute()
finally:
    nbformat.write(notebook, path)
print(f"Executed notebook saved: {path}")
