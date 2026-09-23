"""Execute the course notebook in a fresh kernel and preserve real cell outputs."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=["starter", "expanded"])
    args = parser.parse_args()
    os.chdir(ROOT)
    name = args.experiment
    notebook_path = ROOT / ("custom_llm.ipynb" if name == "starter" else "custom_llm_expanded.ipynb")
    notebook = nbformat.read(ROOT / "source/course_starter.ipynb", as_version=4)
    notebook.cells[0].source = f"# {name.title()} corpus experiment — executed evidence\n\n" + notebook.cells[0].source
    notebook.cells[1].source = (
        'CORPUS = "classroom"\n'
        f'CORPUS_FOLDER = "corpus/{name}"\n'
        'TRAINING_STEPS = 3000\n'
        'LEARNING_RATE = 0.001\n'
    )
    prediction = (ROOT / "docs/pretraining_plan.md").read_text()
    notebook.cells[2].source = "## Prediction recorded before training\n\n" + prediction + "\n\n## 2. Load the tools and network\n"
    # Preserve all course training, evaluation, inspection and sampling code.
    # The three extra chat cells run the very same supplied interface against this model.
    chat_cell = notebook.cells[23].source
    additional = []
    for prompt in ["the nurse", "the small cups", "explain quantum teleportation"]:
        additional.append(nbformat.v4.new_code_cell(chat_cell.replace('CHAT_PROMPT = "the customer"', f'CHAT_PROMPT = {json.dumps(prompt)}')))
    notebook.cells[24:24] = additional
    notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata.pop("colab", None)
    nbformat.write(notebook, notebook_path)
    before = set((ROOT / "llm_runs").glob("*")) if (ROOT / "llm_runs").exists() else set()
    started = time.time()

    def save_progress(cell, cell_index, execute_reply=None):
        nbformat.write(notebook, notebook_path)
        print(f"{name}: executed cell {cell_index + 1}/{len(notebook.cells)}", flush=True)

    client = NotebookClient(notebook, timeout=3600, kernel_name="python3",
                            resources={"metadata": {"path": str(ROOT)}},
                            on_cell_executed=save_progress)
    try:
        client.execute()
    finally:
        nbformat.write(notebook, notebook_path)
    new_runs = [p for p in (ROOT / "llm_runs").iterdir() if p.is_dir() and p not in before]
    if len(new_runs) != 1:
        raise RuntimeError(f"Expected one fresh run, found {new_runs}")
    ledger_path = ROOT / "experiment_runs.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    ledger[name] = {"notebook": str(notebook_path.relative_to(ROOT)),
                    "run_dir": str(new_runs[0].relative_to(ROOT)),
                    "wall_seconds": time.time() - started,
                    "python_executable": sys.executable}
    ledger_path.write_text(json.dumps(ledger, indent=2) + "\n")
    print(json.dumps(ledger[name], indent=2), flush=True)


if __name__ == "__main__":
    main()
