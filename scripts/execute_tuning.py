"""Build and execute the supplemental notebook; preserve original notebooks."""
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
path = ROOT / "custom_llm_tuning.ipynb"
notebook = nbformat.read(path, as_version=4)
md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
notebook.cells = [
    md("# Controlled LLM tuning and a real optimizer update\n\n"
       "This supplemental experiment keeps the expanded corpus, vocabulary, split, and architecture fixed. "
       "It compares six configurations, selects using held-out loss, confirms across initialization seeds, "
       "and records a genuine step-1,000 update. The original two experiments remain available.\n\n"
       "**Prediction:** balanced exposure may improve grammar and contrast loss. Learning-rate changes may "
       "help optimization, but benchmark gains are uncertain. Vocabulary fixes coverage at 27/48.\n\n"
       "**Selection:** minimize final category-macro validation loss, subject to original-panel loss staying "
       "within 5% of the matched uniform, LR 0.001 baseline. Keep every trial, including failures.\n\n"
       "**Reproduction:** run from the repository root with its Python dependencies. Each execution creates "
       "a new study folder. This notebook executes the study functions directly; its outputs are recorded results."),
    code("from datetime import datetime, timezone\nfrom pathlib import Path\nimport sys\n"
         "from IPython.display import Markdown, SVG, display\n"
         "ROOT = Path.cwd()\nassert (ROOT / 'scripts/tuning_study.py').exists(), 'Run from repository root'\n"
         "sys.path.insert(0, str(ROOT))\nfrom scripts.tuning_study import Study, read_json\n"
         "study_path = ROOT / 'tuning_runs' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')\n"
         "study = Study(study_path)\nprint('Study:', study_path)"),
    md("## Freeze controls before training\n\n"
       "The original validation panel is retained. A second panel contains ten starter, five grammar, "
       "and five contrast passages. Each category contributes equally to the selection metric. "
       "The training-document RNG stays fixed across all runs; confirmation seeds change initialization only. "
       "Each sampler uses the same minibatches for all learning rates. The update example is chosen from "
       "the predetermined step-1,000 minibatch before training."),
    code("study.prepare()"),
    md("## Six configurations\n\nPeak learning rates: 0.0003, 0.001, and 0.003. "
       "Samplers: uniform passages; or 16 starter, eight grammar, eight contrast passages per batch. "
       "Every model starts from the same saved seed-42 weights and completes 3,000 updates. "
       "AdamW uses the supplied warmup, cosine decay, moments, weight decay, and gradient clipping. "
       "Losses average non-padding next-token targets. Every run saves its optimizer state and actual update evidence."),
    code("study.sweep()"),
    md("## Select using validation loss\n\nThe language suite has not been loaded or scored by this study. "
       "The selection artifact is written before confirmation and evaluation."),
    code("selection = study.select()"),
    md("## Confirm and test instrumentation\n\nRepeat the baseline and selected setting with seeds 43 and 44. "
       "Replay the selected seed-42 run with update instrumentation disabled and compare all final weights "
       "and recorded validation losses exactly."),
    code("study.confirm()"),
    md("## Evaluate after selection\n\nRun the unchanged public 48-case suite on all instrumented final checkpoints. "
       "Report coverage, correct answers, case gains, and regressions. These scores do not revise the selected configuration."),
    code("study.evaluate()"),
    md("## Verify saved evidence\n\nReconstruct the original baseline, check fixed inputs and initial weights, "
       "replay every saved evaluation, and reproduce each step-1,000 update from its weights, optimizer state, "
       "and minibatch. Compare the selected coordinate with the AdamW equation."),
    code("checks = study.verify()"),
    md("## Results, interpretation, and next experiments\n\n"
       "The report below is generated from the executed study's saved measurements. "
       "All weights change in an optimizer step; the selected coordinate illustrates that update. "
       "Individual target probabilities and validation losses can move in different directions."),
    code("summary = study.report()\ndisplay(Markdown((study_path / 'REPORT.md').read_text().replace('![Validation curves](curves.svg)', '')))\ndisplay(SVG(filename=str(study_path / 'curves.svg')))"),
]
notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbformat.write(notebook, path)


def save(cell, cell_index, execute_reply=None):
    nbformat.write(notebook, path)
    print(f"Executed tuning cell {cell_index + 1}/{len(notebook.cells)}", flush=True)
    for output in cell.get("outputs", []):
        if output.get("output_type") == "stream":
            print(output.get("text", "")[-5000:], flush=True)


client = NotebookClient(notebook, timeout=1800, kernel_name="python3",
                        resources={"metadata": {"path": str(ROOT)}}, on_cell_executed=save)
try:
    client.execute()
finally:
    nbformat.write(notebook, path)
