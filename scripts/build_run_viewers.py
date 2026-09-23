"""Embed each actual run's checkpoint into a copy of the course viewer."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
for name, record in json.loads((ROOT / "experiment_runs.json").read_text()).items():
    run = ROOT / record["run_dir"]
    data = (run / "checkpoint.json").read_bytes()
    checkpoint = json.loads(data)
    payload = {"tokens": checkpoint["vocabulary"],
               "before": checkpoint["initial_embeddings"],
               "after": checkpoint["weights"]["wte"],
               "steps": checkpoint["completed_steps"],
               "name": f"nanoGPT · {name} actual run", "tokenizer": "word",
               "run": run.name, "sha256": hashlib.sha256(data).hexdigest(),
               "initialization": "Actual full embedding table captured before training."}
    packed = json.dumps(payload).replace("<", "\\u003c")
    page, count = re.subn(r'(<script id="model-data" type="application/json">).*?(</script>)',
                         lambda match: match[1] + packed + match[2],
                         (ROOT / "embedding-viewer.html").read_text(), count=1, flags=re.S)
    assert count == 1
    page = page.replace("Use reference model", "Reset to this run")
    destination = ROOT / "evidence" / f"embeddings_{name}.html"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(page)
    print(destination.relative_to(ROOT), payload["sha256"])
