"""Generate colab_runner.ipynb.

The notebook is deliberately thin: it clones this repository and calls the same modules the
GitHub Actions runner calls, so notebook and runner can never drift apart. Regenerate with:

    python scripts/build_colab_notebook.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORCHESTRATION = json.loads(
    (PROJECT_ROOT / "config" / "orchestration.json").read_text(encoding="utf-8")
)
REPOSITORY = ORCHESTRATION["repository"]
NOTEBOOK_PATH = ORCHESTRATION["colab_notebook_path"]
COLAB_URL = f"https://colab.research.google.com/github/{REPOSITORY}/blob/main/{NOTEBOOK_PATH}"

INTRO = f"""# CIC-DDoS2019 LightGBM baseline — Colab runner

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB_URL})

This notebook holds no pipeline logic. It clones
[`{REPOSITORY}`](https://github.com/{REPOSITORY}), reads credentials from **Colab Secrets**,
and runs `data.py` / `train.py` / `make_report.py`. Every artifact is written to S3, so a
disconnected session loses at most one checkpoint block.

**Before running**

1. `Runtime -> Change runtime type -> CPU`, and enable **High-RAM** (Colab Pro).
2. Open the key icon in the left sidebar and add these secrets with *Notebook access* on:
   `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`, `S3_BUCKET`,
   `S3_PREFIX`, and — only for the first preprocessing run — `KAGGLE_USERNAME`, `KAGGLE_KEY`.
3. `Runtime -> Run all`, then **leave this tab open**. Colab Pro has no background
   execution, so closing the tab ends the session.

Exit code `75` means the session paused safely after a checkpoint: just run the notebook
again and it resumes at the next boosting iteration. Exit code `0` means the run reached
iteration 100 and the final report succeeded.
"""

SETUP = '''#@title 1. Clone the pipeline from GitHub
import subprocess, sys, os
from pathlib import Path

REPOSITORY = "{repository}"
BRANCH = "main"
SOURCE_DIR = Path("/content/src")

if SOURCE_DIR.exists():
    subprocess.check_call(["git", "-C", str(SOURCE_DIR), "fetch", "--depth", "1", "origin", BRANCH])
    subprocess.check_call(["git", "-C", str(SOURCE_DIR), "reset", "--hard", f"origin/{{BRANCH}}"])
else:
    subprocess.check_call([
        "git", "clone", "--depth", "1", "--branch", BRANCH,
        f"https://github.com/{{REPOSITORY}}.git", str(SOURCE_DIR),
    ])

commit = subprocess.check_output(
    ["git", "-C", str(SOURCE_DIR), "rev-parse", "HEAD"], text=True
).strip()
# Record the exact commit: the thesis must be able to name the code that produced a run.
print(f"Running {{REPOSITORY}}@{{commit}}")

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
    "-r", str(SOURCE_DIR / "requirements.txt"),
])
import lightgbm
print("LightGBM", lightgbm.__version__, "device=CPU")
'''

SECRETS = '''#@title 2. Load credentials from Colab Secrets
import os, time, json

REQUIRED_SECRETS = [
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION",
    "S3_BUCKET", "S3_PREFIX",
]
OPTIONAL_SECRETS = ["KAGGLE_USERNAME", "KAGGLE_KEY"]
COLAB_SESSION_HOURS = 11.5  #@param {type:"number"}

try:
    from google.colab import userdata
except ImportError:  # allows the notebook to be smoke-tested outside Colab
    userdata = None

missing = []
for name in REQUIRED_SECRETS + OPTIONAL_SECRETS:
    value = ""
    if userdata is not None:
        try:
            value = (userdata.get(name) or "").strip()
        except Exception:
            value = ""
    if not value:
        value = os.environ.get(name, "").strip()
    if value:
        os.environ[name] = value
    elif name in REQUIRED_SECRETS:
        missing.append(name)
if missing:
    raise RuntimeError(
        "Add these Colab Secrets and enable notebook access for each: " + ", ".join(missing)
    )
os.environ.setdefault("AWS_REGION", os.environ["AWS_DEFAULT_REGION"])

# One budget covers preprocessing and training in this session; train.py takes the
# earlier of this deadline and the one configured in config/train.json.
os.environ["PIPELINE_SESSION_DEADLINE_EPOCH"] = str(time.time() + COLAB_SESSION_HOURS * 3600.0)
print(f"Secrets loaded for s3://{os.environ['S3_BUCKET']}/{os.environ['S3_PREFIX']}")
print("Values were not printed.")
'''

DATA = '''#@title 3. Prepared dataset: reuse from S3, or build it once from the raw Parquet
import json, os, subprocess, sys
from pathlib import Path

SOURCE_DIR = Path("/content/src")
PREPARED_DIR = Path("/content/outputs/data")
RAW_DIR = Path("/content/data/cicddos2019-parquet")
KAGGLE_DATASET = "dungnguyen28101991/cicddos2019-parquet"  #@param {type:"string"}

def run(argv, **kwargs):
    print(">>", " ".join(str(item) for item in argv), flush=True)
    return subprocess.call([str(item) for item in argv], cwd=str(SOURCE_DIR), **kwargs)

PREPARED_DIR.mkdir(parents=True, exist_ok=True)
status = run([sys.executable, "scripts/sync_dataset.py", "status", "--output-dir", PREPARED_DIR])
needs_raw_download = status != 0

if needs_raw_download and not any(RAW_DIR.glob("**/*.parquet")):
    if not os.environ.get("KAGGLE_KEY"):
        raise RuntimeError(
            "The prepared dataset is not in S3 yet, so the raw Parquet must be downloaded. "
            "Add the KAGGLE_USERNAME and KAGGLE_KEY Colab Secrets and rerun this cell."
        )
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "kaggle"])
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    credentials = kaggle_dir / "kaggle.json"
    credentials.write_text(json.dumps({
        "username": os.environ["KAGGLE_USERNAME"], "key": os.environ["KAGGLE_KEY"],
    }), encoding="utf-8")
    credentials.chmod(0o600)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.check_call([
        "kaggle", "datasets", "download", "-d", KAGGLE_DATASET,
        "-p", str(RAW_DIR), "--unzip",
    ])
elif not needs_raw_download:
    print("Prepared dataset already exists in S3; the raw Parquet is not needed.")

# Preprocessing the full dataset outlasts a Colab session, so it gets the same time budget
# training does: stop on a source-file boundary before the runtime is pulled, rather than
# being killed mid-file and redoing that file next session.
session = json.loads((SOURCE_DIR / "config" / "train.json").read_text(encoding="utf-8"))["session"]

# data.py restores the shared prepared dataset from S3 when it is complete, and otherwise
# resumes preprocessing file by file, uploading each finished part.
data_code = run([
    sys.executable, "data.py",
    "--config", "config/data.json",
    "--output-dir", PREPARED_DIR,
    "--s3-config", "config/train.json",
    "--full-dataset",
    "--maximum-hours", session["maximum_hours"],
    "--stop-before-minutes", session["stop_before_minutes"],
])
print(f"data.py exit code: {data_code}")
if data_code == 75:
    raise SystemExit(
        "Preprocessing paused safely before the session deadline. Rerun the notebook to continue."
    )
if data_code != 0:
    raise SystemExit(f"data.py failed with exit code {data_code}")
'''

TRAIN = '''#@title 4. Train exactly 100 boosting iterations, resuming from S3
import subprocess, sys
from pathlib import Path

SOURCE_DIR = Path("/content/src")
PREPARED_DIR = Path("/content/outputs/data")
RUNS_DIR = Path("/content/outputs/runs")

train_code = subprocess.call([
    sys.executable, "train.py",
    "--config", "config/train.json",
    "--prepared-data-dir", str(PREPARED_DIR),
    "--output-dir", str(RUNS_DIR),
    "--upload-checkpoints-to-s3",
], cwd=str(SOURCE_DIR))

print(f"train.py exit code: {train_code}")
if train_code == 75:
    print(
        "Session paused safely after a checkpoint. Every artifact is on S3; rerun this "
        "notebook (or let the GitHub Actions fallback take over) to continue."
    )
elif train_code != 0:
    raise SystemExit(f"train.py failed with exit code {train_code}")
else:
    print("Run reached iteration 100 and the final report completed.")
'''

SUMMARY = '''#@title 5. Show accuracy/loss curves, confusion matrix, and summary metrics
import json
from pathlib import Path
from IPython.display import Image, display

RUNS_DIR = Path("/content/outputs/runs")
pointer_path = RUNS_DIR / "active_run.json"
if not pointer_path.exists():
    print("No active run pointer was created in this session.")
else:
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    print(json.dumps(pointer, indent=2))
    run_dir = RUNS_DIR / str(pointer["run_id"])
    figures = run_dir / "figures"
    for name in ("learning_curves.png", "confusion_matrix.png", "confusion_matrix_raw.png"):
        path = figures / name
        if path.exists():
            print(f"--- {name}")
            display(Image(filename=str(path)))
        else:
            print(f"--- {name} is not generated yet (it is produced after iteration 100)")
    summary = run_dir / "metrics" / "summary_metrics.csv"
    if summary.exists():
        import pandas as pd
        display(pd.read_csv(summary).T)
    else:
        print("summary_metrics.csv appears after the final evaluation step.")
'''


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.rstrip("\n").splitlines(keepends=True),
    }


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.rstrip("\n").splitlines(keepends=True),
    }


def build_notebook() -> dict:
    return {
        "cells": [
            markdown_cell(INTRO),
            code_cell(SETUP.format(repository=REPOSITORY)),
            code_cell(SECRETS),
            code_cell(DATA),
            code_cell(TRAIN),
            code_cell(SUMMARY),
        ],
        "metadata": {
            "accelerator": "None",
            "colab": {"name": NOTEBOOK_PATH, "provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(PROJECT_ROOT / NOTEBOOK_PATH))
    args = parser.parse_args()
    notebook = build_notebook()
    Path(args.output).write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.output} with {len(notebook['cells'])} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
