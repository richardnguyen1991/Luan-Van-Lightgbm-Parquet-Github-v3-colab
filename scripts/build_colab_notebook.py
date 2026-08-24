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

**Which experiment to run.** Cell 3 has an `EXPERIMENT` dropdown. Each choice is a separate
prepared dataset, a separate `run_id` and a separate S3 prefix, so the three can be run in
any order without colliding.

| `EXPERIMENT` | Question it answers | Classes | Split |
|---|---|---|---|
| `A_random_split` | How well does the model do in-distribution? | 14 | group-aware random 70/15/15 |
| `B_cross_capture_day` | Does it generalise to another capture day? | 6 shared | train 01-12, test 03-11 |
| `C_open_set` | Where does it send an attack family it never saw? | 13 trained, Portmap held out | train 01-12, test = Portmap only |

A and B report Macro-F1 and balanced accuracy; plain accuracy is secondary, because TFTP
alone is 28.5% of the corpus. C reports no accuracy at all: its test rows carry a class the
model has no output unit for, so it reports the distribution it forces them into instead.
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
RAW_DIR = Path("/content/data/cicddos2019-parquet")
KAGGLE_DATASET = "dungnguyen28101991/cicddos2019-parquet"  #@param {type:"string"}
EXPERIMENT = "A_random_split"  #@param ["A_random_split", "B_cross_capture_day", "C_open_set"]

# Each experiment is a different preprocessing recipe, so it gets its own prepared directory
# and its own S3 prefix suffix. Sharing one would let a 6-class day-split dataset resume onto
# a 14-class random-split checkpoint and die at the feature_schema_hash guard.
EXPERIMENTS = {
    "A_random_split": ("config/data.json", "config/train.json", "data"),
    "B_cross_capture_day": ("config/data.expB.json", "config/train.expB.json", "data-expB"),
    "C_open_set": ("config/data.expC.json", "config/train.expC.json", "data-expC"),
}
DATA_CONFIG, TRAIN_CONFIG, PREPARED_NAME = EXPERIMENTS[EXPERIMENT]
PREPARED_DIR = Path("/content/outputs") / PREPARED_NAME
if EXPERIMENT != "A_random_split":
    # Keep each experiment's artifacts under their own S3 prefix.
    base_prefix = os.environ["S3_PREFIX"].rstrip("/")
    suffix = EXPERIMENT.split("_", 1)[0].lower()
    if not base_prefix.endswith(f"/{suffix}"):
        os.environ["S3_PREFIX"] = f"{base_prefix}/{suffix}"
os.environ["PIPELINE_EXPERIMENT"] = EXPERIMENT
os.environ["PIPELINE_DATA_CONFIG"] = DATA_CONFIG
os.environ["PIPELINE_TRAIN_CONFIG"] = TRAIN_CONFIG
os.environ["PIPELINE_PREPARED_DIR"] = str(PREPARED_DIR)
print(f"Experiment {EXPERIMENT}: {DATA_CONFIG} + {TRAIN_CONFIG} -> {PREPARED_DIR}")
print(f"S3 prefix: {os.environ['S3_PREFIX']}")

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
session = json.loads((SOURCE_DIR / TRAIN_CONFIG).read_text(encoding="utf-8"))["session"]

# data.py restores the shared prepared dataset from S3 when it is complete, and otherwise
# resumes preprocessing file by file, uploading each finished part.
data_code = run([
    sys.executable, "data.py",
    "--config", DATA_CONFIG,
    "--output-dir", PREPARED_DIR,
    "--s3-config", TRAIN_CONFIG,
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
import os, subprocess, sys
from pathlib import Path

SOURCE_DIR = Path("/content/src")
# Cell 3 chose the experiment; read it back rather than duplicating the mapping.
TRAIN_CONFIG = os.environ.get("PIPELINE_TRAIN_CONFIG", "config/train.json")
DATA_CONFIG = os.environ.get("PIPELINE_DATA_CONFIG", "config/data.json")
EXPERIMENT = os.environ.get("PIPELINE_EXPERIMENT", "A_random_split")
PREPARED_DIR = Path(os.environ.get("PIPELINE_PREPARED_DIR", "/content/outputs/data"))
RUNS_DIR = Path("/content/outputs/runs")

# Pin the run to the preprocessing recipe it was trained on. Without --run-id, train.py
# resolves the run from active_run.json on S3, which still points at whatever ran last: after
# a recipe change that resumes onto a checkpoint built from a different feature set and dies
# at the feature_schema_hash guard. Deriving the id from data_version gives both properties a
# resume needs -- identical every session, and new the moment the recipe changes.
sys.path.insert(0, str(SOURCE_DIR))
from data import compute_data_version, load_config

# The experiment tag is part of the id as well as the data_version: two experiments can
# share a recipe hash only by coincidence, and a collision would silently cross their runs.
RUN_ID = "lightgbm_{tag}_{version}".format(
    tag=EXPERIMENT.split("_", 1)[0].lower(),
    version=compute_data_version(load_config(SOURCE_DIR / DATA_CONFIG)),
)
print(f"run_id: {RUN_ID}")

train_code = subprocess.call([
    sys.executable, "train.py",
    "--config", TRAIN_CONFIG,
    "--prepared-data-dir", str(PREPARED_DIR),
    "--output-dir", str(RUNS_DIR),
    "--run-id", RUN_ID,
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
    # open_set_distribution.png only exists for Experiment C; the confusion matrix only for
    # A and B. Missing figures are reported, not treated as failures.
    for name in (
        "learning_curves.png", "confusion_matrix.png", "confusion_matrix_raw.png",
        "open_set_distribution.png",
    ):
        path = figures / name
        if path.exists():
            print(f"--- {name}")
            display(Image(filename=str(path)))
    summary = run_dir / "metrics" / "summary_metrics.csv"
    if summary.exists():
        import pandas as pd
        display(pd.read_csv(summary).T)
    else:
        print("summary_metrics.csv appears after the final evaluation step.")
    distribution = run_dir / "metrics" / "open_set_prediction_distribution.csv"
    if distribution.exists():
        import pandas as pd
        print("--- where the model sent the held-out class")
        display(pd.read_csv(distribution))
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
