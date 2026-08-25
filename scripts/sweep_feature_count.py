"""Choose how many features to keep, by sweeping k on the validation split.

Step 2 of the feature-reduction protocol in the README. The sweep prepares one subsampled
dataset, trains a full-feature baseline on it, ranks the columns by permutation importance
measured on validation, then trains one model per candidate k and compares them.

Two properties matter more than anything else here:

**Every comparison is made on validation.** `summary_metrics.csv` is computed on the test
split, so a sweep that read it would pick k using the held-out set and quietly inflate the
confirmation run that follows. Macro-F1 and balanced accuracy come from the last row of
`history.json`, which LightGBM computed on the full validation split each round, and the
per-class recalls come from `feature_ranking.validation_scores`, which is structurally
unable to read test.

**The tolerance is declared before the sweep runs, not after.** With millions of
validation rows every difference is statistically significant and almost none are
meaningful, so the accept/reject rule is an absolute margin supplied on the command line.

    python scripts/sweep_feature_count.py \\
        --data-config config/data.json --train-config config/train.json \\
        --output-root outputs/sweep --target-total-rows 7000000 \\
        --k 60 40 30 20 15 10

The sweep is resumable: a run whose history already holds 100 iterations is skipped, so a
Colab session that dies mid-sweep continues where it stopped.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from checkpoint import atomic_json_dump  # noqa: E402
from data import load_config, prepare_dataset  # noqa: E402
from feature_ranking import (  # noqa: E402
    RANKING_FILE_NAME,
    validation_permutation_importance,
    validation_scores,
    write_ranking,
)
from train import load_train_config, train  # noqa: E402

LOGGER = logging.getLogger("sweep_feature_count")

BASELINE_TAG = "baseline"


def run_is_complete(run_dir: Path) -> bool:
    history_path = run_dir / "metrics" / "history.json"
    if not history_path.exists():
        return False
    try:
        return len(json.loads(history_path.read_text(encoding="utf-8"))) == 100
    except json.JSONDecodeError:
        return False


def validation_history_scores(run_dir: Path) -> dict[str, float]:
    """Final-round validation metrics, as LightGBM measured them on the whole split."""
    history = json.loads((run_dir / "metrics" / "history.json").read_text(encoding="utf-8"))
    final = history[-1]
    if int(final["iteration"]) != 100:
        raise ValueError(f"{run_dir.name} did not reach iteration 100")
    return {
        "val_macro_f1": float(final["val_macro_f1"]),
        "val_balanced_accuracy": float(final["val_macro_recall"]),
        "val_multi_logloss": float(final["val_multi_logloss"]),
        "val_accuracy": 1.0 - float(final["val_multi_error"]),
        "training_seconds": float(sum(float(item["iteration_seconds"]) for item in history)),
    }


def build_train_config(
    source: Mapping[str, Any],
    prepared: Path,
    ranking_file: Path | None,
    k: int | None,
) -> dict[str, Any]:
    config = json.loads(json.dumps(source))
    config["dataset"]["prepared_data_dir"] = str(prepared)
    # The sweep runs on a deterministic subsample, so the production "every physical row"
    # gate does not apply to it. The confirmation run on B still enforces it.
    config["dataset"]["require_full_dataset_manifest"] = False
    if k is None:
        config["feature_selection"] = "none"
    else:
        config["feature_selection"] = "validation_permutation_top_k"
        screening = dict(config["dataset"].get("feature_screening") or {})
        screening.update({"maximum_features": int(k), "ranking_file": str(ranking_file)})
        config["dataset"]["feature_screening"] = screening
    return config


def execute_run(
    config: Mapping[str, Any], output_root: Path, tag: str, resume: bool
) -> Path:
    run_dir = output_root / f"lightgbm_{tag}"
    if resume and run_is_complete(run_dir):
        LOGGER.info("Skipping %s: already at iteration 100", tag)
        return run_dir
    config_path = output_root / f"train.{tag}.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    code = train(Namespace(
        config=str(config_path),
        prepared_data_dir=None,
        output_dir=str(output_root),
        run_id=f"lightgbm_{tag}",
        max_rounds_this_session=None,
        upload_checkpoints_to_s3=False,
    ))
    if code != 0:
        raise SystemExit(
            f"Training for {tag} exited {code}"
            + (" (session paused; rerun to continue the sweep)" if code == 75 else "")
        )
    return run_dir


def evaluate(run_dir: Path, prepared: Path, scoring_rows: int) -> dict[str, Any]:
    scores = validation_history_scores(run_dir)
    detail = validation_scores(run_dir, prepared, maximum_rows=scoring_rows)
    run_config = json.loads((run_dir / "config" / "run_config.json").read_text(encoding="utf-8"))
    return {
        **scores,
        "feature_count": int(run_config["feature_count"]),
        "per_class_recall": detail["per_class_recall"],
        "per_class_support": detail["per_class_support"],
        "scored_rows": detail["rows_scored"],
    }


def sweep(
    data_config_path: Path,
    train_config_path: Path,
    output_root: Path,
    target_total_rows: int,
    candidates: Sequence[int],
    tolerance_macro_f1: float,
    tolerance_class_recall: float,
    scoring_rows: int,
    resume: bool = True,
) -> pd.DataFrame:
    output_root.mkdir(parents=True, exist_ok=True)
    prepared = output_root / "data"

    data_config = load_config(data_config_path)
    if data_config["dataset"].get("samples_per_file") is not None:
        raise ValueError("Clear dataset.samples_per_file before setting a target row count")
    data_config["dataset"]["target_total_rows"] = int(target_total_rows)
    if not (prepared / "sample_manifest.json").exists():
        LOGGER.info(
            "Preparing a %d-row subsample of %s", target_total_rows, data_config["dataset"]["data_dir"]
        )
        prepare_dataset(data_config, prepared)
    else:
        LOGGER.info("Reusing the prepared subsample at %s", prepared)

    train_config = load_train_config(train_config_path)

    LOGGER.info("=== baseline: every candidate feature ===")
    baseline_dir = execute_run(
        build_train_config(train_config, prepared, None, None), output_root, BASELINE_TAG, resume
    )
    baseline = evaluate(baseline_dir, prepared, scoring_rows)
    LOGGER.info(
        "Baseline %d features: val_macro_f1=%.6f val_balanced_accuracy=%.6f",
        baseline["feature_count"], baseline["val_macro_f1"], baseline["val_balanced_accuracy"],
    )

    ranking_path = baseline_dir / "config" / RANKING_FILE_NAME
    if not ranking_path.exists():
        LOGGER.info("=== permutation importance on the validation split ===")
        table, provenance = validation_permutation_importance(baseline_dir, prepared)
        write_ranking(baseline_dir, table, provenance)
        LOGGER.info(
            "%d of %d features fall within measurement noise",
            provenance["features_within_noise"], provenance["candidate_feature_count"],
        )
    else:
        LOGGER.info("Reusing the ranking at %s", ranking_path)

    rows = [{
        "k": baseline["feature_count"], "tag": BASELINE_TAG, "is_baseline": True,
        **{key: baseline[key] for key in (
            "feature_count", "val_macro_f1", "val_balanced_accuracy",
            "val_multi_logloss", "val_accuracy", "training_seconds", "scored_rows",
        )},
        "delta_macro_f1": 0.0, "delta_balanced_accuracy": 0.0,
        "worst_class": "", "worst_class_recall_drop": 0.0, "accepted": True,
    }]
    for k in candidates:
        if k >= baseline["feature_count"]:
            LOGGER.warning(
                "Skipping k=%d: the baseline only has %d features", k, baseline["feature_count"]
            )
            continue
        LOGGER.info("=== k = %d ===", k)
        run_dir = execute_run(
            build_train_config(train_config, prepared, ranking_path, k),
            output_root, f"k{k:03d}", resume,
        )
        result = evaluate(run_dir, prepared, scoring_rows)
        drops = {
            name: baseline["per_class_recall"][name] - value
            for name, value in result["per_class_recall"].items()
        }
        worst_class = max(drops, key=lambda name: drops[name])
        worst_drop = float(drops[worst_class])
        delta_f1 = baseline["val_macro_f1"] - result["val_macro_f1"]
        delta_balanced = baseline["val_balanced_accuracy"] - result["val_balanced_accuracy"]
        accepted = delta_f1 <= tolerance_macro_f1 and worst_drop <= tolerance_class_recall
        LOGGER.info(
            "k=%d val_macro_f1=%.6f (-%.6f) balanced=%.6f (-%.6f) worst class %s -%.4f -> %s",
            k, result["val_macro_f1"], delta_f1, result["val_balanced_accuracy"],
            delta_balanced, worst_class, worst_drop, "ACCEPT" if accepted else "reject",
        )
        rows.append({
            "k": k, "tag": f"k{k:03d}", "is_baseline": False,
            **{key: result[key] for key in (
                "feature_count", "val_macro_f1", "val_balanced_accuracy",
                "val_multi_logloss", "val_accuracy", "training_seconds", "scored_rows",
            )},
            "delta_macro_f1": delta_f1, "delta_balanced_accuracy": delta_balanced,
            "worst_class": worst_class, "worst_class_recall_drop": worst_drop,
            "accepted": accepted,
        })

    table = pd.DataFrame(rows)
    csv_path = output_root / "sweep_feature_count.csv"
    table.to_csv(csv_path, index=False, float_format="%.6f")

    passing = table[(~table["is_baseline"]) & table["accepted"]]
    chosen = int(passing["k"].min()) if len(passing) else None
    decision = {
        "scored_split": "validation",
        "target_total_rows": int(target_total_rows),
        "prepared_dir": str(prepared),
        "baseline_feature_count": baseline["feature_count"],
        "baseline_val_macro_f1": baseline["val_macro_f1"],
        "baseline_val_balanced_accuracy": baseline["val_balanced_accuracy"],
        "candidates": list(candidates),
        "tolerance_macro_f1": tolerance_macro_f1,
        "tolerance_class_recall": tolerance_class_recall,
        "rule": (
            "accept k when the validation Macro-F1 drop and the worst per-class validation "
            "recall drop both stay within the declared tolerances; choose the smallest "
            "accepted k"
        ),
        "chosen_k": chosen,
        "ranking_file": str(ranking_path),
        "next_step": (
            "confirm the chosen k once on Experiment B (config/data.expB.json + "
            "config/train.expB.json) against B's own full-feature baseline; do not sweep there"
        ),
        "results": table.to_dict(orient="records"),
    }
    decision_path = output_root / "sweep_feature_count.json"
    atomic_json_dump(decision, decision_path)
    LOGGER.info(
        "Chosen k = %s. Wrote %s and %s",
        chosen if chosen is not None else "none (no candidate met the tolerances)",
        csv_path, decision_path,
    )
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default="config/data.json")
    parser.add_argument("--train-config", default="config/train.json")
    parser.add_argument("--output-root", default="outputs/sweep")
    parser.add_argument("--target-total-rows", type=int, default=7_000_000)
    parser.add_argument("--k", type=int, nargs="+", default=[60, 40, 30, 20, 15, 10])
    parser.add_argument(
        "--tolerance-macro-f1", type=float, default=0.005,
        help="Largest acceptable absolute drop in validation Macro-F1",
    )
    parser.add_argument(
        "--tolerance-class-recall", type=float, default=0.02,
        help="Largest acceptable absolute drop in any single class's validation recall",
    )
    parser.add_argument(
        "--scoring-rows", type=int, default=0,
        help="Validation rows used for the per-class pass; 0 scores the whole split",
    )
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    table = sweep(
        Path(args.data_config), Path(args.train_config), Path(args.output_root),
        args.target_total_rows, sorted(set(args.k), reverse=True),
        args.tolerance_macro_f1, args.tolerance_class_recall, args.scoring_rows, args.resume,
    )
    columns = [
        "k", "val_macro_f1", "delta_macro_f1", "val_balanced_accuracy",
        "worst_class", "worst_class_recall_drop", "accepted",
    ]
    print(table[columns].to_string(index=False))


if __name__ == "__main__":
    main()
