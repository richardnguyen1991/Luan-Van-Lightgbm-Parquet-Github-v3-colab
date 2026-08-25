"""Rank features by permutation importance measured on the validation split.

`make_report.py` already computes permutation importance, but it does so on
`raw/explain_sample.parquet`, which is drawn from the **test** split. That table is fine
for explaining a finished model and unusable for *choosing* features: selecting on it
leaks the held-out set into the decision, and every number reported afterwards on that
same test split is then optimistic by an unmeasurable amount.

This module answers the same question against validation instead, so the ranking it
produces can drive `feature_selection = "validation_permutation_top_k"` without touching
test. Gain -- the other available ranking -- is measured on the training split and is
biased toward high-cardinality columns; permutation measures what a trained model
actually loses when a column is destroyed, which is the quantity a feature-reduction
decision is about.

Run it against a finished run:

    python feature_ranking.py --run-dir outputs/runs/<run_id> \\
                              --prepared-data-dir outputs/data
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score

from checkpoint import atomic_json_dump
from model import (
    LazyParquetFeatures,
    _read_split_labels,
    stratified_monitor_indices,
)

LOGGER = logging.getLogger("feature_ranking")

RANKING_FILE_NAME = "feature_ranking_validation.json"
RANKING_FORMAT_VERSION = 1
# The one split this module is allowed to read. Named rather than parameterised: the whole
# point of the module is that the ranking never sees test.
PERMUTATION_SPLIT = "validation"

DEFAULT_SETTINGS = {
    "maximum_rows": 200000,
    "repeats": 5,
    "seed": 2026,
    "predict_chunk_rows": 250000,
}


def load_settings(run_dir: Path) -> dict[str, Any]:
    """Merge `report.json -> validation_permutation` over the defaults."""
    settings = dict(DEFAULT_SETTINGS)
    config_path = run_dir / "config" / "report.json"
    if config_path.exists():
        configured = json.loads(config_path.read_text(encoding="utf-8")).get(
            "validation_permutation"
        )
        if configured:
            unknown = set(configured).difference(DEFAULT_SETTINGS)
            if unknown:
                raise ValueError(
                    f"Unknown keys in report.json -> validation_permutation: {sorted(unknown)}"
                )
            settings.update(configured)
    if int(settings["maximum_rows"]) <= 0:
        raise ValueError("validation_permutation.maximum_rows must be positive")
    if int(settings["repeats"]) < 1:
        raise ValueError("validation_permutation.repeats must be at least 1")
    return settings


def _scores(
    labels: np.ndarray, predicted: np.ndarray, labels_order: Sequence[int]
) -> tuple[float, float]:
    return (
        float(f1_score(labels, predicted, labels=labels_order, average="macro", zero_division=0)),
        # Macro recall is balanced accuracy: the second headline metric, so a feature is
        # judged on both rather than on whichever one happens to move.
        float(recall_score(labels, predicted, labels=labels_order, average="macro", zero_division=0)),
    )


def _predict_classes(booster: Any, frame: pd.DataFrame, chunk_rows: int) -> np.ndarray:
    predicted = np.empty(len(frame), dtype=np.int32)
    for start in range(0, len(frame), chunk_rows):
        stop = min(start + chunk_rows, len(frame))
        block = booster.predict(frame.iloc[start:stop], num_iteration=100)
        predicted[start:stop] = np.asarray(block).argmax(axis=1)
        del block
    return predicted


def _load_validation_context(run_dir: Path, prepared: Path) -> dict[str, Any]:
    """Booster plus the validation split it will be scored on, and nothing else.

    Reading the split by name here rather than taking it as an argument is deliberate:
    every consumer in this module must be structurally incapable of scoring on test.
    """
    import lightgbm as lgb

    manifest = json.loads((prepared / "sample_manifest.json").read_text(encoding="utf-8"))
    label_mapping = json.loads((prepared / "label_mapping.json").read_text(encoding="utf-8"))
    run_config = json.loads((run_dir / "config" / "run_config.json").read_text(encoding="utf-8"))
    feature_names = list(run_config["feature_names"])
    model_feature_names = list(run_config["model_feature_names"])

    booster = lgb.Booster(model_file=str(run_dir / "checkpoints" / "final_model_round_100.txt"))
    if int(booster.current_iteration()) != 100:
        raise ValueError("Scoring requires the round-100 Booster")
    if list(booster.feature_name()) != model_feature_names:
        raise ValueError("Booster feature order disagrees with run_config.json")

    parts = manifest["parts"][PERMUTATION_SPLIT]
    labels = _read_split_labels(prepared, parts)
    if np.any(labels < 0):
        raise ValueError(
            "The validation split contains open-set rows, which have no trainable label to "
            "score against"
        )
    features = LazyParquetFeatures(prepared, parts, feature_names, model_feature_names)
    if len(features) != len(labels):
        raise AssertionError("Validation feature and label counts disagree")
    return {
        "booster": booster,
        "features": features,
        "labels": labels,
        "feature_names": feature_names,
        "model_feature_names": model_feature_names,
        "label_mapping": label_mapping,
        "run_config": run_config,
    }


def validation_scores(
    run_dir: str | Path,
    prepared_data_dir: str | Path,
    maximum_rows: int = 0,
    seed: int = 2026,
    predict_chunk_rows: int = 250000,
) -> dict[str, Any]:
    """Macro-F1, balanced accuracy and per-class recall of a finished run, on validation.

    A feature-count sweep must compare candidates on validation. ``summary_metrics.csv``
    is computed on the **test** split, so reading it to pick k would decide the reduction
    using the held-out set and make every later number on that set optimistic. These
    figures come from the same model but never touch test.

    ``maximum_rows = 0`` scores the whole split, which is what a decision should use; the
    permutation pass subsamples only because it re-predicts hundreds of times.
    """
    run_dir = Path(run_dir)
    context = _load_validation_context(run_dir, Path(prepared_data_dir))
    labels_all = context["labels"]
    indices = (
        np.arange(len(labels_all), dtype=np.int64)
        if maximum_rows <= 0 or len(labels_all) <= maximum_rows
        else stratified_monitor_indices(labels_all, int(maximum_rows), int(seed))
    )
    frame = context["features"].read(indices, as_frame=True)
    labels = labels_all[indices]
    label_mapping = context["label_mapping"]
    labels_order = list(range(len(label_mapping)))
    class_names = [
        name for name, _ in sorted(label_mapping.items(), key=lambda item: int(item[1]))
    ]
    predicted = _predict_classes(context["booster"], frame, int(predict_chunk_rows))
    macro_f1, balanced_accuracy = _scores(labels, predicted, labels_order)
    per_class = recall_score(
        labels, predicted, labels=labels_order, average=None, zero_division=0
    )
    support = np.bincount(labels, minlength=len(labels_order))
    del frame, context
    gc.collect()
    return {
        "scored_split": PERMUTATION_SPLIT,
        "rows_available": int(len(labels_all)),
        "rows_scored": int(len(labels)),
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        "per_class_recall": {
            name: float(value) for name, value in zip(class_names, per_class)
        },
        "per_class_support": {
            name: int(value) for name, value in zip(class_names, support)
        },
    }


def validation_permutation_importance(
    run_dir: str | Path,
    prepared_data_dir: str | Path,
    settings: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Mean drop in Macro-F1 when each feature is destroyed, measured on validation."""
    run_dir = Path(run_dir)
    prepared = Path(prepared_data_dir)
    settings = dict(settings or load_settings(run_dir))

    context = _load_validation_context(run_dir, prepared)
    booster = context["booster"]
    features_all = context["features"]
    labels_all = context["labels"]
    feature_names = context["feature_names"]
    model_feature_names = context["model_feature_names"]
    label_mapping = context["label_mapping"]
    run_config = context["run_config"]

    indices = stratified_monitor_indices(
        labels_all, int(settings["maximum_rows"]), int(settings["seed"])
    )
    frame = features_all.read(indices, as_frame=True)
    labels = labels_all[indices]
    labels_order = list(range(len(label_mapping)))
    LOGGER.info(
        "Permuting %d features x %d repeats over %d of %d validation rows",
        len(feature_names), int(settings["repeats"]), len(frame), len(labels_all),
    )

    chunk_rows = int(settings["predict_chunk_rows"])
    baseline_f1, baseline_balanced = _scores(
        labels, _predict_classes(booster, frame, chunk_rows), labels_order
    )
    LOGGER.info(
        "Validation baseline: macro_f1=%.6f balanced_accuracy=%.6f", baseline_f1, baseline_balanced
    )

    repeats = int(settings["repeats"])
    rng = np.random.default_rng(int(settings["seed"]))
    rows = []
    for feature, model_feature in zip(feature_names, model_feature_names):
        original = frame[model_feature].to_numpy(copy=True)
        f1_drops, balanced_drops = [], []
        for _ in range(repeats):
            frame[model_feature] = rng.permutation(original)
            permuted_f1, permuted_balanced = _scores(
                labels, _predict_classes(booster, frame, chunk_rows), labels_order
            )
            f1_drops.append(baseline_f1 - permuted_f1)
            balanced_drops.append(baseline_balanced - permuted_balanced)
        frame[model_feature] = original
        rows.append({
            "feature": feature,
            "mean_decrease_macro_f1": float(np.mean(f1_drops)),
            "std_decrease_macro_f1": float(np.std(f1_drops, ddof=1)) if repeats > 1 else 0.0,
            "mean_decrease_balanced_accuracy": float(np.mean(balanced_drops)),
        })
        del original
        gc.collect()

    table = pd.DataFrame(rows).sort_values(
        ["mean_decrease_macro_f1", "feature"], ascending=[False, True], ignore_index=True
    )
    table["rank"] = np.arange(1, len(table) + 1)
    # A feature whose destruction costs no more than the run-to-run noise of the measurement
    # is not evidence of usefulness. Flagged rather than dropped: the cut belongs to the
    # person choosing k, not to the ranking.
    table["within_noise"] = table["mean_decrease_macro_f1"] <= table["std_decrease_macro_f1"]

    provenance = {
        "format_version": RANKING_FORMAT_VERSION,
        "method": "permutation_importance",
        "scored_split": PERMUTATION_SPLIT,
        "scored_by": "macro_f1",
        "run_id": str(run_config.get("run_id", run_dir.name)),
        "feature_schema_hash": run_config.get("feature_schema_hash"),
        "data_version": (run_config.get("dataset_provenance") or {}).get("data_version"),
        "split_rows_available": int(len(labels_all)),
        "split_rows_scored": int(len(frame)),
        "sampling": "deterministic class-proportional subsample, minimum one row per class",
        "repeats": repeats,
        "seed": int(settings["seed"]),
        "baseline_macro_f1": baseline_f1,
        "baseline_balanced_accuracy": baseline_balanced,
        "candidate_feature_count": len(feature_names),
        "features_within_noise": int(table["within_noise"].sum()),
        "ranking": [
            {
                "rank": int(row.rank),
                "feature": str(row.feature),
                "mean_decrease_macro_f1": float(row.mean_decrease_macro_f1),
                "std_decrease_macro_f1": float(row.std_decrease_macro_f1),
                "within_noise": bool(row.within_noise),
            }
            for row in table.itertuples(index=False)
        ],
    }
    del frame, features_all, labels_all
    gc.collect()
    return table, provenance


def write_ranking(
    run_dir: str | Path,
    table: pd.DataFrame,
    provenance: Mapping[str, Any],
    callback: Any | None = None,
) -> list[Path]:
    run_dir = Path(run_dir)
    explainability = run_dir / "explainability"
    explainability.mkdir(parents=True, exist_ok=True)
    csv_path = explainability / "permutation_importance_validation.csv"
    table.to_csv(csv_path, index=False, float_format="%.8f")
    json_path = run_dir / "config" / RANKING_FILE_NAME
    json_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(dict(provenance), json_path)
    if callback is not None:
        callback(csv_path, "explainability")
        callback(json_path, "config")
    return [csv_path, json_path]


def load_ranking(path: str | Path) -> dict[str, Any]:
    """Read a ranking file and refuse anything that was not measured on validation."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(payload.get("format_version", 0)) != RANKING_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported feature-ranking format version: {payload.get('format_version')!r}"
        )
    if payload.get("scored_split") != PERMUTATION_SPLIT:
        raise ValueError(
            "Feature selection may only consume a ranking measured on the validation split; "
            f"this one was scored on {payload.get('scored_split')!r}"
        )
    if not payload.get("ranking"):
        raise ValueError("Feature-ranking file contains no ranking")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Finished run directory")
    parser.add_argument("--prepared-data-dir", required=True, help="Prepared split directory")
    parser.add_argument("--maximum-rows", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--upload-to-s3", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    run_dir = Path(args.run_dir)
    settings = load_settings(run_dir)
    for name in ("maximum_rows", "repeats", "seed"):
        value = getattr(args, name)
        if value is not None:
            settings[name] = value
    table, provenance = validation_permutation_importance(
        run_dir, args.prepared_data_dir, settings
    )
    from make_report import local_s3_callback

    callback = local_s3_callback(run_dir, args.upload_to_s3, False)
    written = write_ranking(run_dir, table, provenance, callback)
    top = table.head(15)[["rank", "feature", "mean_decrease_macro_f1"]]
    LOGGER.info(
        "Baseline macro_f1=%.6f; %d/%d features fall within measurement noise\n%s",
        provenance["baseline_macro_f1"], provenance["features_within_noise"],
        provenance["candidate_feature_count"], top.to_string(index=False),
    )
    LOGGER.info("Wrote %s", ", ".join(str(path) for path in written))


if __name__ == "__main__":
    main()
