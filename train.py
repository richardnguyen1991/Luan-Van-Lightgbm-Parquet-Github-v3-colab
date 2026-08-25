"""Train the fixed 100-round CPU LightGBM baseline with resumable checkpoints.

This module intentionally contains no matplotlib imports or plotting code.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shutil
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import psutil
import sklearn

from checkpoint import (
    CheckpointManager,
    atomic_json_dump,
    canonical_hash,
    worker_environment,
    worker_id,
)
from model import (
    InsufficientMemoryError,
    IterationRecorder,
    TrainingPauseRequested,
    build_datasets,
    continue_training,
    multiclass_macro_metrics,
    validate_training_config,
)


LOGGER = logging.getLogger(__name__)
PAUSED_EXIT_CODE = 75

# Gradient-quantization variants, each with the run-id suffix that keeps its checkpoints
# apart. Changing these parameters changes params_hash, so without a distinct run id a
# rerun would resume onto an incompatible checkpoint and stop at the resume guard.
GRADIENT_QUANTIZATION_VARIANTS: dict[str, tuple[dict[str, Any], str]] = {
    "as-configured": ({}, ""),
    "off": ({"use_quantized_grad": False}, "gq-off"),
    "bins-32": ({"use_quantized_grad": True, "num_grad_quant_bins": 32}, "gq32"),
}


def apply_gradient_quantization(config: dict[str, Any], variant: str) -> str:
    """Override the quantization parameters in place; return the run-id suffix."""
    if variant not in GRADIENT_QUANTIZATION_VARIANTS:
        raise ValueError(
            f"Unknown --gradient-quantization {variant!r}; expected one of "
            f"{sorted(GRADIENT_QUANTIZATION_VARIANTS)}"
        )
    overrides, suffix = GRADIENT_QUANTIZATION_VARIANTS[variant]
    if overrides:
        config["model_params"].update(overrides)
        validate_training_config(config)
        LOGGER.info("Gradient quantization variant %r applied: %s", variant, overrides)
    return suffix


def run_id_for_variant(run_id: str | None, suffix: str) -> str | None:
    if not run_id or not suffix:
        return run_id
    return run_id if run_id.endswith(f"_{suffix}") else f"{run_id}_{suffix}"


def variant_from_run_id(run_id: str) -> str:
    """Which quantization variant a run id was created for."""
    for name, (_, suffix) in GRADIENT_QUANTIZATION_VARIANTS.items():
        if suffix and run_id.endswith(f"_{suffix}"):
            return name
    return "as-configured"


def check_variant_matches_run(run_id: str, variant: str) -> None:
    """Fail with an actionable message rather than an opaque params_hash mismatch.

    The GitHub Actions fallback resolves the run id from S3 rather than being told it, so
    it can easily be pointed at a run trained under a different quantization setting. The
    resume guard would catch that, but only as "params_hash differs", which says nothing
    about which flag to pass.
    """
    implied = variant_from_run_id(run_id)
    if implied != variant:
        raise ValueError(
            f"Run {run_id!r} was trained with gradient quantization {implied!r} but this "
            f"session selected {variant!r}. Pass --gradient-quantization {implied} to "
            "continue it, or choose a different run id."
        )


def load_train_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        if config_path.suffix.casefold() == ".json":
            config = json.load(handle)
        else:
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError("YAML configuration requires PyYAML; JSON needs no extra dependency") from exc
            config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Training configuration root must be an object")
    required = {
        "project_name", "model_name", "experiment_role", "seed", "device",
        "num_boost_round", "early_stopping", "imbalance_handling", "feature_selection",
        "use_all_train_rows", "model_params", "dataset", "checkpoint", "session", "s3", "logging",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Missing training configuration keys: {missing}")
    validate_training_config(config)
    return config


def session_deadline(config: Mapping[str, Any]) -> float | None:
    maximum_hours = float(config["session"]["maximum_hours"])
    stop_before_minutes = float(config["session"]["stop_before_minutes"])
    if maximum_hours <= 0:
        return None
    usable_seconds = maximum_hours * 3600.0 - stop_before_minutes * 60.0
    if usable_seconds <= 0:
        raise ValueError("session.stop_before_minutes must be less than session.maximum_hours")
    deadline_seconds = usable_seconds
    external_deadline = os.environ.get("PIPELINE_SESSION_DEADLINE_EPOCH")
    if external_deadline:
        deadline_seconds = min(deadline_seconds, max(0.0, float(external_deadline) - time.time()))
    return time.monotonic() + deadline_seconds


def validate_resume_state(
    state: Mapping[str, Any], run_id: str, params_hash: str, feature_schema_hash: str, target: int
) -> int:
    checks = {
        "run_id": (state.get("run_id"), run_id),
        "params_hash": (state.get("params_hash"), params_hash),
        "feature_schema_hash": (state.get("feature_schema_hash"), feature_schema_hash),
        "target_iteration": (int(state.get("target_iteration", -1)), int(target)),
    }
    failures = {key: {"observed": observed, "expected": expected} for key, (observed, expected) in checks.items() if observed != expected}
    if failures:
        raise ValueError(f"Checkpoint is incompatible with this run: {failures}")
    current = int(state["current_iteration"])
    if not 0 <= current <= int(target):
        raise ValueError(f"Checkpoint iteration {current} is outside 0..{target}")
    return current


def remaining_rounds(current_iteration: int, target_iteration: int) -> int:
    remaining = int(target_iteration) - int(current_iteration)
    if remaining < 0:
        raise ValueError("Checkpoint already exceeds the configured target iteration")
    return remaining


def check_available_memory(config: Mapping[str, Any]) -> str | None:
    """Refuse to start on a worker that is obviously too small for this dataset.

    The GitHub Actions runner has far less RAM than a Colab Pro high-RAM runtime. Pausing
    before building the LightGBM Datasets keeps the S3 checkpoint intact instead of losing
    the session to an OOM kill mid-iteration.
    """
    minimum_gb = config["session"].get("minimum_available_ram_gb")
    if minimum_gb in (None, 0):
        return None
    available_gb = psutil.virtual_memory().available / 1024 ** 3
    if available_gb >= float(minimum_gb):
        return None
    return (
        f"{available_gb:.1f} GiB RAM available on {worker_environment()} is below the configured "
        f"minimum of {float(minimum_gb):.1f} GiB"
    )


class Heartbeat:
    """Refresh active_run.json on a timer so the watchdog can tell live from dead workers.

    A single boosting iteration over the full dataset can take longer than the watchdog's
    staleness window, so the pointer cannot be refreshed only at checkpoint boundaries.
    Every pointer write goes through this object's lock to keep the checkpoint hook and the
    timer thread from racing on the same file.
    """

    def __init__(self, manager: CheckpointManager, run_id: str, interval_seconds: float) -> None:
        self.manager = manager
        self.run_id = run_id
        self.interval_seconds = float(interval_seconds)
        self.status = "running"
        self.current_iteration = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def publish(self, status: str, current_iteration: int) -> None:
        with self._lock:
            self.status = status
            self.current_iteration = int(current_iteration)
            self.manager.set_run_status(
                self.run_id, status, int(current_iteration),
                extra={"heartbeat_interval_seconds": self.interval_seconds},
            )

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self.publish(self.status, self.current_iteration)
            except Exception:
                LOGGER.warning("Heartbeat update failed; training continues", exc_info=True)

    def start(self) -> None:
        if self.interval_seconds <= 0 or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="active-run-heartbeat", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None


def _session_id() -> str:
    return f"session_{datetime.now().strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:8]}"


def _environment_metadata(lightgbm_version: str, num_threads: int) -> dict[str, Any]:
    memory = psutil.virtual_memory()
    return {
        "python_version": platform.python_version(),
        "python_compiler": platform.python_compiler(),
        "lightgbm_version": lightgbm_version,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "configured_num_threads": int(num_threads),
        "total_ram_bytes": int(memory.total),
    }


def _write_run_configuration(
    run_dir: Path,
    config: Mapping[str, Any],
    params: Mapping[str, Any],
    params_hash: str,
    feature_schema_hash: str,
    label_mapping: Mapping[str, int],
    feature_names: list[str],
    model_feature_names: list[str],
    lightgbm_version: str,
    feature_selection: Mapping[str, Any],
    data_version: str | None,
    memory_estimate: Mapping[str, Any],
    monitor_summary: Mapping[str, Any] | None = None,
    gradient_quantization: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    config_dir = run_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((config_dir / "sample_manifest.json").read_text(encoding="utf-8"))
    physical_rows = sum(int(item["physical_rows"]) for item in manifest["source_files"])
    selected_rows = sum(int(value) for value in manifest["split"]["sizes"].values())
    run_config = dict(config)
    run_config.update({
        "run_id": run_dir.name,
        "experiment_role": str(config["experiment_role"]),
        "num_boost_round": 100,
        "early_stopping": False,
        "imbalance_handling": "none",
        "feature_selection": str(config["feature_selection"]),
        "use_all_train_rows": True,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "learning_rate": float(params["learning_rate"]),
        "dataset_provenance": {
            "data_version": data_version,
            "dataset_root": manifest["dataset_root"],
            "sampling_mode": manifest["sampling_mode"],
            "source_file_count": len(manifest["source_files"]),
            "physical_rows": physical_rows,
            "selected_rows": selected_rows,
            "rows_excluded_by_label_policy": int(
                manifest["split"].get("rows_excluded_by_label_policy", 0)
            ),
            "rows_excluded_by_unassigned_split": int(
                manifest["split"].get("rows_excluded_by_unassigned_split", 0)
            ),
            # True only for the unrestricted experiment. B and C exclude rows on purpose,
            # and the two counters above say exactly how many and why.
            "all_physical_rows_used": (
                manifest["sampling_mode"] == "full" and selected_rows == physical_rows
            ),
            "split_sizes": dict(manifest["split"]["sizes"]),
            "leakage_audit_passed": bool(manifest["leakage_audit"]["passed"]),
            "split_strategy": manifest["split"].get("strategy", "auto_group_aware"),
            "capture_day_assignment": manifest["split"].get("capture_day_assignment"),
            "group_audit_scope": manifest["split"].get("group_audit_scope"),
            "label_policy": manifest.get("label_policy"),
            "open_set_labels": manifest["split"].get("open_set_labels", []),
            "open_set_only_splits": manifest["split"].get("open_set_only_splits", []),
        },
        "monitoring": dict(monitor_summary) if monitor_summary else {"enabled": False},
        "gradient_quantization": dict(gradient_quantization) if gradient_quantization else None,
        "params_hash": params_hash,
        "feature_schema_hash": feature_schema_hash,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "model_feature_names": model_feature_names,
        "feature_name_mapping": dict(zip(feature_names, model_feature_names)),
        "feature_selection_summary": {
            "method": feature_selection["method"],
            "fit_split": feature_selection.get("fit_split"),
            "candidate_feature_count": int(feature_selection["candidate_feature_count"]),
            "selected_feature_count": int(feature_selection["selected_feature_count"]),
            "screening_rows_used": feature_selection.get("screening_rows_used"),
            "final_training_rows_discarded": int(
                feature_selection.get("final_training_rows_discarded", 0)
            ),
        },
        "num_classes": len(label_mapping),
        "memory_estimate": dict(memory_estimate),
        "environment": _environment_metadata(lightgbm_version, int(params["num_threads"])),
    })
    run_path = config_dir / "run_config.json"
    params_path = config_dir / "model_params.json"
    atomic_json_dump(run_config, run_path)
    atomic_json_dump(dict(params), params_path)
    return run_path, params_path


def _prepared_data_version(prepared: Path) -> str | None:
    version_path = prepared / "dataset_version.json"
    if not version_path.exists():
        return None
    return str(json.loads(version_path.read_text(encoding="utf-8"))["data_version"])


def _copy_prepared_metadata(
    prepared: Path,
    run_dir: Path,
    selected_feature_names: list[str],
    feature_selection: Mapping[str, Any],
) -> list[Path]:
    config_dir = run_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for name in ("preprocessing.json", "data_profile.json", "sample_manifest.json", "label_mapping.json"):
        source = prepared / name
        if not source.exists():
            raise FileNotFoundError(f"Required prepared-data artifact is missing: {source}")
        destination = config_dir / name
        shutil.copyfile(source, destination)
        copied.append(destination)
    preprocessing_path = config_dir / "preprocessing.json"
    preprocessing = json.loads(preprocessing_path.read_text(encoding="utf-8"))
    candidate_features = list(preprocessing["feature_columns_in_order"])
    preprocessing["candidate_feature_columns_in_order"] = candidate_features
    preprocessing["feature_columns_in_order"] = list(selected_feature_names)
    preprocessing["feature_dtypes"] = {
        name: preprocessing["feature_dtypes"][name] for name in selected_feature_names
    }
    preprocessing["feature_selection"] = {
        "method": feature_selection["method"],
        "fit_split": feature_selection.get("fit_split"),
        "candidate_feature_count": int(feature_selection["candidate_feature_count"]),
        "selected_feature_count": int(feature_selection["selected_feature_count"]),
    }
    atomic_json_dump(preprocessing, preprocessing_path)
    feature_selection_path = config_dir / "feature_selection.json"
    atomic_json_dump(dict(feature_selection), feature_selection_path)
    copied.append(feature_selection_path)
    report_source = Path(__file__).resolve().parent / "config" / "report.json"
    report_destination = config_dir / "report_config.json"
    shutil.copyfile(report_source, report_destination)
    copied.append(report_destination)
    return copied


def _run_final_reporting(
    manager: CheckpointManager,
    run_id: str,
    run_dir: Path,
    booster: Any,
    test_features: Any,
    test_labels: np.ndarray,
) -> bool:
    try:
        from make_report import evaluate_final_model

        evaluate_final_model(
            run_dir,
            booster,
            test_features,
            test_labels,
            callback=lambda path, category: manager.sync_artifact(run_id, path, category),
        )
        return True
    except Exception:
        LOGGER.warning(
            "Final evaluation/reporting failed; final_model_round_100.txt and checkpoints remain intact",
            exc_info=True,
        )
        return False


def train(args: argparse.Namespace) -> int:
    config = load_train_config(args.config)
    variant = str(getattr(args, "gradient_quantization", "as-configured"))
    variant_suffix = apply_gradient_quantization(config, variant)
    if args.prepared_data_dir is not None:
        config["dataset"]["prepared_data_dir"] = args.prepared_data_dir
    if args.max_rounds_this_session is not None:
        if args.max_rounds_this_session <= 0:
            raise ValueError("--max-rounds-this-session must be positive")
        config["session"]["max_rounds_this_session"] = args.max_rounds_this_session
    logging.basicConfig(
        level=getattr(logging, str(config["logging"]["level"]).upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("Install lightgbm>=4.0,<5 before training") from exc

    output_root = Path(args.output_dir)
    manager = CheckpointManager(
        output_root=output_root,
        model_name=str(config["model_name"]),
        checkpoint_config=config["checkpoint"],
        s3_config=config["s3"],
        s3_enabled_override=True if args.upload_checkpoints_to_s3 else None,
    )
    run_id = manager.resolve_run_id(run_id_for_variant(args.run_id, variant_suffix))
    check_variant_matches_run(run_id, variant)
    run_dir = manager.run_dir(run_id)
    prepared = Path(config["dataset"]["prepared_data_dir"])

    blocked = check_available_memory(config)
    if blocked is not None:
        LOGGER.warning("Pausing before dataset construction: %s", blocked)
        return PAUSED_EXIT_CODE
    claimed, claim_reason = manager.claim_run(
        run_id, float(config["session"].get("worker_stale_minutes") or 0.0)
    )
    if not claimed:
        LOGGER.warning("Refusing to start run %s: %s", run_id, claim_reason)
        return PAUSED_EXIT_CODE
    LOGGER.info(
        "Worker %s on %s claimed run %s (%s)",
        worker_id(), worker_environment(), run_id, claim_reason,
    )

    LOGGER.info("Preparing exactly one LightGBM Dataset for each train/validation/test split")
    try:
        bundle = build_datasets(prepared, config)
    except InsufficientMemoryError as exc:
        # Pausing keeps the S3 checkpoint intact and lets the watchdog retry on a larger
        # worker; an OOM kill mid-construction would lose the session for nothing.
        LOGGER.warning("Pausing before dataset construction: %s", exc)
        manager.set_run_status(run_id, "paused", 0)
        return PAUSED_EXIT_CODE
    target = int(config["num_boost_round"])
    manager.download_resume_state(run_id)
    loaded = manager.load_state(run_id)
    if loaded is None:
        current_iteration = 0
        history: list[dict[str, Any]] = []
        init_model: str | None = None
        _copy_prepared_metadata(
            prepared, run_dir, bundle.feature_names, bundle.feature_selection
        )
        _write_run_configuration(
            run_dir, config, bundle.params, bundle.params_hash, bundle.feature_schema_hash,
            bundle.label_mapping, bundle.feature_names, bundle.model_feature_names, lgb.__version__,
            bundle.feature_selection, _prepared_data_version(prepared), bundle.memory_estimate,
            gradient_quantization={
                "variant": variant,
                "use_quantized_grad": bool(config["model_params"]["use_quantized_grad"]),
                "num_grad_quant_bins": (
                    int(config["model_params"]["num_grad_quant_bins"])
                    if config["model_params"]["use_quantized_grad"] else None
                ),
            },
            monitor_summary={
                "enabled": bundle.monitor_dataset is not None,
                "name": bundle.monitor_name,
                "source_split": bundle.monitor_source_split,
                "rows": int(len(bundle.monitor_indices)) if bundle.monitor_indices is not None else 0,
                "selection": "deterministic class-proportional subsample, minimum one row per class",
                "used_for_model_selection": False,
            },
        )
        for path in sorted((run_dir / "config").glob("*.json")):
            manager.sync_artifact(run_id, path, "config")
    else:
        state, history = loaded
        current_iteration = validate_resume_state(
            state, run_id, bundle.params_hash, bundle.feature_schema_hash, target
        )
        init_model = str(run_dir / "checkpoints" / "last_model.txt")
        LOGGER.info("Resuming run %s from completed iteration %d", run_id, current_iteration)

    if remaining_rounds(current_iteration, target) == 0:
        booster = lgb.Booster(model_file=str(run_dir / "checkpoints" / "last_model.txt"))
        if int(booster.current_iteration()) != target:
            raise AssertionError("Serialized final Booster does not contain exactly 100 iterations")
        manager.save_final_model(run_id, booster, target)
        manager.set_run_status(run_id, "ready_for_report", target)
        reporting_complete = _run_final_reporting(
            manager, run_id, run_dir, booster, bundle.features["test"], bundle.labels["test"]
        )
        manager.set_run_status(
            run_id, "complete" if reporting_complete else "ready_for_report", target
        )
        LOGGER.info("Run %s was already complete at iteration %d", run_id, target)
        return 0

    session_id = _session_id()
    heartbeat = Heartbeat(
        manager, run_id, float(config["session"].get("heartbeat_seconds") or 0.0)
    )
    heartbeat.publish("running", current_iteration)
    heartbeat.start()
    checkpoint_state: dict[str, Any] = {}

    def checkpoint_hook(booster: Any, current_history: list[dict[str, Any]], status: str) -> float:
        nonlocal checkpoint_state
        checkpoint_state, checkpoint_seconds = manager.save_checkpoint(
            run_id=run_id,
            session_id=session_id,
            booster=booster,
            history=current_history,
            params_hash=bundle.params_hash,
            feature_schema_hash=bundle.feature_schema_hash,
            target_iteration=target,
            status=status,
        )
        current_history[-1]["checkpoint_seconds"] = checkpoint_seconds
        manager.update_history_after_checkpoint(run_id, checkpoint_state, current_history)
        heartbeat.publish(status, int(booster.current_iteration()))
        try:
            from viz import generate_incremental_reports

            generate_incremental_reports(
                run_dir,
                callback=lambda path, category: manager.sync_artifact(run_id, path, category),
            )
        except Exception:
            LOGGER.warning(
                "Incremental plotting failed after iteration %d; checkpoint remains valid",
                int(booster.current_iteration()),
                exc_info=True,
            )
        return checkpoint_seconds

    recorder = IterationRecorder(
        history=history,
        session_id=session_id,
        target_iteration=target,
        learning_rate=float(bundle.params["learning_rate"]),
        checkpoint_interval=int(config["checkpoint"]["interval_rounds"]),
        checkpoint_hook=checkpoint_hook,
        deadline_monotonic=session_deadline(config),
        max_rounds_this_session=config["session"].get("max_rounds_this_session"),
        session_start_iteration=current_iteration,
        maximum_session_hours=float(config["session"]["maximum_hours"]),
        stop_before_minutes=float(config["session"]["stop_before_minutes"]),
        environment=worker_environment(),
        monitor_name=bundle.monitor_name,
    )
    callbacks = [recorder]
    period = int(config["logging"]["lightgbm_period"])
    if period > 0:
        callbacks.append(lgb.log_evaluation(period=period))
    # A third evaluation set turns the learning-curve figure into the argument it needs to
    # make: train and validation come from the same capture day and will lie on top of each
    # other, while the held-out day separates. It is watched, never selected on -- early
    # stopping is off and the round count is fixed at 100.
    valid_sets = [bundle.train_dataset, bundle.validation_dataset]
    valid_names = ["train", "validation"]
    valid_features = {"validation": bundle.features["validation"]}
    if bundle.monitor_dataset is not None:
        valid_sets.append(bundle.monitor_dataset)
        valid_names.append(bundle.monitor_name)
        valid_features[bundle.monitor_name] = bundle.monitor_features
    try:
        # lightgbm.train(init_model=...) cannot resume a Parquet-Sequence Dataset; see
        # model.continue_training for the equivalent sequence of steps that can.
        booster = continue_training(
            lgb,
            params=bundle.params,
            train_dataset=bundle.train_dataset,
            valid_sets=valid_sets,
            valid_names=valid_names,
            num_boost_round=remaining_rounds(current_iteration, target),
            feval=multiclass_macro_metrics(len(bundle.label_mapping)),
            callbacks=callbacks,
            num_class=len(bundle.label_mapping),
            train_features=bundle.features["train"],
            valid_features=valid_features,
            init_model_path=init_model,
            init_score_chunk_rows=int(config["dataset"].get("init_score_chunk_rows", 250000)),
        )
    except TrainingPauseRequested as exc:
        LOGGER.info("%s; next session will resume at iteration %d", exc, len(history) + 1)
        return PAUSED_EXIT_CODE
    finally:
        heartbeat.stop()

    final_iteration = int(booster.current_iteration())
    if final_iteration != target or len(history) != target:
        raise AssertionError(
            f"Training returned without exactly 100 iterations: booster={final_iteration}, history={len(history)}"
        )
    final_path = manager.save_final_model(run_id, booster, target)
    heartbeat.publish("ready_for_report", target)
    LOGGER.info("Completed fixed baseline at iteration 100: %s", final_path)
    reporting_complete = _run_final_reporting(
        manager, run_id, run_dir, booster, bundle.features["test"], bundle.labels["test"]
    )
    heartbeat.publish("complete" if reporting_complete else "ready_for_report", target)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/train.json")
    parser.add_argument("--prepared-data-dir", default=None)
    parser.add_argument("--output-dir", default="outputs/runs")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-rounds-this-session", type=int, default=None)
    parser.add_argument(
        "--gradient-quantization",
        choices=sorted(GRADIENT_QUANTIZATION_VARIANTS),
        default="as-configured",
        help=(
            "Override LightGBM gradient quantization. 'off' removes it entirely, which is the "
            "decisive test for whether it is what makes the training loss non-monotonic. Any "
            "value but 'as-configured' also extends --run-id, so variants never share a checkpoint."
        ),
    )
    parser.add_argument("--upload-checkpoints-to-s3", action="store_true")
    return parser.parse_args()


def main() -> None:
    raise SystemExit(train(parse_args()))


if __name__ == "__main__":
    main()
