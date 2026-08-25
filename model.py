"""LightGBM Dataset construction, baseline parameters, metric, and callbacks."""

from __future__ import annotations

import ctypes
import gc
import json
import logging
import math
import re
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score

from checkpoint import canonical_hash


LOGGER = logging.getLogger(__name__)
SPLIT_NAMES = ("train", "validation", "test")
INTERNAL_COLUMNS = ("_sample_file_id", "_sample_row_id", "_label")
# Mirrors data.OPEN_SET_LABEL_CODE: rows of a class deliberately withheld from training.
OPEN_SET_LABEL_CODE = -1
RESERVED_VALID_NAMES = frozenset({"train", "validation", "training"})


class TrainingPauseRequested(RuntimeError):
    """Raised only after a durable checkpoint requests a new Kaggle session."""


class InsufficientMemoryError(RuntimeError):
    """Projected LightGBM peak exceeds the budget; pause rather than risk an OOM kill."""


def estimate_training_memory(
    split_sizes: Mapping[str, int],
    num_features: int,
    num_classes: int,
    params: Mapping[str, Any],
    available_bytes: int,
    available_fraction: float,
    monitor_rows: int = 0,
) -> dict[str, Any]:
    """Project the resident set this run needs, per allocation, for the Sequence path.

    ``data_profile.json`` only answers whether the *raw* float32 matrix fits, which is
    always false at full scale and is why the Parquet Sequence path exists at all. The
    number that actually decides whether this run survives is the peak of the binned
    dataset plus LightGBM's per-class score and gradient buffers plus the float64
    prediction buffers the Python layer allocates for every validation set — and, on a
    resumed session, the init scores computed before boosting restarts.
    """
    train_rows = int(split_sizes["train"])
    validation_rows = int(split_sizes["validation"])
    # A cross-day monitoring set is a third scored Dataset: it is binned, scored and
    # predicted every round exactly like validation, so it belongs in every term below.
    monitored_rows = max(0, int(monitor_rows))
    scored_rows = train_rows + validation_rows + monitored_rows
    bytes_per_bin = 1 if int(params.get("max_bin", 255)) <= 255 else 2

    binned_dataset = scored_rows * num_features * bytes_per_bin
    internal_scores = scored_rows * num_classes * 4
    gradients = train_rows * num_classes * 2 * 4
    # valid_sets carries both train and validation, and lightgbm.Booster caches one float64
    # prediction buffer per validation set so the custom Macro-F1 can be evaluated.
    prediction_buffers = scored_rows * num_classes * 8
    metric_workspace = scored_rows * 8
    steady_state = (
        binned_dataset + internal_scores + gradients + prediction_buffers + metric_workspace
    )
    # A resumed session materialises init scores as float64 and LightGBM copies them into
    # its own array before ours can be released, so both exist at once.
    resume_init_scores = scored_rows * num_classes * 8 * 2
    resume_peak = binned_dataset + internal_scores + resume_init_scores
    # The monitoring subsample is materialised once as a dense float32 matrix before
    # LightGBM bins it; that transient allocation coexists with the binned train Dataset.
    monitor_materialization = monitored_rows * num_features * 4
    construction_peak = (
        train_rows + validation_rows
    ) * num_features * bytes_per_bin + monitor_materialization
    estimated_peak = max(steady_state, resume_peak, construction_peak)
    budget = int(available_bytes * float(available_fraction))
    return {
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "monitor_rows": monitored_rows,
        "num_features": num_features,
        "num_classes": num_classes,
        "binned_dataset_bytes": int(binned_dataset),
        "lightgbm_score_bytes": int(internal_scores),
        "gradient_and_hessian_bytes": int(gradients),
        "python_prediction_buffer_bytes": int(prediction_buffers),
        "metric_workspace_bytes": int(metric_workspace),
        "steady_state_bytes": int(steady_state),
        "resume_init_score_bytes": int(resume_init_scores),
        "resume_peak_bytes": int(resume_peak),
        "monitor_materialization_bytes": int(monitor_materialization),
        "construction_peak_bytes": int(construction_peak),
        "estimated_peak_bytes": int(estimated_peak),
        "available_bytes": int(available_bytes),
        "available_fraction": float(available_fraction),
        "budget_bytes": budget,
        "fits_budget": estimated_peak <= budget,
    }


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def validate_training_config(config: Mapping[str, Any]) -> None:
    if int(config["num_boost_round"]) != 100:
        raise ValueError("Baseline num_boost_round must be exactly 100")
    if bool(config["early_stopping"]):
        raise ValueError("Early stopping is forbidden for the baseline")
    if config["device"] != "cpu":
        raise ValueError("The baseline must run on CPU")
    if config["imbalance_handling"] != "none":
        raise ValueError("Training imbalance handling must be 'none'")
    if config["feature_selection"] not in SUPPORTED_FEATURE_SELECTION:
        raise ValueError(f"feature_selection must be one of {list(SUPPORTED_FEATURE_SELECTION)}")
    if config["feature_selection"] == "train_gain_top_k":
        screening = config.get("dataset", {}).get("feature_screening")
        if not isinstance(screening, Mapping):
            raise ValueError("dataset.feature_screening is required for train_gain_top_k")
        for name in ("maximum_features", "balanced_train_rows", "num_boost_round"):
            if int(screening.get(name, 0)) <= 0:
                raise ValueError(f"dataset.feature_screening.{name} must be positive")
        if float(screening.get("learning_rate", 0.0)) <= 0.0:
            raise ValueError("dataset.feature_screening.learning_rate must be positive")
        if not isinstance(screening.get("seed"), int):
            raise ValueError("dataset.feature_screening.seed must be an integer")
    if config["feature_selection"] == "validation_permutation_top_k":
        screening = config.get("dataset", {}).get("feature_screening")
        if not isinstance(screening, Mapping):
            raise ValueError(
                "dataset.feature_screening is required for validation_permutation_top_k"
            )
        if int(screening.get("maximum_features", 0)) <= 0:
            raise ValueError("dataset.feature_screening.maximum_features must be positive")
        if not str(screening.get("ranking_file", "")).strip():
            raise ValueError(
                "dataset.feature_screening.ranking_file must name the feature_ranking_validation"
                ".json produced by feature_ranking.py against a completed full-feature run"
            )
    if not bool(config["use_all_train_rows"]):
        raise ValueError("The baseline must use every row in the train split")
    params = config["model_params"]
    required_exact = {
        "boosting_type": "gbdt",
        "objective": "multiclass",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_data_in_leaf": 20,
        "lambda_l1": 0.0,
        "lambda_l2": 0.1,
        "min_gain_to_split": 0.0,
        "max_bin": 255,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "device_type": "cpu",
        "deterministic": True,
        "is_enable_sparse": False,
        "histogram_pool_size": 128.0,
        "use_quantized_grad": True,
        "num_grad_quant_bins": 16,
        "quant_train_renew_leaf": True,
    }
    mismatches = {
        key: {"expected": expected, "observed": params.get(key)}
        for key, expected in required_exact.items()
        if params.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Baseline parameter contract violated: {mismatches}")
    if bool(params.get("force_col_wise")) == bool(params.get("force_row_wise")):
        raise ValueError("Exactly one of force_col_wise and force_row_wise must be true")
    if int(params.get("num_threads", 0)) <= 0:
        raise ValueError("model_params.num_threads must be a positive CPU thread count")
    for seed_name in ("seed", "bagging_seed", "feature_fraction_seed", "data_random_seed"):
        if seed_name not in params or not isinstance(params[seed_name], int):
            raise ValueError(f"model_params.{seed_name} must be an integer")
    forbidden = {"class_weight", "scale_pos_weight", "is_unbalance", "sample_weight"}
    present = sorted(forbidden.intersection(params))
    if present:
        raise ValueError(f"Imbalance-handling parameters are forbidden: {present}")
    metrics = set(params.get("metric", []))
    if metrics != {"multi_logloss", "multi_error"}:
        raise ValueError("Configured metrics must be exactly multi_logloss and multi_error")
    if int(config["checkpoint"]["interval_rounds"]) != 10:
        raise ValueError("checkpoint.interval_rounds must be exactly 10")
    monitor = config.get("dataset", {}).get("monitor_split")
    if monitor is not None:
        if not isinstance(monitor, Mapping):
            raise ValueError("dataset.monitor_split must be an object")
        unknown = set(monitor).difference({"enabled", "split", "name", "maximum_rows", "seed"})
        if unknown:
            raise ValueError(f"Unknown keys in dataset.monitor_split: {sorted(unknown)}")
        if bool(monitor.get("enabled", False)):
            if str(monitor.get("split", "test")) not in SPLIT_NAMES:
                raise ValueError(f"dataset.monitor_split.split must be one of {list(SPLIT_NAMES)}")
            name = str(monitor.get("name", "crossday"))
            if name in RESERVED_VALID_NAMES:
                raise ValueError(
                    f"dataset.monitor_split.name must not shadow a built-in evaluation set: {name!r}"
                )
            if int(monitor.get("maximum_rows", 0)) <= 0:
                raise ValueError("dataset.monitor_split.maximum_rows must be positive")
            if not isinstance(monitor.get("seed"), int):
                raise ValueError("dataset.monitor_split.seed must be an integer")


def effective_model_params(config: Mapping[str, Any], num_classes: int) -> dict[str, Any]:
    validate_training_config(config)
    if num_classes <= 1:
        raise ValueError("Multiclass LightGBM requires at least two classes")
    params = dict(config["model_params"])
    params["num_class"] = int(num_classes)
    return params


def validate_dataset_manifest(config: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    if not bool(config["dataset"].get("require_full_dataset_manifest", False)):
        return
    if manifest.get("sampling_mode") != "full":
        raise ValueError(
            "Production training requires sampling_mode='full'; sampled prepared data was supplied"
        )
    source_files = list(manifest.get("source_files", []))
    if not source_files:
        raise ValueError("Production manifest contains no source files")
    incomplete = []
    for item in source_files:
        physical = int(item["physical_rows"])
        planned = int(item["planned_sample_rows"])
        # "Full dataset" means every physical row was *read and classified*, not that every
        # row was written: the cross-day and open-set experiments deliberately discard rows
        # whose class or capture day is out of scope. Older manifests have no rows_read.
        read = int(item.get("rows_read", item["rows_processed"]))
        if physical != planned or physical != read:
            incomplete.append(str(item["path"]))
    if incomplete:
        raise ValueError(
            "Production manifest does not read every physical row: " + ", ".join(incomplete[:10])
        )
    written = sum(int(value) for value in manifest["split"]["sizes"].values())
    physical = sum(int(item["physical_rows"]) for item in source_files)
    split = manifest["split"]
    excluded = (
        int(split.get("rows_excluded_by_label_policy", 0))
        + int(split.get("rows_excluded_by_unassigned_split", 0))
    )
    if written + excluded != physical:
        raise ValueError(
            f"Production manifest row count mismatch: written={written}, excluded={excluded}, "
            f"physical={physical}"
        )


@dataclass
class DatasetBundle:
    train_dataset: Any
    validation_dataset: Any
    features: dict[str, Any]
    labels: dict[str, np.ndarray]
    feature_names: list[str]
    model_feature_names: list[str]
    label_mapping: dict[str, int]
    params: dict[str, Any]
    params_hash: str
    feature_schema_hash: str
    feature_selection: dict[str, Any]
    memory_estimate: dict[str, Any]
    # Optional third evaluation set. Under the cross-capture-day experiment this is a
    # stratified subsample of the held-out day, watched every round so the learning-curve
    # figure can show the in-distribution and out-of-day curves side by side.
    monitor_dataset: Any = None
    monitor_features: Any = None
    monitor_labels: np.ndarray | None = None
    monitor_name: str | None = None
    monitor_source_split: str | None = None
    monitor_indices: np.ndarray | None = None


class _ILocIndexer:
    def __init__(self, owner: "LazyParquetFeatures") -> None:
        self.owner = owner

    def __getitem__(self, index: Any) -> pd.DataFrame | pd.Series:
        return self.owner.read(index, as_frame=True)


class LazyParquetFeatures:
    """DataFrame-like, row-addressable view over prepared Parquet parts."""

    def __init__(
        self, prepared_dir: Path, parts: Sequence[Mapping[str, Any]], feature_names: Sequence[str],
        output_feature_names: Sequence[str] | None = None,
    ) -> None:
        if not parts:
            raise ValueError("Prepared split contains no Parquet parts")
        self.prepared_dir = prepared_dir
        self.parts = [dict(part) for part in parts]
        self.feature_names = list(feature_names)
        self.output_feature_names = list(output_feature_names or feature_names)
        if len(self.output_feature_names) != len(self.feature_names):
            raise ValueError("Input and output feature-name counts differ")
        self._lengths = np.asarray([int(part["rows"]) for part in self.parts], dtype=np.int64)
        self._offsets = np.concatenate(([0], np.cumsum(self._lengths)))
        self.iloc = _ILocIndexer(self)

    def __len__(self) -> int:
        return int(self._offsets[-1])

    @property
    def shape(self) -> tuple[int, int]:
        return len(self), len(self.feature_names)

    def _read_part(self, part_index: int) -> pd.DataFrame:
        frame = pd.read_parquet(
            self.prepared_dir / str(self.parts[part_index]["path"]), columns=self.feature_names
        )
        if list(frame.columns) != self.feature_names:
            raise AssertionError("Feature order does not match preprocessing.json")
        return frame.astype(np.float32)

    def read(self, index: Any, as_frame: bool = False) -> Any:
        scalar = isinstance(index, (int, np.integer))
        if scalar:
            positions = np.asarray([int(index)], dtype=np.int64)
        elif isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            positions = np.arange(start, stop, step, dtype=np.int64)
        else:
            positions = np.asarray(index, dtype=np.int64).reshape(-1)
            positions = np.where(positions < 0, positions + len(self), positions)
        if np.any((positions < 0) | (positions >= len(self))):
            raise IndexError("Parquet row index is out of range")
        if not len(positions):
            empty = pd.DataFrame(columns=self.output_feature_names, dtype=np.float32)
            return empty if as_frame else empty.to_numpy(dtype=np.float32)

        result = np.empty((len(positions), len(self.feature_names)), dtype=np.float32)
        part_indices = np.searchsorted(self._offsets[1:], positions, side="right")
        for part_index in np.unique(part_indices):
            output_positions = np.flatnonzero(part_indices == part_index)
            local_positions = positions[output_positions] - self._offsets[part_index]
            frame = self._read_part(int(part_index))
            result[output_positions] = frame.iloc[local_positions].to_numpy(dtype=np.float32, copy=False)
        if scalar:
            if as_frame:
                return pd.Series(result[0], index=self.output_feature_names)
            return result[0]
        if as_frame:
            return pd.DataFrame(result, columns=self.output_feature_names)
        return result


class RowSubsetFeatures:
    """A row-index view over :class:`LazyParquetFeatures`, for a monitoring subsample.

    Holding the subsample as indices rather than a materialised matrix keeps the resume
    path lazy: init scores are recomputed chunk by chunk from Parquet instead of pinning
    a multi-hundred-megabyte float32 array for the whole run.
    """

    def __init__(self, source: "LazyParquetFeatures", indices: np.ndarray) -> None:
        self.source = source
        self.indices = np.asarray(indices, dtype=np.int64)
        if len(self.indices) and (
            self.indices.min() < 0 or self.indices.max() >= len(source)
        ):
            raise IndexError("Monitoring subsample references rows outside the source split")
        self.feature_names = list(source.feature_names)
        self.output_feature_names = list(source.output_feature_names)
        self.iloc = _ILocIndexer(self)

    def __len__(self) -> int:
        return int(len(self.indices))

    @property
    def shape(self) -> tuple[int, int]:
        return len(self), len(self.feature_names)

    def read(self, index: Any, as_frame: bool = False) -> Any:
        if isinstance(index, slice):
            positions = self.indices[index]
        elif isinstance(index, (int, np.integer)):
            positions = self.indices[int(index)]
        else:
            positions = self.indices[np.asarray(index, dtype=np.int64).reshape(-1)]
        return self.source.read(positions, as_frame=as_frame)


def stratified_monitor_indices(
    labels: np.ndarray, maximum_rows: int, seed: int
) -> np.ndarray:
    """Deterministic, class-proportional row sample with at least one row per class.

    The held-out capture day is tens of millions of rows; scoring all of them every round
    would cost more than the boosting itself. A proportional sample keeps the curve's
    shape while bounding the per-round cost, and the minimum of one row per class stops a
    rare class from vanishing out of the monitored macro metrics.
    """
    total = int(len(labels))
    budget = int(maximum_rows)
    if budget <= 0 or total <= budget:
        return np.arange(total, dtype=np.int64)
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) > budget:
        raise ValueError(
            f"monitor_split.maximum_rows={budget} cannot cover {len(classes)} classes"
        )
    exact = counts.astype(np.float64) * (budget / float(total))
    quotas = np.maximum(1, np.floor(exact).astype(np.int64))
    quotas = np.minimum(quotas, counts)
    # Largest-remainder top-up, then trim from the largest classes if the minimum-one rule
    # pushed the total over budget.
    order = np.argsort(-(exact - np.floor(exact)), kind="stable")
    index = 0
    while quotas.sum() < budget and index < len(order) * 4:
        candidate = order[index % len(order)]
        if quotas[candidate] < counts[candidate]:
            quotas[candidate] += 1
        index += 1
    while quotas.sum() > budget:
        candidate = int(np.argmax(quotas))
        if quotas[candidate] <= 1:
            break
        quotas[candidate] -= 1
    rng = np.random.default_rng(int(seed))
    chosen: list[np.ndarray] = []
    for class_value, quota in zip(classes, quotas):
        members = np.flatnonzero(labels == class_value)
        if quota >= len(members):
            chosen.append(members)
            continue
        chosen.append(rng.choice(members, size=int(quota), replace=False))
    return np.sort(np.concatenate(chosen)).astype(np.int64)


def lightgbm_safe_feature_names(feature_names: Sequence[str]) -> list[str]:
    """Create stable, unique ASCII names accepted by LightGBM's JSON serializer."""
    safe = []
    for index, original in enumerate(feature_names):
        slug = re.sub(r"[^A-Za-z0-9_]+", "_", str(original)).strip("_") or "feature"
        safe.append(f"f{index:04d}_{slug[:96]}")
    if len(set(safe)) != len(safe):
        raise AssertionError("Generated LightGBM feature names are not unique")
    return safe


class ParquetRowGroupCache:
    """Small shared LRU for decoded Parquet row groups.

    Prepared Parquet remains the only on-disk feature representation.  The cache
    bounds decoded feature memory across *all* LightGBM Sequences, instead of
    retaining one full part per Sequence or expanding every part into an
    uncompressed NumPy file.
    """

    def __init__(self, feature_names: Sequence[str], max_entries: int) -> None:
        if max_entries <= 0:
            raise ValueError("Parquet row-group cache entry count must be positive")
        self.feature_names = list(feature_names)
        self.max_entries = int(max_entries)
        self.current_bytes = 0
        self.misses = 0
        self._entries: OrderedDict[tuple[Path, int], np.ndarray] = OrderedDict()
        self._reusable_buffer: np.ndarray | None = None

    @staticmethod
    def _trim_process_heap() -> None:
        """Return freed native pages to Linux instead of retaining them until exit."""
        if not sys.platform.startswith("linux"):
            return
        try:
            malloc_trim = ctypes.CDLL(None).malloc_trim
            malloc_trim.argtypes = [ctypes.c_size_t]
            malloc_trim.restype = ctypes.c_int
            malloc_trim(0)
        except (AttributeError, OSError):
            # Non-glibc Linux images do not expose malloc_trim(). Arrow's pool
            # release below remains the portable best-effort fallback.
            return

    @staticmethod
    def _release_arrow_memory() -> int:
        import pyarrow as pa

        gc.collect()
        pool = pa.default_memory_pool()
        pool.release_unused()
        ParquetRowGroupCache._trim_process_heap()
        return int(pool.bytes_allocated())

    def get(self, path: Path, row_group: int) -> np.ndarray:
        key = (path, int(row_group))
        cached = self._entries.pop(key, None)
        if cached is not None:
            self._entries[key] = cached
            return cached

        import pyarrow.parquet as pq

        # LightGBM requests random sample indices monotonically and then pushes
        # all slices monotonically. Evict before decoding the next row group so
        # the old NumPy matrix never overlaps the next Arrow table at peak.
        evicted_any = False
        while len(self._entries) >= self.max_entries:
            _, evicted = self._entries.popitem(last=False)
            self.current_bytes -= evicted.nbytes
            if self.max_entries == 1 and evicted.flags.c_contiguous:
                # LightGBM consumes each returned slice synchronously. Keeping
                # one backing allocation and overwriting it for the next row
                # group avoids severe allocator fragmentation after hundreds
                # of differently-sized Parquet row groups.
                reusable = evicted
                while isinstance(reusable.base, np.ndarray):
                    reusable = reusable.base
                self._reusable_buffer = reusable
            else:
                del evicted
            evicted_any = True
        if evicted_any:
            self._release_arrow_memory()

        parquet_file = pq.ParquetFile(path, memory_map=False, pre_buffer=False)
        try:
            table = parquet_file.read_row_group(
                row_group, columns=self.feature_names, use_threads=False
            )
        finally:
            # Do not leave a file-wide memory map or Arrow RandomAccessFile to
            # the garbage collector. On Kaggle this was retaining many GiB of
            # resident pages even though Arrow reported a zero-byte pool.
            parquet_file.close(force=True)

        required_shape = (table.num_rows, len(self.feature_names))
        reusable = self._reusable_buffer
        if reusable is not None and reusable.shape[0] >= table.num_rows:
            matrix = reusable[:table.num_rows]
            self._reusable_buffer = None
        else:
            self._reusable_buffer = None
            matrix = np.empty(required_shape, dtype=np.float32)
        for column_index in range(len(self.feature_names)):
            matrix[:, column_index] = np.asarray(
                table.column(column_index).to_numpy(zero_copy_only=False), dtype=np.float32
            )
        del table
        arrow_bytes = self._release_arrow_memory()
        self._entries[key] = matrix
        self.current_bytes += matrix.nbytes
        self.misses += 1
        if self.misses == 1 or self.misses % 100 == 0:
            import psutil

            LOGGER.info(
                "Parquet row-group reads=%d; NumPy cache=%.1f MiB; Arrow pool=%.1f MiB; RSS=%.1f MiB",
                self.misses,
                self.current_bytes / (1024 ** 2),
                arrow_bytes / (1024 ** 2),
                psutil.Process().memory_info().rss / (1024 ** 2),
            )
        return matrix


def _sequence_for_part(
    lgb: Any,
    prepared_dir: Path,
    part: Mapping[str, Any],
    feature_names: Sequence[str],
    batch_size: int,
    row_group_cache: ParquetRowGroupCache,
) -> Any:
    class ParquetPartSequence(lgb.Sequence):
        def __init__(self) -> None:
            import pyarrow.parquet as pq

            self.path = prepared_dir / str(part["path"])
            self.rows = int(part["rows"])
            self.columns = list(feature_names)
            self.batch_size = int(batch_size)
            parquet_file = pq.ParquetFile(self.path, memory_map=False, pre_buffer=False)
            try:
                metadata = parquet_file.metadata
                row_group_rows = [
                    metadata.row_group(i).num_rows for i in range(metadata.num_row_groups)
                ]
            finally:
                parquet_file.close(force=True)
            self._row_group_offsets = np.concatenate(
                ([0], np.cumsum(row_group_rows, dtype=np.int64))
            )
            if int(self._row_group_offsets[-1]) != self.rows:
                raise AssertionError(f"Prepared row count disagrees with Parquet metadata: {self.path}")

        def __len__(self) -> int:
            return self.rows

        def _row_group_for_index(self, index: int) -> int:
            return int(np.searchsorted(self._row_group_offsets, index, side="right") - 1)

        def _read_range(self, start: int, stop: int) -> np.ndarray:
            output = np.empty((stop - start, len(self.columns)), dtype=np.float32)
            output_offset = 0
            cursor = start
            while cursor < stop:
                row_group = self._row_group_for_index(cursor)
                group_start = int(self._row_group_offsets[row_group])
                group_stop = int(self._row_group_offsets[row_group + 1])
                source_stop = min(stop, group_stop)
                values = row_group_cache.get(self.path, row_group)
                count = source_stop - cursor
                output[output_offset:output_offset + count] = values[
                    cursor - group_start:source_stop - group_start
                ]
                cursor = source_stop
                output_offset += count
            return output

        def __getitem__(self, index: Any) -> np.ndarray:
            # LightGBM's random Sequence sampler requires double precision rows.
            # Sequential construction accepts float32, halving transient memory.
            if isinstance(index, slice):
                start, stop, step = index.indices(self.rows)
                if step != 1:
                    raise ValueError("LightGBM Parquet Sequence only supports contiguous slices")
                return self._read_range(start, stop)
            row = int(index)
            if row < 0:
                row += self.rows
            if row < 0 or row >= self.rows:
                raise IndexError(row)
            row_group = self._row_group_for_index(row)
            group_start = int(self._row_group_offsets[row_group])
            return np.array(
                row_group_cache.get(self.path, row_group)[row - group_start],
                dtype=np.float64,
                copy=True,
            )

    return ParquetPartSequence()


def _read_split_labels(prepared_dir: Path, parts: Sequence[Mapping[str, Any]]) -> np.ndarray:
    if not parts:
        raise ValueError("Prepared split contains no Parquet parts")
    labels = np.empty(sum(int(part["rows"]) for part in parts), dtype=np.int32)
    offset = 0
    for part in parts:
        values = pd.read_parquet(
            prepared_dir / str(part["path"]), columns=["_label"]
        )["_label"].to_numpy(dtype=np.int32, copy=False)
        expected_rows = int(part["rows"])
        if len(values) != expected_rows:
            raise AssertionError(f"Prepared label metadata mismatch: {part['path']}")
        labels[offset:offset + expected_rows] = values
        offset += expected_rows
    return labels


SUPPORTED_FEATURE_SELECTION = ("none", "train_gain_top_k", "validation_permutation_top_k")


def _select_by_validation_permutation(
    candidates: Sequence[str], config: Mapping[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    """Take the top k of a permutation ranking a previous run measured on validation.

    Gain, the other available ranking, is fitted on the training split and rewards
    high-cardinality columns; permutation measures what a trained model actually loses
    when a column is destroyed. It needs a trained model to exist first, so the ranking is
    produced by ``feature_ranking.py`` against a completed full-feature run and consumed
    here by the reduced-feature run that follows.

    The ranking file is refused unless it was scored on validation. Selecting on a
    test-derived ranking would fold the held-out split into the decision and quietly
    inflate every number reported on it afterwards.
    """
    # Imported lazily: feature_ranking imports this module.
    from feature_ranking import load_ranking

    settings = dict(config["dataset"].get("feature_screening") or {})
    ranking_file = settings.get("ranking_file")
    if not ranking_file:
        raise ValueError(
            "feature_selection='validation_permutation_top_k' requires "
            "dataset.feature_screening.ranking_file"
        )
    maximum_features = int(settings["maximum_features"])
    candidates = list(candidates)
    if not 1 <= maximum_features <= len(candidates):
        raise ValueError("feature_screening.maximum_features is outside the candidate feature range")

    payload = load_ranking(ranking_file)
    ranked = [str(item["feature"]) for item in payload["ranking"]]
    missing = [name for name in candidates if name not in set(ranked)]
    if missing:
        raise ValueError(
            "The feature ranking does not cover every candidate column, so a top-k cut would "
            f"be arbitrary. Missing: {missing[:10]}"
        )
    unknown = [name for name in ranked if name not in set(candidates)]
    if unknown:
        raise ValueError(
            "The feature ranking names columns this dataset does not have; it was produced "
            f"from a different preprocessing recipe. Unexpected: {unknown[:10]}"
        )
    chosen = set(ranked[:maximum_features])
    selected = [name for name in candidates if name in chosen]
    if len(selected) != maximum_features:
        raise AssertionError("Top-k selection did not preserve the requested feature count")
    LOGGER.info(
        "Selected %d/%d features from the validation permutation ranking of run %s",
        len(selected), len(candidates), payload.get("run_id"),
    )
    return selected, {
        "method": "validation_permutation_top_k",
        # The ranking never saw train or test: it is measured on validation only.
        "fit_split": payload["scored_split"],
        "ranking_file": str(ranking_file),
        "ranking_run_id": payload.get("run_id"),
        "ranking_feature_schema_hash": payload.get("feature_schema_hash"),
        "ranking_scored_by": payload.get("scored_by"),
        "ranking_rows_scored": payload.get("split_rows_scored"),
        "ranking_repeats": payload.get("repeats"),
        "ranking_seed": payload.get("seed"),
        "ranking_baseline_macro_f1": payload.get("baseline_macro_f1"),
        "features_within_measurement_noise": payload.get("features_within_noise"),
        "candidate_feature_count": len(candidates),
        "selected_feature_count": len(selected),
        "selected_features_in_model_order": selected,
        "ranking_by_permutation": payload["ranking"],
        "final_training_rows_discarded": 0,
    }


def select_model_features(
    lgb: Any,
    prepared: Path,
    train_parts: Sequence[Mapping[str, Any]],
    candidate_features: Sequence[str],
    train_labels: np.ndarray,
    params: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    method = str(config.get("feature_selection", "none"))
    candidates = list(candidate_features)
    if method == "none":
        return candidates, {
            "method": "none",
            "candidate_feature_count": len(candidates),
            "selected_feature_count": len(candidates),
            "selected_features_in_model_order": candidates,
        }
    if method == "validation_permutation_top_k":
        return _select_by_validation_permutation(candidates, config)
    if method != "train_gain_top_k":
        raise ValueError(f"Unsupported feature selection method: {method}")

    settings = dict(config["dataset"]["feature_screening"])
    maximum_features = int(settings["maximum_features"])
    requested_rows = int(settings["balanced_train_rows"])
    if not 1 <= maximum_features <= len(candidates):
        raise ValueError("feature_screening.maximum_features is outside the candidate feature range")
    if requested_rows <= 0:
        raise ValueError("feature_screening.balanced_train_rows must be positive")

    classes = np.unique(train_labels)
    rows_per_class = max(1, requested_rows // len(classes))
    rng = np.random.default_rng(int(settings["seed"]))
    sampled_indices = []
    sampled_class_counts: dict[str, int] = {}
    for class_index in classes:
        positions = np.flatnonzero(train_labels == class_index)
        retained = min(len(positions), rows_per_class)
        chosen = rng.choice(positions, size=retained, replace=False)
        sampled_indices.append(chosen)
        sampled_class_counts[str(int(class_index))] = int(retained)
    indices = np.concatenate(sampled_indices).astype(np.int64, copy=False)
    rng.shuffle(indices)

    LOGGER.info(
        "Screening %d candidate features on %d deterministic class-balanced train rows",
        len(candidates), len(indices),
    )
    source = LazyParquetFeatures(prepared, train_parts, candidates)
    sample_features = source.read(indices)
    sample_labels = train_labels[indices]
    screening_params = dict(params)
    screening_params.update({
        "learning_rate": float(settings["learning_rate"]),
        "metric": [],
        "verbosity": -1,
    })
    screening_dataset = lgb.Dataset(
        sample_features,
        label=sample_labels,
        feature_name=lightgbm_safe_feature_names(candidates),
        categorical_feature=[],
        params=screening_params,
        free_raw_data=True,
    )
    screening_booster = lgb.train(
        screening_params,
        screening_dataset,
        num_boost_round=int(settings["num_boost_round"]),
    )
    gains = np.asarray(screening_booster.feature_importance(importance_type="gain"), dtype=np.float64)
    ranking_indices = sorted(range(len(candidates)), key=lambda i: (-float(gains[i]), candidates[i]))
    selected_set = set(ranking_indices[:maximum_features])
    selected = [name for index, name in enumerate(candidates) if index in selected_set]
    ranking = [
        {"rank": rank, "feature": candidates[index], "gain": float(gains[index])}
        for rank, index in enumerate(ranking_indices, start=1)
    ]
    result = {
        "method": method,
        "fit_split": "train",
        "screening_sampling": "deterministic class-balanced sample without replacement",
        "screening_seed": int(settings["seed"]),
        "screening_rows_requested": requested_rows,
        "screening_rows_used": len(indices),
        "screening_class_counts": sampled_class_counts,
        "screening_num_boost_round": int(settings["num_boost_round"]),
        "screening_learning_rate": float(settings["learning_rate"]),
        "candidate_feature_count": len(candidates),
        "selected_feature_count": len(selected),
        "selected_features_in_model_order": selected,
        "ranking_by_gain": ranking,
        "final_training_rows_discarded": 0,
    }
    LOGGER.info("Selected %d/%d features for full-dataset training", len(selected), len(candidates))
    del screening_booster, screening_dataset, sample_features, sample_labels, source, indices
    gc.collect()
    ParquetRowGroupCache._trim_process_heap()
    return selected, result


def build_datasets(prepared_data_dir: str | Path, config: Mapping[str, Any]) -> DatasetBundle:
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("Install lightgbm>=4.0,<5 before training") from exc
    prepared = Path(prepared_data_dir)
    manifest = _read_json(prepared / "sample_manifest.json")
    validate_dataset_manifest(config, manifest)
    preprocessing = _read_json(prepared / "preprocessing.json")
    label_mapping = {str(key): int(value) for key, value in _read_json(prepared / "label_mapping.json").items()}
    profile = _read_json(prepared / "data_profile.json")
    candidate_feature_names = list(preprocessing["feature_columns_in_order"])
    if preprocessing.get("scaling") != "none" or preprocessing.get("imbalance_handling") != "none":
        raise ValueError("Prepared data violates the unscaled/unbalanced baseline contract")
    batch_size = int(config["dataset"].get("sequence_batch_rows", 8192))
    if batch_size <= 0:
        raise ValueError("dataset.sequence_batch_rows must be positive")
    row_group_cache_entries = int(
        config["dataset"].get("sequence_row_group_cache_entries", 1)
    )
    if row_group_cache_entries <= 0:
        raise ValueError("dataset.sequence_row_group_cache_entries must be positive")
    labels: dict[str, np.ndarray] = {}
    for split in SPLIT_NAMES:
        parts = manifest["parts"][split]
        labels[split] = _read_split_labels(prepared, parts)
    # Open-set rows carry OPEN_SET_LABEL_CODE: they have no output unit and must never
    # reach a split the model fits or is scored on during boosting.
    open_set_rows = {
        split: int(np.count_nonzero(labels[split] == OPEN_SET_LABEL_CODE)) for split in SPLIT_NAMES
    }
    fitted = [split for split in ("train", "validation") if open_set_rows[split]]
    if fitted:
        raise AssertionError(f"Open-set rows reached the {fitted} split(s); they belong in test only")
    observed_set: set[int] = set()
    for split in SPLIT_NAMES:
        observed_set.update(
            int(value) for value in np.unique(labels[split]) if int(value) != OPEN_SET_LABEL_CODE
        )
    observed = sorted(observed_set)
    expected = list(range(len(label_mapping)))
    if observed != expected:
        raise AssertionError(f"label_mapping.json indices {expected} do not match observed labels {observed}")
    params = effective_model_params(config, len(label_mapping))
    feature_names, feature_selection = select_model_features(
        lgb,
        prepared,
        manifest["parts"]["train"],
        candidate_feature_names,
        labels["train"],
        params,
        config,
    )
    model_feature_names = lightgbm_safe_feature_names(feature_names)

    monitor_cfg = dict(config["dataset"].get("monitor_split") or {})
    monitor_enabled = bool(monitor_cfg.get("enabled", False))
    monitor_source = str(monitor_cfg.get("split", "test"))
    monitor_name = str(monitor_cfg.get("name", "crossday"))
    monitor_budget = int(monitor_cfg.get("maximum_rows", 0))
    monitor_seed = int(monitor_cfg.get("seed", 2026))
    monitor_indices: np.ndarray | None = None
    if monitor_enabled:
        if open_set_rows[monitor_source] == int(manifest["split"]["sizes"][monitor_source]):
            raise ValueError(
                f"dataset.monitor_split cannot watch the {monitor_source!r} split: it holds only "
                "open-set rows, which have no trainable label to score against"
            )
        if open_set_rows[monitor_source]:
            raise ValueError(
                f"dataset.monitor_split cannot watch the {monitor_source!r} split while it mixes "
                f"{open_set_rows[monitor_source]} open-set rows with trainable ones"
            )
        monitor_indices = stratified_monitor_indices(
            labels[monitor_source], monitor_budget, monitor_seed
        )
        LOGGER.info(
            "Monitoring %d of %d %s rows every round as evaluation set %r",
            len(monitor_indices), len(labels[monitor_source]), monitor_source, monitor_name,
        )

    import psutil

    guard = dict(config["dataset"].get("memory_guard") or {})
    memory_estimate = estimate_training_memory(
        manifest["split"]["sizes"], len(feature_names), len(label_mapping), params,
        int(psutil.virtual_memory().available),
        float(guard.get("available_ram_fraction", 0.8)),
        monitor_rows=len(monitor_indices) if monitor_indices is not None else 0,
    )
    memory_estimate["raw_matrix_safe_to_materialize"] = bool(
        profile["safe_to_materialize_for_lightgbm"]
    )
    LOGGER.info(
        "Projected LightGBM peak %.1f GiB (steady %.1f, resume %.1f) against a %.1f GiB budget",
        memory_estimate["estimated_peak_bytes"] / 1024**3,
        memory_estimate["steady_state_bytes"] / 1024**3,
        memory_estimate["resume_peak_bytes"] / 1024**3,
        memory_estimate["budget_bytes"] / 1024**3,
    )
    if bool(config["dataset"].get("require_safe_memory_profile", False)) and not memory_estimate["fits_budget"]:
        raise InsufficientMemoryError(
            "dataset.require_safe_memory_profile is set and the projected LightGBM peak of "
            f"{memory_estimate['estimated_peak_bytes'] / 1024**3:.1f} GiB exceeds the "
            f"{memory_estimate['budget_bytes'] / 1024**3:.1f} GiB budget "
            f"({memory_estimate['available_bytes'] / 1024**3:.1f} GiB available x "
            f"{memory_estimate['available_fraction']}). Use a high-RAM runtime, lower "
            "dataset.feature_screening.maximum_features, or free memory and rerun."
        )

    row_group_cache = ParquetRowGroupCache(feature_names, max_entries=row_group_cache_entries)
    if not profile["safe_to_materialize_for_lightgbm"]:
        LOGGER.info(
            "Raw full-matrix materialization is unsafe; constructing LightGBM Datasets from "
            "Parquet Sequences with %d rows per batch, %d selected features, and a shared "
            "%d-row-group cache",
            batch_size, len(feature_names), row_group_cache_entries,
        )
    features: dict[str, LazyParquetFeatures] = {}
    sequences: dict[str, list[Any]] = {}
    for split in SPLIT_NAMES:
        parts = manifest["parts"][split]
        features[split] = LazyParquetFeatures(prepared, parts, feature_names, model_feature_names)
        if split != "test":
            sequences[split] = [
                _sequence_for_part(
                    lgb, prepared, part, feature_names, batch_size, row_group_cache
                )
                for part in parts
            ]
        if len(features[split]) != int(manifest["split"]["sizes"][split]):
            raise AssertionError(f"Prepared {split} row count disagrees with sample_manifest.json")
    if not bool(params["is_enable_sparse"]):
        train_rows = int(manifest["split"]["sizes"]["train"])
        validation_rows = int(manifest["split"]["sizes"]["validation"])
        dense_cells = (train_rows + validation_rows) * len(feature_names)
        LOGGER.info(
            "Forcing dense LightGBM bins for %d train+validation rows x %d features "
            "(%.2f billion cells); sparse native storage previously exceeded Kaggle RAM",
            train_rows + validation_rows,
            len(feature_names),
            dense_cells / 1_000_000_000,
        )
    free_raw = bool(config["dataset"].get("free_raw_data", True))
    train_dataset = lgb.Dataset(
        sequences["train"], label=labels["train"], feature_name=model_feature_names,
        categorical_feature=[], params=params, free_raw_data=free_raw,
    )
    validation_dataset = lgb.Dataset(
        sequences["validation"], label=labels["validation"], reference=train_dataset,
        feature_name=model_feature_names, categorical_feature=[], params=params,
        free_raw_data=free_raw,
    )
    monitor_dataset = None
    monitor_features = None
    monitor_labels = None
    if monitor_indices is not None:
        monitor_features = RowSubsetFeatures(features[monitor_source], monitor_indices)
        monitor_labels = np.asarray(labels[monitor_source])[monitor_indices]
        # Materialised once, densely, then handed to LightGBM for binning. Unlike the train
        # and validation Datasets this one is small enough by construction (bounded by
        # monitor_split.maximum_rows) that the Sequence path would only add overhead.
        monitor_matrix = features[monitor_source].read(monitor_indices)
        monitor_dataset = lgb.Dataset(
            monitor_matrix, label=monitor_labels, reference=train_dataset,
            feature_name=model_feature_names, categorical_feature=[], params=params,
            free_raw_data=free_raw,
        )
        del monitor_matrix
        gc.collect()

    schema_payload = {
        "feature_names": feature_names,
        "model_feature_names": model_feature_names,
        "feature_dtypes": {
            name: preprocessing["feature_dtypes"][name] for name in feature_names
        },
        "categorical_features": preprocessing["categorical_features"],
        "label_mapping": label_mapping,
    }
    return DatasetBundle(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        features=features,
        labels=labels,
        feature_names=feature_names,
        model_feature_names=model_feature_names,
        label_mapping=label_mapping,
        params=params,
        params_hash=canonical_hash(params),
        feature_schema_hash=canonical_hash(schema_payload),
        feature_selection=feature_selection,
        memory_estimate=memory_estimate,
        monitor_dataset=monitor_dataset,
        monitor_features=monitor_features,
        monitor_labels=monitor_labels,
        monitor_name=monitor_name if monitor_dataset is not None else None,
        monitor_source_split=monitor_source if monitor_dataset is not None else None,
        monitor_indices=monitor_indices,
    )


def multiclass_macro_metrics(
    num_classes: int,
) -> Callable[[np.ndarray, Any], list[tuple[str, float, bool]]]:
    """Per-round Macro-F1 and Macro-Recall for every evaluation set.

    Macro-Recall is balanced accuracy. On CIC-DDoS2019 the largest class is 28.5% of the
    rows, so plain accuracy is close to uninformative; reporting the balanced figure every
    round -- not only in the final table -- is what makes the learning curves readable.
    """
    labels_order = list(range(num_classes))

    def evaluate(predictions: np.ndarray, dataset: Any) -> list[tuple[str, float, bool]]:
        probabilities = np.asarray(predictions)
        labels = np.asarray(dataset.get_label(), dtype=np.int32)
        if probabilities.ndim == 1:
            probabilities = probabilities.reshape(num_classes, -1).T
        if probabilities.shape != (len(labels), num_classes):
            raise ValueError(
                f"Unexpected multiclass prediction shape {probabilities.shape}; "
                f"expected {(len(labels), num_classes)}"
            )
        predicted = np.argmax(probabilities, axis=1)
        return [
            (
                "macro_f1",
                float(f1_score(labels, predicted, labels=labels_order, average="macro", zero_division=0)),
                True,
            ),
            (
                "macro_recall",
                float(recall_score(labels, predicted, labels=labels_order, average="macro", zero_division=0)),
                True,
            ),
        ]

    return evaluate


def macro_f1_metric(num_classes: int) -> Callable[[np.ndarray, Any], tuple[str, float, bool]]:
    labels_order = list(range(num_classes))

    def evaluate(predictions: np.ndarray, dataset: Any) -> tuple[str, float, bool]:
        probabilities = np.asarray(predictions)
        labels = np.asarray(dataset.get_label(), dtype=np.int32)
        if probabilities.ndim == 1:
            probabilities = probabilities.reshape(num_classes, -1).T
        if probabilities.shape != (len(labels), num_classes):
            raise ValueError(
                f"Unexpected multiclass prediction shape {probabilities.shape}; "
                f"expected {(len(labels), num_classes)}"
            )
        predicted = np.argmax(probabilities, axis=1)
        score = f1_score(labels, predicted, labels=labels_order, average="macro", zero_division=0)
        return "macro_f1", float(score), True

    return evaluate


class IterationRecorder:
    """Append-only LightGBM callback with periodic durable checkpoints."""

    order = 50
    before_iteration = False

    def __init__(
        self,
        history: list[dict[str, Any]],
        session_id: str,
        target_iteration: int,
        learning_rate: float,
        checkpoint_interval: int,
        checkpoint_hook: Callable[[Any, list[dict[str, Any]], str], float],
        deadline_monotonic: float | None,
        max_rounds_this_session: int | None,
        session_start_iteration: int,
        maximum_session_hours: float,
        stop_before_minutes: float,
        environment: str = "local",
        monitor_name: str | None = None,
    ) -> None:
        self.history = history
        self.session_id = session_id
        self.environment = str(environment)
        self.target_iteration = int(target_iteration)
        self.learning_rate = float(learning_rate)
        self.checkpoint_interval = int(checkpoint_interval)
        self.checkpoint_hook = checkpoint_hook
        self.deadline_monotonic = deadline_monotonic
        self.max_rounds_this_session = max_rounds_this_session
        self.session_start_iteration = int(session_start_iteration)
        self.maximum_session_hours = float(maximum_session_hours)
        self.stop_before_minutes = float(stop_before_minutes)
        self.monitor_name = str(monitor_name) if monitor_name else None
        self.last_perf = time.perf_counter()
        self.last_timestamp = datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _metrics(evaluation_result_list: Sequence[Sequence[Any]]) -> dict[str, float]:
        return {
            f"{str(item[0])}_{str(item[1])}": float(item[2])
            for item in evaluation_result_list
        }

    def __call__(self, env: Any) -> None:
        now_perf = time.perf_counter()
        now_timestamp = datetime.now(timezone.utc).isoformat()
        current = int(env.model.current_iteration())
        expected = len(self.history) + 1
        if current != expected:
            raise RuntimeError(f"Non-contiguous LightGBM iteration: expected {expected}, observed {current}")
        metrics = self._metrics(env.evaluation_result_list)
        record = {
            "iteration": current,
            "session_id": self.session_id,
            "environment": self.environment,
            "timestamp_start": self.last_timestamp,
            "timestamp_end": now_timestamp,
            "learning_rate": self.learning_rate,
            "train_multi_logloss": metrics.get("train_multi_logloss"),
            "val_multi_logloss": metrics.get("validation_multi_logloss"),
            "train_multi_error": metrics.get("train_multi_error"),
            "val_multi_error": metrics.get("validation_multi_error"),
            "train_macro_f1": metrics.get("train_macro_f1"),
            "val_macro_f1": metrics.get("validation_macro_f1"),
            # Macro-recall is balanced accuracy; on a dataset whose largest class is 28.5%
            # of the rows it is the honest per-round companion to plain accuracy.
            "train_macro_recall": metrics.get("train_macro_recall"),
            "val_macro_recall": metrics.get("validation_macro_recall"),
            "iteration_seconds": now_perf - self.last_perf,
            "checkpoint_seconds": 0.0,
            "is_final_round": current == self.target_iteration,
        }
        required_metrics = [key for key, value in record.items() if key.startswith(("train_", "val_")) and value is None]
        if required_metrics:
            raise RuntimeError(f"LightGBM callback did not receive required metrics: {required_metrics}")
        if self.monitor_name:
            monitored = {
                f"monitor_{suffix}": metrics.get(f"{self.monitor_name}_{suffix}")
                for suffix in ("multi_logloss", "multi_error", "macro_f1", "macro_recall")
            }
            absent = sorted(key for key, value in monitored.items() if value is None)
            if absent:
                raise RuntimeError(
                    f"Monitoring set {self.monitor_name!r} produced no {absent}; check valid_names"
                )
            record.update(monitored)
            record["monitor_name"] = self.monitor_name
        self.history.append(record)
        self.last_perf = now_perf
        self.last_timestamp = now_timestamp

        time_limit = self.deadline_monotonic is not None and time.monotonic() >= self.deadline_monotonic
        round_limit = (
            self.max_rounds_this_session is not None
            and current - self.session_start_iteration >= int(self.max_rounds_this_session)
        )
        final = current == self.target_iteration
        scheduled = current % self.checkpoint_interval == 0
        should_pause = not final and (time_limit or round_limit)
        if scheduled or final or should_pause:
            status = "ready_for_report" if final else ("paused" if should_pause else "running")
            checkpoint_seconds = float(self.checkpoint_hook(env.model, self.history, status))
            record["checkpoint_seconds"] = checkpoint_seconds
            # Exclude checkpoint/S3 synchronization from the following boosting
            # iteration's duration and from accumulated model-training time.
            self.last_perf = time.perf_counter()
            self.last_timestamp = datetime.now(timezone.utc).isoformat()
            if current == self.checkpoint_interval:
                average_iteration = float(np.mean([item["iteration_seconds"] for item in self.history[:current]]))
                blocks = math.ceil(self.target_iteration / self.checkpoint_interval)
                estimated_total = average_iteration * self.target_iteration + checkpoint_seconds * blocks
                usable_session_seconds = max(1.0, self.maximum_session_hours * 3600 - self.stop_before_minutes * 60)
                estimated_sessions = max(1, math.ceil(estimated_total / usable_session_seconds))
                LOGGER.info(
                    "Round-10 timing: avg_iteration=%.3fs checkpoint_block=%.3fs estimated_total_100=%.1fs estimated_sessions=%d",
                    average_iteration, checkpoint_seconds, estimated_total, estimated_sessions,
                )
            if should_pause:
                raise TrainingPauseRequested(f"Session paused safely after iteration {current}")


def raw_init_score(booster: Any, features: Any, num_class: int, chunk_rows: int) -> np.ndarray:
    """Raw scores of the previous model over the train rows, in LightGBM's storage order.

    Computed in chunks so a resumed session never materializes the whole feature matrix,
    and returned class-major because that is how LightGBM stores multiclass init scores.
    """
    total = len(features)
    scores = np.empty((total, num_class), dtype=np.float64)
    for start in range(0, total, max(1, int(chunk_rows))):
        stop = min(start + max(1, int(chunk_rows)), total)
        chunk = booster.predict(features.iloc[start:stop], raw_score=True)
        scores[start:stop] = np.asarray(chunk, dtype=np.float64).reshape(stop - start, num_class)
        del chunk
        gc.collect()
    return scores.ravel(order="F")


def continue_training(
    lgb: Any,
    params: Mapping[str, Any],
    train_dataset: Any,
    valid_sets: Sequence[Any],
    valid_names: Sequence[str],
    num_boost_round: int,
    feval: Any,
    callbacks: Sequence[Any],
    num_class: int,
    train_features: Any,
    valid_features: Mapping[str, Any] | None = None,
    init_model_path: str | None = None,
    init_score_chunk_rows: int = 250000,
) -> Any:
    """Boost `num_boost_round` rounds, optionally continuing a saved Booster.

    `lightgbm.train(init_model=...)` cannot be used here. It attaches the previous model as
    a predictor before the Datasets are constructed, and LightGBM then calls
    `predictor.predict()` on each Dataset's raw data to derive init scores. Our raw data is
    a list of Parquet `Sequence` objects, which `predict()` cannot consume, so a resumed
    session dies with "Cannot convert data list to numpy array".

    This function performs the same steps LightGBM performs, in an order that works with
    Sequence data: it computes every init score itself in chunks, and attaches the previous
    model to the already-constructed Datasets so the Booster still merges the earlier trees.
    Init scores matter for both train and validation, because LightGBM accumulates each
    Dataset's scores from its init score plus the trees grown in this call only.
    """
    from lightgbm import callback as lgb_callback
    from lightgbm.basic import _InnerPredictor

    params = dict(params)
    valid_features = dict(valid_features or {})
    chunk_rows = int(init_score_chunk_rows)
    previous = None
    predictor = None
    init_iteration = 0
    if init_model_path is not None:
        previous = lgb.Booster(model_file=str(init_model_path))
        init_iteration = int(previous.current_iteration())
        predictor = _InnerPredictor.from_model_file(
            model_file=str(init_model_path), pred_parameter=params
        )

    train_dataset._update_params(params)
    if previous is not None:
        train_dataset.set_init_score(
            raw_init_score(previous, train_features, int(num_class), chunk_rows)
        )
    train_dataset.construct()

    # Build the validation Datasets while the train Dataset still has no predictor:
    # Dataset.set_reference copies the reference's predictor, which would drag the
    # validation split into the same predict-on-raw-data failure.
    train_data_name = "training"
    reduced_valid_sets, name_valid_sets = [], []
    for index, valid_data in enumerate(valid_sets):
        if valid_data is train_dataset:
            train_data_name = valid_names[index]
            continue
        name = valid_names[index]
        valid_data._update_params(params).set_reference(train_dataset).construct()
        if previous is not None:
            if name not in valid_features:
                raise ValueError(f"Resuming requires features for validation set {name!r}")
            valid_data.set_init_score(
                raw_init_score(previous, valid_features[name], int(num_class), chunk_rows)
            )
        reduced_valid_sets.append(valid_data)
        name_valid_sets.append(name)

    if predictor is not None:
        # Attach only after construction. Assigning through Dataset._set_predictor would
        # re-enter the predict-on-raw-data path this function exists to avoid; the Booster
        # reads this attribute solely to merge the previous trees, and Booster.add_valid
        # requires every validation Dataset to carry the same predictor object.
        train_dataset._predictor = predictor
        for valid_set in reduced_valid_sets:
            valid_set._predictor = predictor
    del previous
    gc.collect()

    booster = lgb.Booster(params=params, train_set=train_dataset)
    booster.set_train_data_name(train_data_name)
    for valid_set, name in zip(reduced_valid_sets, name_valid_sets):
        booster.add_valid(valid_set, name)
    booster.best_iteration = 0

    ordered = list(callbacks)
    for index, item in enumerate(ordered):
        item.__dict__.setdefault("order", index - len(ordered))
    before_iteration = sorted(
        (item for item in ordered if getattr(item, "before_iteration", False)),
        key=lambda item: item.order,
    )
    after_iteration = sorted(
        (item for item in ordered if not getattr(item, "before_iteration", False)),
        key=lambda item: item.order,
    )

    end_iteration = init_iteration + int(num_boost_round)
    for iteration in range(init_iteration, end_iteration):
        for item in before_iteration:
            item(lgb_callback.CallbackEnv(
                model=booster, params=params, iteration=iteration,
                begin_iteration=init_iteration, end_iteration=end_iteration,
                evaluation_result_list=None,
            ))
        booster.update()
        evaluation_result_list = list(booster.eval_train(feval)) + list(booster.eval_valid(feval))
        for item in after_iteration:
            item(lgb_callback.CallbackEnv(
                model=booster, params=params, iteration=iteration,
                begin_iteration=init_iteration, end_iteration=end_iteration,
                evaluation_result_list=evaluation_result_list,
            ))
    return booster
