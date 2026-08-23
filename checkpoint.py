"""Crash-safe LightGBM Booster checkpoints and verified atomic S3 sync."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


LOGGER = logging.getLogger(__name__)

# Prepared Parquet splits are keyed by data fingerprint, not by run, so a new run_id
# never re-runs preprocessing over the raw dataset.
DATASET_KEY_PREFIX = "datasets"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def worker_environment() -> str:
    """Name the execution environment so history and the watchdog can tell them apart."""
    if os.environ.get("GITHUB_ACTIONS", "").strip().casefold() == "true":
        return "github_actions"
    if os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU") is not None:
        return "colab"
    try:
        import google.colab  # noqa: F401
    except Exception:
        return "local"
    return "colab"


def worker_id() -> str:
    """Stable per-process identity used by the cooperative single-writer lock."""
    cached = os.environ.get("PIPELINE_WORKER_ID", "").strip()
    if cached:
        return cached
    generated = f"{worker_environment()}-{uuid.uuid4().hex[:12]}"
    os.environ["PIPELINE_WORKER_ID"] = generated
    return generated


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json_dump(payload: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def atomic_history_dump(history: Sequence[Mapping[str, Any]], json_path: Path, csv_path: Path) -> None:
    atomic_json_dump(list(history), json_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = csv_path.with_suffix(csv_path.suffix + ".tmp")
    fieldnames: list[str] = []
    for record in history:
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, csv_path)


def validate_history(history: Sequence[Mapping[str, Any]], current_iteration: int) -> None:
    iterations = [int(record["iteration"]) for record in history]
    expected = list(range(1, int(current_iteration) + 1))
    if iterations != expected:
        raise ValueError(
            f"history.json must contain continuous unique iterations 1..{current_iteration}; "
            f"observed head={iterations[:5]} tail={iterations[-5:]}"
        )


class S3Store:
    """S3 adapter using temporary upload, verification, copy, and verification."""

    def __init__(self, config: Mapping[str, Any], enabled_override: bool | None = None) -> None:
        configured = bool(config["enabled"])
        self.enabled = configured if enabled_override is None else bool(enabled_override)
        self.required = bool(config["upload_required"])
        self.max_retries = max(1, int(config["max_retries"]))
        self.retry_base_seconds = float(config["retry_base_seconds"])
        self.bucket = os.environ.get(str(config["bucket_env"]), "").strip()
        self.prefix = os.environ.get(str(config["prefix_env"]), "").strip().strip("/")
        region = os.environ.get(str(config["region_env"]), "").strip()
        if not region:
            region = os.environ.get(str(config["fallback_region_env"]), "").strip()
        self.region = region or None
        self._client: Any | None = None
        if self.enabled and (not self.bucket or not self.prefix):
            message = "S3 synchronization requires S3_BUCKET and S3_PREFIX"
            if self.required:
                raise RuntimeError(message)
            LOGGER.warning("%s; local checkpoints remain enabled", message)
            self.enabled = False

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                import boto3
            except ImportError as exc:
                raise RuntimeError("boto3 is required when S3 synchronization is enabled") from exc
            self._client = boto3.client("s3", region_name=self.region)
        return self._client

    def project_key(self, relative: str) -> str:
        relative = relative.lstrip("/")
        return f"{self.prefix}/{relative}" if self.prefix else relative

    def run_key(self, run_id: str, relative: str) -> str:
        return self.project_key(f"{run_id}/{relative.lstrip('/')}")

    def dataset_key(self, data_version: str, relative: str) -> str:
        """Prepared-dataset keys are shared by every run built from the same source data."""
        return self.project_key(f"{DATASET_KEY_PREFIX}/{data_version}/{relative.lstrip('/')}")

    def _retry(self, operation: str, callback: Any) -> Any:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return callback()
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    delay = self.retry_base_seconds * 2 ** (attempt - 1)
                    LOGGER.warning("S3 %s attempt %d/%d failed (%s); retrying in %.1fs", operation, attempt, self.max_retries, type(exc).__name__, delay)
                    time.sleep(delay)
        assert last_error is not None
        raise last_error

    @staticmethod
    def _not_found(exc: Exception) -> bool:
        response = getattr(exc, "response", {})
        return response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 404 or str(
            response.get("Error", {}).get("Code", "")
        ) in {"404", "NoSuchKey", "NotFound"}

    def object_exists(self, key: str) -> bool:
        if not self.enabled:
            return False
        try:
            self._retry("head_object", lambda: self.client.head_object(Bucket=self.bucket, Key=key))
            return True
        except Exception as exc:
            if self._not_found(exc):
                return False
            raise

    def upload_atomic(self, source: Path, final_key: str) -> bool:
        if not self.enabled:
            return False
        size = int(source.stat().st_size)
        checksum = sha256_file(source)
        temporary_key = f"{final_key}.tmp-{uuid.uuid4().hex}"
        try:
            try:
                self._retry(
                    "upload_file",
                    lambda: self.client.upload_file(
                        str(source), self.bucket, temporary_key,
                        ExtraArgs={"Metadata": {"sha256": checksum}},
                    ),
                )
                temporary_head = self._retry(
                    "head_temporary", lambda: self.client.head_object(Bucket=self.bucket, Key=temporary_key)
                )
                if int(temporary_head["ContentLength"]) != size or temporary_head.get("Metadata", {}).get("sha256") != checksum:
                    raise IOError(f"Temporary S3 verification failed for {temporary_key}")
                self._retry(
                    "copy_object",
                    lambda: self.client.copy_object(
                        Bucket=self.bucket,
                        Key=final_key,
                        CopySource={"Bucket": self.bucket, "Key": temporary_key},
                        MetadataDirective="COPY",
                    ),
                )
                final_head = self._retry(
                    "head_final", lambda: self.client.head_object(Bucket=self.bucket, Key=final_key)
                )
                if int(final_head["ContentLength"]) != size or final_head.get("Metadata", {}).get("sha256") != checksum:
                    raise IOError(f"Final S3 verification failed for {final_key}")
                return True
            except Exception as exc:
                if self.required:
                    raise
                LOGGER.warning("S3 upload failed; local artifact remains at %s: %s", source, type(exc).__name__)
                return False
        finally:
            try:
                self.client.delete_object(Bucket=self.bucket, Key=temporary_key)
            except Exception:
                LOGGER.warning("Could not delete temporary S3 key %s", temporary_key)

    def download_file(self, key: str, destination: Path, required: bool = False) -> bool:
        if not self.enabled:
            return False
        if not self.object_exists(key):
            if required:
                raise FileNotFoundError(f"Required S3 object not found: s3://{self.bucket}/{key}")
            return False
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".download")
        try:
            self._retry("download_file", lambda: self.client.download_file(self.bucket, key, str(temporary)))
            os.replace(temporary, destination)
            return True
        finally:
            if temporary.exists():
                temporary.unlink()

    def read_json(self, key: str) -> dict[str, Any] | None:
        if not self.enabled or not self.object_exists(key):
            return None
        response = self._retry("get_object", lambda: self.client.get_object(Bucket=self.bucket, Key=key))
        return json.loads(response["Body"].read().decode("utf-8"))


class CheckpointManager:
    RESUME_CONFIG_FILES = (
        "run_config.json", "model_params.json", "sample_manifest.json", "preprocessing.json",
        "data_profile.json", "label_mapping.json", "report_config.json",
        "feature_selection.json",
    )

    def __init__(
        self,
        output_root: str | Path,
        model_name: str,
        checkpoint_config: Mapping[str, Any],
        s3_config: Mapping[str, Any],
        s3_enabled_override: bool | None = None,
    ) -> None:
        self.output_root = Path(output_root)
        self.model_name = model_name
        self.checkpoint_config = dict(checkpoint_config)
        self.s3 = S3Store(s3_config, s3_enabled_override)
        self.active_pointer = self.output_root / "active_run.json"

    def resolve_run_id(self, requested: str | None = None) -> str:
        if requested:
            return requested
        remote = self.s3.read_json(self.s3.project_key("active_run.json")) if self.s3.enabled else None
        if remote and remote.get("status") in {"preparing", "running", "paused", "ready_for_report"}:
            return str(remote["run_id"])
        if self.active_pointer.exists():
            with self.active_pointer.open(encoding="utf-8") as handle:
                local = json.load(handle)
            if local.get("status") in {"preparing", "running", "paused", "ready_for_report"}:
                return str(local["run_id"])
        return f"{self.model_name}_{datetime.now().strftime('%Y%m%d-%H%M')}"

    def run_dir(self, run_id: str) -> Path:
        return self.output_root / run_id

    def set_run_status(
        self,
        run_id: str,
        status: str,
        current_iteration: int,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        pointer = {
            "run_id": run_id,
            "status": status,
            "current_iteration": int(current_iteration),
            "updated_at": utc_now(),
            "worker": worker_environment(),
            "worker_id": worker_id(),
        }
        if extra:
            pointer.update(dict(extra))
        atomic_json_dump(pointer, self.active_pointer)
        if self.s3.enabled:
            self.s3.upload_atomic(self.active_pointer, self.s3.project_key("active_run.json"))

    def claim_run(self, run_id: str, stale_minutes: float) -> tuple[bool, str]:
        """Cooperative single-writer lock shared by Colab and the GitHub Actions runner.

        Colab cannot be stopped from outside, so the runner must never train the same run
        concurrently. A worker may start only when the remote pointer is stale, owned by
        this worker, or not in an actively-training state.
        """
        if not self.s3.enabled or stale_minutes <= 0:
            return True, "S3 pointer disabled; local run proceeds"
        remote = self.s3.read_json(self.s3.project_key("active_run.json"))
        if not remote or str(remote.get("run_id")) != run_id:
            return True, "no active pointer for this run"
        if str(remote.get("worker_id", "")) == worker_id():
            return True, "pointer already owned by this worker"
        if str(remote.get("status")) not in {"preparing", "running"}:
            return True, f"pointer status is {remote.get('status')!r}"
        updated_at = parse_timestamp(remote.get("updated_at"))
        if updated_at is None:
            return True, "pointer has no usable timestamp"
        age_minutes = (datetime.now(timezone.utc) - updated_at).total_seconds() / 60.0
        if age_minutes >= stale_minutes:
            return True, f"pointer is stale by {age_minutes:.1f} minutes"
        return False, (
            f"worker {remote.get('worker_id')} on {remote.get('worker')} has held the run "
            f"for {age_minutes:.1f} minutes; refusing to train the same run twice"
        )

    def download_resume_state(self, run_id: str) -> bool:
        run_dir = self.run_dir(run_id)
        state_path = run_dir / "checkpoints" / "training_state.json"
        if self.s3.enabled and self.s3.object_exists(self.s3.run_key(run_id, "checkpoints/training_state.json")):
            self.s3.download_file(self.s3.run_key(run_id, "checkpoints/training_state.json"), state_path, required=True)
            with state_path.open(encoding="utf-8") as handle:
                state = json.load(handle)
            self.s3.download_file(self.s3.run_key(run_id, "checkpoints/last_model.txt"), run_dir / "checkpoints" / "last_model.txt", required=True)
            self.s3.download_file(self.s3.run_key(run_id, "metrics/history.json"), run_dir / "metrics" / "history.json", required=True)
            for name in self.RESUME_CONFIG_FILES:
                self.s3.download_file(
                    self.s3.run_key(run_id, f"config/{name}"),
                    run_dir / "config" / name,
                    required=True,
                )
            self._verify_downloads(run_dir, state)
            return True
        return state_path.exists()

    @staticmethod
    def _verify_downloads(run_dir: Path, state: Mapping[str, Any]) -> None:
        model_path = run_dir / "checkpoints" / "last_model.txt"
        history_path = run_dir / "metrics" / "history.json"
        if sha256_file(model_path) != state["model_sha256"]:
            raise IOError("Downloaded last_model.txt SHA-256 does not match training_state.json")
        if sha256_file(history_path) != state["history_sha256"]:
            raise IOError("Downloaded history.json SHA-256 does not match training_state.json")

    def load_state(self, run_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
        run_dir = self.run_dir(run_id)
        state_path = run_dir / "checkpoints" / "training_state.json"
        if not state_path.exists():
            return None
        with state_path.open(encoding="utf-8") as handle:
            state = json.load(handle)
        with (run_dir / "metrics" / "history.json").open(encoding="utf-8") as handle:
            history = json.load(handle)
        self._verify_downloads(run_dir, state)
        validate_history(history, int(state["current_iteration"]))
        return state, history

    def save_checkpoint(
        self,
        run_id: str,
        session_id: str,
        booster: Any,
        history: list[dict[str, Any]],
        params_hash: str,
        feature_schema_hash: str,
        target_iteration: int,
        status: str,
    ) -> tuple[dict[str, Any], float]:
        started = time.perf_counter()
        run_dir = self.run_dir(run_id)
        checkpoint_dir = run_dir / "checkpoints"
        metrics_dir = run_dir / "metrics"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        current_iteration = int(booster.current_iteration())
        validate_history(history, current_iteration)
        temporary_model = checkpoint_dir / "last_model.txt.tmp"
        booster.save_model(str(temporary_model), num_iteration=current_iteration)
        with temporary_model.open("rb+") as handle:
            os.fsync(handle.fileno())
        last_model = checkpoint_dir / "last_model.txt"
        os.replace(temporary_model, last_model)
        history_json = metrics_dir / "history.json"
        history_csv = metrics_dir / "history.csv"
        atomic_history_dump(history, history_json, history_csv)
        state = {
            "state_version": 1,
            "model_type": "lightgbm.Booster",
            "model": "last_model.txt",
            "optimizer": None,
            "scheduler": None,
            "run_id": run_id,
            "session_id": session_id,
            "current_iteration": current_iteration,
            "target_iteration": int(target_iteration),
            "params_hash": params_hash,
            "feature_schema_hash": feature_schema_hash,
            "model_sha256": sha256_file(last_model),
            "history_sha256": sha256_file(history_json),
            "model_size_bytes": int(last_model.stat().st_size),
            "status": status,
            "saved_at": utc_now(),
        }
        state_path = checkpoint_dir / "training_state.json"
        atomic_json_dump(state, state_path)
        if self.checkpoint_config.get("save_immutable_round_files", True):
            immutable = checkpoint_dir / f"model_round_{current_iteration:03d}.txt"
            shutil.copyfile(last_model, immutable)

        if self.s3.enabled:
            self.s3.upload_atomic(last_model, self.s3.run_key(run_id, "checkpoints/last_model.txt"))
            self.s3.upload_atomic(history_json, self.s3.run_key(run_id, "metrics/history.json"))
            self.s3.upload_atomic(history_csv, self.s3.run_key(run_id, "metrics/history.csv"))
            self.s3.upload_atomic(state_path, self.s3.run_key(run_id, "checkpoints/training_state.json"))
        self._prune_round_files(checkpoint_dir)
        return state, time.perf_counter() - started

    def update_history_after_checkpoint(
        self, run_id: str, state: dict[str, Any], history: list[dict[str, Any]]
    ) -> None:
        run_dir = self.run_dir(run_id)
        history_json = run_dir / "metrics" / "history.json"
        history_csv = run_dir / "metrics" / "history.csv"
        atomic_history_dump(history, history_json, history_csv)
        state["history_sha256"] = sha256_file(history_json)
        state["saved_at"] = utc_now()
        state_path = run_dir / "checkpoints" / "training_state.json"
        atomic_json_dump(state, state_path)
        if self.s3.enabled:
            self.s3.upload_atomic(history_json, self.s3.run_key(run_id, "metrics/history.json"))
            self.s3.upload_atomic(history_csv, self.s3.run_key(run_id, "metrics/history.csv"))
            self.s3.upload_atomic(state_path, self.s3.run_key(run_id, "checkpoints/training_state.json"))

    def save_final_model(self, run_id: str, booster: Any, target_iteration: int) -> Path:
        if int(booster.current_iteration()) != int(target_iteration):
            raise ValueError("Final model may only be saved at the configured target iteration")
        destination = self.run_dir(run_id) / "checkpoints" / f"final_model_round_{target_iteration}.txt"
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        booster.save_model(str(temporary), num_iteration=target_iteration)
        os.replace(temporary, destination)
        if self.s3.enabled:
            self.s3.upload_atomic(destination, self.s3.run_key(run_id, f"checkpoints/{destination.name}"))
        return destination

    def sync_artifact(self, run_id: str, source: Path, category: str) -> None:
        if self.s3.enabled:
            self.s3.upload_atomic(source, self.s3.run_key(run_id, f"{category}/{source.name}"))

    def _prune_round_files(self, directory: Path) -> None:
        keep = max(0, int(self.checkpoint_config["local_keep_round_files"]))
        files = sorted(directory.glob("model_round_*.txt"))
        for old in files[:-keep] if keep else files:
            old.unlink()
