import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from checkpoint import (  # noqa: E402
    CheckpointManager,
    S3Store,
    canonical_hash,
    sha256_file,
    validate_history,
    worker_environment,
    worker_id,
)
from model import (  # noqa: E402
    IterationRecorder,
    TrainingPauseRequested,
    estimate_training_memory,
    macro_f1_metric,
    validate_dataset_manifest,
    validate_training_config,
)
from train import load_train_config, remaining_rounds, validate_resume_state  # noqa: E402


class FakeBooster:
    def __init__(self, iteration: int = 0) -> None:
        self.iteration = iteration

    def current_iteration(self) -> int:
        return self.iteration

    def save_model(self, destination: str, num_iteration: int) -> None:
        Path(destination).write_text(
            json.dumps({"iteration": self.iteration, "num_iteration": num_iteration}),
            encoding="utf-8",
        )


class ContractAndMetricTest(unittest.TestCase):
    def test_fixed_baseline_contract(self) -> None:
        config = load_train_config(PROJECT_ROOT / "config" / "train.json")
        validate_training_config(config)
        self.assertFalse(config["model_params"]["is_enable_sparse"])
        config["model_params"]["is_enable_sparse"] = True
        with self.assertRaisesRegex(ValueError, "is_enable_sparse"):
            validate_training_config(config)
        config["model_params"]["is_enable_sparse"] = False
        config["num_boost_round"] = 99
        with self.assertRaisesRegex(ValueError, "exactly 100"):
            validate_training_config(config)

    def test_every_shipped_train_profile_satisfies_the_contract(self) -> None:
        for name in ("train.json", "train.gha.json", "train.smoke.json"):
            with self.subTest(profile=name):
                config = load_train_config(PROJECT_ROOT / "config" / name)
                validate_training_config(config)
                if config["feature_selection"] == "train_gain_top_k":
                    screening = config["dataset"]["feature_screening"]
                    self.assertEqual(screening["seed"], config["seed"], name)

    def test_production_manifest_must_prove_every_physical_row_was_used(self) -> None:
        config = load_train_config(PROJECT_ROOT / "config" / "train.json")
        full = {
            "sampling_mode": "full",
            "source_files": [
                {
                    "path": "a.parquet", "physical_rows": 60,
                    "planned_sample_rows": 60, "rows_processed": 60,
                },
                {
                    "path": "b.parquet", "physical_rows": 40,
                    "planned_sample_rows": 40, "rows_processed": 40,
                },
            ],
            "split": {"sizes": {"train": 70, "validation": 15, "test": 15}},
        }
        validate_dataset_manifest(config, full)
        sampled = dict(full, sampling_mode="deterministic_proportional_exact_total")
        with self.assertRaisesRegex(ValueError, "sampling_mode='full'"):
            validate_dataset_manifest(config, sampled)

    def test_macro_f1_accepts_matrix_and_flat_class_major_predictions(self) -> None:
        labels = np.array([0, 1, 2, 1], dtype=np.int32)
        probabilities = np.array(
            [[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.2, 0.7, 0.1]],
            dtype=np.float64,
        )
        dataset = SimpleNamespace(get_label=lambda: labels)
        metric = macro_f1_metric(3)
        self.assertEqual(metric(probabilities, dataset), ("macro_f1", 1.0, True))
        name, score, higher = metric(probabilities.T.reshape(-1), dataset)
        self.assertEqual(name, "macro_f1")
        self.assertEqual(score, 1.0)
        self.assertTrue(higher)

    def test_iteration_callback_pauses_only_after_durable_round_10_checkpoint(self) -> None:
        history = []
        checkpoint_calls = []

        def checkpoint_hook(booster, records, status):
            checkpoint_calls.append((booster.current_iteration(), len(records), status))
            return 0.25

        recorder = IterationRecorder(
            history=history,
            session_id="session_test",
            target_iteration=100,
            learning_rate=0.05,
            checkpoint_interval=10,
            checkpoint_hook=checkpoint_hook,
            deadline_monotonic=None,
            max_rounds_this_session=10,
            session_start_iteration=0,
            maximum_session_hours=12,
            stop_before_minutes=20,
        )
        booster = FakeBooster()
        with self.assertRaises(TrainingPauseRequested):
            for iteration in range(1, 11):
                booster.iteration = iteration
                values = [
                    ("train", "multi_logloss", 1.0 / iteration, False),
                    ("validation", "multi_logloss", 1.1 / iteration, False),
                    ("train", "multi_error", 0.3, False),
                    ("validation", "multi_error", 0.4, False),
                    ("train", "macro_f1", 0.7, True),
                    ("validation", "macro_f1", 0.6, True),
                    ("train", "macro_recall", 0.72, True),
                    ("validation", "macro_recall", 0.62, True),
                ]
                recorder(SimpleNamespace(model=booster, evaluation_result_list=values))
        self.assertEqual([item["iteration"] for item in history], list(range(1, 11)))
        self.assertEqual(checkpoint_calls, [(10, 10, "paused")])
        self.assertEqual(history[-1]["checkpoint_seconds"], 0.25)


class CheckpointTest(unittest.TestCase):
    def _configs(self):
        train_config = load_train_config(PROJECT_ROOT / "config" / "train.smoke.json")
        return train_config["checkpoint"], train_config["s3"]

    def test_local_booster_state_history_round_trip_and_hash_validation(self) -> None:
        checkpoint_config, s3_config = self._configs()
        with tempfile.TemporaryDirectory() as temporary:
            manager = CheckpointManager(temporary, "lightgbm", checkpoint_config, s3_config)
            run_id = "lightgbm_20260809-1200"
            history = [
                {"iteration": iteration, "session_id": "one", "iteration_seconds": 0.1}
                for iteration in range(1, 11)
            ]
            booster = FakeBooster(10)
            state, _ = manager.save_checkpoint(
                run_id, "one", booster, history, "params-hash", "schema-hash", 100, "paused"
            )
            loaded_state, loaded_history = manager.load_state(run_id)
            self.assertEqual(loaded_state["current_iteration"], 10)
            self.assertEqual(loaded_state["model"], "last_model.txt")
            self.assertIsNone(loaded_state["optimizer"])
            self.assertIsNone(loaded_state["scheduler"])
            self.assertEqual(len(loaded_history), 10)
            self.assertEqual(state["target_iteration"], 100)
            self.assertTrue((manager.run_dir(run_id) / "checkpoints" / "last_model.txt").exists())
            validate_history(loaded_history, 10)
            self.assertEqual(remaining_rounds(10, 100), 90)
            self.assertEqual(
                validate_resume_state(loaded_state, run_id, "params-hash", "schema-hash", 100), 10
            )
            with self.assertRaisesRegex(ValueError, "incompatible"):
                validate_resume_state(loaded_state, run_id, "changed", "schema-hash", 100)

    def test_resume_downloads_configuration_required_for_final_report(self) -> None:
        checkpoint_config, s3_config = self._configs()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "source-model.txt"
            history = root / "source-history.json"
            model.write_text("model", encoding="utf-8")
            history.write_text('[{"iteration": 1}]', encoding="utf-8")
            state = {
                "current_iteration": 1,
                "model_sha256": sha256_file(model),
                "history_sha256": sha256_file(history),
            }
            remote = {
                "checkpoints/training_state.json": json.dumps(state).encode("utf-8"),
                "checkpoints/last_model.txt": model.read_bytes(),
                "metrics/history.json": history.read_bytes(),
            }
            for name in CheckpointManager.RESUME_CONFIG_FILES:
                remote[f"config/{name}"] = json.dumps({"artifact": name}).encode("utf-8")

            class FakeStore:
                enabled = True

                @staticmethod
                def run_key(run_id, relative):
                    return relative

                @staticmethod
                def object_exists(key):
                    return key in remote

                @staticmethod
                def download_file(key, destination, required=False):
                    if key not in remote:
                        if required:
                            raise FileNotFoundError(key)
                        return False
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(remote[key])
                    return True

            manager = CheckpointManager(root / "runs", "lightgbm", checkpoint_config, s3_config)
            manager.s3 = FakeStore()
            self.assertTrue(manager.download_resume_state("run"))
            for name in CheckpointManager.RESUME_CONFIG_FILES:
                self.assertTrue((manager.run_dir("run") / "config" / name).exists())

    def test_s3_upload_verifies_temporary_and_final_objects(self) -> None:
        class FakeS3Client:
            def __init__(self):
                self.objects = {}
                self.calls = []

            def upload_file(self, source, bucket, key, ExtraArgs=None):
                self.calls.append(("upload", key))
                self.objects[key] = {
                    "body": Path(source).read_bytes(),
                    "metadata": dict((ExtraArgs or {}).get("Metadata", {})),
                }

            def head_object(self, Bucket, Key):
                self.calls.append(("head", Key))
                item = self.objects[Key]
                return {"ContentLength": len(item["body"]), "Metadata": item["metadata"]}

            def copy_object(self, Bucket, Key, CopySource, MetadataDirective):
                self.calls.append(("copy", Key))
                source = self.objects[CopySource["Key"]]
                self.objects[Key] = {"body": source["body"], "metadata": dict(source["metadata"])}

            def delete_object(self, Bucket, Key):
                self.calls.append(("delete", Key))
                self.objects.pop(Key, None)

        old_bucket, old_prefix = os.environ.get("S3_BUCKET"), os.environ.get("S3_PREFIX")
        try:
            os.environ["S3_BUCKET"] = "test-bucket"
            os.environ["S3_PREFIX"] = "test-prefix"
            _, s3_config = self._configs()
            s3_config = dict(s3_config)
            s3_config.update({"enabled": True, "upload_required": True})
            store = S3Store(s3_config)
            fake = FakeS3Client()
            store._client = fake
            with tempfile.TemporaryDirectory() as temporary:
                source = Path(temporary) / "last_model.txt"
                source.write_bytes(b"verified-lightgbm-model")
                self.assertTrue(store.upload_atomic(source, "run/checkpoints/last_model.txt"))
            final = fake.objects["run/checkpoints/last_model.txt"]
            self.assertEqual(final["body"], b"verified-lightgbm-model")
            self.assertEqual(len(final["metadata"]["sha256"]), 64)
            self.assertEqual(
                [name for name, _ in fake.calls],
                ["upload", "head", "copy", "head", "delete"],
            )
            self.assertFalse(any(".tmp-" in key for key in fake.objects))
        finally:
            if old_bucket is None:
                os.environ.pop("S3_BUCKET", None)
            else:
                os.environ["S3_BUCKET"] = old_bucket
            if old_prefix is None:
                os.environ.pop("S3_PREFIX", None)
            else:
                os.environ["S3_PREFIX"] = old_prefix

    def _lock_manager(self, temporary, pointer):
        """Build a manager whose S3 layer only serves the shared active_run.json pointer."""
        checkpoint_config, s3_config = self._configs()
        s3_config = dict(s3_config)
        s3_config.update({"enabled": True, "upload_required": False})
        manager = CheckpointManager(temporary, "lightgbm", checkpoint_config, s3_config)
        manager.s3.enabled = True
        manager.s3.bucket, manager.s3.prefix = "bucket", "prefix"
        manager.s3.read_json = lambda key: pointer
        return manager

    def test_worker_refuses_to_train_a_run_another_live_worker_holds(self) -> None:
        pointer = {
            "run_id": "lightgbm_20260822-1200",
            "status": "running",
            "worker": "colab",
            "worker_id": "colab-abcdef123456",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        with tempfile.TemporaryDirectory() as temporary:
            manager = self._lock_manager(temporary, pointer)
            claimed, reason = manager.claim_run("lightgbm_20260822-1200", 30.0)
        self.assertFalse(claimed)
        self.assertIn("colab-abcdef123456", reason)

    def test_worker_claims_a_run_whose_heartbeat_went_stale(self) -> None:
        pointer = {
            "run_id": "lightgbm_20260822-1200",
            "status": "running",
            "worker": "colab",
            "worker_id": "colab-abcdef123456",
            "updated_at": (datetime.now(timezone.utc) - timedelta(minutes=90)).isoformat(),
        }
        with tempfile.TemporaryDirectory() as temporary:
            manager = self._lock_manager(temporary, pointer)
            claimed, reason = manager.claim_run("lightgbm_20260822-1200", 30.0)
        self.assertTrue(claimed)
        self.assertIn("stale", reason)

    def test_paused_run_is_free_for_any_worker(self) -> None:
        pointer = {
            "run_id": "lightgbm_20260822-1200",
            "status": "paused",
            "worker": "colab",
            "worker_id": "colab-abcdef123456",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        with tempfile.TemporaryDirectory() as temporary:
            manager = self._lock_manager(temporary, pointer)
            claimed, _ = manager.claim_run("lightgbm_20260822-1200", 30.0)
        self.assertTrue(claimed)

    def test_run_status_pointer_names_the_worker_that_wrote_it(self) -> None:
        checkpoint_config, s3_config = self._configs()
        with tempfile.TemporaryDirectory() as temporary:
            manager = CheckpointManager(temporary, "lightgbm", checkpoint_config, s3_config)
            manager.set_run_status("lightgbm_20260822-1200", "running", 20)
            pointer = json.loads(manager.active_pointer.read_text(encoding="utf-8"))
        self.assertEqual(pointer["worker"], worker_environment())
        self.assertEqual(pointer["worker_id"], worker_id())
        self.assertEqual(pointer["current_iteration"], 20)

    def test_canonical_hash_is_order_independent_for_mappings(self) -> None:
        self.assertEqual(canonical_hash({"a": 1, "b": 2}), canonical_hash({"b": 2, "a": 1}))


if __name__ == "__main__":
    unittest.main()



class MemoryGuardTest(unittest.TestCase):
    """dataset.require_safe_memory_profile was configured everywhere but read nowhere."""

    PARAMS = {"max_bin": 255, "num_threads": 8}

    def _estimate(self, train_rows, available_gb, features=20, classes=13):
        return estimate_training_memory(
            {"train": train_rows, "validation": train_rows // 5, "test": train_rows // 5},
            features, classes, self.PARAMS, int(available_gb * 1024**3), 0.8,
        )

    def test_projection_accounts_for_the_buffers_the_raw_profile_ignores(self):
        estimate = self._estimate(49_000_000, available_gb=51)
        # The float64 prediction buffers the Python layer keeps for every validation set
        # dominate the binned dataset, which is the whole reason the raw profile misleads.
        self.assertGreater(
            estimate["python_prediction_buffer_bytes"], estimate["binned_dataset_bytes"]
        )
        self.assertGreaterEqual(estimate["estimated_peak_bytes"], estimate["steady_state_bytes"])
        self.assertGreaterEqual(estimate["estimated_peak_bytes"], estimate["resume_peak_bytes"])

    def test_a_high_ram_runtime_fits_and_a_small_runner_does_not(self):
        self.assertTrue(self._estimate(49_000_000, available_gb=51)["fits_budget"])
        self.assertFalse(self._estimate(49_000_000, available_gb=16)["fits_budget"])

    def test_resume_peak_is_reported_separately_from_steady_state(self):
        estimate = self._estimate(49_000_000, available_gb=51)
        self.assertGreater(estimate["resume_init_score_bytes"], 0)
        self.assertNotEqual(estimate["resume_peak_bytes"], estimate["steady_state_bytes"])

    def test_more_classes_cost_more_memory(self):
        thirteen = self._estimate(10_000_000, available_gb=51, classes=13)
        twenty = self._estimate(10_000_000, available_gb=51, classes=20)
        self.assertGreater(twenty["estimated_peak_bytes"], thirteen["estimated_peak_bytes"])
