import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model import (  # noqa: E402
    InsufficientMemoryError,
    IterationRecorder,
    ParquetRowGroupCache,
    TrainingPauseRequested,
    build_datasets,
    continue_training,
    macro_f1_metric,
    raw_init_score,
)


class ArrayFeatures:
    """Row-sliceable stand-in for LazyParquetFeatures used by the resume tests."""

    def __init__(self, data: np.ndarray) -> None:
        self.data = data
        self.iloc = self

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self) -> int:
        return len(self.data)



SEQUENCE_FIXTURE_SIZES = {"train": 37, "validation": 13, "test": 11}


def write_prepared_fixture(prepared: Path) -> dict[str, int]:
    """Write the smallest prepared dataset that satisfies the production train contract."""
    parts = {name: [] for name in ("train", "validation", "test")}
    sizes = dict(SEQUENCE_FIXTURE_SIZES)
    for split, rows in sizes.items():
        split_dir = prepared / "splits" / split
        split_dir.mkdir(parents=True)
        for number, (start, stop) in enumerate(((0, rows // 2), (rows // 2, rows))):
            count = stop - start
            frame = pd.DataFrame({
                "f[0]": np.arange(start, stop, dtype=np.float32),
                "f\"1:rate": np.linspace(0, 1, count, dtype=np.float32),
                "_sample_file_id": np.zeros(count, dtype=np.uint64),
                "_sample_row_id": np.arange(start, stop, dtype=np.uint64),
                "_label": np.arange(start, stop, dtype=np.int32) % 3,
            })
            path = split_dir / f"part-{number:06d}.parquet"
            frame.to_parquet(path, index=False)
            parts[split].append({
                "path": path.relative_to(prepared).as_posix(), "rows": count, "bytes": path.stat().st_size
            })
    artifacts = {
        "sample_manifest.json": {
            "parts": parts,
            "split": {"sizes": sizes},
            # config/train.json requires a full-dataset manifest, so the fixture
            # must declare that every physical row was processed.
            "sampling_mode": "full",
            "source_files": [{
                "path": "source.parquet",
                "physical_rows": sum(sizes.values()),
                "planned_sample_rows": sum(sizes.values()),
                "rows_processed": sum(sizes.values()),
            }],
        },
        "preprocessing.json": {
            "feature_columns_in_order": ["f[0]", "f\"1:rate"],
            "feature_dtypes": {"f[0]": "float32", "f\"1:rate": "float32"},
            "categorical_features": [], "scaling": "none", "imbalance_handling": "none",
        },
        "label_mapping.json": {"a": 0, "b": 1, "c": 2},
        "data_profile.json": {"safe_to_materialize_for_lightgbm": False},
    }
    for name, payload in artifacts.items():
        (prepared / name).write_text(json.dumps(payload), encoding="utf-8")
    return sizes


def prepared_fixture_config() -> dict:
    config = json.loads((PROJECT_ROOT / "config" / "train.json").read_text(encoding="utf-8"))
    config["dataset"]["sequence_batch_rows"] = 7
    config["dataset"]["sequence_row_group_cache_entries"] = 1
    # The fixture only has two candidate features; screening cannot ask for more.
    config["dataset"]["feature_screening"]["maximum_features"] = 2
    config["dataset"]["feature_screening"]["balanced_train_rows"] = 37
    return config


class LightGBMResumeIntegrationTest(unittest.TestCase):
    def test_row_group_cache_evicts_before_decoding_the_next_group(self) -> None:
        try:
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("PyArrow is required")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "many-row-groups.parquet"
            rows = 4096
            frame = pd.DataFrame({
                "a": np.arange(rows, dtype=np.float32),
                "b": np.linspace(0, 1, rows, dtype=np.float32),
            })
            frame.to_parquet(path, index=False, row_group_size=128)
            parquet = pq.ParquetFile(path)
            self.assertGreater(parquet.num_row_groups, 20)
            cache = ParquetRowGroupCache(["a", "b"], max_entries=1)
            for row_group in range(parquet.num_row_groups):
                values = cache.get(path, row_group)
                self.assertEqual(values.shape[1], 2)
                self.assertEqual(len(cache._entries), 1)
                self.assertEqual(cache.current_bytes, values.nbytes)
            del parquet

    def test_single_entry_cache_reuses_one_numpy_allocation(self) -> None:
        try:
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("PyArrow is required")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "equal-row-groups.parquet"
            frame = pd.DataFrame({
                "a": np.arange(1024, dtype=np.float32),
                "b": np.linspace(0, 1, 1024, dtype=np.float32),
            })
            frame.to_parquet(path, index=False, row_group_size=128)
            cache = ParquetRowGroupCache(["a", "b"], max_entries=1)
            allocation_addresses = []
            parquet = pq.ParquetFile(path)
            row_groups = parquet.num_row_groups
            parquet.close(force=True)
            for row_group in range(row_groups):
                values = cache.get(path, row_group)
                allocation_addresses.append(values.__array_interface__["data"][0])
            self.assertEqual(len(set(allocation_addresses)), 1)
            self.assertEqual(cache.misses, 8)

    def test_parquet_sequence_uses_bounded_memory_cache_and_does_not_build_test_dataset(self) -> None:
        try:
            import lightgbm  # noqa: F401
            import pyarrow  # noqa: F401
        except ImportError:
            self.skipTest("LightGBM and PyArrow are required")
        with tempfile.TemporaryDirectory() as temporary:
            prepared = Path(temporary)
            sizes = write_prepared_fixture(prepared)
            config = prepared_fixture_config()
            bundle = build_datasets(prepared, config)
            self.assertFalse(bundle.params["is_enable_sparse"])
            self.assertFalse(bundle.train_dataset.get_params()["is_enable_sparse"])
            booster = lightgbm.train(
                bundle.params, bundle.train_dataset, num_boost_round=2,
                valid_sets=[bundle.validation_dataset], valid_names=["validation"],
            )
            self.assertEqual(bundle.train_dataset.num_data(), sizes["train"])
            self.assertEqual(bundle.validation_dataset.num_data(), sizes["validation"])
            self.assertFalse(hasattr(bundle, "test_dataset"))
            self.assertFalse((prepared / ".lightgbm_sequence_cache").exists())
            self.assertEqual(bundle.features["test"].shape, (sizes["test"], 2))
            self.assertEqual(bundle.features["test"].iloc[2:5].shape, (3, 2))
            self.assertEqual(booster.feature_name(), ["f0000_f_0", "f0001_f_1_rate"])
            prediction = booster.predict(bundle.features["test"].iloc[:4])
            self.assertEqual(prediction.shape, (4, 3))

    def test_native_booster_resumes_without_repeating_iterations(self) -> None:
        try:
            import lightgbm as lgb
        except ImportError:
            self.skipTest("LightGBM is not installed in this Python environment")

        rng = np.random.default_rng(2026)
        features = rng.normal(size=(240, 6)).astype(np.float32)
        labels = np.repeat(np.arange(3, dtype=np.int32), 80)
        features[:, 0] += labels * 0.8
        train_set = lgb.Dataset(features[:180], label=labels[:180], free_raw_data=False)
        validation_set = lgb.Dataset(
            features[180:], label=labels[180:], reference=train_set, free_raw_data=False
        )
        params = {
            "objective": "multiclass",
            "num_class": 3,
            "learning_rate": 0.05,
            "num_leaves": 7,
            "min_data_in_leaf": 5,
            "metric": ["multi_logloss", "multi_error"],
            "num_threads": 1,
            "deterministic": True,
            "force_col_wise": True,
            "verbosity": -1,
        }
        target_iteration = 100
        history = []
        saved_model = {}
        checkpoint_events = []

        def checkpoint(booster, records, status):
            checkpoint_events.append((booster.current_iteration(), status))
            saved_model["text"] = booster.model_to_string(num_iteration=booster.current_iteration())
            return 0.01

        first = IterationRecorder(
            history, "session_one", target_iteration, 0.05, 10, checkpoint,
            None, 20, 0, 12.0, 20.0,
        )
        with self.assertRaises(TrainingPauseRequested):
            lgb.train(
                params,
                train_set,
                num_boost_round=target_iteration,
                valid_sets=[train_set, validation_set],
                valid_names=["train", "validation"],
                feval=macro_f1_metric(3),
                keep_training_booster=True,
                callbacks=[first],
            )
        self.assertEqual([record["iteration"] for record in history], list(range(1, 21)))
        self.assertEqual(checkpoint_events, [(10, "running"), (20, "paused")])

        resumed_model = lgb.Booster(model_str=saved_model["text"])
        second = IterationRecorder(
            history, "session_two", target_iteration, 0.05, 10,
            checkpoint, None, None, 20, 12.0, 20.0,
        )
        booster = lgb.train(
            params,
            train_set,
            num_boost_round=target_iteration - 20,
            valid_sets=[train_set, validation_set],
            valid_names=["train", "validation"],
            feval=macro_f1_metric(3),
            init_model=resumed_model,
            keep_training_booster=True,
            callbacks=[second],
        )
        self.assertEqual(booster.current_iteration(), target_iteration)
        self.assertEqual(
            [record["iteration"] for record in history], list(range(1, target_iteration + 1))
        )
        self.assertEqual({record["session_id"] for record in history[:20]}, {"session_one"})
        self.assertEqual({record["session_id"] for record in history[20:]}, {"session_two"})
        self.assertEqual(len({record["iteration"] for record in history}), target_iteration)
        self.assertTrue(history[-1]["is_final_round"])
        self.assertEqual(checkpoint_events[-1], (target_iteration, "ready_for_report"))


if __name__ == "__main__":
    unittest.main()


class ContinueTrainingTest(unittest.TestCase):
    """Resuming must reproduce uninterrupted training, not merely run without crashing."""

    def _problem(self):
        rng = np.random.default_rng(11)
        rows, features, classes = 1200, 6, 3
        matrix = rng.normal(size=(rows, features))
        weights = rng.normal(size=(features, classes))
        labels = np.argmax(matrix @ weights + rng.normal(scale=0.5, size=(rows, classes)), axis=1)
        return matrix[:900], labels[:900], matrix[900:], labels[900:], classes

    def _params(self, classes):
        return {
            "objective": "multiclass", "num_class": classes, "verbosity": -1, "num_threads": 1,
            "learning_rate": 0.05, "deterministic": True, "force_col_wise": True, "seed": 2026,
            "bagging_seed": 2026, "feature_fraction_seed": 2026, "data_random_seed": 2026,
            # Quantized gradients are deliberately off: LightGBM's own resume is not
            # bit-exact with them, so they would test LightGBM rather than this function.
            "use_quantized_grad": False,
        }

    def test_resumed_training_matches_an_uninterrupted_run(self) -> None:
        try:
            import lightgbm
        except ImportError:
            self.skipTest("LightGBM is required")
        train_x, train_y, valid_x, valid_y, classes = self._problem()
        params = self._params(classes)

        def datasets():
            train = lightgbm.Dataset(train_x, label=train_y, params=params, free_raw_data=False)
            valid = lightgbm.Dataset(
                valid_x, label=valid_y, params=params, free_raw_data=False, reference=train
            )
            return train, valid

        train_set, valid_set = datasets()
        reference = lightgbm.train(params, train_set, num_boost_round=40, valid_sets=[valid_set])
        expected = reference.predict(valid_x)

        with tempfile.TemporaryDirectory() as temporary:
            model_path = str(Path(temporary) / "round_10.txt")
            train_set, valid_set = datasets()
            lightgbm.train(
                params, train_set, num_boost_round=10, valid_sets=[valid_set],
                keep_training_booster=True,
            ).save_model(model_path, num_iteration=10)

            train_set, valid_set = datasets()
            resumed = continue_training(
                lightgbm, params=params, train_dataset=train_set,
                valid_sets=[train_set, valid_set], valid_names=["train", "validation"],
                num_boost_round=30, feval=None, callbacks=[], num_class=classes,
                train_features=ArrayFeatures(train_x),
                valid_features={"validation": ArrayFeatures(valid_x)},
                init_model_path=model_path,
            )

        self.assertEqual(resumed.current_iteration(), 40)
        self.assertEqual(resumed.num_trees(), reference.num_trees())
        np.testing.assert_allclose(resumed.predict(valid_x), expected, rtol=0, atol=0)

    def test_fresh_training_matches_lightgbm_train(self) -> None:
        try:
            import lightgbm
        except ImportError:
            self.skipTest("LightGBM is required")
        train_x, train_y, valid_x, valid_y, classes = self._problem()
        params = self._params(classes)

        train_set = lightgbm.Dataset(train_x, label=train_y, params=params, free_raw_data=False)
        valid_set = lightgbm.Dataset(
            valid_x, label=valid_y, params=params, free_raw_data=False, reference=train_set
        )
        reference = lightgbm.train(params, train_set, num_boost_round=25, valid_sets=[valid_set])

        train_set = lightgbm.Dataset(train_x, label=train_y, params=params, free_raw_data=False)
        valid_set = lightgbm.Dataset(
            valid_x, label=valid_y, params=params, free_raw_data=False, reference=train_set
        )
        ours = continue_training(
            lightgbm, params=params, train_dataset=train_set,
            valid_sets=[train_set, valid_set], valid_names=["train", "validation"],
            num_boost_round=25, feval=None, callbacks=[], num_class=classes,
            train_features=ArrayFeatures(train_x),
        )
        np.testing.assert_allclose(ours.predict(valid_x), reference.predict(valid_x), rtol=0, atol=0)

    def test_init_score_is_stored_class_major_in_chunks(self) -> None:
        try:
            import lightgbm
        except ImportError:
            self.skipTest("LightGBM is required")
        train_x, train_y, _, _, classes = self._problem()
        params = self._params(classes)
        train_set = lightgbm.Dataset(train_x, label=train_y, params=params, free_raw_data=False)
        booster = lightgbm.train(params, train_set, num_boost_round=5)
        expected = np.asarray(booster.predict(train_x, raw_score=True), dtype=np.float64)

        chunked = raw_init_score(booster, ArrayFeatures(train_x), classes, chunk_rows=97)
        self.assertEqual(chunked.shape, (len(train_x) * classes,))
        np.testing.assert_allclose(chunked, expected.ravel(order="F"), rtol=0, atol=0)


class MemoryGuardIntegrationTest(unittest.TestCase):
    """require_safe_memory_profile has to stop the run, not just describe it."""

    def _prepared(self, stack: tempfile.TemporaryDirectory) -> Path:
        prepared = Path(stack.name)
        write_prepared_fixture(prepared)
        return prepared

    def test_guard_refuses_to_build_datasets_that_cannot_fit(self) -> None:
        try:
            import lightgbm  # noqa: F401
        except ImportError:
            self.skipTest("LightGBM is required")
        with tempfile.TemporaryDirectory() as temporary:
            prepared = Path(temporary)
            write_prepared_fixture(prepared)
            config = prepared_fixture_config()
            config["dataset"]["require_safe_memory_profile"] = True
            config["dataset"]["memory_guard"] = {"available_ram_fraction": 1e-12}
            with self.assertRaises(InsufficientMemoryError) as caught:
                build_datasets(prepared, config)
            self.assertIn("require_safe_memory_profile", str(caught.exception))

    def test_guard_is_advisory_when_the_flag_is_off(self) -> None:
        try:
            import lightgbm  # noqa: F401
        except ImportError:
            self.skipTest("LightGBM is required")
        with tempfile.TemporaryDirectory() as temporary:
            prepared = Path(temporary)
            write_prepared_fixture(prepared)
            config = prepared_fixture_config()
            config["dataset"]["require_safe_memory_profile"] = False
            config["dataset"]["memory_guard"] = {"available_ram_fraction": 1e-12}
            bundle = build_datasets(prepared, config)
            self.assertFalse(bundle.memory_estimate["fits_budget"])

    def test_projection_is_carried_on_the_bundle_for_run_config(self) -> None:
        try:
            import lightgbm  # noqa: F401
        except ImportError:
            self.skipTest("LightGBM is required")
        with tempfile.TemporaryDirectory() as temporary:
            prepared = Path(temporary)
            sizes = write_prepared_fixture(prepared)
            bundle = build_datasets(prepared, prepared_fixture_config())
            self.assertTrue(bundle.memory_estimate["fits_budget"])
            self.assertEqual(bundle.memory_estimate["train_rows"], sizes["train"])
            self.assertEqual(bundle.memory_estimate["num_classes"], 3)


class TrainerPausesInsteadOfDyingTest(unittest.TestCase):
    def test_train_returns_the_pause_exit_code_when_memory_is_short(self) -> None:
        try:
            import lightgbm  # noqa: F401
        except ImportError:
            self.skipTest("LightGBM is required")
        import train as train_module

        with tempfile.TemporaryDirectory() as temporary:
            prepared = Path(temporary) / "prepared"
            prepared.mkdir()
            write_prepared_fixture(prepared)
            config_path = Path(temporary) / "train.json"
            config = prepared_fixture_config()
            config["dataset"]["require_safe_memory_profile"] = True
            config["dataset"]["memory_guard"] = {"available_ram_fraction": 1e-12}
            config["session"]["minimum_available_ram_gb"] = None
            config["s3"]["enabled"] = False
            config["s3"]["upload_required"] = False
            config_path.write_text(json.dumps(config), encoding="utf-8")
            arguments = SimpleNamespace(
                config=str(config_path),
                prepared_data_dir=str(prepared),
                output_dir=str(Path(temporary) / "runs"),
                run_id="memory-guard-run",
                max_rounds_this_session=None,
                upload_checkpoints_to_s3=False,
            )
            self.assertEqual(train_module.train(arguments), train_module.PAUSED_EXIT_CODE)
