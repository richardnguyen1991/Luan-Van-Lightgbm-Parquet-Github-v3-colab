import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data import (  # noqa: E402
    SampledExactLeakageAuditor,
    allocate_proportional_sample_quotas,
    assign_row_split_codes,
    canonical_group_values,
    compute_data_version,
    deterministic_sample_row_ids,
    enforce_leakage_audit,
    group_hashes,
    load_config,
    prepare_dataset,
)


class DataPipelineTest(unittest.TestCase):
    def test_exact_proportional_sampling_is_deterministic(self) -> None:
        quotas = allocate_proportional_sample_quotas({"a": 900, "b": 600, "c": 300}, 1000)
        self.assertEqual(sum(quotas.values()), 1000)
        self.assertEqual(quotas, {"a": 500, "b": 333, "c": 167})
        first = deterministic_sample_row_ids(123, 10_000, 1_417, 2026)
        second = deterministic_sample_row_ids(123, 10_000, 1_417, 2026)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 1_417)
        self.assertEqual(len(np.unique(first)), 1_417)
        self.assertTrue(np.all(first[:-1] < first[1:]))

    def test_row_split_is_deterministic_and_exclusive(self) -> None:
        rows = np.arange(100_000, dtype=np.uint64)
        first = assign_row_split_codes(123, rows, [0.70, 0.15, 0.15], 2026)
        second = assign_row_split_codes(123, rows, [0.70, 0.15, 0.15], 2026)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.isin(first, [0, 1, 2]).all())
        self.assertEqual(len(rows), sum(np.count_nonzero(first == code) for code in range(3)))

    def test_prepare_preserves_distribution_and_writes_leakage_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            output = root / "output"
            dataset.mkdir()
            rng = np.random.default_rng(9)
            for label, rows in (("BENIGN", 900), ("DrDoS_DNS", 600), ("Syn", 300)):
                frame = pd.DataFrame({
                    "Flow ID": [f"{label}-{index}" for index in range(rows)],
                    "Timestamp": pd.date_range("2019-01-01", periods=rows, freq="s").astype(str),
                    " Feature A": rng.normal(size=rows),
                    "Feature B": rng.normal(size=rows),
                    " Label": label,
                })
                frame.loc[::37, " Feature A"] = np.nan
                frame.loc[::53, "Feature B"] = np.inf
                frame.to_parquet(dataset / f"{label}.parquet", index=False, row_group_size=113)

            config = load_config(PROJECT_ROOT / "config" / "data.json")
            config["dataset"]["data_dir"] = str(dataset)
            config["dataset"]["target_total_rows"] = 900
            config["output"]["rows_per_part"] = 127
            manifest = prepare_dataset(config, output)

            self.assertEqual(sum(manifest["split"]["sizes"].values()), 900)
            self.assertEqual(manifest["sampling_mode"], "deterministic_proportional_exact_total")
            self.assertEqual(manifest["target_total_rows"], 900)
            self.assertEqual(
                {item["path"]: item["rows_processed"] for item in manifest["source_files"]},
                {"BENIGN.parquet": 450, "DrDoS_DNS.parquet": 300, "Syn.parquet": 150},
            )
            self.assertTrue(manifest["split"]["group_aware"])
            self.assertTrue(manifest["leakage_audit"]["passed"])
            self.assertTrue(manifest["leakage_audit"]["sample_id_assertion_passed"])
            self.assertTrue(manifest["leakage_audit"]["group_assertion_passed"])
            self.assertTrue(all(not values for values in manifest["split"]["classes_missing_from_split"].values()))

            with (output / "preprocessing.json").open(encoding="utf-8") as handle:
                preprocessing = json.load(handle)
            self.assertEqual(preprocessing["scaling"], "none")
            self.assertEqual(preprocessing["imbalance_handling"], "none")
            self.assertEqual(preprocessing["categorical_features"], [])
            self.assertEqual(preprocessing["target_column"], "Label")
            self.assertEqual(preprocessing["feature_columns_in_order"], ["Feature A", "Feature B"])
            dropped = {item["column"] for item in preprocessing["dropped_columns"]}
            self.assertIn("Flow ID", dropped)
            self.assertIn("Timestamp", dropped)

            split_frames = {}
            for split, parts in manifest["parts"].items():
                split_frames[split] = pd.concat(
                    [pd.read_parquet(output / item["path"]) for item in parts], ignore_index=True
                )
                self.assertIn("_label", split_frames[split])
                self.assertNotIn("_label_name", split_frames[split])
            identities = {
                split: set(zip(frame["_sample_file_id"], frame["_sample_row_id"]))
                for split, frame in split_frames.items()
            }
            self.assertFalse(identities["train"] & identities["validation"])
            self.assertFalse(identities["train"] & identities["test"])
            self.assertFalse(identities["validation"] & identities["test"])

            with (output / "data_profile.json").open(encoding="utf-8") as handle:
                profile = json.load(handle)
            self.assertEqual(profile["total_selected_rows"], 900)
            self.assertEqual(profile["feature_count"], 2)


class LeakageAuditTest(unittest.TestCase):
    """The full-dataset backend used to return passed=True without reading any data."""

    def _auditor(self, root: Path, divisor: int = 4) -> SampledExactLeakageAuditor:
        return SampledExactLeakageAuditor(root / "audit", divisor, 10_000_000)

    def test_clean_split_passes_and_reports_what_it_actually_inspected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            auditor = self._auditor(Path(temporary))
            identities = np.arange(40_000, dtype=np.uint64)
            auditor.add_samples(11, identities[:30_000], 0)
            auditor.add_samples(11, identities[30_000:], 1)
            result = auditor.result(False)
            self.assertTrue(result["passed"])
            self.assertEqual(result["sample_id_cross_split_overlap_count"], 0)
            self.assertEqual(result["sample_identities_seen"], 40_000)
            self.assertGreater(result["sample_identities_tracked_distinct"], 0)
            self.assertNotEqual(result["method"], "deterministic_seeded_hash_function_proof")

    def test_planted_cross_split_overlap_is_detected_and_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            auditor = self._auditor(Path(temporary))
            identities = np.arange(40_000, dtype=np.uint64)
            auditor.add_samples(11, identities[:30_000], 0)
            auditor.add_samples(11, identities[29_000:], 1)
            result = auditor.result(False)
            self.assertFalse(result["passed"])
            self.assertGreater(result["sample_id_cross_split_overlap_count"], 0)
            with self.assertRaises(ValueError):
                enforce_leakage_audit(result, {"fail_on_cross_split_overlap": True}, False)
            enforce_leakage_audit(result, {"fail_on_cross_split_overlap": False}, False)

    def test_group_overlap_is_only_enforced_for_group_aware_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            auditor = self._auditor(Path(temporary))
            hashes = np.arange(20_000, dtype=np.uint64)
            auditor.add_groups(hashes, 0)
            auditor.add_groups(hashes, 2)
            result = auditor.result(True)
            self.assertFalse(result["passed"])
            self.assertGreater(result["group_cross_split_overlap_count"], 0)
            self.assertTrue(auditor.result(False)["passed"])

    def test_audit_state_survives_a_session_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identities = np.arange(20_000, dtype=np.uint64)
            first = self._auditor(root)
            first.add_samples(3, identities[:10_000], 0)
            first.close()
            # A resumed session constructs a new auditor over the restored state directory
            # and must still see the identities the previous session retained.
            second = self._auditor(root)
            second.add_samples(3, identities[:10_000], 1)
            result = second.result(False)
            self.assertFalse(result["passed"])
            self.assertGreater(result["sample_id_cross_split_overlap_count"], 0)

    def test_removed_no_op_backend_is_rejected_with_a_pointer_to_the_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            pd.DataFrame({"Flow ID": ["a", "b"], "F": [1.0, 2.0], "Label": ["x", "y"]}).to_parquet(
                dataset / "part.parquet", index=False
            )
            config = load_config(PROJECT_ROOT / "config" / "data.json")
            config["dataset"]["data_dir"] = str(dataset)
            config["audit"]["backend"] = "deterministic_proof"
            with self.assertRaises(ValueError) as caught:
                prepare_dataset(config, root / "output")
            self.assertIn("sampled_exact", str(caught.exception))


class PreprocessingSessionBudgetTest(unittest.TestCase):
    """data.py used to ignore the session deadline unless given an explicit hour budget."""

    def _deadline_for(self, argv: list[str], environment: dict[str, str]) -> float | None:
        import data as data_module

        captured: dict[str, Any] = {}

        def fake_prepare(config, output_dir, store, deadline):
            captured["deadline"] = deadline
            raise data_module.PreprocessingPauseRequested("stop")

        with mock.patch.object(data_module, "prepare_dataset", fake_prepare), \
                mock.patch.object(sys, "argv", ["data.py", *argv]), \
                mock.patch.dict(os.environ, environment, clear=False):
            self.assertEqual(data_module.main(), 75)
        return captured["deadline"]

    def test_session_deadline_is_honoured_without_an_explicit_hour_budget(self) -> None:
        deadline = self._deadline_for(
            ["--config", str(PROJECT_ROOT / "config" / "data.json"), "--full-dataset"],
            {"PIPELINE_SESSION_DEADLINE_EPOCH": str(time.time() + 600.0)},
        )
        self.assertIsNotNone(deadline)
        self.assertLessEqual(deadline - time.monotonic(), 601.0)

    def test_the_tighter_of_the_two_budgets_wins(self) -> None:
        deadline = self._deadline_for(
            [
                "--config", str(PROJECT_ROOT / "config" / "data.json"), "--full-dataset",
                "--maximum-hours", "11", "--stop-before-minutes", "30",
            ],
            {"PIPELINE_SESSION_DEADLINE_EPOCH": str(time.time() + 300.0)},
        )
        self.assertLessEqual(deadline - time.monotonic(), 301.0)

    def test_no_budget_anywhere_means_no_deadline(self) -> None:
        environment = dict(os.environ)
        environment.pop("PIPELINE_SESSION_DEADLINE_EPOCH", None)
        import data as data_module

        captured: dict[str, Any] = {}

        def fake_prepare(config, output_dir, store, deadline):
            captured["deadline"] = deadline
            raise data_module.PreprocessingPauseRequested("stop")

        with mock.patch.object(data_module, "prepare_dataset", fake_prepare), \
                mock.patch.object(
                    sys, "argv",
                    ["data.py", "--config", str(PROJECT_ROOT / "config" / "data.json"), "--full-dataset"],
                ), \
                mock.patch.dict(os.environ, environment, clear=True):
            self.assertEqual(data_module.main(), 75)
        self.assertIsNone(captured["deadline"])


class GroupIdentityTest(unittest.TestCase):
    """A flow rendered differently in two source files would split into two identities."""

    def test_integer_and_float_renderings_of_one_port_agree(self) -> None:
        self.assertEqual(
            list(canonical_group_values(pd.Series([80, 443], dtype="int64"))),
            list(canonical_group_values(pd.Series([80.0, 443.0], dtype="float64"))),
        )

    def test_surrounding_whitespace_does_not_create_a_second_identity(self) -> None:
        left = pd.DataFrame({"Flow ID": [" flow-1", "flow-2"], "Source Port": [80, 443]})
        right = pd.DataFrame({"Flow ID": ["flow-1 ", "flow-2"], "Source Port": [80.0, 443.0]})
        np.testing.assert_array_equal(
            group_hashes(left, ["Flow ID", "Source Port"]),
            group_hashes(right, ["Flow ID", "Source Port"]),
        )

    def test_distinct_flows_still_hash_apart(self) -> None:
        frame = pd.DataFrame({"Flow ID": ["flow-1", "flow-2"], "Source Port": [80, 80]})
        hashes = group_hashes(frame, ["Flow ID", "Source Port"])
        self.assertNotEqual(hashes[0], hashes[1])

    def test_data_version_tracks_the_split_algorithm(self) -> None:
        import data as data_module

        config = load_config(PROJECT_ROOT / "config" / "data.json")
        baseline = compute_data_version(config)
        original = data_module.SPLIT_ALGORITHM_VERSION
        try:
            data_module.SPLIT_ALGORITHM_VERSION = original + 1
            self.assertNotEqual(compute_data_version(config), baseline)
        finally:
            data_module.SPLIT_ALGORITHM_VERSION = original


class SplitCoveragePreflightTest(unittest.TestCase):
    def _dataset(self, root: Path, rare_rows: int) -> Path:
        dataset = root / "dataset"
        dataset.mkdir()
        rng = np.random.default_rng(4)
        for label, rows in (("BENIGN", 600), ("Syn", 400), ("WebDDoS", rare_rows)):
            pd.DataFrame({
                "Flow ID": [f"{label}-{index}" for index in range(rows)],
                "Feature A": rng.normal(size=rows),
                "Feature B": rng.normal(size=rows),
                "Label": label,
            }).to_parquet(dataset / f"{label}.parquet", index=False)
        return dataset

    def test_a_class_missing_from_a_split_fails_before_any_part_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            config = load_config(PROJECT_ROOT / "config" / "data.json")
            config["dataset"]["data_dir"] = str(self._dataset(root, rare_rows=1))
            config["output"]["rows_per_part"] = 128
            with self.assertRaises(ValueError) as caught:
                prepare_dataset(config, output)
            message = str(caught.exception)
            self.assertIn("Pre-flight", message)
            self.assertIn("WebDDoS", message)
            # The expensive work must not have started: no split parts on disk.
            self.assertFalse(list(output.glob("splits/**/*.parquet")))

    def test_preflight_matrix_is_recorded_and_matches_the_realized_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            config = load_config(PROJECT_ROOT / "config" / "data.json")
            config["dataset"]["data_dir"] = str(self._dataset(root, rare_rows=300))
            config["output"]["rows_per_part"] = 128
            manifest = prepare_dataset(config, output)
            preflight = json.loads(
                (output / "split_coverage_preflight.json").read_text(encoding="utf-8")
            )
            self.assertEqual(preflight["sizes"], manifest["split"]["sizes"])
            self.assertEqual(
                preflight["class_counts"], manifest["split"]["class_counts"]
            )
            self.assertTrue(manifest["split"]["preflight"]["agrees_with_realized_split"])
            self.assertIn("Label", preflight["columns_read"])
            self.assertIn("Flow ID", preflight["columns_read"])


if __name__ == "__main__":
    unittest.main()
