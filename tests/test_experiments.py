"""Guards for the three-experiment design: label merging, day splits and open-set scoring.

Experiment A is the in-distribution baseline on the merged 14-class taxonomy. Experiment B
trains on the 01-12 capture and tests on 03-11, restricted to the classes both days share.
Experiment C withholds one class entirely and reports only where the model sends it.
"""

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data import (  # noqa: E402
    OPEN_SET_LABEL_CODE,
    LabelPolicy,
    build_label_policy,
    capture_day_split_codes,
    closed_class_coverage,
    load_config,
    prepare_dataset,
    resolve_capture_day_column,
    validate_capture_day_assignment,
    validate_split_strategy,
)
from model import (  # noqa: E402
    estimate_training_memory,
    stratified_monitor_indices,
    validate_training_config,
)

# The three provenance columns the CSV-to-Parquet converter appends; its own header says
# they must never become classifier features.
PROVENANCE_COLUMNS = ("__source_row_id", "__source_file_id", "__capture_day")

# A miniature two-day CIC-DDoS2019: the 01-12 files use the DrDoS_ prefix and "UDP-lag",
# the 03-11 files use the bare names and "UDPLag", exactly as the real corpus does.
LAYOUT = (
    ("01-12", "DrDoS_DNS", "DrDoS_DNS", 320),
    ("01-12", "DrDoS_LDAP", "DrDoS_LDAP", 300),
    ("01-12", "Syn", "Syn", 260),
    ("01-12", "UDPLag", "UDP-lag", 180),
    ("03-11", "LDAP", "LDAP", 280),
    ("03-11", "Syn", "Syn", 300),
    ("03-11", "Portmap", "Portmap", 150),
)
BENIGN_ROWS = 60
DAY_SHIFT = {"01-12": 0.0, "03-11": 0.7}


def write_two_day_dataset(dataset: Path) -> None:
    rng = np.random.default_rng(17)
    centres: dict[str, np.ndarray] = {}
    for _, _, label, _ in LAYOUT:
        centres.setdefault(label, rng.normal(scale=2.0, size=4))
    centres["BENIGN"] = rng.normal(scale=2.0, size=4)
    for day, stem, label, rows in LAYOUT:
        names = [label] * rows + ["BENIGN"] * BENIGN_ROWS
        block = np.vstack([centres[name] for name in names])
        block = block + rng.normal(scale=0.85, size=block.shape) + DAY_SHIFT[day]
        frame = pd.DataFrame({
            "Unnamed: 0": np.arange(len(names), dtype=np.int64),
            "Flow ID": [f"{day}-{stem}-{index}" for index in range(len(names))],
            " Source IP": "10.0.0.1",
            " Source Port": rng.integers(1024, 65535, len(names)),
            " Timestamp": "2019-01-01 00:00:00",
            " Feature A": block[:, 0], "Feature B": block[:, 1],
            " Feature C": block[:, 2], "Feature D": block[:, 3],
            " Label": names,
            "__capture_day": day,
            "__source_file_id": f"{day}/{stem}.csv",
            "__source_row_id": np.arange(len(names), dtype=np.int64),
        })
        (dataset / day).mkdir(parents=True, exist_ok=True)
        frame.to_parquet(dataset / day / f"{stem}.parquet", index=False, row_group_size=97)


def prepared_for(config_name: str, dataset: Path, destination: Path) -> dict:
    config = load_config(PROJECT_ROOT / "config" / config_name)
    config["dataset"]["data_dir"] = str(dataset)
    config["output"]["rows_per_part"] = 256
    config["audit"]["identity_sample_divisor"] = 1
    return prepare_dataset(config, destination)


class LabelPolicyTest(unittest.TestCase):
    def test_merge_is_case_insensitive_and_leaves_unlisted_names_alone(self) -> None:
        policy = LabelPolicy({"DrDoS_LDAP": "LDAP", "UDP-lag": "UDPLag"})
        merged = policy.merge(pd.Series([" drdos_ldap ", "UDP-lag", "TFTP"], dtype="string"))
        self.assertEqual(list(merged), ["LDAP", "UDPLag", "TFTP"])

    def test_keep_only_and_open_set_must_be_disjoint(self) -> None:
        with self.assertRaises(ValueError):
            LabelPolicy({}, ["LDAP", "Portmap"], ["Portmap"])

    def test_open_set_rows_survive_a_keep_only_restriction(self) -> None:
        policy = LabelPolicy({}, ["LDAP"], ["Portmap"])
        keep, opened = policy.classify(pd.Series(["LDAP", "Portmap", "TFTP"], dtype="string"))
        np.testing.assert_array_equal(keep, [True, True, False])
        np.testing.assert_array_equal(opened, [False, True, False])

    def test_shipped_merge_map_collapses_nineteen_labels_to_fourteen(self) -> None:
        # The full CIC-DDoS2019 label set, as counted from the raw CSVs.
        raw = pd.Series([
            "BENIGN", "DrDoS_DNS", "DrDoS_LDAP", "DrDoS_MSSQL", "DrDoS_NTP", "DrDoS_NetBIOS",
            "DrDoS_SNMP", "DrDoS_SSDP", "DrDoS_UDP", "LDAP", "MSSQL", "NetBIOS", "Portmap",
            "Syn", "TFTP", "UDP", "UDP-lag", "UDPLag", "WebDDoS",
        ], dtype="string")
        policy = build_label_policy(load_config(PROJECT_ROOT / "config" / "data.json"))
        merged = sorted(set(policy.merge(raw)))
        self.assertEqual(len(raw), 19)
        self.assertEqual(merged, [
            "BENIGN", "DNS", "LDAP", "MSSQL", "NTP", "NetBIOS", "Portmap", "SNMP", "SSDP",
            "Syn", "TFTP", "UDP", "UDPLag", "WebDDoS",
        ])
        self.assertEqual(len(merged), 14)

    def test_unknown_configuration_keys_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_label_policy({"labels": {"merge_map": {}, "typo": []}})


class CaptureDaySplitTest(unittest.TestCase):
    def test_unclaimed_days_are_dropped_rather_than_silently_trained_on(self) -> None:
        days = pd.Series(["01-12", "03-11", "07-04"], dtype="string")
        codes = capture_day_split_codes(
            days, np.array([1, 2, 3], dtype=np.uint64),
            {"01-12": "train_validation", "03-11": "test"}, 0.15, 2026,
        )
        self.assertIn(codes[0], (0, 1))
        self.assertEqual(codes[1], 2)
        self.assertEqual(codes[2], -1)

    def test_validation_fraction_must_be_a_proper_fraction(self) -> None:
        days = pd.Series(["01-12"], dtype="string")
        for fraction in (0.0, 1.0, -0.1):
            with self.assertRaises(ValueError):
                capture_day_split_codes(
                    days, np.array([1], dtype=np.uint64), {"01-12": "train_validation"},
                    fraction, 2026,
                )

    def test_assignment_must_name_a_training_day(self) -> None:
        with self.assertRaises(ValueError):
            validate_capture_day_assignment({"03-11": "test"})
        with self.assertRaises(ValueError):
            validate_capture_day_assignment({"01-12": "holdout"})
        self.assertEqual(
            validate_capture_day_assignment({" 01-12 ": "train_validation"}),
            {"01-12": "train_validation"},
        )

    def test_missing_capture_day_column_is_a_configuration_error(self) -> None:
        with self.assertRaises(ValueError):
            resolve_capture_day_column([" Label", "Flow ID"], "__capture_day")
        self.assertEqual(resolve_capture_day_column(["__CAPTURE_DAY"], "__capture_day"), "__CAPTURE_DAY")

    def test_unknown_strategy_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            validate_split_strategy("by_timestamp")

    def test_an_open_set_only_split_is_exempt_from_class_coverage(self) -> None:
        coverage = closed_class_coverage(
            {"LDAP", "Syn"},
            {"train": {"LDAP": 5, "Syn": 5}, "validation": {"LDAP": 1, "Syn": 1}, "test": {"Portmap": 9}},
            ["test"],
        )
        self.assertEqual(coverage, {"train": [], "validation": [], "test": []})


class MonitorSplitTest(unittest.TestCase):
    def test_subsample_is_proportional_deterministic_and_keeps_every_class(self) -> None:
        labels = np.concatenate([
            np.zeros(10_000, dtype=np.int32), np.ones(1_000, dtype=np.int32),
            np.full(3, 2, dtype=np.int32),
        ])
        first = stratified_monitor_indices(labels, 1_000, 2026)
        second = stratified_monitor_indices(labels, 1_000, 2026)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 1_000)
        self.assertEqual(sorted(set(labels[first])), [0, 1, 2])
        self.assertTrue(np.all(first[:-1] < first[1:]))

    def test_a_budget_at_or_above_the_split_size_keeps_every_row(self) -> None:
        labels = np.arange(50, dtype=np.int32) % 5
        np.testing.assert_array_equal(
            stratified_monitor_indices(labels, 500, 1), np.arange(50, dtype=np.int64)
        )

    def test_a_budget_below_the_class_count_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            stratified_monitor_indices(np.arange(20, dtype=np.int32), 5, 1)

    def test_monitoring_rows_enter_the_memory_budget(self) -> None:
        sizes = {"train": 1_000_000, "validation": 200_000, "test": 5_000_000}
        params = {"max_bin": 255}
        without = estimate_training_memory(sizes, 80, 14, params, 64 * 1024**3, 0.8)
        with_monitor = estimate_training_memory(
            sizes, 80, 14, params, 64 * 1024**3, 0.8, monitor_rows=500_000
        )
        self.assertEqual(with_monitor["monitor_rows"], 500_000)
        self.assertGreater(with_monitor["steady_state_bytes"], without["steady_state_bytes"])
        self.assertGreater(with_monitor["estimated_peak_bytes"], without["estimated_peak_bytes"])
        self.assertEqual(with_monitor["monitor_materialization_bytes"], 500_000 * 80 * 4)

    def test_a_monitor_name_may_not_shadow_a_builtin_evaluation_set(self) -> None:
        config = json.loads((PROJECT_ROOT / "config" / "train.expB.json").read_text(encoding="utf-8"))
        validate_training_config(config)
        config["dataset"]["monitor_split"]["name"] = "validation"
        with self.assertRaises(ValueError):
            validate_training_config(config)

    def test_monitor_configuration_typos_are_refused(self) -> None:
        config = json.loads((PROJECT_ROOT / "config" / "train.expB.json").read_text(encoding="utf-8"))
        config["dataset"]["monitor_split"]["split"] = "holdout"
        with self.assertRaises(ValueError):
            validate_training_config(config)


class FullDatasetManifestGateTest(unittest.TestCase):
    """`require_full_dataset_manifest` must mean "read every row", not "keep every row"."""

    def _manifest(self, written: int, excluded_label: int, excluded_day: int) -> dict:
        return {
            "sampling_mode": "full",
            "source_files": [{
                "path": "01-12/Syn.parquet", "physical_rows": 100,
                "planned_sample_rows": 100, "rows_read": 100, "rows_processed": written,
            }],
            "split": {
                "sizes": {"train": written, "validation": 0, "test": 0},
                "rows_excluded_by_label_policy": excluded_label,
                "rows_excluded_by_unassigned_split": excluded_day,
            },
        }

    def _config(self) -> dict:
        return {"dataset": {"require_full_dataset_manifest": True}}

    def test_rows_dropped_by_the_label_policy_still_count_as_used(self) -> None:
        from model import validate_dataset_manifest

        validate_dataset_manifest(self._config(), self._manifest(60, 30, 10))

    def test_unexplained_missing_rows_are_still_rejected(self) -> None:
        from model import validate_dataset_manifest

        with self.assertRaises(ValueError):
            validate_dataset_manifest(self._config(), self._manifest(60, 10, 10))

    def test_a_file_that_was_not_fully_read_is_rejected(self) -> None:
        from model import validate_dataset_manifest

        manifest = self._manifest(60, 30, 10)
        manifest["source_files"][0]["rows_read"] = 90
        with self.assertRaises(ValueError):
            validate_dataset_manifest(self._config(), manifest)


class ShippedConfigurationTest(unittest.TestCase):
    def test_every_data_config_drops_the_provenance_columns(self) -> None:
        for name in ("data.json", "data.smoke.json", "data.expB.json", "data.expC.json"):
            with self.subTest(config=name):
                config = load_config(PROJECT_ROOT / "config" / name)
                drops = {str(value).casefold() for value in config["preprocessing"]["explicit_drop_columns"]}
                self.assertIn("unnamed: 0", drops)
                for column in PROVENANCE_COLUMNS:
                    self.assertIn(column.casefold(), drops)

    def test_experiment_b_restricts_to_the_classes_both_days_share(self) -> None:
        config = load_config(PROJECT_ROOT / "config" / "data.expB.json")
        self.assertEqual(config["split"]["strategy"], "by_capture_day")
        self.assertEqual(
            config["split"]["capture_day_assignment"],
            {"01-12": "train_validation", "03-11": "test"},
        )
        # UDPLag is shared on paper but the 03-11 side holds 1,873 rows against 366,461 on
        # 01-12, and its file is 97% other traffic; it is excluded deliberately.
        self.assertEqual(
            sorted(config["labels"]["keep_only"]),
            ["BENIGN", "LDAP", "MSSQL", "NetBIOS", "Syn", "UDP"],
        )

    def test_experiment_c_holds_out_portmap_and_claims_no_test_day(self) -> None:
        config = load_config(PROJECT_ROOT / "config" / "data.expC.json")
        self.assertEqual(config["labels"]["open_set_labels"], ["Portmap"])
        self.assertEqual(config["split"]["capture_day_assignment"], {"01-12": "train_validation"})
        self.assertNotIn("Portmap", config["labels"]["keep_only"])
        self.assertEqual(len(config["labels"]["keep_only"]), 13)


class PreparedExperimentTest(unittest.TestCase):
    """One preparation run per experiment, asserted against the split it should produce."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory()
        base = Path(cls._temporary.name)
        cls.dataset = base / "dataset"
        write_two_day_dataset(cls.dataset)
        cls.prepared = {}
        cls.manifests = {}
        for tag, config_name in (
            ("A", "data.json"), ("B", "data.expB.json"), ("C", "data.expC.json")
        ):
            destination = base / f"prepared-{tag}"
            cls.manifests[tag] = prepared_for(config_name, cls.dataset, destination)
            cls.prepared[tag] = destination

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def _mapping(self, tag: str) -> dict:
        return json.loads((self.prepared[tag] / "label_mapping.json").read_text(encoding="utf-8"))

    def test_provenance_columns_never_reach_the_feature_matrix(self) -> None:
        for tag in ("A", "B", "C"):
            with self.subTest(experiment=tag):
                preprocessing = json.loads(
                    (self.prepared[tag] / "preprocessing.json").read_text(encoding="utf-8")
                )
                features = set(preprocessing["feature_columns_in_order"])
                self.assertFalse(features.intersection(PROVENANCE_COLUMNS))
                self.assertNotIn("Unnamed: 0", features)
                reasons = {
                    item["column"]: item["reason"] for item in preprocessing["dropped_columns"]
                }
                for column in ("Unnamed: 0", *PROVENANCE_COLUMNS):
                    self.assertEqual(reasons[column], "explicitly excluded by configuration")

    def test_experiment_a_merges_the_duplicated_label_pairs(self) -> None:
        mapping = self._mapping("A")
        self.assertFalse([label for label in mapping if label.startswith("DrDoS_")])
        self.assertIn("UDPLag", mapping)
        self.assertNotIn("UDP-lag", mapping)
        # DrDoS_LDAP and LDAP are one class now, so both days feed it.
        counts = self.manifests["A"]["split"]["class_counts"]
        total_ldap = sum(counts[split].get("LDAP", 0) for split in ("train", "validation", "test"))
        self.assertEqual(total_ldap, 300 + 280)

    def test_experiment_b_puts_one_capture_day_on_each_side(self) -> None:
        split = self.manifests["B"]["split"]
        self.assertEqual(split["strategy"], "by_capture_day")
        # 01-12 shared rows: DrDoS_LDAP + Syn + one BENIGN block per 01-12 file.
        train_day = 300 + 260 + 4 * BENIGN_ROWS
        test_day = 280 + 300 + 3 * BENIGN_ROWS
        self.assertEqual(split["sizes"]["train"] + split["sizes"]["validation"], train_day)
        self.assertEqual(split["sizes"]["test"], test_day)
        self.assertGreater(split["sizes"]["train"], split["sizes"]["validation"])
        self.assertGreater(split["sizes"]["validation"], 0)
        self.assertFalse(any(split["classes_missing_from_split"].values()))

    def test_experiment_b_audits_groups_only_inside_the_training_day(self) -> None:
        # Across capture days a repeated flow 5-tuple is a property of the network, not a
        # leak, so auditing it would fail the run for the very thing the design intends.
        split = self.manifests["B"]["split"]
        self.assertEqual(split["group_audit_scope"], ["train", "validation"])
        self.assertTrue(self.manifests["B"]["leakage_audit"]["passed"])

    def test_experiment_c_yields_a_test_split_of_nothing_but_unseen_rows(self) -> None:
        split = self.manifests["C"]["split"]
        self.assertEqual(split["open_set_labels"], ["Portmap"])
        self.assertEqual(split["open_set_only_splits"], ["test"])
        self.assertEqual(split["sizes"]["test"], 150)
        self.assertNotIn("Portmap", self._mapping("C"))
        encoded = pd.concat([
            pd.read_parquet(self.prepared["C"] / part["path"])["_label"]
            for part in self.manifests["C"]["parts"]["test"]
        ])
        self.assertEqual(sorted(encoded.unique().tolist()), [OPEN_SET_LABEL_CODE])
        self.assertGreater(split["rows_excluded_by_unassigned_split"], 0)

    def test_row_accounting_balances_across_every_experiment(self) -> None:
        for tag in ("A", "B", "C"):
            with self.subTest(experiment=tag):
                split = self.manifests[tag]["split"]
                written = sum(split["sizes"].values())
                excluded = (
                    split["rows_excluded_by_label_policy"]
                    + split["rows_excluded_by_unassigned_split"]
                )
                self.assertEqual(split["rows_read"] - excluded, written)


class ExperimentTrainingTest(unittest.TestCase):
    """The full loop for Experiment B and C, including the figures and the report."""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            import lightgbm  # noqa: F401
        except ImportError:  # pragma: no cover - environment guard
            raise unittest.SkipTest("LightGBM is required")
        cls._temporary = tempfile.TemporaryDirectory()
        cls.base = Path(cls._temporary.name)
        cls.dataset = cls.base / "dataset"
        write_two_day_dataset(cls.dataset)
        cls.runs = cls.base / "runs"
        cls.runs.mkdir()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def _train(self, tag: str, data_config: str, train_config: str, monitor: bool) -> Path:
        import train as train_module

        prepared = self.base / f"prepared-{tag}"
        if not prepared.exists():
            prepared_for(data_config, self.dataset, prepared)
        config = train_module.load_train_config(PROJECT_ROOT / "config" / train_config)
        smoke = train_module.load_train_config(PROJECT_ROOT / "config" / "train.smoke.json")
        config["dataset"]["prepared_data_dir"] = str(prepared)
        # Left at the production value on purpose: a label policy that discards rows must
        # still satisfy the "every physical row was read" gate.
        config["dataset"]["require_full_dataset_manifest"] = True
        config["dataset"]["require_safe_memory_profile"] = False
        config["dataset"]["monitor_split"] = {
            "enabled": monitor, "split": "test", "name": "crossday",
            "maximum_rows": 400, "seed": 2026,
        }
        config["model_params"]["num_threads"] = 1
        config["model_params"]["verbosity"] = -1
        config["logging"]["lightgbm_period"] = 0
        config["session"] = smoke["session"]
        config["s3"] = smoke["s3"]
        path = self.runs / f"{tag}.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        code = train_module.train(Namespace(
            config=str(path), prepared_data_dir=None, output_dir=str(self.runs),
            run_id=f"lightgbm_{tag}", max_rounds_this_session=None,
            upload_checkpoints_to_s3=False,
        ))
        self.assertEqual(code, 0)
        return self.runs / f"lightgbm_{tag}"

    def test_experiment_b_draws_the_held_out_day_as_its_own_curve(self) -> None:
        run_dir = self._train("expB", "data.expB.json", "train.expB.json", monitor=True)
        history = json.loads((run_dir / "metrics" / "history.json").read_text(encoding="utf-8"))
        self.assertEqual(len(history), 100)
        self.assertTrue(all(row["monitor_macro_f1"] is not None for row in history))
        self.assertEqual(history[-1]["monitor_name"], "crossday")
        self.assertTrue(all(row["val_macro_recall"] is not None for row in history))

        table = pd.read_csv(run_dir / "metrics" / "learning_curves.csv")
        for column in (
            "monitor_multi_logloss", "monitor_accuracy",
            "val_minus_train_multi_logloss", "monitor_minus_val_multi_logloss",
        ):
            self.assertIn(column, table.columns)
        # The point of the experiment: the in-day gap stays small while the cross-day gap
        # does not. If this ever inverts, the day split has stopped being a day split.
        self.assertGreater(
            float(table["monitor_minus_val_multi_logloss"].iloc[-1]),
            float(table["val_minus_train_multi_logloss"].iloc[-1]),
        )
        self.assertTrue((run_dir / "figures" / "learning_curves.png").exists())

        summary = pd.read_csv(run_dir / "metrics" / "summary_metrics.csv")
        self.assertEqual(list(summary.columns)[:2], ["Macro F1", "Balanced Accuracy"])
        metrics = json.loads((run_dir / "metrics" / "test_metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(metrics["primary_metrics"], ["macro_f1", "balanced_accuracy"])

        run_config = json.loads((run_dir / "config" / "run_config.json").read_text(encoding="utf-8"))
        self.assertTrue(run_config["monitoring"]["enabled"])
        self.assertFalse(run_config["monitoring"]["used_for_model_selection"])
        self.assertEqual(
            run_config["dataset_provenance"]["split_strategy"], "by_capture_day"
        )

    def test_experiment_c_reports_a_distribution_instead_of_an_accuracy(self) -> None:
        run_dir = self._train("expC", "data.expC.json", "train.expC.json", monitor=False)
        metrics = json.loads((run_dir / "metrics" / "test_metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(metrics["evaluation_mode"], "open_set")
        self.assertNotIn("accuracy", metrics)
        self.assertNotIn("macro_f1", metrics)
        self.assertEqual(metrics["open_set"]["open_set_rows"], 150)

        distribution = pd.read_csv(run_dir / "metrics" / "open_set_prediction_distribution.csv")
        mapping = json.loads((run_dir / "config" / "label_mapping.json").read_text(encoding="utf-8"))
        self.assertEqual(len(distribution), len(mapping))
        self.assertAlmostEqual(float(distribution["rows"].sum()), 150.0)
        self.assertTrue((run_dir / "figures" / "open_set_distribution.png").exists())
        # A closed-set model must still emit a probability for every unseen row, so the
        # entropy is bounded by log(number of trained classes).
        self.assertLessEqual(
            metrics["open_set"]["mean_predictive_entropy"],
            metrics["open_set"]["maximum_entropy_for_reference"] + 1e-9,
        )


if __name__ == "__main__":
    unittest.main()
