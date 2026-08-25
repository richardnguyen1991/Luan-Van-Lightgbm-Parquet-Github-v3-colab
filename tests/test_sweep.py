"""Guards for the feature-count sweep that chooses k.

The sweep's whole value rests on one property: it must never consult the test split. A
sweep that ranked candidates by `summary_metrics.csv` would look identical, run just as
fast, and silently decide the reduction using the held-out set. Most of what is asserted
here is therefore about *where* the numbers came from.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from feature_ranking import validation_scores  # noqa: E402
from sweep_feature_count import (  # noqa: E402
    build_train_config,
    run_is_complete,
    sweep,
    validation_history_scores,
)
from test_experiments import write_two_day_dataset  # noqa: E402


class SweepUnitTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self._temporary.name)
        self.addCleanup(self._temporary.cleanup)

    def _history(self, rounds: int) -> Path:
        run_dir = self.directory / "run"
        (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
        history = [{
            "iteration": index, "val_macro_f1": 0.7, "val_macro_recall": 0.68,
            "val_multi_logloss": 0.5, "val_multi_error": 0.2, "iteration_seconds": 1.5,
        } for index in range(1, rounds + 1)]
        (run_dir / "metrics" / "history.json").write_text(json.dumps(history), encoding="utf-8")
        return run_dir

    def test_only_a_finished_run_counts_as_complete(self) -> None:
        self.assertFalse(run_is_complete(self.directory / "missing"))
        self.assertFalse(run_is_complete(self._history(99)))
        self.assertTrue(run_is_complete(self._history(100)))

    def test_scores_are_read_from_the_validation_columns_of_history(self) -> None:
        scores = validation_history_scores(self._history(100))
        self.assertEqual(scores["val_macro_f1"], 0.7)
        self.assertEqual(scores["val_balanced_accuracy"], 0.68)
        self.assertAlmostEqual(scores["val_accuracy"], 0.8)
        self.assertAlmostEqual(scores["training_seconds"], 150.0)

    def test_an_unfinished_run_is_refused_rather_than_scored(self) -> None:
        with self.assertRaisesRegex(ValueError, "iteration 100"):
            validation_history_scores(self._history(60))

    def test_the_baseline_config_selects_nothing_and_each_k_selects_by_ranking(self) -> None:
        source = json.loads((PROJECT_ROOT / "config" / "train.json").read_text(encoding="utf-8"))
        baseline = build_train_config(source, Path("prepared"), None, None)
        self.assertEqual(baseline["feature_selection"], "none")
        reduced = build_train_config(source, Path("prepared"), Path("rank.json"), 20)
        self.assertEqual(reduced["feature_selection"], "validation_permutation_top_k")
        self.assertEqual(reduced["dataset"]["feature_screening"]["maximum_features"], 20)
        self.assertEqual(
            reduced["dataset"]["feature_screening"]["ranking_file"], "rank.json"
        )
        # The source config must not be mutated: the sweep builds many configs from one.
        self.assertEqual(source["feature_selection"], "none")
        self.assertNotIn("ranking_file", source["dataset"].get("feature_screening", {}))


class SweepEndToEndTest(unittest.TestCase):
    """One real sweep, small enough to run in the suite."""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            import lightgbm  # noqa: F401
        except ImportError:  # pragma: no cover - environment guard
            raise unittest.SkipTest("LightGBM is required")
        cls._temporary = tempfile.TemporaryDirectory()
        base = Path(cls._temporary.name)
        cls.dataset = base / "dataset"
        write_two_day_dataset(cls.dataset)

        # A data config pointed at the fixture, and a train config small enough to be fast.
        cls.data_config = base / "data.json"
        data = json.loads((PROJECT_ROOT / "config" / "data.json").read_text(encoding="utf-8"))
        data["dataset"]["data_dir"] = str(cls.dataset)
        data["output"]["rows_per_part"] = 256
        data["audit"]["identity_sample_divisor"] = 1
        cls.data_config.write_text(json.dumps(data), encoding="utf-8")

        cls.train_config = base / "train.json"
        train = json.loads((PROJECT_ROOT / "config" / "train.json").read_text(encoding="utf-8"))
        smoke = json.loads((PROJECT_ROOT / "config" / "train.smoke.json").read_text(encoding="utf-8"))
        train["dataset"]["require_safe_memory_profile"] = False
        train["model_params"]["num_threads"] = 1
        train["model_params"]["verbosity"] = -1
        train["logging"]["lightgbm_period"] = 0
        train["session"] = smoke["session"]
        train["s3"] = smoke["s3"]
        cls.train_config.write_text(json.dumps(train), encoding="utf-8")

        cls.output_root = base / "sweep"
        cls.table = sweep(
            cls.data_config, cls.train_config, cls.output_root,
            target_total_rows=2000, candidates=[3, 2],
            tolerance_macro_f1=0.005, tolerance_class_recall=0.02, scoring_rows=0,
        )
        cls.decision = json.loads(
            (cls.output_root / "sweep_feature_count.json").read_text(encoding="utf-8")
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def test_the_sweep_covers_the_baseline_and_every_candidate(self) -> None:
        self.assertEqual(list(self.table["k"]), [4, 3, 2])
        self.assertTrue(self.table.iloc[0]["is_baseline"])
        self.assertFalse(self.table.iloc[1:]["is_baseline"].any())
        self.assertEqual(list(self.table["feature_count"]), [4, 3, 2])

    def test_every_reported_number_comes_from_validation(self) -> None:
        self.assertEqual(self.decision["scored_split"], "validation")
        manifest = json.loads(
            (self.output_root / "data" / "sample_manifest.json").read_text(encoding="utf-8")
        )
        validation_rows = manifest["split"]["sizes"]["validation"]
        test_rows = manifest["split"]["sizes"]["test"]
        self.assertNotEqual(validation_rows, test_rows, "fixture cannot distinguish the splits")
        for rows in self.table["scored_rows"]:
            self.assertEqual(int(rows), validation_rows)

    def test_the_sweep_never_reads_the_test_metrics(self) -> None:
        # The figures in the sweep table must differ from the test-split report, otherwise
        # the assertion above proves nothing about which file was read.
        baseline_dir = self.output_root / "lightgbm_baseline"
        summary = pd.read_csv(baseline_dir / "metrics" / "summary_metrics.csv")
        from_validation = float(self.table.iloc[0]["val_macro_f1"])
        self.assertAlmostEqual(
            from_validation,
            validation_scores(baseline_dir, self.output_root / "data")["macro_f1"],
            places=6,
        )
        self.assertNotAlmostEqual(from_validation, float(summary["Macro F1"][0]), places=9)

    def test_the_ranking_is_produced_once_and_reused_by_every_k(self) -> None:
        ranking_path = self.output_root / "lightgbm_baseline" / "config" / "feature_ranking_validation.json"
        self.assertTrue(ranking_path.exists())
        self.assertEqual(self.decision["ranking_file"], str(ranking_path))
        for k in (3, 2):
            selection = json.loads(
                (self.output_root / f"lightgbm_k{k:03d}" / "config" / "feature_selection.json")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(selection["method"], "validation_permutation_top_k")
            self.assertEqual(selection["ranking_file"], str(ranking_path))
            self.assertEqual(selection["selected_feature_count"], k)

    def test_the_decision_records_the_tolerances_it_was_given(self) -> None:
        self.assertEqual(self.decision["tolerance_macro_f1"], 0.005)
        self.assertEqual(self.decision["tolerance_class_recall"], 0.02)
        self.assertEqual(self.decision["candidates"], [3, 2])
        self.assertIn("smallest accepted k", self.decision["rule"])
        chosen = self.decision["chosen_k"]
        accepted = [int(row["k"]) for row in self.decision["results"]
                    if not row["is_baseline"] and row["accepted"]]
        self.assertEqual(chosen, min(accepted) if accepted else None)

    def test_deltas_are_measured_against_the_full_feature_baseline(self) -> None:
        baseline_f1 = float(self.table.iloc[0]["val_macro_f1"])
        for _, row in self.table.iloc[1:].iterrows():
            self.assertAlmostEqual(
                float(row["delta_macro_f1"]), baseline_f1 - float(row["val_macro_f1"]), places=9
            )
            self.assertIn(row["worst_class"], self.decision["results"][0]["per_class_recall"]
                          if "per_class_recall" in self.decision["results"][0] else
                          [row["worst_class"]])

    def test_a_second_sweep_reuses_the_finished_runs(self) -> None:
        marker = self.output_root / "lightgbm_baseline" / "metrics" / "history.json"
        before = marker.stat().st_mtime_ns
        repeated = sweep(
            self.data_config, self.train_config, self.output_root,
            target_total_rows=2000, candidates=[3, 2],
            tolerance_macro_f1=0.005, tolerance_class_recall=0.02, scoring_rows=0,
        )
        self.assertEqual(marker.stat().st_mtime_ns, before, "a finished run was retrained")
        np.testing.assert_allclose(repeated["val_macro_f1"], self.table["val_macro_f1"])


if __name__ == "__main__":
    unittest.main()
