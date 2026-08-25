"""Guards for permutation importance measured on validation, and the selector it feeds.

The ranking exists so feature reduction can be decided without consulting the test split.
Most of what is worth testing here is therefore about refusal: a ranking scored anywhere
but validation, a ranking from a different preprocessing recipe, or a selector asked to
cut without one must all fail loudly rather than quietly bias the reported numbers.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import load_config, prepare_dataset  # noqa: E402
from feature_ranking import (  # noqa: E402
    RANKING_FORMAT_VERSION,
    load_ranking,
    validation_permutation_importance,
    write_ranking,
)
from model import (  # noqa: E402
    SUPPORTED_FEATURE_SELECTION,
    select_model_features,
    validate_training_config,
)
from test_experiments import write_two_day_dataset  # noqa: E402

FEATURES = ["Feature A", "Feature B", "Feature C", "Feature D"]


def ranking_payload(order=FEATURES, split="validation", version=RANKING_FORMAT_VERSION):
    return {
        "format_version": version,
        "method": "permutation_importance",
        "scored_split": split,
        "scored_by": "macro_f1",
        "run_id": "lightgbm_test",
        "repeats": 5,
        "seed": 2026,
        "baseline_macro_f1": 0.9,
        "features_within_noise": 0,
        "split_rows_scored": 1000,
        "ranking": [
            {
                "rank": index + 1, "feature": name,
                "mean_decrease_macro_f1": 1.0 / (index + 1),
                "std_decrease_macro_f1": 0.001, "within_noise": False,
            }
            for index, name in enumerate(order)
        ],
    }


def write_payload(directory: Path, payload: dict, name: str = "ranking.json") -> Path:
    path = directory / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def selection_config(ranking_file, k):
    return {
        "feature_selection": "validation_permutation_top_k",
        "dataset": {"feature_screening": {"maximum_features": k, "ranking_file": str(ranking_file)}},
    }


class RankingFileTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self._temporary.name)
        self.addCleanup(self._temporary.cleanup)

    def test_a_valid_validation_ranking_loads(self) -> None:
        path = write_payload(self.directory, ranking_payload())
        self.assertEqual(load_ranking(path)["scored_split"], "validation")

    def test_a_ranking_scored_on_test_is_refused(self) -> None:
        # The whole reason this module exists: selecting on a test-derived ranking folds
        # the held-out split into the decision and inflates everything reported after it.
        path = write_payload(self.directory, ranking_payload(split="test"))
        with self.assertRaisesRegex(ValueError, "validation split"):
            load_ranking(path)

    def test_a_ranking_scored_on_train_is_refused(self) -> None:
        path = write_payload(self.directory, ranking_payload(split="train"))
        with self.assertRaisesRegex(ValueError, "validation split"):
            load_ranking(path)

    def test_an_unknown_format_version_is_refused(self) -> None:
        path = write_payload(self.directory, ranking_payload(version=99))
        with self.assertRaisesRegex(ValueError, "format version"):
            load_ranking(path)

    def test_an_empty_ranking_is_refused(self) -> None:
        payload = ranking_payload()
        payload["ranking"] = []
        with self.assertRaisesRegex(ValueError, "no ranking"):
            load_ranking(write_payload(self.directory, payload))


class PermutationSelectorTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self._temporary.name)
        self.addCleanup(self._temporary.cleanup)

    def _select(self, payload, k, candidates=FEATURES):
        path = write_payload(self.directory, payload)
        return select_model_features(
            None, self.directory, [], list(candidates), np.zeros(1, dtype=np.int32), {},
            selection_config(path, k),
        )

    def test_top_k_is_taken_from_the_ranking_and_kept_in_candidate_order(self) -> None:
        # Ranked B, D, A, C; the model must still see them in preprocessing order.
        payload = ranking_payload(["Feature B", "Feature D", "Feature A", "Feature C"])
        selected, summary = self._select(payload, 3)
        self.assertEqual(selected, ["Feature A", "Feature B", "Feature D"])
        self.assertEqual(summary["method"], "validation_permutation_top_k")
        self.assertEqual(summary["fit_split"], "validation")
        self.assertEqual(summary["selected_feature_count"], 3)
        self.assertEqual(summary["candidate_feature_count"], 4)

    def test_a_ranking_missing_a_candidate_is_refused(self) -> None:
        payload = ranking_payload(FEATURES[:-1])
        with self.assertRaisesRegex(ValueError, "does not cover every candidate"):
            self._select(payload, 2)

    def test_a_ranking_naming_an_absent_column_is_refused(self) -> None:
        # This is what a ranking from a different preprocessing recipe looks like.
        payload = ranking_payload(FEATURES + ["Unnamed: 0"])
        with self.assertRaisesRegex(ValueError, "does not have"):
            self._select(payload, 2)

    def test_k_outside_the_candidate_range_is_refused(self) -> None:
        for k in (0, len(FEATURES) + 1):
            with self.subTest(k=k), self.assertRaises(ValueError):
                self._select(ranking_payload(), k)

    def test_the_method_must_be_given_a_ranking_file(self) -> None:
        with self.assertRaisesRegex(ValueError, "ranking_file"):
            select_model_features(
                None, self.directory, [], list(FEATURES), np.zeros(1, dtype=np.int32), {},
                {"feature_selection": "validation_permutation_top_k",
                 "dataset": {"feature_screening": {"maximum_features": 2}}},
            )

    def test_the_training_contract_accepts_the_method_only_with_a_ranking_file(self) -> None:
        self.assertIn("validation_permutation_top_k", SUPPORTED_FEATURE_SELECTION)
        config = json.loads((PROJECT_ROOT / "config" / "train.json").read_text(encoding="utf-8"))
        config["feature_selection"] = "validation_permutation_top_k"
        config["dataset"]["feature_screening"] = {"maximum_features": 20}
        with self.assertRaisesRegex(ValueError, "ranking_file"):
            validate_training_config(config)
        config["dataset"]["feature_screening"]["ranking_file"] = "config/ranking.json"
        validate_training_config(config)

    def test_an_unknown_selection_method_is_still_refused(self) -> None:
        config = json.loads((PROJECT_ROOT / "config" / "train.json").read_text(encoding="utf-8"))
        config["feature_selection"] = "shap_top_k"
        with self.assertRaises(ValueError):
            validate_training_config(config)


class ValidationPermutationEndToEndTest(unittest.TestCase):
    """Train on every feature, rank on validation, retrain on the top k."""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            import lightgbm  # noqa: F401
        except ImportError:  # pragma: no cover - environment guard
            raise unittest.SkipTest("LightGBM is required")
        import train as train_module

        cls._temporary = tempfile.TemporaryDirectory()
        base = Path(cls._temporary.name)
        dataset = base / "dataset"
        write_two_day_dataset(dataset)
        cls.prepared = base / "prepared"
        config = load_config(PROJECT_ROOT / "config" / "data.json")
        config["dataset"]["data_dir"] = str(dataset)
        config["output"]["rows_per_part"] = 256
        config["audit"]["identity_sample_divisor"] = 1
        cls.manifest = prepare_dataset(config, cls.prepared)
        cls.runs = base / "runs"
        cls.runs.mkdir()
        cls.settings = {
            "maximum_rows": 400, "repeats": 3, "seed": 2026, "predict_chunk_rows": 250000,
        }

        cls.full_dir = cls._train(train_module, "full")
        cls.table, cls.provenance = validation_permutation_importance(
            cls.full_dir, cls.prepared, cls.settings
        )
        cls.ranking_path = write_ranking(cls.full_dir, cls.table, cls.provenance)[1]

    @classmethod
    def _train(cls, train_module, tag, selection=None, ranking_file=None, k=None) -> Path:
        config = train_module.load_train_config(PROJECT_ROOT / "config" / "train.json")
        smoke = train_module.load_train_config(PROJECT_ROOT / "config" / "train.smoke.json")
        config["dataset"]["prepared_data_dir"] = str(cls.prepared)
        config["dataset"]["require_full_dataset_manifest"] = False
        config["dataset"]["require_safe_memory_profile"] = False
        if selection:
            config["feature_selection"] = selection
            config["dataset"]["feature_screening"] = {
                "maximum_features": k, "ranking_file": str(ranking_file), "seed": 2026,
            }
        config["model_params"]["num_threads"] = 1
        config["model_params"]["verbosity"] = -1
        config["logging"]["lightgbm_period"] = 0
        config["session"] = smoke["session"]
        config["s3"] = smoke["s3"]
        path = cls.runs / f"{tag}.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        code = train_module.train(Namespace(
            config=str(path), prepared_data_dir=None, output_dir=str(cls.runs),
            run_id=f"lightgbm_{tag}", max_rounds_this_session=None,
            upload_checkpoints_to_s3=False,
        ))
        if code != 0:
            raise AssertionError(f"training {tag} exited {code}")
        return cls.runs / f"lightgbm_{tag}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def test_the_ranking_is_measured_on_validation_and_covers_every_feature(self) -> None:
        run_config = json.loads(
            (self.full_dir / "config" / "run_config.json").read_text(encoding="utf-8")
        )
        self.assertEqual(sorted(self.table["feature"]), sorted(run_config["feature_names"]))
        self.assertEqual(self.provenance["scored_split"], "validation")
        self.assertEqual(
            self.provenance["split_rows_available"],
            self.manifest["split"]["sizes"]["validation"],
        )
        self.assertGreater(self.provenance["baseline_macro_f1"], 0.0)
        self.assertLessEqual(self.provenance["baseline_macro_f1"], 1.0)

    def test_the_table_is_ordered_by_the_damage_each_feature_does_when_destroyed(self) -> None:
        decreases = list(self.table["mean_decrease_macro_f1"])
        self.assertEqual(decreases, sorted(decreases, reverse=True))
        self.assertEqual(list(self.table["rank"]), list(range(1, len(self.table) + 1)))

    def test_the_same_seed_reproduces_the_same_ranking(self) -> None:
        repeated, _ = validation_permutation_importance(
            self.full_dir, self.prepared, self.settings
        )
        self.assertEqual(list(repeated["feature"]), list(self.table["feature"]))
        np.testing.assert_allclose(
            repeated["mean_decrease_macro_f1"], self.table["mean_decrease_macro_f1"]
        )

    def test_a_reduced_run_trains_on_exactly_the_top_of_that_ranking(self) -> None:
        import train as train_module

        k = len(self.table) - 1
        reduced = self._train(
            train_module, "topk", "validation_permutation_top_k", self.ranking_path, k
        )
        selection = json.loads(
            (reduced / "config" / "feature_selection.json").read_text(encoding="utf-8")
        )
        self.assertEqual(selection["method"], "validation_permutation_top_k")
        self.assertEqual(selection["fit_split"], "validation")
        self.assertEqual(selection["selected_feature_count"], k)
        self.assertEqual(
            set(selection["selected_features_in_model_order"]),
            {item["feature"] for item in self.provenance["ranking"][:k]},
        )
        run_config = json.loads(
            (reduced / "config" / "run_config.json").read_text(encoding="utf-8")
        )
        self.assertEqual(run_config["feature_count"], k)
        # Both runs still produce a comparable headline number, which is the point: the
        # reduction is accepted or rejected by comparing these two.
        for directory in (self.full_dir, reduced):
            summary = pd.read_csv(directory / "metrics" / "summary_metrics.csv")
            self.assertEqual(list(summary.columns)[:2], ["Macro F1", "Balanced Accuracy"])


if __name__ == "__main__":
    unittest.main()
