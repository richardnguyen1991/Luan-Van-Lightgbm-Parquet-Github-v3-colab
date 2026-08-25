"""Guards for the one learning parameter allowed to vary, and the run ids that keep it honest.

The 14-class baseline run made the training loss non-monotonic after round 40 - it swung
between roughly 0.4 and 0.8 for the remaining sixty rounds - which makes the model
delivered at the fixed hundredth round a matter of luck. Gradient quantization is the
suspect, so it has to be adjustable for that to be measurable. Everything else in the
baseline contract stays locked, and each variant trains under its own run id so a rerun
cannot resume onto a checkpoint fitted with different gradients.
"""

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model import ALLOWED_GRAD_QUANT_BINS, validate_training_config  # noqa: E402
from train import (  # noqa: E402
    GRADIENT_QUANTIZATION_VARIANTS,
    apply_gradient_quantization,
    check_variant_matches_run,
    load_train_config,
    run_id_for_variant,
    variant_from_run_id,
)


def base_config() -> dict:
    return json.loads((PROJECT_ROOT / "config" / "train.json").read_text(encoding="utf-8"))


class ContractTest(unittest.TestCase):
    def test_the_shipped_configs_still_declare_the_baseline_quantization(self) -> None:
        for name in ("train.json", "train.gha.json", "train.smoke.json",
                     "train.expB.json", "train.expC.json"):
            with self.subTest(config=name):
                params = load_train_config(PROJECT_ROOT / "config" / name)["model_params"]
                self.assertTrue(params["use_quantized_grad"])
                self.assertEqual(params["num_grad_quant_bins"], 16)

    def test_turning_quantization_off_is_accepted(self) -> None:
        config = base_config()
        config["model_params"]["use_quantized_grad"] = False
        validate_training_config(config)

    def test_every_allowed_bin_count_is_accepted(self) -> None:
        for bins in ALLOWED_GRAD_QUANT_BINS:
            with self.subTest(bins=bins):
                config = base_config()
                config["model_params"]["num_grad_quant_bins"] = bins
                validate_training_config(config)

    def test_a_bin_count_outside_the_list_is_refused(self) -> None:
        for bins in (0, 3, 17, 128, "16", None):
            with self.subTest(bins=bins):
                config = base_config()
                config["model_params"]["num_grad_quant_bins"] = bins
                with self.assertRaisesRegex(ValueError, "num_grad_quant_bins"):
                    validate_training_config(config)

    def test_a_non_boolean_quantization_flag_is_refused(self) -> None:
        config = base_config()
        config["model_params"]["use_quantized_grad"] = "true"
        with self.assertRaisesRegex(ValueError, "use_quantized_grad"):
            validate_training_config(config)

    def test_renew_leaf_is_required_only_while_quantization_is_on(self) -> None:
        config = base_config()
        config["model_params"]["quant_train_renew_leaf"] = False
        with self.assertRaisesRegex(ValueError, "quant_train_renew_leaf"):
            validate_training_config(config)
        # With quantization off the parameter is inert, so it stops being a contract term.
        config["model_params"]["use_quantized_grad"] = False
        validate_training_config(config)

    def test_the_rest_of_the_learning_contract_is_still_locked(self) -> None:
        for key, value in (
            ("learning_rate", 0.1), ("num_leaves", 63), ("max_bin", 511),
            ("min_data_in_leaf", 5), ("bagging_fraction", 0.8), ("lambda_l2", 1.0),
        ):
            with self.subTest(parameter=key):
                config = base_config()
                config["model_params"][key] = value
                with self.assertRaises(ValueError):
                    validate_training_config(config)


class VariantTest(unittest.TestCase):
    def test_each_variant_sets_what_its_name_says(self) -> None:
        expected = {
            "as-configured": (True, 16),
            "off": (False, 16),
            "bins-32": (True, 32),
        }
        for variant, (quantized, bins) in expected.items():
            with self.subTest(variant=variant):
                config = load_train_config(PROJECT_ROOT / "config" / "train.json")
                apply_gradient_quantization(config, variant)
                params = config["model_params"]
                self.assertEqual(params["use_quantized_grad"], quantized)
                self.assertEqual(params["num_grad_quant_bins"], bins)

    def test_an_unknown_variant_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "gradient-quantization"):
            apply_gradient_quantization(base_config(), "bins-13")

    def test_only_the_baseline_variant_leaves_the_run_id_alone(self) -> None:
        base = "lightgbm_a_1225d5b5f1b142ca"
        ids = {
            variant: run_id_for_variant(base, suffix)
            for variant, (_, suffix) in GRADIENT_QUANTIZATION_VARIANTS.items()
        }
        self.assertEqual(ids["as-configured"], base)
        self.assertEqual(ids["off"], base + "_gq-off")
        self.assertEqual(ids["bins-32"], base + "_gq32")
        # Distinct ids are the whole point: variants must never share a checkpoint.
        self.assertEqual(len(set(ids.values())), len(ids))

    def test_appending_a_suffix_is_idempotent(self) -> None:
        once = run_id_for_variant("lightgbm_a_x", "gq-off")
        self.assertEqual(run_id_for_variant(once, "gq-off"), once)

    def test_a_run_id_reports_the_variant_it_was_created_for(self) -> None:
        self.assertEqual(variant_from_run_id("lightgbm_a_x"), "as-configured")
        self.assertEqual(variant_from_run_id("lightgbm_a_x_gq-off"), "off")
        self.assertEqual(variant_from_run_id("lightgbm_a_x_gq32"), "bins-32")

    def test_continuing_a_run_under_the_wrong_variant_is_refused_by_name(self) -> None:
        # The GitHub Actions fallback resolves the run id from S3 rather than being told
        # it, so this is the realistic way the two drift apart. The message has to say
        # which flag to pass, not just that a hash differs.
        with self.assertRaises(ValueError) as caught:
            check_variant_matches_run("lightgbm_a_x_gq-off", "as-configured")
        self.assertIn("--gradient-quantization off", str(caught.exception))
        with self.assertRaises(ValueError):
            check_variant_matches_run("lightgbm_a_x", "off")

    def test_a_matching_variant_passes(self) -> None:
        for run_id, variant in (
            ("lightgbm_a_x", "as-configured"),
            ("lightgbm_a_x_gq-off", "off"),
            ("lightgbm_a_x_gq32", "bins-32"),
        ):
            with self.subTest(run_id=run_id):
                check_variant_matches_run(run_id, variant)


class NotebookAndWorkflowTest(unittest.TestCase):
    def test_the_notebook_exposes_the_variant_and_passes_it_to_train(self) -> None:
        notebook = json.loads(
            (PROJECT_ROOT / "colab_runner.ipynb").read_text(encoding="utf-8")
        )
        # Selected by its #@title, not by mentioning train.py: cell 2's comments do too.
        train_cell = next(
            "".join(cell["source"]) for cell in notebook["cells"]
            if cell["cell_type"] == "code"
            and "".join(cell["source"]).startswith("#@title 4.")
        )
        self.assertIn('GRADIENT_QUANTIZATION = "off"', train_cell)
        self.assertIn('"--gradient-quantization", GRADIENT_QUANTIZATION', train_cell)
        # The printed run id must already carry the suffix, so what the cell reports is
        # what the checkpoints are actually written under.
        self.assertIn("run_id_for_variant", train_cell)

    def test_the_fallback_worker_can_be_told_which_variant_to_continue(self) -> None:
        workflow = (PROJECT_ROOT / ".github" / "workflows" / "fallback-worker.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("gradient_quantization:", workflow)
        self.assertIn(
            '--gradient-quantization "${{ inputs.gradient_quantization }}"', workflow
        )


if __name__ == "__main__":
    unittest.main()
