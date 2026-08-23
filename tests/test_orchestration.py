"""Contract tests for the Colab notebook, the watchdog, and the shared dataset key."""

import json
import sys
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_colab_notebook import build_notebook  # noqa: E402
from colab_orchestrator import (  # noqa: E402
    colab_url,
    decide_next_action,
    orchestration_state_after_action,
)
from data import compute_data_version  # noqa: E402

ORCHESTRATION = json.loads(
    (PROJECT_ROOT / "config" / "orchestration.json").read_text(encoding="utf-8")
)
NOW = time.time()


def minutes_ago(minutes: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()


class ColabNotebookTest(unittest.TestCase):
    def test_checked_in_notebook_matches_the_builder(self):
        checked_in = json.loads((PROJECT_ROOT / "colab_runner.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(checked_in, build_notebook())

    def test_notebook_is_cpu_only_and_carries_no_credentials(self):
        notebook = json.loads((PROJECT_ROOT / "colab_runner.ipynb").read_text(encoding="utf-8"))
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        self.assertEqual(notebook["metadata"]["accelerator"], "None")
        self.assertIn("device=CPU", source)
        self.assertIn("from google.colab import userdata", source)
        self.assertIn("PIPELINE_SESSION_DEADLINE_EPOCH", source)
        self.assertIn("--upload-checkpoints-to-s3", source)
        self.assertIn('"git", "clone"', source)
        self.assertNotIn("AKIA", source)
        self.assertNotIn("AWS_SECRET_ACCESS_KEY=", source)

    def test_notebook_clones_the_repository_named_by_the_orchestration_config(self):
        notebook = json.loads((PROJECT_ROOT / "colab_runner.ipynb").read_text(encoding="utf-8"))
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        self.assertIn(ORCHESTRATION["repository"], source)
        self.assertIn(colab_url(ORCHESTRATION), source)

    def test_preprocessing_is_given_the_same_session_budget_as_training(self):
        """Without a deadline, data.py runs until Colab kills it mid-source-file."""
        notebook = json.loads((PROJECT_ROOT / "colab_runner.ipynb").read_text(encoding="utf-8"))
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        self.assertIn("--maximum-hours", source)
        self.assertIn("--stop-before-minutes", source)
        self.assertIn('session["maximum_hours"]', source)

    def test_notebook_names_the_kaggle_dataset_actually_used(self):
        notebook = json.loads((PROJECT_ROOT / "colab_runner.ipynb").read_text(encoding="utf-8"))
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        self.assertIn("dungnguyen28101991/cicddos2019-parquet", source)


class ConfigurationContractTest(unittest.TestCase):
    def test_training_stays_on_cpu_with_the_fixed_baseline_contract(self):
        train = json.loads((PROJECT_ROOT / "config" / "train.json").read_text(encoding="utf-8"))
        self.assertEqual(train["device"], "cpu")
        self.assertEqual(train["model_params"]["device_type"], "cpu")
        self.assertEqual(train["num_boost_round"], 100)
        self.assertFalse(train["early_stopping"])
        self.assertEqual(train["imbalance_handling"], "none")
        self.assertEqual(train["model_params"]["learning_rate"], 0.05)
        self.assertTrue(train["model_params"]["deterministic"])
        self.assertTrue(train["model_params"]["force_col_wise"])

    def test_runner_profile_differs_only_in_resource_parameters(self):
        colab = json.loads((PROJECT_ROOT / "config" / "train.json").read_text(encoding="utf-8"))
        runner = json.loads((PROJECT_ROOT / "config" / "train.gha.json").read_text(encoding="utf-8"))
        for key, value in colab["model_params"].items():
            if key == "num_threads":
                # LightGBM documents deterministic=true as stable across num_threads,
                # so the two workers may size their thread pools differently.
                continue
            self.assertEqual(runner["model_params"][key], value, key)
        self.assertLess(runner["session"]["maximum_hours"], colab["session"]["maximum_hours"])
        self.assertIsNotNone(runner["session"]["minimum_available_ram_gb"])
        self.assertIsNotNone(colab["session"]["minimum_available_ram_gb"])
        # Screening decides which features reach the model, so it belongs to the learning
        # contract rather than to the resource budget: a run that migrates between the two
        # workers mid-training would otherwise screen a different feature set and abort on
        # the feature_schema_hash guard instead of resuming.
        self.assertEqual(runner["feature_selection"], colab["feature_selection"])
        self.assertEqual(
            runner["dataset"]["feature_screening"], colab["dataset"]["feature_screening"]
        )

    def test_memory_guard_is_configured_wherever_it_is_required(self):
        for name in ("train.json", "train.gha.json", "train.smoke.json"):
            config = json.loads((PROJECT_ROOT / "config" / name).read_text(encoding="utf-8"))
            dataset = config["dataset"]
            if dataset.get("require_safe_memory_profile"):
                self.assertIn("memory_guard", dataset, name)
                self.assertGreater(dataset["memory_guard"]["available_ram_fraction"], 0.0, name)
                self.assertLessEqual(dataset["memory_guard"]["available_ram_fraction"], 1.0, name)

    def test_full_dataset_audit_backend_actually_inspects_data(self):
        data = json.loads((PROJECT_ROOT / "config" / "data.json").read_text(encoding="utf-8"))
        self.assertNotEqual(data["audit"]["backend"], "deterministic_proof")
        self.assertEqual(data["audit"]["backend"], "sampled_exact")
        self.assertGreater(data["audit"]["identity_sample_divisor"], 0)
        self.assertTrue(data["audit"]["fail_on_cross_split_overlap"])
        self.assertTrue(data["audit"]["fail_on_group_cross_split_overlap"])

    def test_data_config_points_at_the_colab_runtime_path(self):
        data = json.loads((PROJECT_ROOT / "config" / "data.json").read_text(encoding="utf-8"))
        self.assertTrue(str(data["dataset"]["data_dir"]).startswith("/content/"))
        self.assertIsNone(data["dataset"]["samples_per_file"])
        self.assertIsNone(data["dataset"]["target_total_rows"])

    def test_orchestration_config_has_every_watchdog_threshold(self):
        for key in (
            "repository", "colab_notebook_path", "target_iteration", "heartbeat_stale_minutes",
            "fallback_after_minutes", "recent_fallback_guard_minutes",
            "maximum_session_attempts", "maximum_stagnant_restarts", "maximum_report_restarts",
        ):
            self.assertIn(key, ORCHESTRATION)
        self.assertLess(
            ORCHESTRATION["heartbeat_stale_minutes"], ORCHESTRATION["fallback_after_minutes"]
        )


class WorkflowTest(unittest.TestCase):
    def test_watchdog_decides_notifies_and_dispatches_the_fallback(self):
        workflow = (PROJECT_ROOT / ".github" / "workflows" / "watchdog.yml").read_text(encoding="utf-8")
        self.assertIn("scripts/colab_orchestrator.py decide", workflow)
        self.assertIn("scripts/colab_orchestrator.py record-action", workflow)
        self.assertIn("gh workflow run fallback-worker.yml", workflow)
        self.assertIn("colab-watchdog", workflow)
        self.assertIn("AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}", workflow)
        self.assertIn("GITHUB_STEP_SUMMARY", workflow)
        self.assertNotIn("kaggle", workflow.casefold())

    def test_fallback_worker_shares_one_concurrency_group_and_tolerates_exit_75(self):
        workflow = (PROJECT_ROOT / ".github" / "workflows" / "fallback-worker.yml").read_text(encoding="utf-8")
        self.assertIn("group: lightgbm-v3-worker", workflow)
        self.assertIn("cancel-in-progress: false", workflow)
        self.assertIn("config/train.gha.json", workflow)
        self.assertIn('"75"', workflow)
        self.assertIn("scripts/sync_dataset.py pull", workflow)
        self.assertIn("make_report.py", workflow)
        self.assertIn("PIPELINE_SESSION_DEADLINE_EPOCH", workflow)


class DatasetVersionTest(unittest.TestCase):
    def base_config(self) -> dict:
        return json.loads((PROJECT_ROOT / "config" / "data.json").read_text(encoding="utf-8"))

    def test_version_ignores_where_the_raw_data_is_mounted(self):
        colab = self.base_config()
        runner = self.base_config()
        runner["dataset"]["data_dir"] = "/home/runner/work/raw"
        self.assertEqual(compute_data_version(colab), compute_data_version(runner))

    def test_version_changes_when_the_recipe_changes(self):
        original = self.base_config()
        modified = self.base_config()
        modified["split"]["seed"] = original["split"]["seed"] + 1
        self.assertNotEqual(compute_data_version(original), compute_data_version(modified))

    def test_explicit_override_wins(self):
        config = self.base_config()
        config["dataset"]["data_version_override"] = "manual-version"
        self.assertEqual(compute_data_version(config), "manual-version")


class WatchdogDecisionTest(unittest.TestCase):
    config = {
        "target_iteration": 100,
        "heartbeat_stale_minutes": 30,
        "fallback_after_minutes": 90,
        "recent_fallback_guard_minutes": 45,
        "maximum_session_attempts": 200,
        "maximum_stagnant_restarts": 3,
        "maximum_report_restarts": 12,
        "run_id": None,
    }

    def decide(self, active, state=None, force=None):
        return decide_next_action(active, state or {}, self.config, NOW, force)

    def test_live_colab_worker_is_left_alone(self):
        decision = self.decide({
            "run_id": "r1", "status": "running", "current_iteration": 40,
            "worker": "colab", "updated_at": minutes_ago(4),
        })
        self.assertEqual(decision.action, "wait")
        self.assertEqual(decision.worker_status, "alive")

    def test_recently_stalled_worker_only_notifies(self):
        decision = self.decide({
            "run_id": "r1", "status": "running", "current_iteration": 40,
            "worker": "colab", "updated_at": minutes_ago(45),
        })
        self.assertEqual(decision.action, "notify")
        self.assertEqual(decision.worker_status, "stale")

    def test_long_silence_starts_the_github_fallback(self):
        decision = self.decide({
            "run_id": "r1", "status": "running", "current_iteration": 40,
            "worker": "colab", "updated_at": minutes_ago(120),
        })
        self.assertEqual(decision.action, "fallback_train")

    def test_paused_run_is_resumable_by_the_fallback(self):
        decision = self.decide({
            "run_id": "r1", "status": "paused", "current_iteration": 40,
            "worker": "colab", "updated_at": minutes_ago(120),
        })
        self.assertEqual(decision.action, "fallback_train")

    def test_recent_fallback_guard_downgrades_to_notify(self):
        decision = self.decide(
            {
                "run_id": "r1", "status": "running", "current_iteration": 40,
                "worker": "colab", "updated_at": minutes_ago(120),
            },
            {"last_fallback_at": minutes_ago(10), "last_observed_iteration": 40,
             "last_active_status": "running"},
        )
        self.assertEqual(decision.action, "notify")

    def test_iteration_100_without_a_report_runs_the_report(self):
        decision = self.decide({
            "run_id": "r1", "status": "ready_for_report", "current_iteration": 100,
            "worker": "colab", "updated_at": minutes_ago(60),
        })
        self.assertEqual(decision.action, "fallback_report")

    def test_completed_run_stops_the_loop(self):
        decision = self.decide({
            "run_id": "r1", "status": "complete", "current_iteration": 100,
            "worker": "colab", "updated_at": minutes_ago(60),
        })
        self.assertEqual(decision.action, "complete")

    def test_report_budget_is_bounded(self):
        decision = self.decide(
            {"run_id": "r1", "status": "ready_for_report", "current_iteration": 100,
             "worker": "colab", "updated_at": minutes_ago(60)},
            {"report_restarts": 12, "last_observed_iteration": 100,
             "last_active_status": "ready_for_report"},
        )
        self.assertEqual(decision.action, "stop")

    def test_stagnation_budget_stops_pointless_restarts(self):
        decision = self.decide(
            {"run_id": "r1", "status": "running", "current_iteration": 40,
             "worker": "colab", "updated_at": minutes_ago(200)},
            {"stagnant_restarts": 3, "last_observed_iteration": 40,
             "last_active_status": "running"},
        )
        self.assertEqual(decision.action, "stop")

    def test_durable_progress_releases_the_stagnation_lock(self):
        decision = self.decide(
            {"run_id": "r1", "status": "running", "current_iteration": 60,
             "worker": "colab", "updated_at": minutes_ago(200)},
            {"stagnant_restarts": 3, "last_observed_iteration": 40,
             "last_active_status": "running"},
        )
        self.assertEqual(decision.action, "fallback_train")
        self.assertEqual(decision.stagnant_restarts, 0)

    def test_missing_pointer_asks_the_human_to_start_the_first_session(self):
        decision = self.decide(None)
        self.assertEqual(decision.action, "notify")
        self.assertEqual(decision.worker_status, "missing")

    def test_force_overrides_the_decision(self):
        decision = self.decide(
            {"run_id": "r1", "status": "running", "current_iteration": 40,
             "worker": "colab", "updated_at": minutes_ago(1)},
            force="fallback_report",
        )
        self.assertEqual(decision.action, "fallback_report")


class OrchestrationStateTest(unittest.TestCase):
    active = {"run_id": "r1", "status": "running", "current_iteration": 40}

    def test_notification_consumes_no_budget(self):
        previous = {"session_attempts": 2, "stagnant_restarts": 2, "last_observed_iteration": 40,
                    "last_active_status": "running"}
        state = orchestration_state_after_action(
            previous, self.active, "notify", "stalled", 40, "2026-08-22T00:00:00+00:00"
        )
        self.assertEqual(state["session_attempts"], 2)
        self.assertEqual(state["stagnant_restarts"], 2)
        self.assertIsNone(state["last_fallback_at"])

    def test_fallback_consumes_budget_and_stamps_the_guard(self):
        previous = {"session_attempts": 2, "stagnant_restarts": 2, "last_observed_iteration": 40,
                    "last_active_status": "running"}
        state = orchestration_state_after_action(
            previous, self.active, "fallback_train", "silent", 40, "2026-08-22T00:00:00+00:00"
        )
        self.assertEqual(state["session_attempts"], 3)
        self.assertEqual(state["stagnant_restarts"], 3)
        self.assertEqual(state["last_fallback_at"], "2026-08-22T00:00:00+00:00")

    def test_progress_resets_budgets_even_when_a_worker_is_launched(self):
        previous = {"session_attempts": 5, "stagnant_restarts": 3, "last_observed_iteration": 30,
                    "last_active_status": "running"}
        state = orchestration_state_after_action(
            previous, self.active, "fallback_train", "silent", 40, "2026-08-22T00:00:00+00:00"
        )
        self.assertEqual(state["session_attempts"], 0)
        self.assertEqual(state["stagnant_restarts"], 0)

    def test_report_restarts_only_accumulate_while_reporting(self):
        active = {"run_id": "r1", "status": "ready_for_report", "current_iteration": 100}
        state = orchestration_state_after_action(
            {"report_restarts": 4, "last_observed_iteration": 100,
             "last_active_status": "ready_for_report"},
            active, "fallback_report", "report", 100, "2026-08-22T00:00:00+00:00",
        )
        self.assertEqual(state["report_restarts"], 5)
        cleared = orchestration_state_after_action(
            {"report_restarts": 4, "last_observed_iteration": 40, "last_active_status": "running"},
            self.active, "fallback_train", "train", 40, "2026-08-22T00:00:00+00:00",
        )
        self.assertEqual(cleared["report_restarts"], 0)


if __name__ == "__main__":
    unittest.main()
