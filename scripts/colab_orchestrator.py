"""Decide what the GitHub Actions watchdog should do about a stalled Colab run.

Colab notebooks cannot be started through a public API, so this watchdog cannot simply
push the next session the way the Kaggle pipeline did. Instead it reads the S3 pointer
that every worker heartbeats into and chooses one of:

    wait             a worker is alive, or a guard window is still open
    notify           Colab has gone quiet; ask the human to reopen the notebook
    fallback_train   Colab has been quiet long enough that the runner should train
    fallback_report  iteration 100 is durable but the final report is missing
    complete         iteration 100 and the final report both exist
    stop             a safety budget has been exhausted; a human must intervene
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SECRET_ENV_NAMES = (
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
    "AWS_REGION", "AWS_DEFAULT_REGION", "S3_BUCKET", "S3_PREFIX",
)
ACTIONS = ("wait", "notify", "fallback_train", "fallback_report", "complete", "stop")
LIVE_STATUSES = {"preparing", "running"}


def normalize_secret_environment() -> None:
    """Strip stray newlines pasted into GitHub Secrets before boto3 sees them."""
    for name in SECRET_ENV_NAMES:
        value = os.environ.get(name)
        if value is not None:
            os.environ[name] = (
                value.replace("\r", "").replace("\n", "")
                .replace("\\r", "").replace("\\n", "").strip()
            )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_timestamp(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


@dataclass(frozen=True)
class Decision:
    action: str
    reason: str
    run_id: str
    current_iteration: int
    worker_status: str
    worker: str
    heartbeat_age_minutes: float
    session_attempts: int
    stagnant_restarts: int
    report_restarts: int


def made_durable_progress(active: Mapping[str, Any], state: Mapping[str, Any]) -> bool:
    """True when something irreversible happened since the last recorded action."""
    current = int(active.get("current_iteration", 0))
    previous_iteration = int(state.get("last_observed_iteration", 0))
    preprocessing_progress = int(active.get("preprocessing_completed_files", 0))
    previous_preprocessing = int(state.get("last_preprocessing_completed_files", 0))
    active_status = str(active.get("status", "missing")).casefold()
    previous_status = str(state.get("last_active_status", "missing")).casefold()
    lifecycle_progress = (
        active_status != previous_status
        and active_status in {"paused", "ready_for_report", "complete"}
    )
    return (
        current > previous_iteration
        or preprocessing_progress > previous_preprocessing
        or lifecycle_progress
    )


def orchestration_state_after_action(
    previous: Mapping[str, Any],
    active: Mapping[str, Any],
    action: str,
    reason: str,
    observed_iteration: int,
    now_utc: str,
) -> dict[str, Any]:
    active_status = str(active.get("status", "missing")).casefold()
    progress = made_durable_progress(active, previous)
    state = {
        "last_action": action,
        "last_action_at": now_utc,
        "last_action_reason": reason,
        "last_observed_iteration": observed_iteration,
        "last_preprocessing_completed_files": int(active.get("preprocessing_completed_files", 0)),
        "last_active_status": active_status,
        "session_attempts": 0 if progress else int(previous.get("session_attempts", 0)),
        "stagnant_restarts": 0 if progress else int(previous.get("stagnant_restarts", 0)),
        "report_restarts": (
            int(previous.get("report_restarts", 0)) if active_status == "ready_for_report" else 0
        ),
        "last_fallback_at": previous.get("last_fallback_at"),
    }
    if action in {"fallback_train", "fallback_report"}:
        # Only a launched worker consumes a budget; a notification costs nothing.
        state["last_fallback_at"] = now_utc
        state["session_attempts"] = 0 if progress else int(previous.get("session_attempts", 0)) + 1
        state["stagnant_restarts"] = 0 if progress else int(previous.get("stagnant_restarts", 0)) + 1
        if action == "fallback_report":
            state["report_restarts"] = int(previous.get("report_restarts", 0)) + 1
    return state


def decide_next_action(
    active: Mapping[str, Any] | None,
    state: Mapping[str, Any] | None,
    config: Mapping[str, Any],
    now_timestamp: float,
    force: str | None = None,
) -> Decision:
    active, state = dict(active or {}), dict(state or {})
    target = int(config["target_iteration"])
    current = int(active.get("current_iteration", 0))
    active_status = str(active.get("status", "missing")).casefold()
    worker = str(active.get("worker", "unknown"))
    configured_run_id = str(config.get("run_id") or "").strip()
    active_run_id = str(active.get("run_id", "")).strip()
    run_id = configured_run_id or active_run_id
    different_run = bool(configured_run_id and active_run_id and configured_run_id != active_run_id)

    attempts = int(state.get("session_attempts", 0))
    stagnant = int(state.get("stagnant_restarts", 0))
    report_restarts = int(state.get("report_restarts", 0))
    if different_run:
        current, active_status = 0, "different_run"
        attempts = stagnant = report_restarts = 0
    if made_durable_progress(active, state):
        # These budgets guard against repeated no-op restarts. They must never become
        # lifetime counters that lock out a healthy run after many successful resumes.
        attempts = stagnant = 0

    heartbeat = parse_timestamp(active.get("updated_at"))
    age_minutes = (now_timestamp - heartbeat) / 60.0 if heartbeat is not None else float("inf")
    stale_minutes = float(config["heartbeat_stale_minutes"])
    if not active:
        worker_status = "missing"
    elif active_status == "complete":
        worker_status = "complete"
    elif active_status in LIVE_STATUSES and age_minutes < stale_minutes:
        worker_status = "alive"
    else:
        worker_status = "stale"

    def decide(action: str, reason: str) -> Decision:
        assert action in ACTIONS
        return Decision(
            action=action, reason=reason, run_id=run_id, current_iteration=current,
            worker_status=worker_status, worker=worker,
            heartbeat_age_minutes=round(age_minutes, 2) if heartbeat is not None else -1.0,
            session_attempts=attempts, stagnant_restarts=stagnant, report_restarts=report_restarts,
        )

    if force:
        return decide(force, f"manual force: {force}")
    if active_status == "complete" and current >= target:
        return decide("complete", "iteration 100 and the final report are complete")
    if worker_status == "alive":
        return decide("wait", f"{worker} worker heartbeat is {age_minutes:.1f} minutes old")
    if not active:
        return decide(
            "notify",
            "no run exists yet; start the first session on Colab so preprocessing can run there",
        )
    if active_status == "ready_for_report" or (current >= target and active_status != "complete"):
        if report_restarts >= int(config["maximum_report_restarts"]):
            return decide("stop", "maximum report restarts reached; inspect the report failure")
        return decide("fallback_report", "iteration 100 is durable but the final report is missing")
    if attempts >= int(config["maximum_session_attempts"]):
        return decide("stop", "maximum session attempts reached")
    if stagnant >= int(config["maximum_stagnant_restarts"]):
        return decide("stop", "maximum stagnant restarts reached; no durable progress is being made")

    last_fallback = parse_timestamp(state.get("last_fallback_at"))
    guard_minutes = float(config["recent_fallback_guard_minutes"])
    guarded = last_fallback is not None and (now_timestamp - last_fallback) / 60.0 < guard_minutes
    if age_minutes >= float(config["fallback_after_minutes"]) and not guarded:
        return decide(
            "fallback_train",
            f"no worker heartbeat for {age_minutes:.1f} minutes; the runner continues training",
        )
    if guarded:
        return decide("notify", "recent fallback guard is active; asking the human to reopen Colab")
    return decide("notify", f"worker is {worker_status}; reopen the Colab notebook to resume")


class S3State:
    def __init__(self) -> None:
        import boto3

        self.bucket = os.environ["S3_BUCKET"].strip()
        self.prefix = os.environ["S3_PREFIX"].strip().strip("/")
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        self.client = boto3.client("s3", region_name=region or None)

    def key(self, name: str) -> str:
        return f"{self.prefix}/{name}" if self.prefix else name

    def read_json(self, name: str) -> dict[str, Any] | None:
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=self.key(name))
        except Exception as exc:
            error = getattr(exc, "response", {}) or {}
            not_found = error.get("Error", {}).get("Code") in {"NoSuchKey", "NotFound", "404"}
            if not_found or error.get("ResponseMetadata", {}).get("HTTPStatusCode") == 404:
                return None
            raise
        return json.loads(response["Body"].read().decode("utf-8"))

    def write_json(self, name: str, payload: Mapping[str, Any]) -> None:
        body = (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode("utf-8")
        self.client.put_object(Bucket=self.bucket, Key=self.key(name), Body=body)


def colab_url(config: Mapping[str, Any], branch: str = "main") -> str:
    repository = str(config["repository"]).strip("/")
    notebook = str(config["colab_notebook_path"]).lstrip("/")
    return f"https://colab.research.google.com/github/{repository}/blob/{branch}/{notebook}"


def write_github_output(path: str | None, decision: Decision, config: Mapping[str, Any]) -> None:
    values = {
        "action": decision.action,
        "reason": decision.reason.replace("\n", " "),
        "run_id": decision.run_id,
        "current_iteration": str(decision.current_iteration),
        "worker_status": decision.worker_status,
        "worker": decision.worker,
        "heartbeat_age_minutes": str(decision.heartbeat_age_minutes),
        "colab_url": colab_url(config),
    }
    if path:
        with Path(path).open("a", encoding="utf-8") as handle:
            for key, value in values.items():
                handle.write(f"{key}={value}\n")
    print(json.dumps(asdict(decision), indent=2))


def load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def command_decide(args: argparse.Namespace) -> None:
    config, store = load_config(args.config), S3State()
    decision = decide_next_action(
        store.read_json("active_run.json"),
        store.read_json("orchestration_state.json"),
        config,
        time.time(),
        args.force,
    )
    write_github_output(args.github_output, decision, config)


def command_record(args: argparse.Namespace) -> None:
    store = S3State()
    previous = store.read_json("orchestration_state.json") or {}
    active = store.read_json("active_run.json") or {}
    observed = int(active.get("current_iteration", args.observed_iteration))
    state = orchestration_state_after_action(
        previous, active, args.action, args.reason, observed, utc_now()
    )
    store.write_json("orchestration_state.json", state)
    print(json.dumps(state, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "orchestration.json"))
    commands = parser.add_subparsers(dest="command", required=True)
    decide = commands.add_parser("decide")
    decide.add_argument("--github-output", default=None)
    decide.add_argument("--force", choices=ACTIONS, default=None)
    decide.set_defaults(func=command_decide)
    record = commands.add_parser("record-action")
    record.add_argument("--action", required=True, choices=ACTIONS)
    record.add_argument("--reason", required=True)
    record.add_argument("--observed-iteration", type=int, default=0)
    record.set_defaults(func=command_record)
    return parser.parse_args()


def main() -> None:
    normalize_secret_environment()
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
