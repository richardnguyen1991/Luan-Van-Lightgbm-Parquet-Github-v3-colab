"""Push or pull the prepared Parquet splits that S3 keeps as intermediate data.

`data.py` already synchronizes prepared splits while it runs. This script exposes the same
store as a standalone command so a dataset can be uploaded from one machine, inspected, or
pulled into a fresh Colab runtime without re-reading the raw CIC-DDoS2019 Parquet files.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import PreprocessingStore, compute_data_version, load_config  # noqa: E402

ARTIFACT_NAMES = (
    "data_profile.json",
    "label_mapping.json",
    "preprocessing.json",
    "sample_manifest.json",
    "dataset_version.json",
)


def build_store(args: argparse.Namespace) -> tuple[PreprocessingStore, str]:
    data_config = load_config(args.data_config)
    if args.data_version:
        data_config["dataset"]["data_version_override"] = args.data_version
    data_version = compute_data_version(data_config)
    s3_document = json.loads(Path(args.s3_config).read_text(encoding="utf-8"))
    store = PreprocessingStore(Path(args.output_dir), data_version, s3_document["s3"])
    return store, data_version


def command_pull(args: argparse.Namespace) -> int:
    store, data_version = build_store(args)
    progress = store.restore()
    if progress is None:
        print(f"No prepared dataset {data_version} exists in S3 yet")
        return 1
    status = str(progress.get("status", "unknown"))
    parts = sum(len(items) for items in progress.get("parts", {}).values())
    print(json.dumps({
        "data_version": data_version,
        "status": status,
        "parts_downloaded": parts,
        "output_dir": str(Path(args.output_dir).resolve()),
    }, indent=2))
    return 0 if status == "complete" else 1


def command_push(args: argparse.Namespace) -> int:
    store, data_version = build_store(args)
    destination = Path(args.output_dir)
    manifest_path = destination / "sample_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} is missing; run data.py to completion before pushing a dataset"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    version_path = destination / "dataset_version.json"
    if not version_path.exists():
        # data.py writes this file; recreate it when pushing a dataset prepared elsewhere.
        version_path.write_text(
            json.dumps(
                {
                    "data_version": data_version,
                    "prepared_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    uploaded = 0
    for split, parts in manifest["parts"].items():
        for part in parts:
            relative = str(part["path"])
            store.upload_part(destination / relative, relative)
            uploaded += 1
    for name in ARTIFACT_NAMES:
        store.upload_artifact(destination / name)

    local_progress = destination / "progress.json"
    fingerprint: Any = None
    if local_progress.exists():
        fingerprint = json.loads(local_progress.read_text(encoding="utf-8")).get("fingerprint")
    store.save_progress({
        "format_version": 1,
        "status": "complete",
        "fingerprint": fingerprint,
        "completed_files": [item["path"] for item in manifest["source_files"]],
        "labels_seen": sorted(manifest["split"]["class_counts"]["train"]),
        "split_counts": manifest["split"]["class_counts"],
        "source_inventory": manifest["source_files"],
        "parts": manifest["parts"],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })
    print(json.dumps({
        "data_version": data_version,
        "parts_uploaded": uploaded,
        "artifacts_uploaded": len(ARTIFACT_NAMES),
    }, indent=2))
    return 0


def command_status(args: argparse.Namespace) -> int:
    store, data_version = build_store(args)
    progress = store.s3.read_json(store.key("progress.json"))
    if progress is None:
        print(json.dumps({"data_version": data_version, "status": "absent"}, indent=2))
        return 1
    print(json.dumps({
        "data_version": data_version,
        "status": progress.get("status"),
        "completed_files": len(progress.get("completed_files", [])),
        "parts": {split: len(items) for split, items in progress.get("parts", {}).items()},
        "updated_at": progress.get("updated_at"),
    }, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default=str(PROJECT_ROOT / "config" / "data.json"))
    parser.add_argument("--s3-config", default=str(PROJECT_ROOT / "config" / "train.json"))
    parser.add_argument("--output-dir", default="outputs/data")
    parser.add_argument("--data-version", default=None)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("pull").set_defaults(func=command_pull)
    commands.add_parser("push").set_defaults(func=command_push)
    commands.add_parser("status").set_defaults(func=command_status)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
