#!/usr/bin/env python3
"""Write an immutable platform-configuration record for one training launch."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_training_provenance(config_path: Path, output_dir: Path, *, launch_id: str) -> Path:
    """Snapshot the exact validated platform config and return its manifest path."""
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"platform configuration does not exist: {config_path}")
    if not launch_id:
        raise ValueError("launch_id must not be empty")

    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_name = "platform_config_snapshot.json"
    snapshot_path = output_dir / snapshot_name
    snapshot_path.write_bytes(config_path.read_bytes())
    payload = {
        "schema_version": 1,
        "launch_id": launch_id,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform_config": {
            "source_path": str(config_path),
            "snapshot_file": snapshot_name,
            "sha256": file_sha256(config_path),
        },
    }
    artifact = output_dir / "training_provenance.json"
    artifact.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--launch-id", required=True)
    args = parser.parse_args()
    print(write_training_provenance(args.config, args.output_dir, launch_id=args.launch_id))


if __name__ == "__main__":
    main()
