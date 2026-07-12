"""Persistent wall-clock timing records for CoDA experiments."""

import json
import os
from datetime import datetime, timezone


def _load_record(path, method):
    if not os.path.isfile(path):
        return {
            "format_version": 1,
            "method": method,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "stages_seconds": {},
            "metadata": {},
        }

    with open(path, "r", encoding="utf-8") as file:
        record = json.load(file)
    existing_method = record.get("method")
    if existing_method != method:
        raise ValueError(
            f"Timing record {path} belongs to method={existing_method!r}, not {method!r}."
        )
    return record


def _write_record(path, record):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(record, file, ensure_ascii=False, indent=2)
        file.write("\n")
    os.replace(tmp_path, path)


def record_stage_timing(path, method, stage, seconds, metadata=None):
    """Record one completed stage, using an atomic write to keep partial runs useful."""
    if not path:
        return

    record = _load_record(path, method)
    record["stages_seconds"][stage] = round(float(seconds), 3)
    if metadata:
        record["metadata"].update(metadata)
    record["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    record["total_recorded_seconds"] = round(sum(record["stages_seconds"].values()), 3)
    _write_record(path, record)


def read_timing_record(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)
