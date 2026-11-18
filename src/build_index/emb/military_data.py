"""Load the military corpus without changing text or record order."""

import json
from pathlib import Path


def load_military_records(path):
    path = Path(path)
    with path.open(encoding="utf-8-sig") as stream:
        if path.suffix.lower() == ".jsonl":
            records = [json.loads(line) for line in stream if line.strip()]
        else:
            records = json.load(stream)
    if not isinstance(records, list) or not records:
        raise ValueError("Military input must contain a nonempty JSON array or JSONL records")
    for i, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Military record {i} must be an object")
        if not isinstance(record.get("text"), str) or not record["text"].strip():
            raise ValueError(f"Military record {i} must have nonempty text")
        key = record.get("id")
        if isinstance(key, bool) or not isinstance(key, (str, int)) or key == "":
            raise ValueError(f"Military record {i} must have a string/integer id")
    return records
