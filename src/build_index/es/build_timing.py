"""Wall-clock timing shared by the BM25 indexing scripts."""

import json
import time
from pathlib import Path


class BuildTiming:
    def __init__(self):
        self.started = self.previous = time.perf_counter()
        self.stages = {}

    def checkpoint(self, stage):
        now = time.perf_counter()
        self.stages[stage] = self.stages.get(stage, 0.0) + now - self.previous
        self.previous = now

    def finish(self, es, index, count, dry):
        refresh_seconds = 0.0
        if not dry:
            started = time.perf_counter()
            result = es.indices.refresh(index=index)
            if result.get("_shards", {}).get("failed", 0):
                raise RuntimeError(f"Index refresh failed: {result}")
            refresh_seconds = time.perf_counter() - started
        stats = {
            "dataset": index,
            "index": index,
            "method": "bm25",
            "dry_run": dry,
            "prepared_document_count": count,
            "submitted_document_count": 0 if dry else count,
            **self.stages,
            "refresh_seconds": refresh_seconds,
            "total_seconds": time.perf_counter() - self.started,
        }
        suffix = "bm25.dry.stats.json" if dry else "bm25.stats.json"
        path = Path(__file__).resolve().parents[3] / "data" / "corpus" / index / suffix
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(json.dumps(stats, indent=2))
        print(f"Timing report: {path}")
