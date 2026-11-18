"""Question input, thread-local timing, and a single prediction report."""

import json
from pathlib import Path
import threading
import time


def load_questions(path, require_answer=False):
    text = Path(path).read_text(encoding="utf-8-sig")
    rows = json.loads(text) if text.lstrip().startswith("[") else [
        json.loads(line) for line in text.splitlines() if line.strip()
    ]
    if not isinstance(rows, list) or not rows:
        raise ValueError("Question file must be a nonempty JSON array or JSONL")
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("question"), str) or not row["question"].strip():
            raise ValueError(f"Record {i} requires a nonempty question")
        if require_answer and "answer" not in row:
            raise ValueError(f"Record {i} requires answer for evaluation")
    return rows


class PerformanceTracker:
    def __init__(self):
        self.local = threading.local()

    def wrap(self, stage, function):
        def timed(*args, **kwargs):
            started = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                metrics = getattr(self.local, "metrics", None)
                if metrics is not None:
                    metrics[stage + "_seconds"] += time.perf_counter() - started
                    metrics[stage + "_calls"] += 1
        return timed

    def run(self, function, idx, question, args):
        self.local.metrics = {"retrieval_seconds": 0.0, "retrieval_calls": 0,
                              "generation_seconds": 0.0, "generation_calls": 0}
        started = time.perf_counter()
        try:
            try:
                row = function(idx, question, args)
                row["status"] = "success"
            except Exception as exc:
                if args.eval_backend != "none":
                    raise
                row = {"id": idx, "question": question["question"],
                       "answer": question.get("answer"), "output": None,
                       "status": "error", "error": str(exc), "ref_log": []}
            metrics = dict(self.local.metrics)
            metrics["total_seconds"] = time.perf_counter() - started
            row["performance"] = metrics
            if "id" in question:
                row["source_id"] = question["id"]
            documents = []
            for step in row.get("ref_log", []):
                for ref in step.get("refs", []):
                    documents.append(ref if isinstance(ref, str) else
                                     ref.get("paragraph_text", ref.get("text", "")))
            row["retrieved_materials"] = list(dict.fromkeys(documents))
            row["retrieved_material_count"] = len(documents)
            row["unique_retrieved_material_count"] = len(row["retrieved_materials"])
            return row
        finally:
            self.local.metrics = None


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def save_report(path, results, run, elapsed, complete=False):
    successful = [r for r in results if r.get("status") == "success"]
    latencies = [r["performance"]["total_seconds"] for r in successful]
    summary = {
        "complete": complete,
        "completed_questions": len(results), "successful_questions": len(successful),
        "failed_questions": len(results) - len(successful),
        "run_wall_seconds": elapsed,
        "successful_questions_per_second": len(successful) / elapsed if elapsed > 0 else 0,
        "average_success_latency_seconds": sum(latencies) / len(latencies) if latencies else None,
        "p50_success_latency_seconds": percentile(latencies, 0.5),
        "p95_success_latency_seconds": percentile(latencies, 0.95),
        "retrieval_seconds_sum": sum(r["performance"]["retrieval_seconds"] for r in results),
        "generation_seconds_sum": sum(r["performance"]["generation_seconds"] for r in results),
        "token_usage": None,
        "timing_note": "Wall time starts before question processing; excludes model/index loading and scoring, includes report saves. Stage sums can exceed wall time with concurrency. Retrieval includes embedding and search; generation includes all LLM workflow calls and retries. No warmup exclusion.",
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps({"run": run, "summary": summary, "results": results},
                                    ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)
    return summary
