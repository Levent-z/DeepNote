"""Summarize existing per-question performance records without rerunning QA."""

import argparse
import json
import math
from pathlib import Path


FIELDS = {
    "qa_seconds": "total_seconds",
    "generation_seconds": "generation_seconds",
    "generation_calls": "generation_calls",
    "retrieval_seconds": "retrieval_seconds",
    "retrieval_calls": "retrieval_calls",
}


def statistics(values):
    return {"count": len(values), "mean": sum(values) / len(values) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None}


def summarize(rows):
    output = {"questions": len(rows), "statuses": {}}
    for row in rows:
        status = row.get("status", "unknown")
        output["statuses"][status] = output["statuses"].get(status, 0) + 1
    for name, field in FIELDS.items():
        values = []
        for row in rows:
            performance = row.get("performance") or {}
            value = performance.get(field)
            if name == "qa_seconds" and value is None:
                value = row.get("qa_elapsed_s")
            if (isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value) and value >= 0):
                values.append(value)
        output[name] = statistics(values)
    return output


def read_report(path):
    report = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(report, dict) or not isinstance(report.get("results"), list):
        raise ValueError(f"Expected a JSON report with a results list: {path}")
    run = report.get("run") or report.get("generation_run") or {}
    dataset = run.get("dataset")
    if not dataset:
        raise ValueError(f"Missing run.dataset or generation_run.dataset: {path}")
    if not all(isinstance(row, dict) for row in report["results"]):
        raise ValueError(f"Invalid result row: {path}")
    return dataset, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", help="Generation or scored JSON reports, one per run")
    parser.add_argument("--output", help="Write summary JSON to this path (default: stdout)")
    parser.add_argument("--group-by-parent", action="store_true",
                        help="Group reports by their parent directory instead of their dataset field")
    args = parser.parse_args()
    paths = [Path(name).resolve() for name in args.reports]
    if len(set(paths)) != len(paths):
        parser.error("Each report may be supplied only once")
    if args.output and Path(args.output).resolve() in paths:
        parser.error("Output must not overwrite an input report")
    datasets = {}
    sources = []
    for path in paths:
        dataset, report = read_report(path)
        group = path.parent.name if args.group_by_parent else dataset
        datasets.setdefault(group, []).extend(report["results"])
        sources.append({"path": str(path), "dataset": dataset,
                        "group": group,
                        "questions": len(report["results"]),
                        "complete": report.get("summary", {}).get("complete")})
    all_rows = [row for rows in datasets.values() for row in rows]
    result = {"note": "Means are per recorded question; null means unavailable. Overall is question-weighted. Failures are included. Supply one report per run to avoid double counting.",
              "sources": sources, "datasets": {name: summarize(rows) for name, rows in datasets.items()},
              "overall": summarize(all_rows)}
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + "\n", encoding="utf-8")
        print(f"Saved: {target}")
    else:
        print(text)


if __name__ == "__main__":
    main()
