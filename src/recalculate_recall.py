"""Recalculate recall in saved DeepNote scored reports without calling a judge."""

import argparse
import asyncio
import copy
import json
from pathlib import Path

from score_deepnote_reports import calc_recal_rate, write_json


async def recalculate(path, assume_record=False):
    path = Path(path).resolve()
    original = path.read_bytes()
    report = json.loads(original.decode("utf-8-sig"))
    rows = report.get("results")
    summary = report.get("summary")
    if not isinstance(rows, list) or not rows or not isinstance(summary, dict) or not summary.get("complete"):
        raise ValueError("Expected a complete scored report with results")
    if [row.get("id") for row in rows] != list(range(len(rows))):
        raise ValueError("Report results must have sequential IDs in original order")
    backup = Path(str(path) + ".recall.bak")
    if backup.exists():
        raise FileExistsError(f"Backup already exists: {backup}; move it aside before recalculating")
    updated = copy.deepcopy(report)
    run = report.get("generation_run", {})
    mapped_required = run.get("retrieve_method") == "emb" and run.get("retrieval_unit") != "record" and not assume_record
    for row in updated["results"]:
        if row.get("status") == "qa_error":
            row["retrieved_rate"] = 0.0
            continue
        if mapped_required and "retrieved_paragraphs" not in row:
            raise ValueError(f"Question {row['id']} has no mapped paragraphs; use --assume-record only if retrieved texts are complete source records")
        gold = row.get("gold_docs")
        retrieved = row.get("retrieved_paragraphs", row.get("retrieved_results"))
        if not isinstance(gold, list) or any(not isinstance(doc, str) for doc in gold):
            raise ValueError(f"Question {row['id']} has invalid gold_docs")
        if not isinstance(retrieved, list) or any(not isinstance(doc, str) for doc in retrieved):
            raise ValueError(f"Question {row['id']} has invalid retrieved documents")
        row["retrieved_rate"] = await calc_recal_rate(gold, retrieved)
    updated["summary"]["avg_multiround_recall_on_answerable"] = sum(
        row["retrieved_rate"] for row in updated["results"]
    ) / len(rows)
    updated["summary"]["recall_policy"] = "Average over all selected questions; QA errors count as zero."
    if path.read_bytes() != original:
        raise RuntimeError("Report changed during recalculation; refusing to overwrite it")
    with backup.open("xb") as target:
        target.write(original)
    write_json(path, updated)
    print(f"{path}: recall={updated['summary']['avg_multiround_recall_on_answerable']:.6f}; backup={backup}")
    return updated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", required=True, help="Existing .scored.json reports")
    parser.add_argument("--assume-record", action="store_true", help="Treat retrieved embedding texts as complete source records")
    args = parser.parse_args()
    paths = [Path(path).resolve() for path in args.reports]
    if len(set(paths)) != len(paths):
        parser.error("Report paths must be distinct")
    for path in paths:
        asyncio.run(recalculate(path, args.assume_record))


if __name__ == "__main__":
    main()
