"""Retry failed judgments in a saved DeepNote scored report."""

import argparse
import asyncio
import copy
import json
import os
from pathlib import Path

import score_deepnote_reports as scorer


async def retry_report(path, settings):
    path = Path(path).resolve()
    original = path.read_bytes()
    report = json.loads(original.decode("utf-8-sig"))
    rows = report.get("results")
    if not isinstance(rows, list) or not rows or not report.get("summary", {}).get("complete"):
        raise ValueError("Expected a complete scored report with results")
    if [row.get("id") for row in rows] != list(range(len(rows))):
        raise ValueError("Report results must have sequential IDs in original order")
    judge = report.get("judge", {})
    if not settings.get("model") or not settings.get("base_url"):
        raise ValueError("Configure evaluation.model and evaluation.base_url")
    if judge.get("model") != settings["model"] or judge.get("base_url") != settings["base_url"]:
        raise ValueError("Configured judge differs from the report; use the original judge or rescore the whole report")
    failed = [row for row in rows if row.get("status") == "judge_error"]
    if not failed:
        print(f"{path}: no judge_error records; no changes made")
        return report
    backup = Path(str(path) + ".bak")
    if backup.exists():
        raise FileExistsError(f"Backup already exists: {backup}; move it aside before retrying")
    for row in rows:
        if row.get("status") in ("ok", "judge_error"):
            if not isinstance(row.get("gold_docs"), list) or not all(isinstance(doc, str) for doc in row["gold_docs"]):
                raise ValueError(f"Question {row.get('id')} has invalid gold_docs")
            retrieved = row.get("retrieved_paragraphs", row.get("retrieved_results"))
            if not isinstance(retrieved, list) or any(not isinstance(doc, str) for doc in retrieved):
                raise ValueError(f"Question {row.get('id')} has invalid retrieved documents")
            run = report.get("generation_run", {})
            if run.get("retrieve_method") == "emb" and run.get("retrieval_unit") != "record" and "retrieved_paragraphs" not in row:
                raise ValueError(f"Question {row.get('id')} is missing mapped paragraphs; rescore the source report")
    updated = copy.deepcopy(report)
    scorer.cache_data = {}
    for row in updated["results"]:
        if row.get("status") not in ("ok", "judge_error"):
            continue
        retrieved = row.get("retrieved_paragraphs", row.get("retrieved_results"))
        row["retrieved_rate"] = await scorer.calc_recal_rate(row["gold_docs"], retrieved)
        if row["status"] != "judge_error":
            continue
        try:
            right, reason, answerable, answerable_reason = await scorer.check_result(
                base_url=settings["base_url"],
                api_key=os.environ.get("EVAL_JUDGE_API_KEY") or settings.get("api_key") or "EMPTY",
                model=settings["model"], question=row["question"], answer=row["answer"],
                llm_answer=row["llm_answer"], gold_docs=row["gold_docs"],
            )
            row.update(llm_judge=right, llm_judge_reason=reason,
                       llm_judge_answerable=answerable, answerable_reason=answerable_reason, status="ok")
            row.pop("error", None)
            print(f"Question {row['id']}: scored", flush=True)
        except Exception as exc:
            row["error"] = str(exc)
            print(f"Question {row['id']}: still failed: {exc}", flush=True)
    new_summary = scorer.summarize_results(updated["results"])
    new_summary["total_tokens"] = new_summary["avg_tokens"] = None
    new_summary.update(complete=True, completed=len(rows), token_usage_note="Not recorded by DeepNote")
    updated["summary"] = new_summary
    if path.read_bytes() != original:
        raise RuntimeError("Report changed during scoring; refusing to overwrite it")
    with backup.open("xb") as target:
        target.write(original)
    scorer.write_json(path, updated)
    print(f"Updated {path}; backup: {backup}")
    print(json.dumps(new_summary, ensure_ascii=False, indent=2))
    return updated


def main():
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", required=True, help="Existing .scored.json reports to update")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config/config.yaml"))
    args = parser.parse_args()
    paths = [Path(path).resolve() for path in args.reports]
    if len(set(paths)) != len(paths) or Path(args.config).resolve() in paths:
        parser.error("Report paths must be distinct from each other and from the config")
    settings = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))["evaluation"]
    for path in paths:
        asyncio.run(retry_report(path, settings))


if __name__ == "__main__":
    main()
