"""Rerun failed predictions and replace them in a saved report and JSONL."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from prediction_report import load_questions, save_report


def retry(report_path, question_ids):
    if isinstance(question_ids, int):
        question_ids = [question_ids]
    if not question_ids or len(set(question_ids)) != len(question_ids):
        raise ValueError("Provide at least one distinct question ID")
    report_path = Path(report_path).resolve()
    jsonl_path = report_path.with_suffix(".jsonl")
    report_bytes = report_path.read_bytes()
    jsonl_bytes = jsonl_path.read_bytes()
    report = json.loads(report_bytes.decode("utf-8"))
    run, summary, rows = report["run"], report["summary"], report["results"]
    if not summary.get("complete") or run.get("question_count") != len(rows):
        raise ValueError("Only complete reports with all selected questions can be retried")
    if [row.get("id") for row in rows] != list(range(len(rows))):
        raise ValueError("Report results must be in original question order")
    for question_id in question_ids:
        if not 0 <= question_id < len(rows):
            raise ValueError(f"Question ID {question_id} is outside the report")
        if rows[question_id].get("status") != "error":
            raise ValueError(f"Question {question_id} is not failed; only failed questions can be retried")
    jsonl_rows = [json.loads(line) for line in jsonl_bytes.decode("utf-8").splitlines() if line.strip()]
    if jsonl_rows != rows:
        raise ValueError("JSON report and JSONL differ; refusing to overwrite either file")
    questions = load_questions(run["question_file"])
    selected = []
    for question_id in question_ids:
        if question_id >= len(questions):
            raise ValueError(f"Question {question_id} is missing from the question file")
        question, original = questions[question_id], rows[question_id]
        if question.get("question") != original.get("question") or question.get("answer") != original.get("answer"):
            raise ValueError(f"Question file no longer matches failed result {question_id}")
        selected.append(question)
    backups = [Path(str(path) + ".bak") for path in (report_path, jsonl_path)]
    if any(path.exists() for path in backups):
        raise FileExistsError("Backup already exists; move it aside before retrying")
    script_dir = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix=".retry-", dir=report_path.parent) as temporary:
        temp_dir = Path(temporary)
        question_path = temp_dir / "question.json"
        question_path.write_text(json.dumps(selected, ensure_ascii=False), encoding="utf-8")
        retry_path = temp_dir / "retry.json"
        command = [sys.executable, str(script_dir / "main.py"), "--eval_backend", "none",
                   "--question_file", str(question_path), "--output_path", str(retry_path)]
        for field in ("dataset", "method", "model", "model_backend", "retrieve_method", "max_step",
                      "max_fail_step", "max_top_k", "retrieve_top_k"):
            if field not in run or run[field] is None:
                raise ValueError(f"Report is missing required run parameter: {field}")
            command.extend(["--" + field, str(run[field])])
        command.extend(["--MaxClients", "1"])
        subprocess.run(command, cwd=script_dir, check=True)
        retried_report = json.loads(retry_path.read_text(encoding="utf-8"))
        replacements = retried_report["results"]
        if (not retried_report["summary"].get("complete") or len(replacements) != len(question_ids)
                or [row.get("id") for row in replacements] != list(range(len(question_ids)))):
            raise ValueError("Retry report is incomplete or has unexpected result IDs")
        if retried_report["run"].get("retrieval_unit") != run.get("retrieval_unit"):
            raise ValueError("Index retrieval unit differs from the original report")
        for question_id, replacement in zip(question_ids, replacements):
            original = rows[question_id]
            if replacement.get("status") != "success":
                raise RuntimeError(f"Retry for question {question_id} failed; original files unchanged: {replacement.get('error')}")
            if replacement.get("question") != original.get("question") or replacement.get("answer") != original.get("answer"):
                raise ValueError(f"Retry output does not match original question {question_id}")
        for question_id, replacement in zip(question_ids, replacements):
            replacement["id"] = question_id
            rows[question_id] = replacement
        elapsed = summary["run_wall_seconds"] + retried_report["summary"]["run_wall_seconds"]
        updated_report = temp_dir / "updated.json"
        save_report(updated_report, rows, run, elapsed, complete=True)
        payload = json.loads(updated_report.read_text(encoding="utf-8"))
        payload["summary"]["timing_note"] += " Wall time includes original run plus retry, excluding time between runs; stage sums reflect current results only."
        payload["summary"]["retried_question_ids"] = list(dict.fromkeys(summary.get("retried_question_ids", []) + question_ids))
        updated_report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        updated_jsonl = temp_dir / "updated.jsonl"
        with updated_jsonl.open("w", encoding="utf-8", newline="\n") as output:
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
        if report_path.read_bytes() != report_bytes or jsonl_path.read_bytes() != jsonl_bytes:
            raise RuntimeError("Original files changed during retry; refusing to overwrite them")
        for original_path, backup in zip((report_path, jsonl_path), backups):
            with backup.open("xb") as target, original_path.open("rb") as source:
                shutil.copyfileobj(source, target)
        try:
            updated_report.replace(report_path)
            updated_jsonl.replace(jsonl_path)
        except Exception:
            shutil.copy2(backups[0], report_path)
            shutil.copy2(backups[1], jsonl_path)
            raise
    print(f"Updated questions {question_ids}; backups: {backups[0]}, {backups[1]}")
    print("Existing scored/comparison reports are unchanged; run scoring again.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, help="Original complete generation JSON report")
    ids = parser.add_mutually_exclusive_group(required=True)
    ids.add_argument("--id", type=int, help="One zero-based question ID")
    ids.add_argument("--ids", type=int, nargs="+", help="One or more zero-based question IDs")
    args = parser.parse_args()
    retry(args.report, args.ids if args.ids is not None else [args.id])


if __name__ == "__main__":
    main()
