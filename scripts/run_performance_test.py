"""Run generation-only benchmarks using consecutive questions."""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "2wikimultihopqa": ("2wikimultihopqa", "2wiki_questions.json"),
    "musique": ("musique", "musique_questions.json"),
    "military_15": ("military", "military_15.json"),
    "military_generate": ("military", "military_generate.json"),
}


def select_questions(path, start_index, total_run):
    questions = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(questions, list):
        raise ValueError(f"Question file must contain a JSON array: {path}")
    selected = questions[start_index:start_index + total_run]
    if len(selected) != total_run:
        raise ValueError(f"{path}: only {len(selected)} questions available from index {start_index}; need {total_run}")
    if any(not isinstance(row, dict) or not isinstance(row.get("question"), str)
           or not row["question"].strip() for row in selected):
        raise ValueError(f"{path}: selected questions must contain a nonempty question string")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question-dir", type=Path, default=Path("/data/shared/data/question"))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--retrieval", nargs="+", choices=["emb", "bm25"], default=["emb", "bm25"])
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--total-run", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, help="New output directory")
    parser.add_argument("--max-clients", type=int, default=1)
    parser.add_argument("--model", help="Override generation.model")
    args = parser.parse_args()
    if args.start_index < 0 or args.total_run < 1 or args.max_clients < 1:
        parser.error("start index must be nonnegative; total run and max clients must be positive")
    if len(set(args.datasets)) != len(args.datasets) or len(set(args.retrieval)) != len(args.retrieval):
        parser.error("datasets and retrieval modes must not contain duplicates")
    selected = {}
    for name in args.datasets:
        source = args.question_dir.resolve() / DATASETS[name][1]
        selected[name] = (source, select_questions(source, args.start_index, args.total_run))
    output = (args.output_dir or ROOT / "output" / f"performance_test_{datetime.now():%Y%m%d_%H%M%S}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"start_index": args.start_index, "total_run": args.total_run,
                "retrieval": args.retrieval, "max_clients": args.max_clients,
                "datasets": {name: {"dataset": DATASETS[name][0], "question_file": str(source),
                                     "selected_count": len(rows)}
                             for name, (source, rows) in selected.items()}}
    (output / "selection.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    reports = {retrieval: [] for retrieval in args.retrieval}
    for name, (source, rows) in selected.items():
        dataset_dir = output / name
        dataset_dir.mkdir()
        question_file = dataset_dir / "selected_questions.json"
        question_file.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{name}: {source} [{args.start_index}:{args.start_index + args.total_run}] -> {len(rows)} questions", flush=True)
        for retrieval in args.retrieval:
            report = dataset_dir / f"{name}_deepnote_{retrieval}.json"
            command = [sys.executable, "main.py", "--model_backend", "api", "--dataset", DATASETS[name][0],
                       "--method", "deepnote", "--question_file", str(question_file), "--eval_backend", "none",
                       "--retrieve_top_k", "15", "--max_step", "4", "--max_top_k", "60",
                       "--MaxClients", str(args.max_clients), "--retrieve_method",
                       "es" if retrieval == "bm25" else retrieval, "--output_path", str(report)]
            if args.model:
                command.extend(["--model", args.model])
            print(f"Running {name} / {retrieval}; report: {report}", flush=True)
            subprocess.run(command, cwd=ROOT / "src", check=True)
            reports[retrieval].append(report)
    for retrieval, paths in reports.items():
        summary_path = output / f"{retrieval}_performance_summary.json"
        subprocess.run([sys.executable, str(ROOT / "src" / "summarize_performance.py"),
                        *map(str, paths), "--group-by-parent", "--output", str(summary_path)],
                       cwd=ROOT, check=True)
    print(f"Finished. Reports: {output}")


if __name__ == "__main__":
    main()
