#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
QUESTION_FILE="${QUESTION_FILE:-/data/shared/data/question/musique_questions.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/output/musique}"
MAX_QUESTIONS="${MAX_QUESTIONS:-200}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
  else
    PYTHON_BIN="$(command -v python)"
  fi
fi
export PYTHONUNBUFFERED=1
mkdir -p -- "$OUTPUT_DIR"
OUTPUT_DIR="$(cd -- "$OUTPUT_DIR" && pwd)"
EMB_REPORT="${OUTPUT_DIR}/musique_deepnote_emb.json"
BM25_REPORT="${OUTPUT_DIR}/musique_deepnote_bm25.json"
if [[ -e "$EMB_REPORT" || -e "$BM25_REPORT" || -e "${EMB_REPORT%.json}.jsonl" || -e "${BM25_REPORT%.json}.jsonl" ]]; then
  echo "Output reports already exist. Set OUTPUT_DIR to a fresh directory to preserve prior runs." >&2
  exit 1
fi
cd -- "${ROOT_DIR}/src"

# Read-only preflight: catch missing inputs/wrong ES service before a long run.
"$PYTHON_BIN" - "$QUESTION_FILE" "$MAX_QUESTIONS" <<'PY'
import sys
from pathlib import Path
from prediction_report import load_questions, limit_questions
questions = load_questions(sys.argv[1], require_answer=True)
selected = limit_questions(questions, int(sys.argv[2]))
from score_deepnote_reports import validate_questions
validate_questions(selected)
import yaml
settings = yaml.safe_load(Path('../config/config.yaml').read_text(encoding='utf-8'))
for name in ('generation', 'embedding', 'evaluation'):
    if not settings.get(name, {}).get('base_url') or not settings[name].get('model'):
        raise ValueError(f'Configure {name}.base_url and {name}.model first')
for name in ('musique.index', 'chunk.json', 'musique.index.embedding.json'):
    if not (Path('../data/corpus/musique') / name).is_file():
        raise FileNotFoundError(name)
sys.path.insert(0, 'build_index/es')
from es_support import create_client
es = create_client()
if not es.indices.exists(index='musique'):
    raise ValueError('musique BM25 index is missing from the configured Elasticsearch service')
print(f'Preflight passed: {len(questions)} input questions; selecting {len(selected)}')
PY

COMMON_ARGS=(
  --model_backend api --dataset musique --method deepnote
  --question_file "$QUESTION_FILE" --max_questions "$MAX_QUESTIONS"
  --eval_backend none --retrieve_top_k 15 --max_step 4 --max_top_k 60
)
echo "[1/3] Generating DeepNote vector answers"
"$PYTHON_BIN" main.py "${COMMON_ARGS[@]}" --retrieve_method emb --output_path "$EMB_REPORT"
echo "[2/3] Generating DeepNote BM25 answers"
"$PYTHON_BIN" main.py "${COMMON_ARGS[@]}" --retrieve_method es --output_path "$BM25_REPORT"
echo "[3/3] Scoring both saved reports with run_test rules"
"$PYTHON_BIN" score_deepnote_reports.py \
  --reports "$EMB_REPORT" "$BM25_REPORT" \
  --question-file "$QUESTION_FILE" \
  --comparison-output "${OUTPUT_DIR}/musique_deepnote_comparison.json"
echo "Finished. Reports and scores: $OUTPUT_DIR"
