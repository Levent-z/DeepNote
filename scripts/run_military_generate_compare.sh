#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
QUESTION_FILE="${QUESTION_FILE:-/data/shared/data/question/military_generate.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/output/military_generate}"
MAX_QUESTIONS="${MAX_QUESTIONS:-}"
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
EMB_REPORT="${OUTPUT_DIR}/military_generate_deepnote_emb.json"
BM25_REPORT="${OUTPUT_DIR}/military_generate_deepnote_bm25.json"
if [[ -e "$EMB_REPORT" || -e "$BM25_REPORT" || -e "${EMB_REPORT%.json}.jsonl" || -e "${BM25_REPORT%.json}.jsonl" ]]; then
  echo "Output reports already exist. Set OUTPUT_DIR to a fresh directory to preserve prior runs." >&2
  exit 1
fi
cd -- "${ROOT_DIR}/src"

# Read-only preflight: catch missing inputs/wrong ES service before a long run.
"$PYTHON_BIN" - "$QUESTION_FILE" "$MAX_QUESTIONS" "$OUTPUT_DIR" <<'PY'
import sys
from pathlib import Path
from prediction_report import load_questions, limit_questions
questions = load_questions(sys.argv[1], require_answer=True)
selected = limit_questions(questions, int(sys.argv[2]) if sys.argv[2] else None)
from score_deepnote_reports import skip_empty_references, write_json
selected_count = len(selected)
selected, skipped = skip_empty_references(selected)
filtered_path = Path(sys.argv[3]) / 'military_generate_questions.filtered.json'
if filtered_path.resolve() == Path(sys.argv[1]).resolve():
    raise ValueError('Filtered output must not overwrite the original question file')
write_json(filtered_path, selected)
write_json(Path(sys.argv[3]) / 'military_generate_questions.skipped.json', {
    'source_question_file': str(Path(sys.argv[1]).resolve()),
    'input_count': len(questions), 'selected_before_filter': selected_count,
    'kept_count': len(selected), 'skipped_count': len(skipped), 'skipped': skipped,
})
for item in skipped:
    print(f"Skipping original question #{item['original_index'] + 1}: {item['reason']} ({item['question']})")
import yaml
settings = yaml.safe_load(Path('../config/config.yaml').read_text(encoding='utf-8'))
for name in ('generation', 'embedding', 'evaluation'):
    if not settings.get(name, {}).get('base_url') or not settings[name].get('model'):
        raise ValueError(f'Configure {name}.base_url and {name}.model first')
for name in ('military.index', 'chunk.json', 'military.index.embedding.json'):
    if not (Path('../data/index/military') / name).is_file():
        raise FileNotFoundError(name)
sys.path.insert(0, 'build_index/es')
from es_support import create_client
es = create_client()
if not es.indices.exists(index='military'):
    raise ValueError('military BM25 index is missing from the configured Elasticsearch service')
print(f'Preflight passed: {len(questions)} input questions; selecting {len(selected)}')
PY
QUESTION_FILE="${OUTPUT_DIR}/military_generate_questions.filtered.json"

COMMON_ARGS=(
  --model_backend api --dataset military --method deepnote
  --question_file "$QUESTION_FILE"
  --eval_backend none --retrieve_top_k 15 --max_step 4 --max_top_k 60
)
if [[ -n "$MAX_QUESTIONS" ]]; then
  COMMON_ARGS+=(--max_questions "$MAX_QUESTIONS")
fi
echo "[1/3] Generating DeepNote vector answers"
"$PYTHON_BIN" main.py "${COMMON_ARGS[@]}" --retrieve_method emb --output_path "$EMB_REPORT"
echo "[2/3] Generating DeepNote BM25 answers"
"$PYTHON_BIN" main.py "${COMMON_ARGS[@]}" --retrieve_method es --output_path "$BM25_REPORT"
echo "[3/3] Scoring both saved reports with run_test rules"
"$PYTHON_BIN" score_deepnote_reports.py \
  --reports "$EMB_REPORT" "$BM25_REPORT" \
  --question-file "$QUESTION_FILE" \
  --comparison-output "${OUTPUT_DIR}/military_generate_deepnote_comparison.json"
echo "Finished. Reports and scores: $OUTPUT_DIR"
