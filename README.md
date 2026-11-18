<h1 align="center">
    DeepNote: Note-Centric Deep Retrieval-Augmented Generation
</h1>


We develop **DeepNote**, an adaptive RAG framework that achieves in-depth
and robust exploration of knowledge sources through note-centric adaptive retrieval. DeepNote employs notes as carriers for refining and accumulating knowledge. During in-depth exploration, it uses these notes to determine retrieval timing, formulate retrieval queries, and iteratively assess knowledge growth, ultimately leveraging the best note for answer generation.

![RNote](assets/DeepNote.png)

# Prepare Datasets

All corpus and evaluation files should be placed in the `/data` directory. You can download the experimental data [here](https://drive.google.com/drive/folders/1NeEm-r7l43MQxGS1n7jJ8tPvltgcaPjY?usp=sharing).

We use Wikipedia as the corpus for ASQA and StrategyQA. Due to its large size, please download it separately [here](https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz) and place it in `/data/corpus/wiki/`.

# Retrieval Settings

For different datasets, we employ various retrieval methods:

For 2WikiMQA, MusiQue, and HotpotQA:
- BM25 retrieval based on ElasticSearch
- Dense retrieval with FAISS index using embeddings from BGE model

For ASQA and StrategyQA:
- Dense retrieval with FAISS index using embeddings from GTR model

## Setup ElasticSearch

Install Elasticsearch:
```bash
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.10.2-linux-x86_64.tar.gz
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.10.2-linux-x86_64.tar.gz.sha512
shasum -a 512 -c elasticsearch-7.10.2-linux-x86_64.tar.gz.sha512
tar -xzf elasticsearch-7.10.2-linux-x86_64.tar.gz
cd elasticsearch-7.10.2/
./bin/elasticsearch # Start the server
pkill -f elasticsearch # To stop the server
```

## Build Indices 

### For BM25
```bash
cd src/build_index/es

# 2WikiMQA
python index_2wiki.py

# MusiQue
python index_musique.py

# HotpotQA
python index_hotpotqa.py
```

All ES builders read `es.url` from `config/config.yaml` (ES 7.x and the pinned
7.x Python client). Legacy 2Wiki/MusiQue builders still require all corpus
train/dev/test files listed in their source, and replace existing same-name
indices only when explicitly passed `--reindex`. Without it they refuse to
overwrite. Builders check server identity before writing and reject OpenSearch
and non-7.x servers. Verify the dedicated Elasticsearch endpoint before use.
Missing files must be supplied before using those builders.

Military BM25, from the same `src/build_index/es` directory:

```bash
# One input record per document:
python index_military.py --input_path ../../../data/military_input.json
# Or reuse exact vector chunks for matching retrieval units:
python index_military.py --chunk_file ../../../data/corpus/military/chunk.json
```

Military defaults to the built-in `cjk` analyzer, requiring no IK plugin. To
use an installed IK analyzer pass `--analyzer ik_max_word`. Replacing an existing
military ES index requires adding `--reindex`. Original record IDs may repeat;
ES document IDs use sequential positions to preserve all records. The
`military` index can be queried with `main.py --dataset military
--retrieve_method es`. No embedding or generation service is needed to build
BM25. Military and the other builders save timings under the corpus directory.

BM25 scripts print wall-clock timings and save `bm25.stats.json` under
`data/corpus/{dataset}/`. Reports include data processing, index setup
(including replacement of an existing index), bulk indexing, explicit ES
refresh, and total time through refresh completion. Bulk time includes client
request preparation and network/server processing. `--dry` writes a separate
`bm25.dry.stats.json`, skips ES writes/refresh, and is not a build benchmark.
Repeated runs overwrite the corresponding report. Existing ES endpoint and
index mapping settings in these scripts are unchanged.

### For Dense Retrieval

#### For HotpotQA, 2WikiMQA, and MusiQue
```bash
cd src/build_index/emb
python index.py --dataset hotpotqa --model bge-base-en-v1.5 # e.g., for HotpotQA dataset
```

#### For ASQA and StrategyQA
Since generating GTR embeddings for Wikipedia corpus is time-consuming, you can download the pre-computed GTR embeddings and place them in `data/corpus/wiki/`:
```bash
wget https://huggingface.co/datasets/princeton-nlp/gtr-t5-xxl-wikipedia-psgs_w100-index/resolve/main/gtr_wikipedia_index.pkl
```

Then build FAISS index:
```bash
cd src/build_index/emb
python index.py --dataset asqa --model gtr-t5-xxl
```

# Configuration

## Qwen3-Embedding via an existing vLLM service

### Military corpus

From `src/build_index/emb`, run:

```bash
python index.py --dataset military
# Optional custom JSON array or JSONL file:
python index.py --dataset military --input_path /path/to/records.jsonl
```

The default input is `data/military_input.json`. Each record must have a
string/integer `id` and nonempty `text`; `metadata` is optional. By default each
record is exactly one chunk: text is preserved unchanged, with no extra title
and no deduplication. Ensure each text fits the embedding service input limit;
oversized requests are not silently truncated by this client. To split long
records instead, add `--split_military --chunk_size 512 --chunk_overlap 0`.

Outputs go to `data/corpus/military/`: `military.index`, `chunk.json`,
`chunk_records.json` (ordered source IDs, record positions and metadata; repeated
source IDs are preserved), embedding metadata and timing
statistics. This path generates new embeddings even if the input has an
`embedding` field; importing precomputed vectors is not implemented here.
API mode with one record per chunk avoids importing LlamaIndex and
SentenceTransformers. This addition builds the military vector index only;
the main evaluation CLI supports this index via `--dataset military
--retrieve_method emb`; BM25 builders do not yet support military data.

`config/config.yaml` now defaults to the embedding API at
`http://127.0.0.1:8200/v1`, with served model name `Qwen3-Embedding-8B`.
Change this address if DeepNote runs on a different host/container. No
Elasticsearch service is required when running with `--retrieve_method emb`.
The generation model configuration remains separate and unchanged.

Build an index from the original corpus (run from the project root):

```bash
cd src/build_index/emb
python index.py --dataset 2wikimultihopqa
# Or: python index.py --dataset musique
# Or: python index.py --dataset hotpotqa
```

Then run evaluation from `src/`, selecting your configured generation model:

```bash
cd ../..
python main.py --dataset 2wikimultihopqa --retrieve_method emb --method deepnote --model qwen2.5-7b-instruct --MaxClients 1
```

The example generation model above still uses the original local loading path;
it does not connect to a separately deployed G9v3 generation service.

The API client batches documents using `embedding.batch_size`, L2-normalizes
both document/query vectors. `embedding.query_instruction` defaults to an empty
string, so queries are sent unchanged, with no instruction or Query prefix.
An optional nonempty instruction is applied only to queries, following the
[Qwen3-Embedding model card](https://huggingface.co/Qwen/Qwen3-Embedding-8B).
Rebuild existing BGE/GTR indices before using Qwen embeddings. A companion
`*.index.embedding.json` records the embedding settings and must stay with the
index. Changing the embedding model requires rebuilding the index. Changing
only the query instruction does not require rebuilding document vectors.
Building writes the usual dataset index and replaces any existing index there;
back up old indices first if needed. ASQA/StrategyQA use the shared `wiki.index`;
in API mode their precomputed GTR embeddings are intentionally not reused.

To restore the original local embeddings, set `embedding.provider: local` and
rebuild with `--model bge-base-en-v1.5` (or `--model gtr-t5-xxl` for Wikipedia).
Only the service needs a Qwen3-compatible runtime; the API client does not load
Qwen3 weights in DeepNote's Python environment.

You can configure your API key, URL, and other settings in the `./config/config.yaml` file.


# Training DeepNote

The training process consists of three main steps:

## 1. Generate Training Data
Generate the initial training data using LLaMA model:
```bash
python gen_dpo_data.py \
    --model llama-3.1-8b-instruct \
    --batch_size 9 \
    --output_path ../data/dpo_data \
    --device 0,1,2,3
```

## 2. Data Selection
Filter and process the generated data:
```bash
python select_dpo_data.py \
    --output_path ../data/dpo/processed/train.jsonl \
    --init_num 1900 \
    --refine_num 1900 \
    --query_num 1900
```

## 3. Start Training
Launch the training process:
```bash
bash train.sh
```


# Running DeepNote and Evaluation

## Save answers, retrieved materials and timings without scoring

### Jiugе / G9 generation API

Set `generation.base_url` in `config/config.yaml` to the deployed chat service
URL ending in `/v1`, and `generation.model` to the exact served name (for
example `9g` or `G9v3-39A5B`, as returned by `/v1/models`). Configure its API key
locally or set the `DEEPNOTE_GENERATION_API_KEY` environment variable. This is
separate from the embedding endpoint at port 8200 and the evaluation judge.

```bash
cd src
python main.py --model_backend api --dataset military --retrieve_method emb --method deepnote --question_file /path/to/questions.json --eval_backend none --output_path ../output/military_jiuge.json --MaxClients 1
```

API mode uses `generation.model` unless overridden with `--model 9g`. All
DeepNote generation steps call `/v1/chat/completions` using user messages;
the service handles the chat template. No local vLLM weights are loaded.
API generation uses `temperature: 0.6`, omits top_p, output token limits,
penalties, seed, stop and stream, and sends `enable_thinking: false` inside
`chat_template_kwargs`. Timeout is 120 seconds per request. Retryable network,
timeout, rate-limit and server failures allow at most 3 total attempts with
2 seconds between attempts; SDK retries are disabled to avoid multiplying calls.
With `strip_thinking: true`, any text through the last `</think>` is removed
before a response is used as a query, note or final answer. The original local
model path remains available with `--model_backend auto` (default) or `local`.

Run from `src/`:

```bash
python main.py --dataset military --retrieve_method emb --method deepnote --model qwen2.5-7b-instruct --question_file /path/to/questions.json --eval_backend none --output_path ../output/military_run.json --MaxClients 1
```

The generation model still uses the existing configured loading/calling path;
replace the example model with a supported configured model. Input may be a
JSON array or JSONL with a `question` string in each record. `answer` and gold
documents are optional in this mode. Example: `[{"id":"q1","question":"Your question"}]`.
The dataset selects the existing corpus/index, not a different retrieval flow.

One JSON report contains `run`, `summary`, and `results`. Each result has the
answer in `output`, full per-round materials in `ref_log`, deduplicated document
texts in `retrieved_materials`, source ID (if supplied), and `performance`.
Performance records per-question total, retrieval and generation seconds plus
call counts. Retrieval includes query embedding and index search; generation
includes initial/refined notes, new queries, comparisons and final answers.
Calls count workflow invocations rather than SDK-internal retries. Retries and
waiting contribute to measured time. This mode does not call the judge.

Summary includes elapsed question-processing wall time, successful-question
throughput, mean/P50/P95 successful latency and stage time sums. Model/index
loading and scoring are excluded; report-writing time between batches is
included, and no warmup is removed. Under concurrency, summed stage times can
exceed wall time. Token usage is unavailable. Failed questions are recorded
with errors; their partial retrieved materials are not retained. Reports are
atomically updated after each completed batch; completed batches survive a
later interruption. Resuming these reports is not yet supported. An existing
output report at the selected path is replaced.

Add `--max_questions 100` to process at most the first 100 questions in file
order; omit the option to process all. Values must be positive integers, and
limits larger than the input simply use all available questions. Logs show
input/selected counts and resumed/remaining counts. Generation reports record
`run.input_question_count`, `run.max_questions`, `run.question_count`, and
actual completed/successful/failed counts in `summary`. The same option is
available in the standalone `eval_jybigdata.py` scorer, limiting the first N
prediction records without renumbering their question IDs.

## Unified jybigdata evaluation

`main.py` now defaults to `--eval_backend jybigdata` for the DeepNote method.
Generation remains the original DeepNote pipeline; scoring uses the supplied
jybigdata answerability/correctness prompts, answer substring shortcut, and
document recall rules. No jybigdata RAG server or memory service is required.
Use `--eval_backend original` to retain the previous Acc/F1/EM evaluation (also
required for the `base` and `base_wo_retri` methods).

Set `evaluation.base_url`, `evaluation.model`, `evaluation.api_key`, and
`evaluation.max_workers` in `config/config.yaml` to match the comparison
framework's judge. These settings are separate from the answer generation
model and embedding service. No real API key is included in the adapted code.
The judge SDK call sends the original user prompt without overriding sampling
settings; matching judge server defaults is also necessary for comparison.

Use the same question JSON array as the comparison framework, for example:

```json
[
  {
    "question": "Example question",
    "answer": "Reference answer",
    "gold_docs": ["Reference document text"]
  }
]
```

Alternatively supply `references: [["Reference document text", "source"]]`.
As in the supplied evaluator, `references` overrides `gold_docs`, and literal
`...` is removed from reference text. Gold documents are used by the evaluator
only, not passed to the DeepNote answer-generation prompts.

Run from `src/` (replace the generation model with a supported configured one):

```bash
python main.py --dataset musique --retrieve_method emb --method deepnote --model qwen2.5-7b-instruct --question_file /path/to/musique_questions.json --eval_backend jybigdata
```

Military vector retrieval is now accepted by the evaluation runner too:
use `--dataset military --retrieve_method emb --question_file /path/to/military_questions.json`.
The military corpus input is not an evaluation question file.

To score existing DeepNote predictions without running generation again:

```bash
python eval_jybigdata.py --pred_path /path/to/predictions.jsonl --question_file /path/to/the_same_questions.json
```

Prediction IDs must be zero-based positions in that question file; question
text and gold answers are checked to prevent mismatched evaluation. If the
prediction file already contains `gold_docs`, `--question_file` can be omitted.
Old Acc/F1/EM summary lines are ignored. Scores are written beside predictions:
`*.jybigdata.json` contains per-question judgments, reasons and recalled texts;
`*.jybigdata.json.summary.json` contains aggregate metrics; a companion judge
cache is saved after each completed question.

Metrics follow the supplied evaluator exactly:

- `accuracy`: correct AND answerable questions divided by answerable questions.
- Per-question `retrieved_rate`: gold document matches divided by
  `(gold document count + 0.001)`. A match is a substring of any retrieved
  document after replacing newlines with spaces. All DeepNote retrieval rounds
  are included, even when the corresponding note was not selected.
- `recall_rate`: sum of retrieved rates for answerable questions divided by
  `(answerable question count + 0.001)`. Rates are in 0–1, not percentages.

The original parser fallbacks are retained: malformed answerability JSON
defaults to answerable, and malformed correctness JSON defaults to wrong;
reasons identify these cases. API failures are saved as `evaluation_error`
and counted separately; such questions lack an answerability verdict and are
excluded from that denominator. Inspect errors before comparing results.
DeepNote does not currently record the comparison framework's
`n_token_retrieval`: token statistics are explicitly unavailable, not zero.

```bash
python main.py --method deepnote --retrieve_top_k 5 --dataset hotpotqa --max_step 3 --max_fail_step 2 --MaxClients 5 --model gpt-4o-mini-2024-07-18 --device cuda:0 
```
The predicted results and evaluation metrics will be automatically saved in the `output/{dataset}/` directory. The evaluation results can be found at the end of the file.
