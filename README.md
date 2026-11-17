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

```bash
python main.py --method deepnote --retrieve_top_k 5 --dataset hotpotqa --max_step 3 --max_fail_step 2 --MaxClients 5 --model gpt-4o-mini-2024-07-18 --device cuda:0 
```
The predicted results and evaluation metrics will be automatically saved in the `output/{dataset}/` directory. The evaluation results can be found at the end of the file.
