import time
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from embedding_api import EmbeddingAPI
import json
from llama_index import SimpleDirectoryReader
import csv
from sentence_transformers import SentenceTransformer
import faiss
import argparse
import yaml
from glob import glob
from itertools import chain
from tqdm import tqdm
from llama_index import Document
from llama_index.node_parser import SimpleNodeParser

with open("../../../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    choices=["bge-base-en-v1.5", "gtr-t5-xxl", "Qwen3-Embedding-8B"],
    default="Qwen3-Embedding-8B",
    help="Model to use",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="2wikimultihopqa",
    choices=["asqa", "strategyqa", "2wikimultihopqa", "hotpotqa", "musique"],
    help="Dataset to use",
)
parser.add_argument("--chunk_size", type=int, default=512, help="chunk size")
parser.add_argument("--chunk_overlap", type=int, default=0, help="chunk overlap")
parser.add_argument("--device", type=str, default="cuda:7", help="Device to use")
args = parser.parse_args()


def split_text(data):

    documents = []
    for record in data:
        if record["title"]:
            combined_text = record["title"] + "\n" + record["content"]
        else:
            combined_text = record["content"]
        documents.append(Document(text=combined_text))

    node_parser = SimpleNodeParser.from_defaults(
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
    )
    nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)

    contents = [node.text for node in nodes]
    return contents


def build_index(embeddings, vectorstore_path):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    faiss.write_index(index, vectorstore_path)


if __name__ == "__main__":

    total_start = time.perf_counter()
    embedding_seconds = 0.0
    reused_gtr_embeddings = False
    use_api = config.get("embedding", {}).get("provider", "local") == "api"
    if use_api:
        if args.model and args.model != config["embedding"]["model"]:
            raise ValueError("--model must match embedding.model when provider is api")
        model = EmbeddingAPI(config["embedding"])
    else:
        model = SentenceTransformer(config["model"][args.model or "bge-base-en-v1.5"], device=args.device)
    if args.dataset == "asqa" or args.dataset == "strategyqa":
        dataset_name = "wiki"
    else:
        dataset_name = args.dataset
    vectorstore_path = f"../../../data/corpus/{dataset_name}/{dataset_name}.index"
    contents = []
    print("loading document ...")
    start = time.time()
    if dataset_name == "wiki":
        if not use_api and os.path.exists("../../../data/corpus/wiki/gtr_wikipedia_index.pkl"):
            reused_gtr_embeddings = True
            import pickle

            with open(
                "../../../data/corpus/wiki/gtr_wikipedia_index.pkl", "rb"
            ) as file:
                embeddings = pickle.load(file)
        else:
            with open(
                "../../../data/corpus/wiki/psgs_w100.tsv", "r", encoding="utf-8"
            ) as file:
                tsv_data = csv.DictReader(file, delimiter="\t")
                raw_data = [row["title"] + "\n" + row["text"] for row in tsv_data]
            print("dataset length", len(raw_data))
            embedding_start = time.perf_counter()
            embeddings = model.encode(raw_data, batch_size=100)
            embedding_seconds = time.perf_counter() - embedding_start
    elif dataset_name == "2wikimultihopqa":
        train = json.load(open("../../../data/corpus/2wikimultihopqa/train.json", "r"))
        dev = json.load(open("../../../data/corpus/2wikimultihopqa/dev.json", "r"))
        test = json.load(open("../../../data/corpus/2wikimultihopqa/test.json", "r"))

        data = {}
        for item in tqdm(chain(train, dev, test)):
            for title, sentences in item["context"]:
                para = " ".join(sentences)
                data[para] = title
        contents = [
            {"id": i, "content": text, "title": title}
            for i, (text, title) in enumerate(data.items())
        ]
    elif dataset_name == "hotpotqa":
        import bz2
        from multiprocessing import Pool

        def process_line(line):
            data = json.loads(line)
            item = {
                "id": data["id"],
                "title": data["title"],
                "content": "".join(data["text"]),
            }
            return item

        def generate_indexing_queries_from_bz2(bz2file, dry=False):
            if dry:
                return

            with bz2.open(bz2file, "rt") as f:
                body = [process_line(line) for line in f]

            return body

        filelist = glob("../../../data/corpus/hotpotqa/*/wiki_*.bz2")

        print("Making indexing queries...")
        pool = Pool()

        for result in tqdm(
            pool.imap(generate_indexing_queries_from_bz2, filelist), total=len(filelist)
        ):
            contents.extend(result)
    elif dataset_name == "musique":
        train = [
            json.loads(line.strip())
            for line in open(
                "../../../data/corpus/musique/musique_ans_v1.0_train.jsonl"
            )
        ] + [
            json.loads(line.strip())
            for line in open(
                "../../../data/corpus/musique/musique_full_v1.0_train.jsonl"
            )
        ]
        dev = [
            json.loads(line.strip())
            for line in open("../../../data/corpus/musique/musique_ans_v1.0_dev.jsonl")
        ] + [
            json.loads(line.strip())
            for line in open("../../../data/corpus/musique/musique_full_v1.0_dev.jsonl")
        ]
        test = [
            json.loads(line.strip())
            for line in open("../../../data/corpus/musique/musique_ans_v1.0_test.jsonl")
        ] + [
            json.loads(line.strip())
            for line in open(
                "../../../data/corpus/musique/musique_full_v1.0_test.jsonl"
            )
        ]

        tot = 0
        hist = set()
        for item in tqdm(chain(train, dev, test)):
            for p in item["paragraphs"]:
                stamp = p["title"] + " " + p["paragraph_text"]
                if not stamp in hist:
                    contents.append(
                        {"id": tot, "content": p["paragraph_text"], "title": p["title"]}
                    )
                    hist.add(stamp)
                    tot += 1

    if dataset_name != "wiki":
        contents = split_text(contents)
        embedding_start = time.perf_counter()
        embeddings = model.encode(contents, batch_size=600)
        embedding_seconds = time.perf_counter() - embedding_start
        with open(
            f"../../../data/corpus/{dataset_name}/chunk.json", "w", encoding="utf-8"
        ) as fout:
            json.dump(contents, fout, ensure_ascii=False)
    print("Building index ...")
    faiss_start = time.perf_counter()
    build_index(embeddings, vectorstore_path)
    faiss_seconds = time.perf_counter() - faiss_start
    if use_api:
        model.save_metadata(vectorstore_path)
    else:
        Path(vectorstore_path + ".embedding.json").unlink(missing_ok=True)
    total_seconds = time.perf_counter() - total_start
    stats = {
        "dataset": args.dataset,
        "provider": "api" if use_api else "local",
        "model": config["embedding"]["model"] if use_api else (args.model or "bge-base-en-v1.5"),
        "chunk_size": args.chunk_size if dataset_name != "wiki" else None,
        "chunk_overlap": args.chunk_overlap if dataset_name != "wiki" else None,
        "batch_size": model.batch_size if use_api else (100 if dataset_name == "wiki" else 600),
        "vector_count": int(embeddings.shape[0]),
        "vector_dimension": int(embeddings.shape[1]),
        "reused_gtr_embeddings": reused_gtr_embeddings,
        "embedding_seconds": embedding_seconds,
        "faiss_build_and_save_seconds": faiss_seconds,
        "total_seconds": total_seconds,
    }
    stats_path = vectorstore_path + ".stats.json"
    with open(stats_path, "w", encoding="utf-8") as fout:
        json.dump(stats, fout, ensure_ascii=False, indent=2)
    print(f"Vectors: {stats['vector_count']}, dimension: {stats['vector_dimension']}")
    print(f"Embedding: {embedding_seconds:.2f}s; FAISS build/save: {faiss_seconds:.2f}s")
    print(f"Total: {total_seconds:.2f}s ({total_seconds / 60:.2f} min)")
    print(f"Timing report: {stats_path}")
