"""Build a DeepNote FAISS index from precomputed vectors and matching text."""

import argparse
import json
from pathlib import Path

import numpy as np


def load_input(vector_path, chunk_path=None):
    vector_path = Path(vector_path)
    if vector_path.suffix.lower() == ".npy":
        vectors = np.load(vector_path, allow_pickle=False)
        records = None
    elif vector_path.suffix.lower() == ".jsonl":
        with vector_path.open(encoding="utf-8-sig") as source:
            records = [json.loads(line) for line in source if line.strip()]
        vectors = None
    elif vector_path.suffix.lower() == ".json":
        records = json.loads(vector_path.read_text(encoding="utf-8-sig"))
        vectors = None
    else:
        raise ValueError("Vector file must be .npy, .json, or .jsonl")

    if records is not None:
        if not isinstance(records, list) or not records:
            raise ValueError("Vector file must contain a nonempty array")
        if isinstance(records[0], dict):
            if any(not isinstance(record, dict) or "embedding" not in record for record in records):
                raise ValueError("Every record must contain an embedding")
            vectors = [record["embedding"] for record in records]
            embedded_texts = [record.get("text") for record in records]
        else:
            vectors = records
            embedded_texts = None
    else:
        embedded_texts = None

    if chunk_path:
        texts = json.loads(Path(chunk_path).read_text(encoding="utf-8-sig"))
    else:
        texts = embedded_texts
    if not isinstance(texts, list) or any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Provide nonempty text for each vector, either in records or via --chunk-file")
    try:
        vectors = np.asarray(vectors, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("Vectors must form a rectangular numeric matrix") from exc
    if vectors.ndim != 2 or not vectors.shape[0] or not vectors.shape[1] or len(texts) != vectors.shape[0]:
        raise ValueError("Vector matrix must be nonempty and have one row per text")
    if not np.isfinite(vectors).all():
        raise ValueError("Vectors must contain only finite values")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms == 0):
        raise ValueError("Vectors must have finite, nonzero norms")
    return np.ascontiguousarray(vectors / norms, dtype=np.float32), texts


def import_vectors(vector_path, dataset, model, output_root, chunk_path=None, overwrite=False):
    if not dataset or dataset in (".", "..") or Path(dataset).name != dataset or "/" in dataset or "\\" in dataset:
        raise ValueError("Dataset must be a single directory name")
    if not model.strip():
        raise ValueError("The embedding model name is required")
    vector_path = Path(vector_path)
    package = vector_path if vector_path.is_dir() else None
    manifest = None
    if package:
        if chunk_path:
            raise ValueError("--chunk-file cannot be used with a portable package")
        manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8-sig"))
        vector_path = package / "records.jsonl"
        declared_model = manifest.get("embedding_model_from_registry")
        if declared_model and declared_model != model:
            raise ValueError("Model differs from the portable package manifest")
    directory = Path(output_root) / dataset
    index_path = directory / f"{dataset}.index"
    chunk_output = directory / "chunk.json"
    metadata_path = Path(str(index_path) + ".embedding.json")
    stats_path = Path(str(index_path) + ".stats.json")
    records_path = directory / "chunk_records.json"
    outputs = (index_path, chunk_output, metadata_path, stats_path, records_path)
    inputs = {Path(path).resolve() for path in (vector_path, chunk_path) if path}
    if any(path.resolve() in inputs for path in outputs):
        raise ValueError("Output paths must not overwrite input files")
    if not overwrite and any(path.exists() for path in outputs):
        raise FileExistsError("Output already exists; pass --overwrite to replace it")
    vectors, texts = load_input(vector_path, chunk_path)
    records_metadata = None
    if package:
        with vector_path.open(encoding="utf-8-sig") as source:
            records = [json.loads(line) for line in source if line.strip()]
        if manifest.get("count") != len(records) or manifest.get("dimension") != vectors.shape[1]:
            raise ValueError("Portable package count or dimension does not match its records")
        if any(not isinstance(record.get("id"), str) or not isinstance(record.get("metadata"), dict)
               for record in records):
            raise ValueError("Portable records must contain a string id and object metadata")
        records_metadata = [{"chunk_index": position, "id": record["id"], "metadata": record["metadata"]}
                            for position, record in enumerate(records)]
    import faiss
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    directory.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    chunk_output.write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
    metadata_path.write_text(json.dumps({"provider": "openai-compatible", "model": model, "normalize": True},
                                        ensure_ascii=False, indent=2), encoding="utf-8")
    stats_path.write_text(json.dumps({"dataset": dataset, "chunk_strategy": "record" if package else "unknown",
                                      "vector_count": vectors.shape[0], "vector_dimension": vectors.shape[1],
                                      "source_package": str(package.resolve()) if package else None},
                                     ensure_ascii=False, indent=2), encoding="utf-8")
    if records_metadata is not None:
        records_path.write_text(json.dumps(records_metadata, ensure_ascii=False), encoding="utf-8")
    elif records_path.exists():
        records_path.unlink()
    return index_path, vectors.shape


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vectors", required=True, help="Portable package directory, .npy matrix, or JSON/JSONL vectors")
    parser.add_argument("--chunk-file", help="JSON array of texts in exactly the vector row order")
    parser.add_argument("--dataset", required=True, help="Dataset name used by main.py")
    parser.add_argument("--model", required=True, help="Embedding model that produced the vectors")
    parser.add_argument("--output-root", default=str(Path(__file__).resolve().parents[3] / "data" / "index"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    path, shape = import_vectors(args.vectors, args.dataset, args.model, args.output_root, args.chunk_file, args.overwrite)
    print(f"Built {path}: {shape[0]} vectors, dimension {shape[1]}")


if __name__ == "__main__":
    main()
