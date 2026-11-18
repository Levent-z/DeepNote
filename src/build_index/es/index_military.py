"""Build a military BM25 index from original records or vector-index chunks."""
import argparse
import html
import json
from pathlib import Path
import sys

from build_timing import BuildTiming
from es_support import create_client

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "emb"))
from military_data import load_military_records

ROOT = Path(__file__).resolve().parents[3]


def documents(input_path, chunk_file=None):
    if chunk_file:
        chunks = json.loads(Path(chunk_file).read_text(encoding="utf-8-sig"))
        if not isinstance(chunks, list) or not chunks or any(not isinstance(t, str) or not t.strip() for t in chunks):
            raise ValueError("chunk_file must contain a nonempty JSON array of text strings")
        records = [{"id": i, "text": text} for i, text in enumerate(chunks)]
    else:
        records = load_military_records(input_path)
    for position, record in enumerate(records):
        # Use position as ES _id: the supplied corpus contains repeated source IDs.
        title = record["text"].splitlines()[0]
        title = title.removeprefix("标题:").removeprefix("标题：").strip() or "military"
        yield position, {
            "id": str(record["id"]), "url": "empty", "title": title,
            "title_unescape": html.unescape(title), "text": record["text"],
            "title_bigram": title, "title_unescape_bigram": html.unescape(title),
            "text_bigram": record["text"], "metadata": record.get("metadata", {}),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_path", default=str(ROOT / "data/military_input.json"))
    parser.add_argument("--chunk_file", help="Use existing vector chunk.json for identical BM25/vector text units")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--analyzer", default="cjk", help="Built-in cjk or an installed analyzer, e.g. ik_max_word")
    parser.add_argument("--reindex", action="store_true", help="Allow replacing an existing military index")
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch_size must be positive")
    timing = BuildTiming()
    docs = list(documents(args.input_path, args.chunk_file))
    timing.checkpoint("data_processing_seconds")
    es = None
    if not args.dry:
        es = create_client()
        if es.indices.exists(index="military"):
            if not args.reindex:
                raise ValueError("military index already exists; add --reindex to replace it")
            es.indices.delete(index="military")
        properties = {name: {"type": "text", "analyzer": args.analyzer} for name in
                      ("title", "title_unescape", "text", "title_bigram", "title_unescape_bigram", "text_bigram")}
        properties.update(id={"type": "keyword"}, url={"type": "keyword"}, metadata={"type": "object", "enabled": False})
        es.indices.create(index="military", body={"mappings": {"properties": properties}})
    timing.checkpoint("index_setup_seconds")
    if not args.dry:
        from tqdm import tqdm
        for start in tqdm(range(0, len(docs), args.batch_size), desc="Indexing military BM25"):
            lines = []
            for position, source in docs[start:start + args.batch_size]:
                lines.extend([json.dumps({"index": {"_id": str(position)}}), json.dumps(source, ensure_ascii=False)])
            response = es.bulk(index="military", body="\n".join(lines) + "\n", timeout="100s")
            if response.get("errors"):
                raise RuntimeError(f"ES bulk write failed: {response}")
    timing.checkpoint("bulk_index_seconds")
    timing.finish(es, "military", len(docs), args.dry)
    print(f"{len(docs)} documents {'prepared' if args.dry else 'indexed'}")


if __name__ == "__main__":
    main()
