#!/usr/bin/env python3
"""Export one OpenSearch index as a portable vector dataset."""

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen


def request_json(base_url, method, path, body=None, timeout=120):
    headers = {"Accept": "application/json"}
    token = os.environ.get("OPENSEARCH_AUTHORIZATION")
    if token:
        headers["Authorization"] = token
    data = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body).encode("utf-8")
    request = Request(base_url.rstrip("/") + path, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as exc:
        raise RuntimeError(f"OpenSearch returned HTTP {exc.code} for {method} {path}") from exc


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def export_index(base_url, index, output, dataset_id=None, model=None, text_field="text",
                 vector_field="embedding", batch_size=100, scroll="5m"):
    if not index or any(char in index for char in "/,*?\\") or index in (".", ".."):
        raise ValueError("Specify exactly one concrete index name")
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    if text_field == vector_field:
        raise ValueError("Text and vector fields must differ")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    index_path = quote(index, safe="")
    mapping = request_json(base_url, "GET", f"/{index_path}/_mapping")
    if set(mapping) != {index}:
        raise ValueError("Index name must resolve to exactly one concrete index")
    properties = mapping[index]["mappings"].get("properties", {})
    vector_mapping = properties.get(vector_field, {})
    dimension = vector_mapping.get("dimension")
    if vector_mapping.get("type") != "knn_vector" or not isinstance(dimension, int) or dimension < 1:
        raise ValueError(f"{vector_field} must be a top-level knn_vector field with a dimension")
    cluster = request_json(base_url, "GET", "/")
    version = cluster.get("version", {}).get("number")
    space = vector_mapping.get("method", {}).get("space_type")
    distance = {"cosinesimil": "cosine", "l2": "l2", "innerproduct": "innerproduct"}.get(space, space)
    dataset_id = dataset_id or index
    with tempfile.TemporaryDirectory(prefix=".opensearch-export-", dir=output.parent) as temp:
        package = Path(temp) / "package"
        package.mkdir()
        scroll_id = None
        count = 0
        try:
            response = request_json(base_url, "POST", f"/{index_path}/_search?scroll={quote(scroll)}",
                                    {"size": batch_size, "sort": ["_doc"], "track_total_hits": True,
                                     "query": {"match_all": {}}})
            scroll_id = response.get("_scroll_id")
            total = response.get("hits", {}).get("total")
            if isinstance(total, dict):
                if total.get("relation") != "eq":
                    raise ValueError("OpenSearch did not return an exact document count")
                total = total.get("value")
            if not isinstance(total, int) or isinstance(total, bool) or total < 0:
                raise ValueError("OpenSearch did not return a valid document count")
            with (package / "records.jsonl").open("w", encoding="utf-8", newline="\n") as records:
                while True:
                    scroll_id = response.get("_scroll_id", scroll_id)
                    if response.get("timed_out") or response.get("_shards", {}).get("failed", 0):
                        raise RuntimeError("OpenSearch search timed out or had failed shards")
                    hits = response.get("hits", {}).get("hits")
                    if not isinstance(hits, list):
                        raise ValueError("Invalid search response: missing hits")
                    if not hits:
                        break
                    if not scroll_id:
                        raise ValueError("OpenSearch did not return a scroll ID")
                    for hit in hits:
                        source = hit.get("_source")
                        if not isinstance(source, dict) or not isinstance(hit.get("_id"), str):
                            raise ValueError("Every document must have _id and _source")
                        text, vector = source.get(text_field), source.get(vector_field)
                        if not isinstance(text, str) or not text.strip():
                            raise ValueError(f"Document {hit['_id']} has missing or empty text")
                        if not isinstance(vector, list) or len(vector) != dimension or any(
                            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                            for value in vector
                        ):
                            raise ValueError(f"Document {hit['_id']} has an invalid {dimension}-dimensional vector")
                        metadata = {key: value for key, value in source.items() if key not in (text_field, vector_field)}
                        record = {"id": hit["_id"], "text": text, "embedding": vector, "metadata": metadata}
                        records.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                        count += 1
                    response = request_json(base_url, "POST", "/_search/scroll",
                                            {"scroll": scroll, "scroll_id": scroll_id})
            if count != total:
                raise ValueError(f"Exported {count} documents but search reported {total}")
        finally:
            if scroll_id:
                try:
                    request_json(base_url, "DELETE", "/_search/scroll", {"scroll_id": scroll_id})
                except Exception as exc:
                    print(f"Warning: could not clear scroll context: {exc}")
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        manifest = {"format": "jsonl", "encoding": "UTF-8", "dataset_id": dataset_id, "count": count,
                    "dimension": dimension, "distance": distance, "embedding_model_from_registry": model,
                    "model_weights_and_preprocessing_verified": False, "source_index": index,
                    "source_snapshot_uuid": None, "source_snapshot_end_time": None, "exported_at": now,
                    "fields": {"id": "OpenSearch _id", "text": f"_source.{text_field}",
                               "embedding": f"_source.{vector_field} (original vector)",
                               "metadata": "Other complete _source fields"}, "asset_files_included": False}
        write_json(package / "manifest.json", manifest)
        write_json(package / "source-manifest.json", {"dataset_id": dataset_id, "index": index,
                                                       "opensearch_version": version, "count": count,
                                                       "snapshot": None, "source_type": "live_index"})
        write_json(package / "source-mapping.json", mapping)
        (package / "README.md").write_text(
            f"# {dataset_id} 通用数据包\n\n来自 OpenSearch 索引 `{index}`，共 {count} 条记录。"
            "`records.jsonl` 为 UTF-8，每行包含 `id`、`text`、`embedding` 和 `metadata`。\n\n"
            "`metadata` 保留除正文和向量以外的完整 `_source` 字段，包括嵌套对象。"
            "向量原样导出，模型权重及预处理未独立核实；查询端需要使用兼容模型及预处理。\n\n"
            "本包来自在线索引，不是 OpenSearch 快照，不包含独立附件。"
            "BM25 倒排索引需在目标库重建。校验命令：`sha256sum -c SHA256SUMS`。\n", encoding="utf-8")
        with (package / "SHA256SUMS").open("w", encoding="utf-8", newline="\n") as sums:
            for path in sorted(package.iterdir()):
                if path.name == "SHA256SUMS":
                    continue
                digest = hashlib.sha256()
                with path.open("rb") as source:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(block)
                sums.write(f"{digest.hexdigest()}  {path.name}\n")
        if output.exists():
            raise FileExistsError(f"Output already exists: {output}")
        package.rename(output)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:9201")
    parser.add_argument("--index", required=True)
    parser.add_argument("--output", required=True, help="New output directory")
    parser.add_argument("--dataset-id")
    parser.add_argument("--model", help="Documented embedding model name; not independently verified")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--vector-field", default="embedding")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--scroll", default="5m")
    args = parser.parse_args()
    count = export_index(args.url, args.index, args.output, args.dataset_id, args.model,
                         args.text_field, args.vector_field, args.batch_size, args.scroll)
    print(f"Exported {count} records to {args.output}")


if __name__ == "__main__":
    main()
