"""Shared ES configuration and compatibility with the README's ES 7 server."""
import json
from pathlib import Path


def create_client(timeout=100):
    import yaml
    from elasticsearch import Elasticsearch
    config_path = Path(__file__).resolve().parents[3] / "config/config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    client = Elasticsearch(hosts=config["es"]["url"], timeout=timeout)
    validate_server(client.info())
    return client


def validate_server(info):
    version = info.get("version", {})
    if version.get("distribution", "").lower() == "opensearch" or "opensearch" in info.get("tagline", "").lower():
        raise ValueError("Refusing to build on OpenSearch. Check es.url; use the dedicated Elasticsearch service.")
    if not str(version.get("number", "")).startswith("7."):
        raise ValueError("BM25 builders require the dedicated Elasticsearch 7.x service; verify es.url before writing")


def prepare_index_body(body):
    # Legacy scripts contain typed mappings and the removed `string` field type.
    mappings = body.get("mappings", {})
    if "doc" in mappings:
        body["mappings"] = mappings["doc"]
    for field in body.get("mappings", {}).get("properties", {}).values():
        if field.get("type") == "string":
            field.update(type="text", index=False)
    analysis = body.get("settings", {}).get("analysis", {})
    if analysis and "analyzer" not in analysis:
        body["settings"]["analysis"] = {"analyzer": analysis}
    return json.dumps(body)
