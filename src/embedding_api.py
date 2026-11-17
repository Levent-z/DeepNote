"""Shared vLLM embedding client for indexing and retrieval."""

import json
from pathlib import Path

import numpy as np
from openai import OpenAI


class EmbeddingAPI:
    def __init__(self, settings):
        self.model = settings["model"]
        self.batch_size = int(settings.get("batch_size", 8))
        if self.batch_size < 1:
            raise ValueError("embedding.batch_size must be positive")
        self.instruction = settings.get("query_instruction", "")
        self.client = OpenAI(
            base_url=settings["base_url"],
            api_key=settings.get("api_key") or "EMPTY",
            timeout=float(settings.get("timeout", 120)),
            max_retries=3,
        )

    def encode(self, texts, batch_size=None, query=False):
        texts = list(texts)
        if not texts:
            raise ValueError("Cannot embed an empty corpus")
        if query and self.instruction:
            texts = [f"Instruct: {self.instruction}\nQuery:{text}" for text in texts]
        batches = []
        size = self.batch_size  # Limit requests even when legacy callers pass 600.
        for start in range(0, len(texts), size):
            inputs = texts[start:start + size]
            response = self.client.embeddings.create(
                model=self.model, input=inputs, encoding_format="float"
            )
            items = sorted(response.data, key=lambda item: item.index)
            if [item.index for item in items] != list(range(len(inputs))):
                raise ValueError("Embedding response does not match input order/count")
            vectors = np.asarray([item.embedding for item in items], dtype=np.float32)
            if vectors.ndim != 2 or vectors.shape[1] == 0 or not np.isfinite(vectors).all():
                raise ValueError("Invalid embedding vectors")
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            if np.any(norms == 0):
                raise ValueError("Embedding service returned a zero vector")
            batches.append(vectors / norms)
        return np.ascontiguousarray(np.concatenate(batches), dtype=np.float32)

    def metadata(self):
        return {"provider": "openai-compatible", "model": self.model,
                "normalize": True}

    def save_metadata(self, index_path):
        Path(str(index_path) + ".embedding.json").write_text(
            json.dumps(self.metadata(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def validate_index(self, index_path):
        path = Path(str(index_path) + ".embedding.json")
        if not path.exists():
            raise ValueError("Missing embedding metadata; rebuild the index")
        stored = json.loads(path.read_text(encoding="utf-8"))
        # Query-only formatting does not change document vectors. Also accept
        # metadata written before this field was removed.
        stored.pop("query_instruction", None)
        if stored != self.metadata():
            raise ValueError("Embedding configuration differs from index metadata; rebuild the index")
