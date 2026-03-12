from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from research_ai.indexing.chunking import PaperChunk

logger = logging.getLogger(__name__)

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None


class FaissVectorStore:
    def __init__(self, index_dir: str | Path, index_name: str = "paper_chunks") -> None:
        if faiss is None:
            raise ImportError("faiss is required to use the vector store")

        self.index_dir = Path(index_dir)
        self.index_name = index_name
        self.index_path = self.index_dir / f"{index_name}.faiss"
        self.metadata_path = self.index_dir / f"{index_name}_metadata.json"
        self._index = None
        self._dimension: int | None = None
        self._records: list[dict[str, Any]] = []
        self._chunk_ids: set[str] = set()

    def add_embeddings(self, chunks: list[PaperChunk], embeddings: np.ndarray) -> None:
        if not chunks:
            return

        vectors = embeddings.astype("float32")
        if vectors.ndim != 2:
            raise ValueError("Embeddings must be a 2D array")

        if self._index is None:
            self._dimension = vectors.shape[1]
            self._index = faiss.IndexFlatIP(self._dimension)
        elif vectors.shape[1] != self._dimension:
            raise ValueError("Embedding dimension does not match existing index")

        filtered_vectors: list[np.ndarray] = []
        filtered_records: list[dict[str, Any]] = []

        for chunk, vector in zip(chunks, vectors, strict=False):
            if chunk.chunk_id in self._chunk_ids:
                continue
            filtered_vectors.append(vector)
            filtered_records.append(chunk.model_dump())
            self._chunk_ids.add(chunk.chunk_id)

        if not filtered_vectors:
            return

        matrix = np.vstack(filtered_vectors).astype("float32")
        faiss.normalize_L2(matrix)
        self._index.add(matrix)
        self._records.extend(filtered_records)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        fetch_k: int | None = None,
    ) -> list[dict[str, Any]]:
        if self._index is None or not self._records:
            return []

        vector = query_vector.astype("float32").reshape(1, -1)
        faiss.normalize_L2(vector)
        candidate_count = fetch_k or max(top_k * 5, top_k)
        scores, indices = self._index.search(vector, candidate_count)

        results: list[dict[str, Any]] = []
        for score, index in zip(scores[0], indices[0], strict=False):
            if index < 0 or index >= len(self._records):
                continue
            record = self._records[index]
            if not _matches_filters(record, filters):
                continue
            results.append(
                {
                    "paper_id": record["paper_id"],
                    "paper_title": record["paper_title"],
                    "section": record["section_name"],
                    "score": float(score),
                    "text": record["text"],
                    "year": record.get("year"),
                    "venue": record.get("venue"),
                    "keywords": record.get("keywords", []),
                    "chunk_id": record["chunk_id"],
                }
            )
            if len(results) >= top_k:
                break

        return results

    def save(self) -> None:
        if self._index is None:
            return

        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_path))
        payload = {
            "dimension": self._dimension,
            "records": self._records,
        }
        self.metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Saved FAISS index to %s", self.index_path)

    def load(self) -> "FaissVectorStore":
        if not self.index_path.exists() or not self.metadata_path.exists():
            raise FileNotFoundError("Saved FAISS index or metadata sidecar not found")

        self._index = faiss.read_index(str(self.index_path))
        payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self._dimension = payload.get("dimension")
        self._records = payload.get("records", [])
        self._chunk_ids = {record["chunk_id"] for record in self._records}
        logger.info("Loaded FAISS index from %s", self.index_path)
        return self

    @property
    def size(self) -> int:
        return len(self._records)


def _matches_filters(record: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True

    if not _match_year(record.get("year"), filters.get("year")):
        return False
    if not _match_keywords(record.get("keywords", []), filters.get("keywords")):
        return False
    if not _match_venue(record.get("venue"), filters.get("venue")):
        return False
    return True


def _match_year(value: int | None, rule: Any) -> bool:
    if rule is None:
        return True
    if value is None:
        return False
    if isinstance(rule, dict):
        min_year = rule.get("min")
        max_year = rule.get("max")
        if min_year is not None and value < int(min_year):
            return False
        if max_year is not None and value > int(max_year):
            return False
        return True
    if isinstance(rule, str):
        rule = rule.strip()
        if rule.startswith(">="):
            return value >= int(rule[2:])
        if rule.startswith("<="):
            return value <= int(rule[2:])
        if rule.startswith(">"):
            return value > int(rule[1:])
        if rule.startswith("<"):
            return value < int(rule[1:])
        if "-" in rule:
            start, end = rule.split("-", 1)
            return int(start) <= value <= int(end)
        return value == int(rule)
    return value == int(rule)


def _match_keywords(values: list[str], rule: Any) -> bool:
    if not rule:
        return True
    if isinstance(rule, str):
        requested = [rule.lower()]
    else:
        requested = [str(item).lower() for item in rule]
    available = [value.lower() for value in values]
    return any(any(keyword in item or item in keyword for item in available) for keyword in requested)


def _match_venue(value: str | None, rule: Any) -> bool:
    if not rule:
        return True
    if value is None:
        return False
    return str(rule).lower() in value.lower()

