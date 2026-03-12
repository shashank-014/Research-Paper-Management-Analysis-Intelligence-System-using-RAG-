from __future__ import annotations

import json
import logging
from hashlib import sha1
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class BaseEmbedder:
    model_name: str

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query], batch_size=1)[0]


class EmbeddingCache:
    def __init__(self, cache_path: str | Path | None = None) -> None:
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: dict[str, list[float]] = {}
        if self.cache_path and self.cache_path.exists():
            self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))

    def get(self, key: str) -> list[float] | None:
        return self._cache.get(key)

    def set(self, key: str, value: list[float]) -> None:
        self._cache[key] = value

    def flush(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._cache), encoding="utf-8")


def create_embedder(
    provider: str = "sentence_transformers",
    model_name: str | None = None,
    cache_path: str | Path | None = None,
    api_key: str | None = None,
) -> BaseEmbedder:
    provider_name = provider.lower()
    if provider_name == "openai":
        return OpenAIEmbedder(
            model_name=model_name or "text-embedding-3-small",
            api_key=api_key,
            cache_path=cache_path,
        )
    return SentenceTransformerEmbedder(
        model_name=model_name or "all-MiniLM-L6-v2",
        cache_path=cache_path,
    )


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_path: str | Path | None = None) -> None:
        self.model_name = model_name
        self.cache = EmbeddingCache(cache_path)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ImportError("sentence-transformers is required for local embeddings") from exc

        self._model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        vectors: list[np.ndarray | None] = [None] * len(texts)
        missing_texts: list[str] = []
        missing_positions: list[int] = []

        for index, text in enumerate(texts):
            cache_key = _cache_key(self.model_name, text)
            cached = self.cache.get(cache_key)
            if cached is not None:
                vectors[index] = np.array(cached, dtype="float32")
            else:
                missing_texts.append(text)
                missing_positions.append(index)

        if missing_texts:
            generated = self._model.encode(
                missing_texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
            for offset, vector in enumerate(generated):
                position = missing_positions[offset]
                vectors[position] = vector.astype("float32")
                self.cache.set(_cache_key(self.model_name, missing_texts[offset]), vectors[position].tolist())
            self.cache.flush()

        return np.vstack([vector for vector in vectors if vector is not None]).astype("float32")


class OpenAIEmbedder(BaseEmbedder):
    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: str | None = None,
        cache_path: str | Path | None = None,
    ) -> None:
        self.model_name = model_name
        self.cache = EmbeddingCache(cache_path)
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError("openai package is required for OpenAI embeddings") from exc

        self._client = OpenAI(api_key=api_key)

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        vectors: list[np.ndarray | None] = [None] * len(texts)
        missing_texts: list[str] = []
        missing_positions: list[int] = []

        for index, text in enumerate(texts):
            cache_key = _cache_key(self.model_name, text)
            cached = self.cache.get(cache_key)
            if cached is not None:
                vectors[index] = np.array(cached, dtype="float32")
            else:
                missing_texts.append(text)
                missing_positions.append(index)

        for start in range(0, len(missing_texts), batch_size):
            batch = missing_texts[start : start + batch_size]
            if not batch:
                continue
            response = self._client.embeddings.create(model=self.model_name, input=batch)
            for offset, item in enumerate(response.data):
                position = missing_positions[start + offset]
                vector = np.array(item.embedding, dtype="float32")
                vectors[position] = vector
                self.cache.set(_cache_key(self.model_name, batch[offset]), vector.tolist())

        self.cache.flush()
        return np.vstack([vector for vector in vectors if vector is not None]).astype("float32")


def _cache_key(model_name: str, text: str) -> str:
    return sha1(f"{model_name}:{text}".encode("utf-8")).hexdigest()
