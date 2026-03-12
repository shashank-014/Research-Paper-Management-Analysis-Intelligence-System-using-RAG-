from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field


class AppSettings(BaseModel):
    base_dir: Path = Field(default_factory=lambda: Path.cwd())
    data_dir: Path = Field(default_factory=lambda: Path.cwd() / "data")
    processed_dir: Path = Field(default_factory=lambda: Path.cwd() / "data" / "processed")
    indices_dir: Path = Field(default_factory=lambda: Path.cwd() / "data" / "indices")
    raw_pdf_dir: Path = Field(default_factory=lambda: Path.cwd() / "data" / "raw_pdfs")
    logs_dir: Path = Field(default_factory=lambda: Path.cwd() / "data" / "logs")
    embedding_provider: str = "sentence_transformers"
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "llama-3.1-8b-instant"

    def ensure_directories(self) -> None:
        for path in [self.data_dir, self.processed_dir, self.indices_dir, self.raw_pdf_dir, self.logs_dir]:
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    settings = AppSettings()
    settings.ensure_directories()
    return settings
