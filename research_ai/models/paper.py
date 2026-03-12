from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PaperSection(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    section_id: str
    paper_id: str
    section_name: str = Field(alias="section_type")
    heading: str | None = None
    content: str
    page_start: int | None = None
    page_end: int | None = None
    chunk_ids: list[str] = Field(default_factory=list)
    citations_mentioned: list[str] = Field(default_factory=list)


class CitationRelation(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    citing_paper_id: str = Field(alias="source_paper_id")
    cited_paper_id: str | None = Field(default=None, alias="target_paper_id")
    raw_reference_text: str = ""
    cited_title: str | None = Field(default=None, alias="reference_title")
    reference_doi: str | None = None
    reference_authors: list[str] = Field(default_factory=list)
    reference_year: int | None = None
    resolved: bool = False
    confidence: float | None = None


class ResearchPaper(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    paper_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    affiliations: list[str] = Field(default_factory=list)
    abstract: str | None = None
    keywords: list[str] = Field(default_factory=list)
    venue: str | None = None
    year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    pdf_path: str
    checksum: str | None = None
    sections: list[PaperSection] = Field(default_factory=list)
    citations: list[CitationRelation] = Field(default_factory=list, alias="references")
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def export_json(self, output_path: str | Path, by_alias: bool = False) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            self.model_dump_json(indent=2, by_alias=by_alias),
            encoding="utf-8",
        )
        return path
