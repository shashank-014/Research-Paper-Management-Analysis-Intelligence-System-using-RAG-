from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from uuid import uuid4

from research_ai.config import get_settings
from research_ai.ingestion.pdf_loader import load_pdf_pages
from research_ai.models import CitationRelation, PaperSection, ResearchPaper
from research_ai.parsing.metadata_extractor import extract_metadata
from research_ai.parsing.section_parser import extract_references, parse_sections
from research_ai.utils.logging_config import configure_logging

logger = logging.getLogger(__name__)


def parse_paper(pdf_path: str | Path, export_json: bool = False, output_dir: str | Path | None = None) -> ResearchPaper:
    configure_logging()

    path = Path(pdf_path)
    logger.info("Parsing paper %s", path.name)

    pages = load_pdf_pages(path)
    parsed_sections = parse_sections(pages)
    metadata = extract_metadata(path, pages, parsed_sections)

    paper_id = _build_paper_id(path)
    sections = _build_sections(paper_id, parsed_sections)
    references = _build_citations(paper_id, parsed_sections)

    paper = ResearchPaper(
        paper_id=paper_id,
        title=str(metadata.get("title") or path.stem),
        authors=list(metadata.get("authors", [])),
        affiliations=list(metadata.get("affiliations", [])),
        abstract=metadata.get("abstract"),
        year=metadata.get("year"),
        venue=metadata.get("venue"),
        doi=metadata.get("doi"),
        arxiv_id=metadata.get("arxiv_id"),
        keywords=list(metadata.get("keywords", [])),
        pdf_path=str(path),
        checksum=_compute_checksum(path),
        sections=sections,
        references=references,
        metadata={
            "source_file": path.name,
            "page_count": len(pages),
        },
    )

    if export_json:
        destination = Path(output_dir) if output_dir else get_settings().processed_dir
        paper.export_json(destination / f"{path.stem}.json")

    logger.info("Built ResearchPaper object for %s", paper.title)
    return paper


def batch_parse_papers(
    folder_path: str | Path,
    export_json: bool = False,
    output_dir: str | Path | None = None,
) -> list[ResearchPaper]:
    configure_logging()

    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    papers: list[ResearchPaper] = []
    for pdf_path in sorted(folder.glob("*.pdf")):
        try:
            paper = parse_paper(pdf_path, export_json=export_json, output_dir=output_dir)
            papers.append(paper)
        except Exception as exc:
            logger.error("Failed to parse %s: %s", pdf_path.name, exc)

    logger.info("Batch parsed %s paper(s) from %s", len(papers), folder)
    return papers


def _build_sections(
    paper_id: str,
    parsed_sections: dict[str, dict[str, object]],
) -> list[PaperSection]:
    sections: list[PaperSection] = []

    for section_name, payload in parsed_sections.items():
        sections.append(
            PaperSection(
                section_id=f"{paper_id}:{section_name}",
                paper_id=paper_id,
                section_name=section_name,
                heading=payload.get("heading"),
                content=str(payload.get("content", "")),
                page_start=payload.get("page_start"),
                page_end=payload.get("page_end"),
            )
        )

    return sections


def _build_citations(
    paper_id: str,
    parsed_sections: dict[str, dict[str, object]],
) -> list[CitationRelation]:
    reference_payload = parsed_sections.get("references")
    if not reference_payload:
        return []

    reference_text = str(reference_payload.get("content", ""))
    parsed_references = extract_references(reference_text)

    citations: list[CitationRelation] = []
    for item in parsed_references:
        citations.append(
            CitationRelation(
                citing_paper_id=paper_id,
                raw_reference_text=str(item.get("raw_reference_text", "")),
                cited_title=item.get("cited_title"),
            )
        )

    return citations


def _build_paper_id(path: Path) -> str:
    return f"paper-{path.stem.lower().replace(' ', '-')}-{uuid4().hex[:8]}"


def _compute_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()
