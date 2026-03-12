from __future__ import annotations

import logging
import re
from pathlib import Path

from research_ai.ingestion.pdf_loader import PageContent

logger = logging.getLogger(__name__)

SECTION_MARKERS = {
    "abstract", "introduction", "background", "related work", "methods", "methodology", "experiments",
    "results", "discussion", "limitations", "conclusion", "references",
}


def extract_metadata(
    pdf_path: str | Path,
    pages: list[PageContent],
    parsed_sections: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    first_page_text = pages[0].text if pages else ""
    first_page_lines = [line.strip() for line in first_page_text.splitlines() if line.strip()]
    raw_text = "\n".join(page.raw_text for page in pages[:2])

    title = _extract_title(first_page_lines, pdf_path)
    authors = _extract_authors(first_page_lines)
    affiliations = _extract_affiliations(first_page_lines)
    year = _extract_year("\n".join(page.text for page in pages[:2]))
    abstract = _extract_abstract(parsed_sections)
    keywords = _extract_keywords(first_page_text, parsed_sections)
    venue = _extract_venue(first_page_lines, title, authors)
    doi = _extract_doi(raw_text)
    arxiv_id = _extract_arxiv_id(raw_text)

    metadata = {
        "title": title,
        "authors": authors,
        "affiliations": affiliations,
        "abstract": abstract,
        "year": year,
        "venue": venue,
        "keywords": keywords,
        "doi": doi,
        "arxiv_id": arxiv_id,
    }
    logger.info("Extracted metadata for %s", Path(pdf_path).name)
    return metadata


def _extract_title(first_page_lines: list[str], pdf_path: str | Path) -> str:
    candidates: list[str] = []
    for line in first_page_lines[:14]:
        if _is_bad_title_line(line):
            continue
        candidates.append(line)
        if len(candidates) == 3:
            break
    return max(candidates, key=len) if candidates else Path(pdf_path).stem


def _extract_authors(first_page_lines: list[str]) -> list[str]:
    author_candidates: list[str] = []
    seen_title = False
    for line in first_page_lines[:25]:
        lowered = line.lower()
        if not seen_title and not _is_bad_title_line(line):
            seen_title = True
            continue
        if lowered in SECTION_MARKERS:
            break
        if _looks_like_author_line(line):
            author_candidates.append(line)
        if author_candidates and ("@" in line or "university" in lowered or "institute" in lowered):
            break

    authors: list[str] = []
    for line in author_candidates[:4]:
        parts = re.split(r",| and |;", re.sub(r"\d", "", line), flags=re.IGNORECASE)
        for part in parts:
            cleaned = part.strip(" ,;*")
            if _looks_like_person_name(cleaned):
                authors.append(cleaned)
    return list(dict.fromkeys(authors))


def _extract_affiliations(first_page_lines: list[str]) -> list[str]:
    affiliations = []
    for line in first_page_lines[:30]:
        lowered = line.lower()
        if re.search(r"\b(university|institute|department|school|laboratory|lab|college|centre|center)\b", lowered):
            affiliations.append(line)
    return list(dict.fromkeys(affiliations))[:6]


def _extract_year(text: str) -> int | None:
    matches = re.findall(r"\b(19\d{2}|20\d{2})\b", text)
    valid_years = [int(year) for year in matches if 1990 <= int(year) <= 2100]
    return min(valid_years) if valid_years else None


def _extract_abstract(parsed_sections: dict[str, dict[str, object]] | None) -> str | None:
    if not parsed_sections or "abstract" not in parsed_sections:
        return None
    content = str(parsed_sections["abstract"].get("content", "")).strip()
    return content or None


def _extract_keywords(text: str, parsed_sections: dict[str, dict[str, object]] | None) -> list[str]:
    match = re.search(r"\bkeywords?\b\s*[:\-]?\s*(.+)", text, flags=re.IGNORECASE)
    if match:
        raw_keywords = match.group(1).split("\n", 1)[0]
        return [part.strip(" .;") for part in re.split(r",|;", raw_keywords) if part.strip()][:12]

    if parsed_sections and parsed_sections.get("abstract"):
        abstract_text = str(parsed_sections["abstract"].get("content", ""))
        candidate_terms = re.findall(r"\b[A-Za-z][A-Za-z\-]{3,}\b", abstract_text)
        return list(dict.fromkeys(term.lower() for term in candidate_terms[:8]))
    return []


def _extract_venue(first_page_lines: list[str], title: str, authors: list[str]) -> str | None:
    for line in first_page_lines[:18]:
        lowered = line.lower()
        if line == title or line in authors:
            continue
        if any(token in lowered for token in ["conference", "journal", "proceedings", "workshop", "transactions", "symposium"]):
            return line
    return None


def _extract_doi(text: str) -> str | None:
    match = re.search(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", text)
    return match.group(0) if match else None


def _extract_arxiv_id(text: str) -> str | None:
    match = re.search(r"arXiv[:\s]+([A-Za-z\-\.]+/\d{7}|\d{4}\.\d{4,5})", text, flags=re.IGNORECASE)
    return match.group(1) if match else None


def _is_bad_title_line(line: str) -> bool:
    lowered = line.lower()
    if lowered in SECTION_MARKERS or len(line.split()) < 3 or "@" in line:
        return True
    return bool(re.search(r"\b(university|institute|department|school)\b", lowered))


def _looks_like_author_line(line: str) -> bool:
    lowered = line.lower()
    if "@" in line:
        return True
    if re.search(r"\b(university|institute|department|school|lab|laboratory)\b", lowered):
        return True
    title_case_words = sum(1 for token in line.split() if token[:1].isupper())
    return title_case_words >= 2 and len(line.split()) <= 14


def _looks_like_person_name(value: str) -> bool:
    if not value or len(value.split()) < 2:
        return False
    if re.search(r"@|\b(university|institute|department|school|lab)\b", value, flags=re.IGNORECASE):
        return False
    return all(part[:1].isupper() for part in value.split()[:3])
