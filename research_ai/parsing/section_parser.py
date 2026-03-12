from __future__ import annotations

import logging
import re
from collections import OrderedDict

from research_ai.ingestion.pdf_loader import PageContent

logger = logging.getLogger(__name__)

SECTION_PATTERNS: "OrderedDict[str, list[re.Pattern[str]]]" = OrderedDict(
    {
        "abstract": [re.compile(r"^\s*(\d+[\.\)]\s*)?abstract\s*(?:[:\-]\s*(?P<rest>.+))?$", re.IGNORECASE)],
        "introduction": [re.compile(r"^\s*(\d+[\.\)]\s*)?(introduction|background)\s*$", re.IGNORECASE)],
        "related_work": [re.compile(r"^\s*(\d+[\.\)]\s*)?(related work|literature review|prior work)\s*$", re.IGNORECASE)],
        "methods": [re.compile(r"^\s*(\d+[\.\)]\s*)?(methods?|methodology|materials and methods|approach|proposed method)\s*$", re.IGNORECASE)],
        "results": [re.compile(r"^\s*(\d+[\.\)]\s*)?(experiments?|experimental results|results|evaluation|results and discussion)\s*$", re.IGNORECASE)],
        "discussion": [re.compile(r"^\s*(\d+[\.\)]\s*)?discussion\s*$", re.IGNORECASE)],
        "limitations": [re.compile(r"^\s*(\d+[\.\)]\s*)?(limitations|threats to validity)\s*$", re.IGNORECASE)],
        "conclusion": [re.compile(r"^\s*(\d+[\.\)]\s*)?(conclusion|conclusions|future work|conclusion and future work)\s*$", re.IGNORECASE)],
        "references": [re.compile(r"^\s*(\d+[\.\)]\s*)?(references|bibliography)\s*$", re.IGNORECASE)],
        "appendix": [re.compile(r"^\s*(\d+[\.\)]\s*)?(appendix|supplementary material)\s*$", re.IGNORECASE)],
    }
)


def parse_sections(pages: list[PageContent]) -> dict[str, dict[str, object]]:
    sections: dict[str, dict[str, object]] = {}
    current_name: str | None = None
    current_heading: str | None = None
    buffer: list[str] = []
    start_page: int | None = None
    end_page: int | None = None

    for page in pages:
        lines = [line.strip() for line in page.text.splitlines() if line.strip()]
        for line in lines:
            detected = match_section_heading(line)
            if detected:
                section_name, inline_content = detected
                if current_name and buffer:
                    sections[current_name] = _build_section_payload(current_heading, buffer, start_page, end_page or page.page_number)
                current_name = section_name
                current_heading = line.strip()
                buffer = [inline_content] if inline_content else []
                start_page = page.page_number
                end_page = page.page_number
                continue

            if current_name:
                buffer.append(line)
                end_page = page.page_number

        if current_name and end_page is None:
            end_page = page.page_number

    if current_name and buffer:
        sections[current_name] = _build_section_payload(current_heading, buffer, start_page, end_page)

    logger.info("Detected sections: %s", list(sections.keys()))
    return sections


def detect_section_name(line: str) -> str | None:
    match = match_section_heading(line)
    return match[0] if match else None


def match_section_heading(line: str) -> tuple[str, str | None] | None:
    candidate = _normalize_heading(line)
    for canonical_name, patterns in SECTION_PATTERNS.items():
        for pattern in patterns:
            match = pattern.match(candidate)
            if match:
                return canonical_name, (match.groupdict().get("rest") or "").strip() or None
    return None


def extract_references(reference_text: str) -> list[dict[str, object]]:
    if not reference_text.strip():
        return []

    text = re.sub(r"\n{2,}", "\n", reference_text.strip())
    entries = re.split(r"\n(?=\[\d+\]|\d+\.\s)", text)
    if len(entries) == 1:
        entries = [part.strip() for part in re.split(r"(?<=\.)\s(?=[A-Z][a-z]+,)", text) if part.strip()]

    references: list[dict[str, object]] = []
    for entry in entries:
        cleaned = entry.strip()
        if not cleaned:
            continue
        title = _extract_reference_title(cleaned)
        references.append({"raw_reference_text": cleaned, "cited_title": title})

    return references


def _build_section_payload(heading: str | None, content_lines: list[str], page_start: int | None, page_end: int | None) -> dict[str, object]:
    content = _clean_section_content(content_lines)
    return {"heading": heading, "content": content, "page_start": page_start, "page_end": page_end}


def _clean_section_content(lines: list[str]) -> str:
    text = "\n".join(lines)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalize_heading(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip())


def _extract_reference_title(reference_text: str) -> str:
    cleaned = re.sub(r"^\s*(\[\d+\]|\d+\.)\s*", "", reference_text)
    quoted = re.search(r'["\']([^"\']+)["\']', cleaned)
    if quoted:
        return quoted.group(1).strip()
    parts = [part.strip() for part in cleaned.split(".") if part.strip()]
    if len(parts) >= 3:
        return parts[1]
    if len(parts) >= 2:
        return parts[-1]
    return cleaned[:200]
