from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import fitz  # type: ignore
except ImportError:  # pragma: no cover
    fitz = None

try:
    import pdfplumber  # type: ignore
except ImportError:  # pragma: no cover
    pdfplumber = None


@dataclass(slots=True)
class PageContent:
    page_number: int
    text: str
    raw_text: str
    source: str


def load_pdf_pages(pdf_path: str | Path, preferred_backend: str = "pymupdf") -> list[PageContent]:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    backends = _backend_order(preferred_backend)
    errors: list[str] = []

    for backend in backends:
        try:
            pages = _extract_with_backend(path, backend)
            if _looks_usable(pages):
                logger.info("Loaded %s with %s", path.name, backend)
                return pages
            errors.append(f"{backend}: low quality extraction")
        except Exception as exc:  # pragma: no cover
            logger.exception("Failed to load %s with %s", path.name, backend)
            errors.append(f"{backend}: {exc}")

    raise ValueError(f"Could not extract text from {path.name}. Errors: {' | '.join(errors)}")


def batch_load_pdfs(folder_path: str | Path, pattern: str = "*.pdf") -> dict[Path, list[PageContent]]:
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    results: dict[Path, list[PageContent]] = {}
    for pdf_path in sorted(folder.glob(pattern)):
        try:
            results[pdf_path] = load_pdf_pages(pdf_path)
        except Exception as exc:
            logger.error("Skipping %s due to load error: %s", pdf_path.name, exc)

    return results


def _backend_order(preferred_backend: str) -> list[str]:
    preferred = preferred_backend.lower()
    if preferred == "pdfplumber":
        return ["pdfplumber", "pymupdf"]
    return ["pymupdf", "pdfplumber"]


def _extract_with_backend(pdf_path: Path, backend: str) -> list[PageContent]:
    if backend == "pymupdf":
        if fitz is None:
            raise ImportError("PyMuPDF is not installed")
        return _extract_with_pymupdf(pdf_path)
    if backend == "pdfplumber":
        if pdfplumber is None:
            raise ImportError("pdfplumber is not installed")
        return _extract_with_pdfplumber(pdf_path)
    raise ValueError(f"Unsupported backend: {backend}")


def _extract_with_pymupdf(pdf_path: Path) -> list[PageContent]:
    pages: list[PageContent] = []
    with fitz.open(pdf_path) as doc:
        for index, page in enumerate(doc, start=1):
            raw_text = page.get_text("text") or ""
            pages.append(
                PageContent(
                    page_number=index,
                    text=clean_page_text(raw_text),
                    raw_text=raw_text,
                    source="pymupdf",
                )
            )
    return pages


def _extract_with_pdfplumber(pdf_path: Path) -> list[PageContent]:
    pages: list[PageContent] = []
    with pdfplumber.open(pdf_path) as doc:
        for index, page in enumerate(doc.pages, start=1):
            raw_text = page.extract_text() or ""
            pages.append(
                PageContent(
                    page_number=index,
                    text=clean_page_text(raw_text),
                    raw_text=raw_text,
                    source="pdfplumber",
                )
            )
    return pages


def clean_page_text(text: str) -> str:
    if not text:
        return ""

    lines = [line.rstrip() for line in text.splitlines()]
    cleaned_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue
        if _is_page_number(stripped):
            continue
        if _looks_like_header_footer(stripped):
            continue
        stripped = re.sub(r"\s+", " ", stripped)
        cleaned_lines.append(stripped)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _looks_usable(pages: list[PageContent]) -> bool:
    if not pages:
        return False
    word_count = sum(len(page.text.split()) for page in pages)
    return word_count >= 150


def _is_page_number(line: str) -> bool:
    return bool(re.fullmatch(r"(page\s+)?\d{1,4}", line.strip(), flags=re.IGNORECASE))


def _looks_like_header_footer(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) > 120:
        return False
    if "copyright" in stripped.lower():
        return True
    if re.search(r"https?://|www\.", stripped, flags=re.IGNORECASE):
        return True
    if re.search(r"\b(arxiv|doi)\b", stripped, flags=re.IGNORECASE) and len(stripped.split()) <= 8:
        return True
    return False
