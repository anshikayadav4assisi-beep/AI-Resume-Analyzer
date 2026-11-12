"""
Resume parsing.

Two responsibilities, kept separate on purpose:
  1. Raw text extraction from PDF/DOCX (format-specific, I/O heavy).
  2. Structured field extraction from that text (format-agnostic, pure logic).

Resume layouts vary enormously, so structured extraction here is
heuristic (regex + section-header detection) rather than assuming a
fixed template. It won't be perfect on every resume — see README
limitations — but it degrades gracefully: worst case, a field is empty
rather than the app crashing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from src.logger import get_logger

logger = get_logger(__name__)

ALLOWED_EXTENSIONS = {".pdf", ".docx"}

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
PHONE_RE = re.compile(
    r"(\+?\d{1,3}[\s.-]?)?(\(?\d{2,4}\)?[\s.-]?){2,4}\d{3,4}"
)
YEARS_EXP_RE = re.compile(
    r"(\d+(?:\.\d+)?)\+?\s*(?:years|yrs)\s*(?:of)?\s*(?:experience|exp)?",
    re.IGNORECASE,
)

# Common section header variants seen across resume templates.
SECTION_ALIASES = {
    "education": ["education", "academic background", "qualifications"],
    "experience": [
        "experience", "work experience", "professional experience",
        "employment history", "work history",
    ],
    "projects": ["projects", "personal projects", "academic projects"],
    "skills": [
        "skills", "technical skills", "core competencies", "key skills",
    ],
    "certifications": ["certifications", "certificates", "licenses"],
    "languages": ["languages", "spoken languages"],
    "summary": ["summary", "objective", "profile", "about"],
}


class UnsupportedFileTypeError(Exception):
    pass


class FileTooLargeError(Exception):
    pass


class CorruptDocumentError(Exception):
    pass


@dataclass
class ParsedResume:
    raw_text: str
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    years_experience: Optional[float] = None
    sections: dict = field(default_factory=dict)  # section name -> text


def validate_file(file_path: Path, max_mb: int = 10) -> None:
    """Validate extension and size before attempting to parse."""
    if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{file_path.suffix}'. "
            f"Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    size_mb = file_path.stat().st_size / (1024 * 1024)
    if size_mb > max_mb:
        raise FileTooLargeError(
            f"File is {size_mb:.1f}MB, exceeds limit of {max_mb}MB"
        )


def extract_text_from_pdf(file_path: Path) -> str:
    """Try PyMuPDF first (fast, robust); fall back to pdfplumber."""
    text_parts: List[str] = []
    try:
        import fitz  # PyMuPDF

        with fitz.open(file_path) as doc:
            for page in doc:
                text_parts.append(page.get_text())
        text = "\n".join(text_parts).strip()
        if text:
            return text
        logger.warning("PyMuPDF extracted empty text, trying pdfplumber")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"PyMuPDF extraction failed ({e}), trying pdfplumber")

    try:
        import pdfplumber

        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
        text = "\n".join(text_parts).strip()
        if not text:
            raise CorruptDocumentError(
                "No extractable text found (resume may be a scanned image)."
            )
        return text
    except CorruptDocumentError:
        raise
    except Exception as e:  # noqa: BLE001
        raise CorruptDocumentError(f"Could not read PDF: {e}") from e


def extract_text_from_docx(file_path: Path) -> str:
    try:
        import docx

        document = docx.Document(str(file_path))
        parts = [p.text for p in document.paragraphs if p.text.strip()]
        # Tables often hold skills/education in resume templates.
        for table in document.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        parts.append(cell.text)
        text = "\n".join(parts).strip()
        if not text:
            raise CorruptDocumentError("DOCX contains no readable text.")
        return text
    except CorruptDocumentError:
        raise
    except Exception as e:  # noqa: BLE001
        raise CorruptDocumentError(f"Could not read DOCX: {e}") from e


def extract_text(file_path: Path, max_mb: int = 10) -> str:
    validate_file(file_path, max_mb=max_mb)
    suffix = file_path.suffix.lower()
    logger.info(f"Extracting text from {file_path.name} ({suffix})")
    if suffix == ".pdf":
        return extract_text_from_pdf(file_path)
    if suffix == ".docx":
        return extract_text_from_docx(file_path)
    raise UnsupportedFileTypeError(suffix)


def clean_text(text: str) -> str:
    """Normalize whitespace and strip odd control characters."""
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _find_sections(text: str) -> dict:
    """
    Split resume text into sections using fuzzy header matching.

    Approach: scan lines, and whenever a line matches a known section
    alias (case-insensitive, short line, likely a header), start a new
    section. Everything until the next header belongs to it.
    """
    lines = text.split("\n")
    alias_to_canonical = {}
    for canonical, aliases in SECTION_ALIASES.items():
        for alias in aliases:
            alias_to_canonical[alias.lower()] = canonical

    sections: dict = {}
    current_section = "header"
    buffer: List[str] = []

    def flush():
        if buffer:
            sections.setdefault(current_section, "")
            sections[current_section] += "\n".join(buffer).strip() + "\n"

    for line in lines:
        stripped = line.strip().lower().strip(":")
        is_header = (
            len(stripped) < 40
            and stripped in alias_to_canonical
        )
        if is_header:
            flush()
            current_section = alias_to_canonical[stripped]
            buffer = []
        else:
            buffer.append(line)
    flush()
    return {k: v.strip() for k, v in sections.items() if v.strip()}


def _guess_name(text: str) -> Optional[str]:
    """Heuristic: the name is usually the first non-empty line that
    isn't an email/phone and isn't all lowercase boilerplate."""
    for line in text.split("\n")[:5]:
        candidate = line.strip()
        if not candidate:
            continue
        if EMAIL_RE.search(candidate) or PHONE_RE.search(candidate):
            continue
        if len(candidate.split()) <= 5 and len(candidate) < 60:
            return candidate
    return None


def _guess_years_experience(text: str) -> Optional[float]:
    matches = YEARS_EXP_RE.findall(text)
    if matches:
        try:
            return max(float(m) for m in matches)
        except ValueError:
            return None
    return None


def parse_resume(file_path: Path, max_mb: int = 10) -> ParsedResume:
    """Full pipeline: extract text -> clean -> extract structured fields."""
    raw = extract_text(file_path, max_mb=max_mb)
    cleaned = clean_text(raw)

    email_match = EMAIL_RE.search(cleaned)
    phone_match = PHONE_RE.search(cleaned)

    resume = ParsedResume(
        raw_text=cleaned,
        name=_guess_name(cleaned),
        email=email_match.group(0) if email_match else None,
        phone=phone_match.group(0) if phone_match else None,
        years_experience=_guess_years_experience(cleaned),
        sections=_find_sections(cleaned),
    )
    logger.info(
        f"Parsed resume: name={resume.name!r}, "
        f"sections_found={list(resume.sections.keys())}"
    )
    return resume
