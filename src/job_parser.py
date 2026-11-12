"""
Job description parsing.

Unlike resumes, job descriptions are plain text pasted by the user, so
there's no file-format concern here — the work is purely NLP: splitting
into responsibilities vs requirements, and separating "required" from
"preferred/nice-to-have" language, since those are treated differently
in skill-gap analysis.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from src.logger import get_logger

logger = get_logger(__name__)

REQUIRED_MARKERS = [
    "required", "requirements", "must have", "you have", "you must",
    "minimum qualifications", "basic qualifications",
]
PREFERRED_MARKERS = [
    "preferred", "nice to have", "bonus", "plus", "desired",
    "preferred qualifications", "good to have",
]
RESPONSIBILITY_MARKERS = [
    "responsibilities", "you will", "role", "what you'll do",
    "duties", "day to day",
]

EDUCATION_RE = re.compile(
    r"(bachelor'?s?|master'?s?|phd|doctorate|b\.?tech|m\.?tech|b\.?sc|m\.?sc|"
    r"associate'?s? degree)[^.\n]*",
    re.IGNORECASE,
)
EXPERIENCE_RE = re.compile(
    r"(\d+)\+?\s*(?:-\s*(\d+)\s*)?years?\s*(?:of)?\s*(?:relevant\s*)?experience",
    re.IGNORECASE,
)


@dataclass
class ParsedJob:
    raw_text: str
    required_text: str = ""
    preferred_text: str = ""
    responsibilities_text: str = ""
    education_requirements: List[str] = field(default_factory=list)
    min_years_experience: Optional[int] = None


ALL_SECTION_MARKERS = list(
    dict.fromkeys(REQUIRED_MARKERS + PREFERRED_MARKERS + RESPONSIBILITY_MARKERS)
)


def _split_by_markers(text: str, markers: List[str]) -> str:
    """
    Return the portion of text following any of the given markers, cut
    off at whichever comes first: the next *other* section marker, a
    blank-line-delimited heading, or a length cap. This keeps sections
    from bleeding into each other on short/densely formatted JDs.
    """
    lowered = text.lower()
    collected = []
    for marker in markers:
        idx = lowered.find(marker)
        if idx == -1:
            continue
        start = idx + len(marker)
        # Find the nearest following occurrence of any OTHER marker
        # (from the full set) to use as the cutoff.
        cutoff = len(text)
        for other in ALL_SECTION_MARKERS:
            if other == marker:
                continue
            other_idx = lowered.find(other, start)
            if other_idx != -1:
                cutoff = min(cutoff, other_idx)
        chunk = text[idx: min(cutoff, idx + 1200)]
        # Also stop at a clear new heading (blank line + short title line).
        end = re.search(r"\n\s*\n[A-Z][A-Za-z ]{2,30}\n", chunk[50:])
        if end:
            chunk = chunk[: 50 + end.start()]
        collected.append(chunk)
    return "\n".join(collected)


def _extract_education(text: str) -> List[str]:
    matches = EDUCATION_RE.findall(text)
    # findall with groups returns tuples; re-search properly to get full spans
    found = []
    for m in EDUCATION_RE.finditer(text):
        found.append(m.group(0).strip().rstrip(".,"))
    # de-duplicate, preserve order
    seen = set()
    result = []
    for item in found:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _extract_min_experience(text: str) -> Optional[int]:
    matches = EXPERIENCE_RE.findall(text)
    if not matches:
        return None
    years = []
    for low, _high in matches:
        try:
            years.append(int(low))
        except ValueError:
            continue
    return min(years) if years else None


def clean_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_job_description(text: str) -> ParsedJob:
    if not text or not text.strip():
        raise ValueError("Job description text is empty.")

    cleaned = clean_text(text)

    required = _split_by_markers(cleaned, REQUIRED_MARKERS)
    preferred = _split_by_markers(cleaned, PREFERRED_MARKERS)
    responsibilities = _split_by_markers(cleaned, RESPONSIBILITY_MARKERS)

    # Fallback: if no explicit "required" section was found, treat the
    # whole JD as required context (better to over-include than drop
    # everything on a poorly formatted JD).
    if not required:
        required = cleaned

    job = ParsedJob(
        raw_text=cleaned,
        required_text=required,
        preferred_text=preferred,
        responsibilities_text=responsibilities or cleaned,
        education_requirements=_extract_education(cleaned),
        min_years_experience=_extract_min_experience(cleaned),
    )
    logger.info(
        f"Parsed job description: min_years={job.min_years_experience}, "
        f"education_reqs={job.education_requirements}"
    )
    return job
