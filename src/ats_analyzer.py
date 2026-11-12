"""
ATS (Applicant Tracking System) style analysis.

Real ATS software varies widely, so this doesn't claim to replicate any
specific vendor's algorithm. Instead it checks the well-documented,
common factors ATS-style parsers care about: keyword coverage, section
completeness, job-title alignment, and length/format signals — and
combines them into a single 0-100 compatibility score.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from src.logger import get_logger

logger = get_logger(__name__)

EXPECTED_SECTIONS = ["experience", "education", "skills"]
IRRELEVANT_MARKERS = [
    "hobbies", "interests", "references available upon request",
    "marital status", "date of birth",
]


@dataclass
class ATSReport:
    ats_score: float
    keyword_coverage_pct: float
    missing_keywords: List[str] = field(default_factory=list)
    sections_found: List[str] = field(default_factory=list)
    sections_missing: List[str] = field(default_factory=list)
    title_aligned: bool = False
    irrelevant_content_flags: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def _keyword_coverage(resume_text: str, job_keywords: List[str]) -> tuple[float, List[str]]:
    if not job_keywords:
        return 100.0, []
    resume_lower = resume_text.lower()
    missing = [kw for kw in job_keywords if kw.lower() not in resume_lower]
    covered = len(job_keywords) - len(missing)
    pct = round((covered / len(job_keywords)) * 100, 2)
    return pct, missing


def _section_completeness(sections_found: Dict[str, str]) -> tuple[List[str], List[str]]:
    found = [s for s in EXPECTED_SECTIONS if s in sections_found]
    missing = [s for s in EXPECTED_SECTIONS if s not in sections_found]
    return found, missing


def _title_alignment(resume_text: str, job_title_keywords: List[str]) -> bool:
    if not job_title_keywords:
        return True
    resume_lower = resume_text.lower()
    return any(kw.lower() in resume_lower for kw in job_title_keywords)


def _flag_irrelevant_content(resume_text: str) -> List[str]:
    resume_lower = resume_text.lower()
    return [marker for marker in IRRELEVANT_MARKERS if marker in resume_lower]


def analyze_ats_compatibility(
    resume_text: str,
    resume_sections: Dict[str, str],
    job_keywords: List[str],
    job_title_keywords: List[str],
) -> ATSReport:
    coverage_pct, missing_keywords = _keyword_coverage(resume_text, job_keywords)
    sections_found, sections_missing = _section_completeness(resume_sections)
    title_aligned = _title_alignment(resume_text, job_title_keywords)
    irrelevant_flags = _flag_irrelevant_content(resume_text)

    section_score = (len(sections_found) / len(EXPECTED_SECTIONS)) * 100
    title_score = 100.0 if title_aligned else 40.0
    irrelevant_penalty = min(20.0, len(irrelevant_flags) * 10)

    ats_score = (
        0.5 * coverage_pct
        + 0.3 * section_score
        + 0.2 * title_score
        - irrelevant_penalty
    )
    ats_score = round(max(0.0, min(100.0, ats_score)), 2)

    notes = []
    if sections_missing:
        notes.append(f"Missing standard sections: {', '.join(sections_missing)}")
    if not title_aligned:
        notes.append("Resume doesn't clearly mention the target job title or close variant.")
    if irrelevant_flags:
        notes.append("Contains information ATS systems typically ignore or flag as noise.")
    if coverage_pct < 60:
        notes.append("Low keyword overlap with job description — consider incorporating more JD language naturally.")

    report = ATSReport(
        ats_score=ats_score,
        keyword_coverage_pct=coverage_pct,
        missing_keywords=missing_keywords,
        sections_found=sections_found,
        sections_missing=sections_missing,
        title_aligned=title_aligned,
        irrelevant_content_flags=irrelevant_flags,
        notes=notes,
    )
    logger.info(f"ATS score: {ats_score}")
    return report
