"""
Pipeline orchestrator.

Wires together resume_parser -> job_parser -> skill_extractor ->
matcher -> scoring -> skill_gap -> ats_analyzer -> llm_service into a
single call, producing one AnalysisResult the Streamlit UI can render
directly. Keeping this separate from app.py means the whole analysis
is unit-testable without touching Streamlit.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.ats_analyzer import ATSReport, analyze_ats_compatibility
from src.config import CONFIG
from src.job_parser import ParsedJob, parse_job_description
from src.llm_service import LLMFeedback, generate_feedback
from src.logger import get_logger
from src.matcher import MatchResult, compute_match
from src.resume_parser import ParsedResume, parse_resume
from src.scoring import ScoreBreakdown, compute_overall_score
from src.skill_extractor import ExtractedSkills, extract_skills
from src.skill_gap import SkillGapReport, analyze_skill_gap

logger = get_logger(__name__)


@dataclass
class AnalysisResult:
    resume: ParsedResume
    job: ParsedJob
    resume_skills: ExtractedSkills
    job_skills: ExtractedSkills
    match: MatchResult
    score: ScoreBreakdown
    skill_gap: SkillGapReport
    ats: ATSReport
    llm_feedback: LLMFeedback


def run_full_analysis(
    resume_file_path: Path,
    job_description_text: str,
    job_title_keywords: Optional[list[str]] = None,
    use_llm: bool = True,
) -> AnalysisResult:
    logger.info("Starting full analysis pipeline")

    resume = parse_resume(resume_file_path, max_mb=CONFIG.max_upload_mb)
    job = parse_job_description(job_description_text)

    resume_skills = extract_skills(resume.raw_text)
    required_skills = extract_skills(job.required_text)
    preferred_skills = extract_skills(job.preferred_text)
    # Union used for overall semantic/skill-overlap purposes.
    all_job_skills = extract_skills(job.raw_text)

    match = compute_match(
        resume_text=resume.raw_text,
        job_text=job.raw_text,
        resume_skills=resume_skills.skills,
        job_skills=all_job_skills.skills,
    )

    gap = analyze_skill_gap(
        resume_skills=resume_skills.skills,
        required_skills=required_skills.skills,
        preferred_skills=preferred_skills.skills,
        job_full_text=job.raw_text,
    )

    score = compute_overall_score(
        skill_overlap_ratio=match.skill_overlap_ratio,
        semantic_similarity=match.semantic_similarity,
        candidate_years=resume.years_experience,
        required_years=job.min_years_experience,
        resume_text=resume.raw_text,
        education_requirements=job.education_requirements,
    )

    ats = analyze_ats_compatibility(
        resume_text=resume.raw_text,
        resume_sections=resume.sections,
        job_keywords=all_job_skills.skills,
        job_title_keywords=job_title_keywords or [],
    )

    if use_llm:
        context = {
            "overall_score": score.overall_score,
            "skill_match_score": score.skill_match_score,
            "semantic_score": score.semantic_score,
            "experience_score": score.experience_score,
            "education_score": score.education_score,
            "ats_score": ats.ats_score,
            "matching_skills": gap.matching_skills,
            "missing_required_skills": gap.missing_required,
            "missing_preferred_skills": gap.missing_preferred,
        }
        feedback = generate_feedback(context)
    else:
        from src.llm_service import _fallback_feedback
        feedback = _fallback_feedback({
            "overall_score": score.overall_score,
            "missing_required_skills": gap.missing_required,
            "ats_score": ats.ats_score,
        })

    logger.info("Analysis pipeline complete")
    return AnalysisResult(
        resume=resume,
        job=job,
        resume_skills=resume_skills,
        job_skills=all_job_skills,
        match=match,
        score=score,
        skill_gap=gap,
        ats=ats,
        llm_feedback=feedback,
    )
