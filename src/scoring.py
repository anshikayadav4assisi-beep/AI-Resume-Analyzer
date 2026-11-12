"""
Transparent, configurable weighted scoring.

Overall Score = w_skill * skill_match
              + w_semantic * semantic_similarity
              + w_experience * experience_match
              + w_education * education_match

Weights come from src.config.ScoringWeights (env-var overridable) and
are always shown alongside the score in the UI, so the number is never
a black box. This is an ESTIMATED compatibility score, not a hiring
decision — that framing is enforced in the UI copy, not just in code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.config import CONFIG, ScoringWeights
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ScoreBreakdown:
    overall_score: float          # 0-100
    skill_match_score: float      # 0-100
    semantic_score: float         # 0-100
    experience_score: float       # 0-100
    education_score: float        # 0-100
    weights: ScoringWeights


def experience_match_score(
    candidate_years: Optional[float], required_years: Optional[int]
) -> float:
    """
    Score 0-100. If the JD states no minimum, or the resume states no
    years, treat as neutral (70) rather than penalizing unfairly for
    missing/unparsed data.
    """
    if required_years is None:
        return 70.0
    if candidate_years is None:
        return 50.0  # can't verify — mildly penalized, not zeroed
    if candidate_years >= required_years:
        return 100.0
    if required_years == 0:
        return 100.0
    # Linear partial credit for being close to the requirement.
    ratio = candidate_years / required_years
    return round(max(0.0, min(100.0, ratio * 100)), 2)


def education_match_score(
    resume_text: str, education_requirements: list[str]
) -> float:
    """
    Score 0-100 based on whether resume text mentions a qualifying
    degree level. Neutral score if the JD specifies no requirement.
    """
    if not education_requirements:
        return 70.0
    resume_lower = resume_text.lower()
    degree_rank = {
        "phd": 4, "doctorate": 4,
        "master": 3, "m.tech": 3, "m.sc": 3,
        "bachelor": 2, "b.tech": 2, "b.sc": 2,
        "associate": 1,
    }

    def rank_of(text: str) -> int:
        text_l = text.lower()
        best = 0
        for key, rank in degree_rank.items():
            if key in text_l:
                best = max(best, rank)
        return best

    required_rank = max((rank_of(req) for req in education_requirements), default=0)
    candidate_rank = rank_of(resume_lower)

    if required_rank == 0:
        return 70.0
    if candidate_rank >= required_rank:
        return 100.0
    if candidate_rank == 0:
        return 30.0
    return round(50.0 + (candidate_rank / required_rank) * 30, 2)


def compute_overall_score(
    skill_overlap_ratio: float,
    semantic_similarity: float,
    candidate_years: Optional[float],
    required_years: Optional[int],
    resume_text: str,
    education_requirements: list[str],
    weights: Optional[ScoringWeights] = None,
) -> ScoreBreakdown:
    weights = weights or CONFIG.weights
    weights.validate()

    skill_score = round(skill_overlap_ratio * 100, 2)
    semantic_score = round(semantic_similarity * 100, 2)
    exp_score = experience_match_score(candidate_years, required_years)
    edu_score = education_match_score(resume_text, education_requirements)

    overall = (
        weights.skill_match * skill_score
        + weights.semantic_similarity * semantic_score
        + weights.experience_match * exp_score
        + weights.education_match * edu_score
    )

    breakdown = ScoreBreakdown(
        overall_score=round(overall, 2),
        skill_match_score=skill_score,
        semantic_score=semantic_score,
        experience_score=exp_score,
        education_score=edu_score,
        weights=weights,
    )
    logger.info(f"Computed overall score: {breakdown.overall_score}")
    return breakdown
