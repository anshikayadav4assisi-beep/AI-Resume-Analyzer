"""
Skill gap analysis.

Compares resume skills against required and preferred job skills,
producing:
  - skills the candidate has that are relevant
  - required skills present / missing
  - preferred (nice-to-have) skills missing
  - a priority level for each missing skill, based on how often it's
    mentioned in the job text (more mentions ~= more important to the
    role) and whether it's in the "required" vs "preferred" section.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from src.config import CONFIG
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SkillGapReport:
    matching_skills: List[str] = field(default_factory=list)
    missing_required: List[str] = field(default_factory=list)
    missing_preferred: List[str] = field(default_factory=list)
    high_priority: List[str] = field(default_factory=list)
    medium_priority: List[str] = field(default_factory=list)
    low_priority: List[str] = field(default_factory=list)


def _mention_count(skill: str, text: str) -> int:
    pattern = r"(?<![a-zA-Z0-9])" + re.escape(skill.lower()) + r"(?![a-zA-Z0-9])"
    return len(re.findall(pattern, text.lower()))


def analyze_skill_gap(
    resume_skills: List[str],
    required_skills: List[str],
    preferred_skills: List[str],
    job_full_text: str,
) -> SkillGapReport:
    resume_set = {s.lower() for s in resume_skills}
    required_set = {s.lower(): s for s in required_skills}
    preferred_set = {s.lower(): s for s in preferred_skills}

    matching = sorted(
        s for s in required_skills + preferred_skills
        if s.lower() in resume_set
    )
    # de-duplicate while preserving original casing
    matching = sorted(set(matching))

    missing_required = [
        original for key, original in required_set.items()
        if key not in resume_set
    ]
    missing_preferred = [
        original for key, original in preferred_set.items()
        if key not in resume_set and key not in required_set
    ]

    high, medium, low = [], [], []
    for skill in missing_required:
        mentions = _mention_count(skill, job_full_text)
        if mentions >= CONFIG.high_priority_min_mentions:
            high.append(skill)
        else:
            medium.append(skill)
    for skill in missing_preferred:
        low.append(skill)

    report = SkillGapReport(
        matching_skills=matching,
        missing_required=sorted(missing_required),
        missing_preferred=sorted(missing_preferred),
        high_priority=sorted(high),
        medium_priority=sorted(medium),
        low_priority=sorted(low),
    )
    logger.info(
        f"Skill gap: {len(matching)} matching, "
        f"{len(missing_required)} missing required, "
        f"{len(missing_preferred)} missing preferred"
    )
    return report
