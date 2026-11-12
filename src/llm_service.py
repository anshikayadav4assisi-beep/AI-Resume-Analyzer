"""
LLM-based narrative feedback layer.

This module is intentionally the LAST thing called in the pipeline and
is fully optional: every other module produces a complete, useful
report without it. If ANTHROPIC_API_KEY is missing or the API call
fails for any reason, callers get a deterministic, template-based
fallback instead of a crash — see generate_feedback()'s try/except.

The LLM is asked only to explain and suggest, never to invent the
numeric scores — those come from scoring.py.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional

from src.config import CONFIG
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LLMFeedback:
    explanation: str
    resume_improvements: List[str] = field(default_factory=list)
    skills_to_learn: List[str] = field(default_factory=list)
    bullet_point_suggestions: List[str] = field(default_factory=list)
    source: str = "llm"  # "llm" or "fallback"


class LLMUnavailableError(Exception):
    pass


def _build_prompt(context: dict) -> str:
    return f"""You are an expert technical recruiter and resume coach.

A candidate was scored against a job description using deterministic
NLP/ML analysis. Here is the analysis data (do NOT change these numbers,
just explain and build on them):

{json.dumps(context, indent=2)}

Respond ONLY with valid JSON (no markdown fences, no preamble) in this
exact shape:
{{
  "explanation": "2-3 sentence plain-English explanation of why the candidate got this score",
  "resume_improvements": ["specific improvement 1", "specific improvement 2", "..."],
  "skills_to_learn": ["skill 1", "skill 2", "..."],
  "bullet_point_suggestions": ["a rewritten example bullet point", "..."]
}}

Keep each list to at most 5 items. Be specific and reference the actual
missing skills and scores given above. Do not claim this score reflects
an actual hiring decision — frame it as an estimate."""


def _call_anthropic(prompt: str) -> str:
    import anthropic

    if not CONFIG.anthropic_api_key:
        raise LLMUnavailableError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=CONFIG.anthropic_api_key)
    response = client.messages.create(
        model=CONFIG.llm_model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    text_blocks = [b.text for b in response.content if b.type == "text"]
    return "\n".join(text_blocks)


def _fallback_feedback(context: dict) -> LLMFeedback:
    """Deterministic, template-based feedback when the LLM is unavailable.
    Keeps the app fully functional without any API key."""
    missing = context.get("missing_required_skills", [])[:5]
    overall = context.get("overall_score", 0)

    explanation = (
        f"Estimated compatibility score is {overall}/100, based on skill "
        f"overlap, semantic similarity between the resume and job "
        f"description, experience level, and education match. "
        f"(Generated without LLM — deterministic template.)"
    )
    improvements = []
    if missing:
        improvements.append(
            f"Add or highlight experience with: {', '.join(missing)}"
        )
    if context.get("ats_score", 100) < 70:
        improvements.append(
            "Improve ATS compatibility by including more exact keywords from the job description."
        )
    if not improvements:
        improvements.append("Resume already aligns well with this job description.")

    skills_to_learn = missing[:3]

    bullet_suggestions = [
        "Quantify achievements: 'Improved API response time by 40% by optimizing database queries' "
        "reads stronger than 'Worked on API performance.'"
    ]

    return LLMFeedback(
        explanation=explanation,
        resume_improvements=improvements,
        skills_to_learn=skills_to_learn,
        bullet_point_suggestions=bullet_suggestions,
        source="fallback",
    )


def generate_feedback(context: dict) -> LLMFeedback:
    """
    context should include: overall_score, skill_match_score,
    semantic_score, experience_score, education_score, ats_score,
    matching_skills, missing_required_skills, missing_preferred_skills.
    """
    try:
        prompt = _build_prompt(context)
        raw = _call_anthropic(prompt)
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.replace("json\n", "", 1)
        parsed = json.loads(cleaned)
        return LLMFeedback(
            explanation=parsed.get("explanation", ""),
            resume_improvements=parsed.get("resume_improvements", []),
            skills_to_learn=parsed.get("skills_to_learn", []),
            bullet_point_suggestions=parsed.get("bullet_point_suggestions", []),
            source="llm",
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"LLM feedback unavailable, using fallback: {e}")
        return _fallback_feedback(context)
