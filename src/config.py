"""
Central configuration for the Resume Analyzer.

Keeping weights and thresholds here (instead of scattering magic numbers
through the codebase) is what makes the scoring system "transparent and
configurable" as required. Everything can be overridden via environment
variables without touching code.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SKILLS_TAXONOMY_PATH = DATA_DIR / "skills_taxonomy.json"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class ScoringWeights:
    """Weights for the overall compatibility score. Must sum to 1.0."""

    skill_match: float = _env_float("WEIGHT_SKILL_MATCH", 0.40)
    semantic_similarity: float = _env_float("WEIGHT_SEMANTIC_SIMILARITY", 0.30)
    experience_match: float = _env_float("WEIGHT_EXPERIENCE_MATCH", 0.20)
    education_match: float = _env_float("WEIGHT_EDUCATION_MATCH", 0.10)

    def validate(self) -> None:
        total = (
            self.skill_match
            + self.semantic_similarity
            + self.experience_match
            + self.education_match
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Scoring weights must sum to 1.0, got {total:.3f}. "
                "Check WEIGHT_* environment variables."
            )


@dataclass(frozen=True)
class AppConfig:
    embedding_model_name: str = os.getenv(
        "EMBEDDING_MODEL", "all-MiniLM-L6-v2"
    )
    llm_provider: str = os.getenv("LLM_PROVIDER", "anthropic")
    llm_model: str = os.getenv("LLM_MODEL", "claude-sonnet-4-6")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "10"))
    weights: ScoringWeights = field(default_factory=ScoringWeights)

    # Priority thresholds for missing-skill categorization.
    # A required skill mentioned this many times (or more) in the JD text
    # is treated as high priority; see skill_gap.py for full logic.
    high_priority_min_mentions: int = 2


CONFIG = AppConfig()
CONFIG.weights.validate()


def load_skills_taxonomy() -> Dict[str, List[str]]:
    """
    Load the skills taxonomy from data/skills_taxonomy.json.

    Structure: {"category": ["skill1", "skill2", ...], ...}
    This keeps skills out of code (no hardcoding) and lets users extend
    the taxonomy by editing a JSON file — no redeploy needed.
    """
    if not SKILLS_TAXONOMY_PATH.exists():
        raise FileNotFoundError(
            f"Skills taxonomy not found at {SKILLS_TAXONOMY_PATH}"
        )
    with open(SKILLS_TAXONOMY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def flat_skill_list() -> List[str]:
    """All known skills across all taxonomy categories, lowercased."""
    taxonomy = load_skills_taxonomy()
    skills = set()
    for category_skills in taxonomy.values():
        for skill in category_skills:
            skills.add(skill.lower())
    return sorted(skills)
