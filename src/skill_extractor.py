"""
Skill extraction.

Uses the configurable taxonomy in data/skills_taxonomy.json rather than
hardcoding skills in Python. Matching is done two ways:
  1. Exact/word-boundary match against taxonomy entries (handles most cases).
  2. Fuzzy match (via rapidfuzz) to catch minor variants and typos, e.g.
     "Node" matching "Node.js", "postgres" matching "PostgreSQL".

This module is shared by both resume_parser output and job_parser output,
so skill sets extracted from each side are directly comparable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Set

from src.config import load_skills_taxonomy
from src.logger import get_logger

logger = get_logger(__name__)

FUZZY_THRESHOLD = 88  # 0-100, rapidfuzz score cutoff for a "match"


@dataclass
class ExtractedSkills:
    skills: List[str]  # canonical taxonomy names found
    by_category: Dict[str, List[str]]


def _build_alias_index(taxonomy: Dict[str, List[str]]) -> Dict[str, str]:
    """Map lowercased skill string -> canonical (original-cased) skill name."""
    index = {}
    for skills in taxonomy.values():
        for skill in skills:
            index[skill.lower()] = skill
    return index


def _exact_matches(text_lower: str, alias_index: Dict[str, str]) -> Set[str]:
    found = set()
    for alias_lower, canonical in alias_index.items():
        # Word-boundary match to avoid "R" matching inside "Robert".
        pattern = r"(?<![a-zA-Z0-9])" + re.escape(alias_lower) + r"(?![a-zA-Z0-9])"
        if re.search(pattern, text_lower):
            found.add(canonical)
    return found


def _fuzzy_matches(
    text_lower: str, alias_index: Dict[str, str], already_found: Set[str]
) -> Set[str]:
    """
    Catch near-misses (typos, minor variants) using token-level fuzzy
    matching. Only checked for skills not already found exactly, and
    only against tokens/short n-grams to keep this fast.
    """
    try:
        from rapidfuzz import fuzz
    except ImportError:
        logger.warning("rapidfuzz not installed; skipping fuzzy skill matching")
        return set()

    found = set()
    tokens = set(re.findall(r"[a-zA-Z0-9+#.\-]{2,}", text_lower))
    remaining_aliases = {
        alias: canonical
        for alias, canonical in alias_index.items()
        if canonical not in already_found
    }
    for alias, canonical in remaining_aliases.items():
        # Only fuzzy-check single-word aliases against tokens; multi-word
        # aliases ("machine learning") are too ambiguous for fuzzy match.
        if " " in alias:
            continue
        for token in tokens:
            if abs(len(token) - len(alias)) > 3:
                continue
            if fuzz.ratio(token, alias) >= FUZZY_THRESHOLD:
                found.add(canonical)
                break
    return found


def extract_skills(text: str) -> ExtractedSkills:
    """Extract known skills (exact + fuzzy) from arbitrary text."""
    if not text:
        return ExtractedSkills(skills=[], by_category={})

    taxonomy = load_skills_taxonomy()
    alias_index = _build_alias_index(taxonomy)
    text_lower = text.lower()

    exact = _exact_matches(text_lower, alias_index)
    fuzzy = _fuzzy_matches(text_lower, alias_index, exact)
    all_found = exact | fuzzy

    by_category: Dict[str, List[str]] = {}
    for category, skills in taxonomy.items():
        matched = sorted(s for s in skills if s in all_found)
        if matched:
            by_category[category] = matched

    result = ExtractedSkills(skills=sorted(all_found), by_category=by_category)
    logger.info(f"Extracted {len(result.skills)} skills")
    return result
