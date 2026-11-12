"""
Matching algorithms: TF-IDF baseline vs semantic (Sentence Transformer).

Why two implementations?
-------------------------
TF-IDF + cosine similarity is a solid, fast, dependency-light baseline.
It works well when the resume and job description share vocabulary, but
it fails on paraphrase: "Built REST APIs using FastAPI" vs "developing
backend services using Python web frameworks" share almost no exact
tokens, so TF-IDF similarity is low even though the concepts are the
same. Sentence embeddings capture meaning rather than surface tokens,
so semantic similarity correctly recognizes these as related.

Both are computed and shown side-by-side in the UI to make this
difference concrete, and the final score uses the semantic score.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

from src.embeddings import EmbeddingUnavailableError, semantic_similarity
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MatchResult:
    tfidf_similarity: float          # 0-1, baseline
    semantic_similarity: float       # 0-1, embedding-based (0 if unavailable)
    semantic_available: bool
    skill_overlap_ratio: float       # 0-1, |resume ∩ job| / |job|


def tfidf_similarity(text_a: str, text_b: str) -> float:
    """Baseline lexical similarity via TF-IDF + cosine similarity."""
    if not text_a.strip() or not text_b.strip():
        return 0.0
    try:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        matrix = vectorizer.fit_transform([text_a, text_b])
        sim = sk_cosine(matrix[0:1], matrix[1:2])[0][0]
        return float(max(0.0, min(1.0, sim)))
    except ValueError:
        # Happens if vocabulary is empty after stopword removal (very
        # short texts) — treat as no similarity rather than crashing.
        return 0.0


def skill_overlap_ratio(resume_skills: Set[str], job_skills: Set[str]) -> float:
    if not job_skills:
        return 0.0
    overlap = resume_skills & job_skills
    return len(overlap) / len(job_skills)


def compute_match(
    resume_text: str,
    job_text: str,
    resume_skills: List[str],
    job_skills: List[str],
) -> MatchResult:
    """Run both matchers and skill overlap; never raises even if the
    embedding model is unavailable (falls back to TF-IDF only)."""
    tfidf_score = tfidf_similarity(resume_text, job_text)

    semantic_score = 0.0
    semantic_ok = True
    try:
        semantic_score = semantic_similarity(resume_text, job_text)
    except EmbeddingUnavailableError as e:
        logger.warning(f"Semantic matching unavailable, using TF-IDF only: {e}")
        semantic_ok = False
        semantic_score = tfidf_score  # graceful degradation

    overlap = skill_overlap_ratio(
        {s.lower() for s in resume_skills}, {s.lower() for s in job_skills}
    )

    return MatchResult(
        tfidf_similarity=round(tfidf_score, 4),
        semantic_similarity=round(semantic_score, 4),
        semantic_available=semantic_ok,
        skill_overlap_ratio=round(overlap, 4),
    )
