"""
Evaluation script.

Runs the small labeled dataset in data/eval_samples/dataset.json through:
  - raw keyword overlap
  - TF-IDF cosine similarity
  - sentence-embedding cosine similarity (if available)
  - skill extraction (precision/recall against hand-labeled expected skills)

and prints a comparison table plus a summary of accuracy on skill
extraction. See README.md "Limitations" for caveats on this evaluation.

Run with: python evaluate.py
"""
from __future__ import annotations

import json
from pathlib import Path

from src.embeddings import EmbeddingUnavailableError, semantic_similarity
from src.matcher import tfidf_similarity
from src.skill_extractor import extract_skills

DATASET_PATH = Path(__file__).parent / "data" / "eval_samples" / "dataset.json"


def keyword_overlap_score(text_a: str, text_b: str) -> float:
    """Naive raw keyword overlap — the crudest possible baseline, included
    to show why even TF-IDF is an improvement over plain word overlap."""
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_b)


def skill_extraction_metrics(extracted: list[str], expected: list[str]) -> dict:
    extracted_set = {s.lower() for s in extracted}
    expected_set = {s.lower() for s in expected}
    if not expected_set:
        # Nothing expected: precision undefined if nothing extracted either.
        precision = 1.0 if not extracted_set else 0.0
        return {"precision": precision, "recall": 1.0, "f1": precision}
    true_positives = extracted_set & expected_set
    precision = len(true_positives) / len(extracted_set) if extracted_set else 0.0
    recall = len(true_positives) / len(expected_set)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": round(precision, 2), "recall": round(recall, 2), "f1": round(f1, 2)}


def main() -> None:
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"{'ID':<8}{'Keyword':>10}{'TF-IDF':>10}{'Embedding':>12}")
    print("-" * 42)

    embedding_available = True
    all_skill_metrics = []

    for pair in dataset:
        resume_text = pair["resume_text"]
        job_text = pair["job_text"]

        kw_score = keyword_overlap_score(resume_text, job_text)
        tfidf_score = tfidf_similarity(resume_text, job_text)

        emb_score_str = "n/a"
        if embedding_available:
            try:
                emb_score = semantic_similarity(resume_text, job_text)
                emb_score_str = f"{emb_score:.2f}"
            except EmbeddingUnavailableError:
                embedding_available = False
                emb_score_str = "unavailable"

        print(f"{pair['id']:<8}{kw_score:>10.2f}{tfidf_score:>10.2f}{emb_score_str:>12}")

        resume_skills = extract_skills(resume_text).skills
        metrics = skill_extraction_metrics(resume_skills, pair["expected_resume_skills"])
        all_skill_metrics.append(metrics)

    print("\nSkill extraction accuracy (precision / recall / F1), averaged:")
    avg_precision = sum(m["precision"] for m in all_skill_metrics) / len(all_skill_metrics)
    avg_recall = sum(m["recall"] for m in all_skill_metrics) / len(all_skill_metrics)
    avg_f1 = sum(m["f1"] for m in all_skill_metrics) / len(all_skill_metrics)
    print(f"  Precision: {avg_precision:.2f}  Recall: {avg_recall:.2f}  F1: {avg_f1:.2f}")

    print(
        "\nNote: this dataset has only 4 hand-written pairs — results here "
        "indicate directional trends, not statistically robust accuracy. "
        "See README.md 'Limitations' for details."
    )


if __name__ == "__main__":
    main()
