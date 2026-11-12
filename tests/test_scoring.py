from src.config import ScoringWeights
from src.scoring import (
    compute_overall_score,
    education_match_score,
    experience_match_score,
)


def test_experience_match_full_credit_when_exceeds_requirement():
    assert experience_match_score(candidate_years=5, required_years=3) == 100.0


def test_experience_match_partial_credit():
    score = experience_match_score(candidate_years=1.5, required_years=3)
    assert 40 < score < 60


def test_experience_match_neutral_when_no_requirement():
    assert experience_match_score(candidate_years=None, required_years=None) == 70.0


def test_experience_match_penalized_when_unknown_candidate_years():
    assert experience_match_score(candidate_years=None, required_years=3) == 50.0


def test_education_match_full_credit_when_meets_requirement():
    resume_text = "I hold a Master's degree in Computer Science."
    reqs = ["Bachelor's degree in Computer Science"]
    assert education_match_score(resume_text, reqs) == 100.0


def test_education_match_low_when_no_degree_mentioned():
    resume_text = "Experienced software engineer with 5 years in industry."
    reqs = ["Bachelor's degree required"]
    assert education_match_score(resume_text, reqs) == 30.0


def test_education_match_neutral_when_no_requirement():
    assert education_match_score("Anything here", []) == 70.0


def test_compute_overall_score_weights_sum_and_bounds():
    weights = ScoringWeights(
        skill_match=0.4, semantic_similarity=0.3,
        experience_match=0.2, education_match=0.1,
    )
    breakdown = compute_overall_score(
        skill_overlap_ratio=0.8,
        semantic_similarity=0.7,
        candidate_years=5,
        required_years=3,
        resume_text="Master's degree in CS",
        education_requirements=["Bachelor's degree"],
        weights=weights,
    )
    assert 0 <= breakdown.overall_score <= 100
    assert breakdown.skill_match_score == 80.0
    assert breakdown.semantic_score == 70.0
    assert breakdown.experience_score == 100.0
    assert breakdown.education_score == 100.0
