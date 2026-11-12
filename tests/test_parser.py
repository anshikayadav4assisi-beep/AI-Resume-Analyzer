import pytest

from src.job_parser import parse_job_description
from src.resume_parser import clean_text, _guess_name, _guess_years_experience


def test_clean_text_collapses_whitespace():
    dirty = "Hello   world\n\n\n\nBye"
    assert clean_text(dirty) == "Hello world\n\nBye"


def test_guess_name_skips_email_lines():
    text = "jane.doe@example.com\nJane Doe\nSoftware Engineer"
    assert _guess_name(text) == "Jane Doe"


def test_guess_years_experience_extracts_max():
    text = "I have 3 years experience in Python and 5 years experience in SQL."
    assert _guess_years_experience(text) == 5.0


def test_guess_years_experience_none_when_absent():
    assert _guess_years_experience("No experience mentioned here.") is None


def test_parse_job_description_empty_raises():
    with pytest.raises(ValueError):
        parse_job_description("   ")


def test_parse_job_description_extracts_min_years():
    jd = "Requirements: 4+ years of experience with Python and FastAPI."
    job = parse_job_description(jd)
    assert job.min_years_experience == 4


def test_parse_job_description_extracts_education():
    jd = "Requirements: Bachelor's degree in Computer Science or related field."
    job = parse_job_description(jd)
    assert any("bachelor" in req.lower() for req in job.education_requirements)
