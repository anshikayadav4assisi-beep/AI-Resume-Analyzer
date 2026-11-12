from src.matcher import skill_overlap_ratio, tfidf_similarity


def test_tfidf_similarity_identical_text_is_high():
    text = "Experienced Python developer with FastAPI and PostgreSQL skills."
    assert tfidf_similarity(text, text) > 0.99


def test_tfidf_similarity_unrelated_text_is_low():
    a = "Experienced chef specializing in French pastry and baking."
    b = "Senior backend engineer with Kubernetes and Go experience."
    assert tfidf_similarity(a, b) < 0.3


def test_tfidf_similarity_empty_text_is_zero():
    assert tfidf_similarity("", "some text") == 0.0
    assert tfidf_similarity("some text", "") == 0.0


def test_skill_overlap_ratio_full_overlap():
    resume = {"python", "sql", "docker"}
    job = {"python", "sql"}
    assert skill_overlap_ratio(resume, job) == 1.0


def test_skill_overlap_ratio_partial_overlap():
    resume = {"python"}
    job = {"python", "sql", "docker", "kubernetes"}
    assert skill_overlap_ratio(resume, job) == 0.25


def test_skill_overlap_ratio_no_job_skills():
    assert skill_overlap_ratio({"python"}, set()) == 0.0


def test_semantic_matching_demo_case():
    """
    Illustrates why semantic matching is needed: these two sentences
    describe the same underlying concept with almost no shared tokens,
    so TF-IDF similarity should be noticeably low.
    """
    resume_line = "Built REST APIs using FastAPI."
    job_line = "Experience developing backend services using Python web frameworks."
    score = tfidf_similarity(resume_line, job_line)
    assert score < 0.4  # TF-IDF baseline should struggle here
