"""
Streamlit dashboard for the AI Resume Analyzer & Job Matcher.

Run with: streamlit run app.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import CONFIG
from src.logger import get_logger
from src.pipeline import run_full_analysis
from src.resume_parser import (
    CorruptDocumentError,
    FileTooLargeError,
    UnsupportedFileTypeError,
)

logger = get_logger(__name__)

st.set_page_config(
    page_title="AI Resume Analyzer & Job Matcher",
    page_icon="🧭",
    layout="wide",
)


def render_score_gauge(label: str, value: float, key: str) -> None:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": label, "font": {"size": 14}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#4C6EF5"},
                "steps": [
                    {"range": [0, 40], "color": "#FFE3E3"},
                    {"range": [40, 70], "color": "#FFF3BF"},
                    {"range": [70, 100], "color": "#D3F9D8"},
                ],
            },
        )
    )
    fig.update_layout(height=220, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True, key=key)


def main() -> None:
    st.title("🧭 AI Resume Analyzer & Job Matcher")
    st.caption(
        "Upload a resume, paste a job description, and get an "
        "**estimated** compatibility analysis. This score is not a "
        "hiring decision — it's a diagnostic to help you tailor your resume."
    )

    with st.sidebar:
        st.header("Inputs")
        uploaded_file = st.file_uploader(
            "Resume (PDF or DOCX)", type=["pdf", "docx"]
        )
        job_title = st.text_input("Target job title (optional)", "")
        job_description = st.text_area(
            "Job description", height=300,
            placeholder="Paste the full job description here...",
        )
        use_llm = st.checkbox(
            "Use LLM for narrative feedback", value=True,
            help="If unchecked, or if no API key is configured, a "
                 "deterministic template-based summary is used instead.",
        )
        run_clicked = st.button("Analyze", type="primary", use_container_width=True)

    if not run_clicked:
        st.info("Upload a resume and paste a job description in the sidebar, then click **Analyze**.")
        return

    if uploaded_file is None:
        st.error("Please upload a resume file.")
        return
    if not job_description.strip():
        st.error("Please paste a job description.")
        return

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / uploaded_file.name
        tmp_path.write_bytes(uploaded_file.getbuffer())

        try:
            with st.spinner("Analyzing resume against job description..."):
                result = run_full_analysis(
                    resume_file_path=tmp_path,
                    job_description_text=job_description,
                    job_title_keywords=[job_title] if job_title else [],
                    use_llm=use_llm,
                )
        except (UnsupportedFileTypeError, FileTooLargeError, CorruptDocumentError) as e:
            st.error(f"Could not process resume: {e}")
            return
        except Exception as e:  # noqa: BLE001
            logger.exception("Unexpected error during analysis")
            st.error(f"An unexpected error occurred: {e}")
            return

    # ---------------- Overall Score ----------------
    st.header("Overall Match")
    score = result.score
    cols = st.columns(5)
    with cols[0]:
        render_score_gauge("Overall", score.overall_score, "gauge_overall")
    with cols[1]:
        render_score_gauge("Skill Match", score.skill_match_score, "gauge_skill")
    with cols[2]:
        render_score_gauge("Semantic Similarity", score.semantic_score, "gauge_semantic")
    with cols[3]:
        render_score_gauge("Experience", score.experience_score, "gauge_exp")
    with cols[4]:
        render_score_gauge("Education", score.education_score, "gauge_edu")

    st.caption(
        f"Weights used — Skill: {score.weights.skill_match:.0%}, "
        f"Semantic: {score.weights.semantic_similarity:.0%}, "
        f"Experience: {score.weights.experience_match:.0%}, "
        f"Education: {score.weights.education_match:.0%}. "
        "*This is an estimated compatibility score, not a hiring decision.*"
    )

    if not result.match.semantic_available:
        st.warning(
            "Semantic (embedding) model was unavailable — semantic score "
            "fell back to TF-IDF similarity."
        )

    # ---------------- TF-IDF vs Semantic comparison ----------------
    st.subheader("Baseline vs. Semantic Matching")
    comp_df = pd.DataFrame(
        {
            "Method": ["TF-IDF (baseline)", "Sentence Embeddings (semantic)"],
            "Similarity (%)": [
                result.match.tfidf_similarity * 100,
                result.match.semantic_similarity * 100,
            ],
        }
    )
    st.bar_chart(comp_df.set_index("Method"))
    st.caption(
        "Semantic similarity often scores higher than TF-IDF when the "
        "resume and job description describe the same concepts with "
        "different words (e.g. 'built REST APIs with FastAPI' vs "
        "'developed backend services using Python web frameworks')."
    )

    # ---------------- Resume & Job side by side ----------------
    left, right = st.columns(2)
    with left:
        st.subheader("📄 Resume")
        r = result.resume
        st.write(f"**Name:** {r.name or 'Not detected'}")
        st.write(f"**Email:** {r.email or 'Not detected'}")
        st.write(f"**Phone:** {r.phone or 'Not detected'}")
        st.write(f"**Years of experience (detected):** {r.years_experience or 'Not detected'}")
        st.write("**Sections found:**", ", ".join(r.sections.keys()) or "None")
        st.write("**Extracted skills:**")
        for category, skills in result.resume_skills.by_category.items():
            st.markdown(f"- **{category.replace('_', ' ').title()}**: {', '.join(skills)}")

    with right:
        st.subheader("💼 Job Description")
        j = result.job
        st.write(f"**Minimum years experience (detected):** {j.min_years_experience or 'Not specified'}")
        st.write(f"**Education requirements (detected):** {', '.join(j.education_requirements) or 'Not specified'}")
        st.write("**Required skills detected:**")
        for category, skills in result.job_skills.by_category.items():
            st.markdown(f"- **{category.replace('_', ' ').title()}**: {', '.join(skills)}")

    # ---------------- Skill Gap ----------------
    st.header("Skill Gap Analysis")
    gap = result.skill_gap
    gcols = st.columns(3)
    with gcols[0]:
        st.metric("Matching Skills", len(gap.matching_skills))
        st.write(", ".join(gap.matching_skills) or "None")
    with gcols[1]:
        st.metric("Missing Required", len(gap.missing_required))
        st.write(", ".join(gap.missing_required) or "None")
    with gcols[2]:
        st.metric("Missing Preferred", len(gap.missing_preferred))
        st.write(", ".join(gap.missing_preferred) or "None")

    priority_df = pd.DataFrame(
        {
            "Priority": ["High", "Medium", "Low"],
            "Count": [
                len(gap.high_priority),
                len(gap.medium_priority),
                len(gap.low_priority),
            ],
        }
    )
    st.bar_chart(priority_df.set_index("Priority"))
    with st.expander("Missing skills by priority"):
        st.write("**High priority:**", ", ".join(gap.high_priority) or "None")
        st.write("**Medium priority:**", ", ".join(gap.medium_priority) or "None")
        st.write("**Low priority:**", ", ".join(gap.low_priority) or "None")

    # ---------------- ATS Analysis ----------------
    st.header("ATS Compatibility")
    ats = result.ats
    acols = st.columns(2)
    with acols[0]:
        render_score_gauge("ATS Score", ats.ats_score, "gauge_ats")
    with acols[1]:
        st.metric("Keyword Coverage", f"{ats.keyword_coverage_pct:.1f}%")
        st.write("**Sections found:**", ", ".join(ats.sections_found) or "None")
        st.write("**Sections missing:**", ", ".join(ats.sections_missing) or "None")
        st.write("**Title aligned:**", "✅ Yes" if ats.title_aligned else "⚠️ No")
    if ats.notes:
        st.write("**Notes:**")
        for note in ats.notes:
            st.markdown(f"- {note}")

    # ---------------- Recommendations ----------------
    st.header("Recommendations")
    fb = result.llm_feedback
    if fb.source == "fallback":
        st.caption("ℹ️ Generated with deterministic templates (LLM unavailable or disabled).")
    st.write(fb.explanation)

    rcols = st.columns(3)
    with rcols[0]:
        st.subheader("Resume Improvements")
        for item in fb.resume_improvements:
            st.markdown(f"- {item}")
    with rcols[1]:
        st.subheader("Skills to Learn")
        for item in fb.skills_to_learn:
            st.markdown(f"- {item}")
    with rcols[2]:
        st.subheader("Bullet Point Suggestions")
        for item in fb.bullet_point_suggestions:
            st.markdown(f"- {item}")


if __name__ == "__main__":
    main()
