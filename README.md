# AI Resume Analyzer & Job Matcher

An application that analyzes a resume against a job description using
real NLP/ML techniques — TF-IDF, sentence embeddings, cosine similarity,
and taxonomy-driven skill extraction — and produces a transparent,
weighted compatibility score with actionable feedback. An LLM is used
**only** to explain the results and suggest improvements; it never
generates the numeric score itself.

## Table of Contents
- [Problem Statement](#problem-statement)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [How the NLP Pipeline Works](#how-the-nlp-pipeline-works)
- [How Semantic Matching Works](#how-semantic-matching-works)
- [Scoring Methodology](#scoring-methodology)
- [Installation](#installation)
- [Environment Variables](#environment-variables)
- [Running the Application](#running-the-application)
- [Example Workflow](#example-workflow)
- [Screenshots](#screenshots)
- [Testing](#testing)
- [Evaluation](#evaluation)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)

## Problem Statement

Job seekers rarely know *why* their resume didn't match a job, and
keyword-matching tools are too crude to notice that "built REST APIs
with FastAPI" and "developed backend services with Python web
frameworks" describe the same skill. This project builds a matcher
that understands meaning, not just vocabulary, and pairs it with
transparent scoring and concrete, actionable feedback.

## Features

- PDF/DOCX resume upload with robust text extraction (PyMuPDF →
  pdfplumber fallback for PDFs, python-docx for Word files).
- Structured resume field extraction: name, email, phone, years of
  experience, and section detection (education, experience, projects,
  skills, certifications).
- Job description parsing that separates required vs. preferred
  requirements, responsibilities, education, and minimum experience.
- Taxonomy-driven skill extraction (`data/skills_taxonomy.json`) shared
  by both resume and job parsing, with fuzzy matching for variants/typos.
- Dual matching: a TF-IDF baseline and a Sentence-Transformer semantic
  matcher, shown side by side so you can see the difference.
- Transparent, configurable weighted overall score (0–100).
- Skill gap analysis with High/Medium/Low priority missing skills.
- ATS-style compatibility check (keyword coverage, section
  completeness, title alignment, irrelevant-content flags).
- LLM-generated explanation and suggestions — with a deterministic
  fallback so the app works fully even without an API key.
- Streamlit dashboard with score gauges, bar charts, and a full
  resume/job/gap breakdown.

## Architecture

```
Resume (PDF/DOCX) ──► resume_parser.py ──► raw text + structured fields
Job Description    ──► job_parser.py   ──► raw text + required/preferred split
                              │
                              ▼
                    skill_extractor.py (taxonomy-driven, shared)
                              │
              ┌───────────────┼────────────────┐
              ▼               ▼                ▼
      TF-IDF baseline   Sentence embeddings   Skill sets
      (matcher.py)      (embeddings.py +      (skill_gap.py)
                          matcher.py)
              │               │                │
              └───────────────┴────────────────┘
                              ▼
                       scoring.py (weighted, configurable)
                              ▼
                    ats_analyzer.py (ATS-style checks)
                              ▼
                  llm_service.py (narrative layer only —
                  never the source of the numeric score)
                              ▼
                  pipeline.py orchestrates all of the above
                              ▼
                        app.py (Streamlit dashboard)
```

Project layout:

```
project/
├── app.py                 # Streamlit UI
├── evaluate.py             # Standalone evaluation script
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
├── src/
│   ├── config.py           # Weights, taxonomy loader, env config
│   ├── logger.py
│   ├── resume_parser.py     # PDF/DOCX text + field extraction
│   ├── job_parser.py        # JD text + requirement extraction
│   ├── skill_extractor.py   # Taxonomy-driven skill matching
│   ├── embeddings.py        # Sentence Transformer wrapper
│   ├── matcher.py           # TF-IDF baseline + semantic matching
│   ├── scoring.py           # Weighted overall score
│   ├── skill_gap.py         # Missing skills + priority
│   ├── ats_analyzer.py      # ATS-style compatibility check
│   ├── llm_service.py       # LLM feedback + deterministic fallback
│   └── pipeline.py          # Orchestrates the full analysis
├── data/
│   ├── skills_taxonomy.json # Editable skills taxonomy
│   └── eval_samples/dataset.json
├── models/                  # Sentence-transformer cache (gitignored)
└── tests/
    ├── test_parser.py
    ├── test_matcher.py
    └── test_scoring.py
```

## Tech Stack

Python, Streamlit, Pandas, NumPy, scikit-learn (TF-IDF, cosine
similarity), Sentence Transformers, rapidfuzz, PyMuPDF, pdfplumber,
python-docx, Plotly, python-dotenv, Anthropic API (optional LLM layer),
pytest.

## How the NLP Pipeline Works

1. **Text preprocessing** (`resume_parser.clean_text`,
   `job_parser.clean_text`): normalizes whitespace, strips control
   characters, collapses excessive blank lines.
2. **Skill extraction** (`skill_extractor.py`): matches text against a
   configurable JSON taxonomy using word-boundary regex for exact
   matches, plus `rapidfuzz`-based fuzzy matching for near-variants
   (e.g. "postgres" → "PostgreSQL"). Nothing is hardcoded in Python —
   extend `data/skills_taxonomy.json` to add skills.
3. **TF-IDF baseline** (`matcher.tfidf_similarity`): vectorizes resume
   and job text with `TfidfVectorizer` (English stopwords removed) and
   computes cosine similarity between the two vectors.
4. **Sentence embeddings** (`embeddings.py`): encodes resume and job
   text with a Sentence Transformer model (`all-MiniLM-L6-v2` by
   default) and computes cosine similarity between the normalized
   embedding vectors.
5. **Skill matching** (`skill_gap.py`): set operations between resume
   skills and job required/preferred skills, with priority assigned by
   how often a missing skill is mentioned in the JD.
6. **Weighted scoring** (`scoring.py`): combines skill match, semantic
   similarity, experience match, and education match into one
   transparent score.

## How Semantic Matching Works

TF-IDF only sees exact tokens. Given:

> Resume: *"Built REST APIs using FastAPI."*
> Job: *"Experience developing backend services using Python web frameworks."*

these two sentences share almost no words, so TF-IDF cosine similarity
is low (verified in `tests/test_matcher.py::test_semantic_matching_demo_case`
and `evaluate.py`) even though they describe the same underlying skill.

Sentence Transformers embed each sentence into a dense vector that
captures *meaning* rather than surface wording — sentences about
similar concepts land close together in the embedding space regardless
of exact vocabulary. Cosine similarity between these vectors is
therefore a much better proxy for "does this candidate have the
experience this job wants" than raw keyword or n-gram overlap.

The app computes and displays **both** scores so the difference is
visible, but uses the semantic score in the final weighted total.

## Scoring Methodology

```
Overall Score = 40% Skill Match
              + 30% Semantic Similarity
              + 20% Experience Match
              + 10% Education Match
```

All four weights are configurable via environment variables
(`WEIGHT_SKILL_MATCH`, `WEIGHT_SEMANTIC_SIMILARITY`,
`WEIGHT_EXPERIENCE_MATCH`, `WEIGHT_EDUCATION_MATCH`) and must sum to
1.0 — this is validated at startup in `config.py`.

- **Skill Match** = |matched skills| / |job skills required or preferred|
- **Semantic Similarity** = cosine similarity of resume/JD embeddings
- **Experience Match** = ratio of candidate years to required years
  (capped at 100%, neutral score if data is missing/unparsed)
- **Education Match** = degree-level comparison against JD requirements

> **This is an estimated compatibility score for self-improvement
> purposes. It does not represent any real company's hiring decision
> or algorithm.**

## Installation

```bash
git clone <this-repo>
cd project
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env       # then fill in ANTHROPIC_API_KEY if you want LLM feedback
```

The first run will download the Sentence Transformer model
(`all-MiniLM-L6-v2`, ~80MB) — this requires internet access once; it's
cached locally afterward.

## Environment Variables

See `.env.example` for the full list. Key ones:

| Variable | Purpose | Required? |
|---|---|---|
| `ANTHROPIC_API_KEY` | Enables LLM-generated narrative feedback | No — fallback used if absent |
| `EMBEDDING_MODEL` | Sentence Transformer model name | No — defaults to `all-MiniLM-L6-v2` |
| `WEIGHT_*` | Scoring weights (must sum to 1.0) | No — sensible defaults provided |
| `MAX_UPLOAD_MB` | Max resume file size | No — defaults to 10 |

## Running the Application

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints, upload a resume, paste a job
description, and click **Analyze**.

Run the evaluation script standalone:

```bash
python evaluate.py
```

## Example Workflow

1. Upload `jane_doe_resume.pdf`.
2. Paste a job description for "Backend Engineer."
3. Click **Analyze**.
4. Review: overall score, skill/semantic/experience/education
   breakdown, TF-IDF vs. semantic comparison chart, missing skills by
   priority, ATS compatibility notes, and LLM (or fallback)
   recommendations for resume improvements and bullet-point rewrites.

## Screenshots

*(Add screenshots of the Streamlit dashboard here after running the
app locally — e.g. `docs/screenshot_overview.png`,
`docs/screenshot_skill_gap.png`.)*

## Testing

```bash
pytest tests/ -v
```

Covers: text cleaning and field extraction heuristics, JD requirement
parsing, TF-IDF similarity behavior (including the paraphrase case
that motivates semantic matching), skill overlap math, and the
weighted scoring formula (bounds, neutral fallbacks, weight
application).

## Evaluation

`evaluate.py` runs a small hand-labeled dataset
(`data/eval_samples/dataset.json`, 4 resume/JD pairs) through:

- Raw keyword overlap (crudest baseline)
- TF-IDF cosine similarity
- Sentence-embedding cosine similarity
- Skill extraction precision/recall/F1 against hand-labeled expected skills

### Limitations of this evaluation

- **Sample size is tiny (4 pairs).** This demonstrates the pipeline
  and directional trends (e.g. semantic > TF-IDF on paraphrased text),
  not statistically significant accuracy figures. A real evaluation
  would need dozens-to-hundreds of labeled pairs across more job
  families.
- **Skill-extraction ground truth is hand-labeled by the author**, so
  it inherits any bias in what "should" count as a matching skill.
- **No real-world resume layouts were tested** (only clean, well
  formatted synthetic text) — actual resumes vary far more.

## Limitations

- **Resume/JD parsing is heuristic**, not a trained layout model.
  Unusual resume formats (heavy graphics, multi-column layouts, scanned
  images without OCR) may extract poorly or not at all.
- **Skill taxonomy is finite.** Skills not in
  `data/skills_taxonomy.json` won't be detected unless you add them.
- **Experience/education parsing uses regex heuristics** and can
  misjudge candidates whose resumes phrase these unusually.
- **The compatibility score is an estimate for self-improvement**, not
  a hiring decision, and doesn't reflect any specific company's actual
  applicant-tracking algorithm.
- **LLM feedback quality depends on the underlying model** and API
  availability; the deterministic fallback is intentionally simpler.
- No OCR support for scanned/image-only PDFs in this version.

## Future Improvements

- OCR fallback for scanned/image-based PDF resumes.
- Fine-tuned or larger embedding models for domain-specific matching
  (e.g. legal, medical, creative fields).
- Resume rewriting assistant that edits bullet points in place.
- Multi-resume batch comparison against one job description.
- A trained named-entity model for resume field extraction instead of
  regex heuristics, to handle more varied layouts.
- Persistent storage of past analyses for tracking resume improvement
  over time.
