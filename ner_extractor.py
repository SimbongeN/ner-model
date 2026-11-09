"""
ner_extractor.py
----------------
Rule-based Named Entity Recognition using spaCy's EntityRuler.

Extracts three entity types from job descriptions:
  SKILL       – technical and soft skills
  DEGREE      – education / qualification requirements
  EXPERIENCE  – years-of-experience mentions

Why rule-based?
---------------
The dataset has no manual NER labels, so supervised training is not
possible without annotation. A rule-based EntityRuler is the standard
approach for structured domains like job postings, where entities follow
predictable linguistic patterns. It is also fully interpretable and
serves as a strong baseline before any fine-tuning.

Outputs
-------
  dataset/ner_results.csv   – one row per job description with extracted entities
"""

import re
import os
import pandas as pd
import spacy
from typing import List, Dict


# ── Entity pattern lists ─────────────────────────────────────────────────────────

# Technical skills (exact token sequences, lowercased)
TECHNICAL_SKILLS = [
    "python", "java", "javascript", "typescript", "sql", "nosql", "r",
    "c", "c++", "c#", "ruby", "php", "swift", "kotlin", "go", "rust",
    "html", "css", "react", "angular", "vue", "node", "nodejs",
    "django", "flask", "spring", "tensorflow", "pytorch", "keras",
    "scikit learn", "scikit-learn", "pandas", "numpy", "matplotlib",
    "machine learning", "deep learning", "natural language processing",
    "nlp", "computer vision", "data analysis", "data science",
    "data engineering", "big data", "statistics", "aws", "azure", "gcp",
    "docker", "kubernetes", "git", "linux", "bash", "spark", "hadoop",
    "tableau", "power bi", "excel", "word", "powerpoint",
    "agile", "scrum", "devops", "ci/cd", "rest", "api", "microservices",
    "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
]

# Soft skills
SOFT_SKILLS = [
    "communication", "leadership", "teamwork", "problem solving",
    "critical thinking", "time management", "attention to detail",
    "collaboration", "adaptability", "creativity", "analytical skills",
    "interpersonal skills", "organisational skills", "organizational skills",
    "written communication", "verbal communication", "presentation skills",
    "project management", "stakeholder management", "decision making",
]

# Degree patterns (regex, applied to cleaned lowercase text)
DEGREE_REGEX_PATTERNS = [
    r"bachelor[s']?\s+(?:of\s+)?(?:science|arts|engineering|commerce|business|technology|education)?\s*(?:degree|qualification)?",
    r"master[s']?\s+(?:of\s+)?(?:science|arts|engineering|business|technology|education|administration)?\s*(?:degree|qualification)?",
    r"(?:ph\.?\s*d|doctorate)\s*(?:degree|qualification)?",
    r"b\.?\s*sc\.?(?:\s+degree)?",
    r"m\.?\s*sc\.?(?:\s+degree)?",
    r"m\.?\s*b\.?\s*a\.?(?:\s+degree)?",
    r"b\.?\s*eng\.?(?:\s+degree)?",
    r"honours\s+degree",
    r"undergraduate\s+degree",
    r"postgraduate\s+(?:degree|diploma|certificate)",
    r"associate[s']?\s+degree",
    r"national\s+diploma",
    r"higher\s+certificate",
    r"hnd",
    r"hnc",
]

# Experience patterns (regex)
EXPERIENCE_REGEX_PATTERNS = [
    r"\d+\s*\+\s*years?\s+(?:of\s+)?(?:relevant\s+)?(?:work\s+)?experience",
    r"\d+\s+to\s+\d+\s+years?\s+(?:of\s+)?(?:work\s+)?experience",
    r"\d+\s*[-–]\s*\d+\s+years?\s+(?:of\s+)?(?:work\s+)?experience",
    r"minimum\s+(?:of\s+)?\d+\s+years?\s+(?:of\s+)?(?:work\s+)?experience",
    r"at\s+least\s+\d+\s+years?\s+(?:of\s+)?(?:work\s+)?experience",
    r"\d+\s+years?\s+(?:of\s+)?(?:industry|professional|relevant|work|proven)?\s*experience",
    r"(?:proven|demonstrated|relevant)\s+experience\s+of\s+\d+\+?\s+years?",
]


# ── Build spaCy pipeline ─────────────────────────────────────────────────────────

def build_nlp_pipeline() -> spacy.Language:
    """Construct a blank spaCy pipeline with EntityRuler for all three entity types."""
    nlp = spacy.blank("en")
    ruler = nlp.add_pipe("entity_ruler", config={"overwrite_ents": True})

    patterns = []

    # SKILL patterns — token-level (works on cleaned lowercase text)
    for skill in TECHNICAL_SKILLS + SOFT_SKILLS:
        tokens = skill.split()
        if len(tokens) == 1:
            patterns.append({"label": "SKILL", "pattern": tokens[0]})
        else:
            patterns.append({"label": "SKILL", "pattern": [{"LOWER": t} for t in tokens]})

    ruler.add_patterns(patterns)
    return nlp


def extract_regex_entities(text: str) -> List[Dict]:
    """Extract DEGREE and EXPERIENCE entities using regex on cleaned text."""
    entities = []

    for pat in DEGREE_REGEX_PATTERNS:
        for m in re.finditer(pat, text, re.IGNORECASE):
            entities.append({
                "text":  m.group().strip(),
                "label": "DEGREE",
                "start": m.start(),
                "end":   m.end(),
            })

    for pat in EXPERIENCE_REGEX_PATTERNS:
        for m in re.finditer(pat, text, re.IGNORECASE):
            entities.append({
                "text":  m.group().strip(),
                "label": "EXPERIENCE",
                "start": m.start(),
                "end":   m.end(),
            })

    return entities


def deduplicate(items: List[str]) -> List[str]:
    """Remove duplicates while preserving order."""
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# ── Main extraction function ─────────────────────────────────────────────────────

def extract_entities_from_df(df: pd.DataFrame,
                              text_col: str = "Long Description",
                              nlp: spacy.Language = None) -> pd.DataFrame:
    """
    Run NER over all rows and add columns:
      extracted_skills, extracted_degrees, extracted_experience
    """
    if nlp is None:
        nlp = build_nlp_pipeline()

    skills_list     = []
    degrees_list    = []
    experience_list = []

    for text in df[text_col]:
        text = str(text)

        # SKILL via EntityRuler
        doc    = nlp(text)
        skills = deduplicate([ent.text for ent in doc.ents if ent.label_ == "SKILL"])

        # DEGREE + EXPERIENCE via regex
        regex_ents = extract_regex_entities(text)
        degrees    = deduplicate([e["text"] for e in regex_ents if e["label"] == "DEGREE"])
        experience = deduplicate([e["text"] for e in regex_ents if e["label"] == "EXPERIENCE"])

        skills_list.append(" | ".join(skills))
        degrees_list.append(" | ".join(degrees))
        experience_list.append(" | ".join(experience))

    df = df.copy()
    df["extracted_skills"]     = skills_list
    df["extracted_degrees"]    = degrees_list
    df["extracted_experience"] = experience_list

    return df


# ── Entry point ──────────────────────────────────────────────────────────────────

def run(cleaned_csv: str = "dataset/cleaned_job_descriptions.csv",
        output_csv:  str = "dataset/ner_results.csv"):

    print(f"Loading cleaned data: {cleaned_csv}")
    df = pd.read_csv(cleaned_csv)
    print(f"  {len(df):,} rows")

    print("Building NER pipeline...")
    nlp = build_nlp_pipeline()

    print("Extracting entities...")
    df_out = extract_entities_from_df(df, nlp=nlp)

    os.makedirs("dataset", exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    print(f"Saved → {output_csv}")

    # Quick summary stats
    has_skill = (df_out["extracted_skills"] != "").sum()
    has_deg   = (df_out["extracted_degrees"] != "").sum()
    has_exp   = (df_out["extracted_experience"] != "").sum()
    total     = len(df_out)
    print(f"\nCoverage over {total:,} descriptions:")
    print(f"  SKILL      found in {has_skill:,} ({100*has_skill/total:.1f}%)")
    print(f"  DEGREE     found in {has_deg:,}   ({100*has_deg/total:.1f}%)")
    print(f"  EXPERIENCE found in {has_exp:,}  ({100*has_exp/total:.1f}%)")

    return df_out


if __name__ == "__main__":
    run()
