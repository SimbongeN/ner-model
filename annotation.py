"""
annotation.py
-------------
Converts the manually annotated CSV (or auto-pre-annotated data) into
spaCy's binary .spacy training format.

Entity categories
-----------------
  SKILL         – technical / soft skills  (e.g. "python", "communication")
  DEGREE        – education requirements   (e.g. "bachelor's degree", "phd")
  EXPERIENCE    – years of experience      (e.g. "3 years", "5+ years experience")

Workflow
--------
1. Rule-based pre-annotation automatically labels obvious patterns.
2. Manual annotations from the CSV override / extend rule predictions.
3. Data is split 80/20 train/test and saved as .spacy files.
"""

import re
import json
import random
import os
import pandas as pd
import spacy
from spacy.tokens import DocBin
from spacy.training import Example
from typing import List, Tuple

random.seed(42)


# ── Entity patterns ─────────────────────────────────────────────────────────────

SKILL_KEYWORDS = [
    "python", "java", "javascript", "sql", "r ", "excel", "tableau",
    "machine learning", "deep learning", "nlp", "natural language processing",
    "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy",
    "communication", "leadership", "teamwork", "problem solving",
    "data analysis", "data science", "aws", "azure", "gcp", "docker",
    "kubernetes", "git", "agile", "scrum", "project management",
    "c\\+\\+", "c#", "ruby", "php", "html", "css", "react", "node",
]

DEGREE_PATTERNS = [
    r"bachelor['\u2019]?s?\s+(?:of\s+)?(?:science|arts|engineering|commerce|business)?(?:\s+degree)?",
    r"master['\u2019]?s?\s+(?:of\s+)?(?:science|arts|engineering|business|mba)?(?:\s+degree)?",
    r"(?:ph\.?d|doctorate)(?:\s+degree)?",
    r"b\.?sc\.?",
    r"m\.?sc\.?",
    r"m\.?b\.?a\.?",
    r"associate['\u2019]?s?\s+degree",
    r"postgraduate\s+(?:diploma|degree|certificate)",
    r"honours\s+degree",
    r"undergraduate\s+degree",
]

EXPERIENCE_PATTERNS = [
    r"\d+\s*\+?\s*years?\s+(?:of\s+)?(?:work\s+)?experience",
    r"\d+\s*[-\u2013]\s*\d+\s+years?\s+(?:of\s+)?experience",
    r"minimum\s+\d+\s+years?",
    r"at\s+least\s+\d+\s+years?",
    r"\d+\s+years?\s+in\s+\w+",
]


def _find_matches(text: str, patterns: List[str], label: str) -> List[Tuple[int, int, str]]:
    """Return (start, end, label) spans for all pattern matches in text."""
    spans = []
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            spans.append((m.start(), m.end(), label))
    return spans


def auto_annotate(text: str) -> List[Tuple[int, int, str]]:
    """Rule-based pre-annotation returning a list of (start, end, label) spans."""
    spans = []

    # Skills: match keyword as whole word / phrase
    for kw in SKILL_KEYWORDS:
        pattern = r"\b" + kw + r"\b"
        for m in re.finditer(pattern, text, re.IGNORECASE):
            spans.append((m.start(), m.end(), "SKILL"))

    # Degree
    for pat in DEGREE_PATTERNS:
        spans.extend(_find_matches(text, [pat], "DEGREE"))

    # Experience
    spans.extend(_find_matches(text, EXPERIENCE_PATTERNS, "EXPERIENCE"))

    # Remove overlapping spans (keep longest)
    spans = _remove_overlaps(sorted(spans, key=lambda x: (x[0], -(x[1]-x[0]))))
    return spans


def _remove_overlaps(spans):
    """Keep non-overlapping spans (greedy, longest-first)."""
    result = []
    last_end = -1
    for start, end, label in spans:
        if start >= last_end:
            result.append((start, end, label))
            last_end = end
    return result


def build_training_data(csv_path: str) -> List[Tuple[str, dict]]:
    """
    Load descriptions from CSV and apply auto-annotation.
    Returns list of (text, {"entities": [(start, end, label), ...]}) tuples.
    """
    df = pd.read_csv(csv_path)
    training_data = []

    for _, row in df.iterrows():
        text = str(row.get("Long Description", ""))
        if len(text) < 20:
            continue
        entities = auto_annotate(text)
        training_data.append((text, {"entities": entities}))

    print(f"Built training data: {len(training_data)} examples")
    return training_data


def to_spacy_format(training_data, output_dir: str = "dataset"):
    """
    Convert (text, entities) pairs → spaCy DocBin (.spacy) files.
    Splits 80% train / 20% test.
    """
    os.makedirs(output_dir, exist_ok=True)
    nlp = spacy.blank("en")

    random.shuffle(training_data)
    split = int(0.8 * len(training_data))
    splits = {"train": training_data[:split], "test": training_data[split:]}

    for split_name, data in splits.items():
        db = DocBin()
        skipped = 0
        for text, annotations in data:
            doc = nlp.make_doc(text)
            ents = []
            for start, end, label in annotations["entities"]:
                span = doc.char_span(start, end, label=label, alignment_mode="contract")
                if span is not None:
                    ents.append(span)
            try:
                doc.ents = spacy.util.filter_spans(ents)
                db.add(doc)
            except Exception:
                skipped += 1

        out_path = os.path.join(output_dir, f"{split_name}.spacy")
        db.to_disk(out_path)
        print(f"Saved {split_name}: {len(data) - skipped} docs → {out_path}  (skipped {skipped})")


# ── Annotation statistics ───────────────────────────────────────────────────────

def annotation_stats(training_data):
    """Print a summary of entity distribution."""
    from collections import Counter
    counts = Counter()
    for _, ann in training_data:
        for _, _, label in ann["entities"]:
            counts[label] += 1
    print("\nEntity distribution:")
    for label, cnt in counts.most_common():
        print(f"  {label:12s}: {cnt:,}")


if __name__ == "__main__":
    training_data = build_training_data("dataset/annotation_sample.csv")
    annotation_stats(training_data)
    to_spacy_format(training_data, output_dir="dataset")
