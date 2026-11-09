"""
inference.py
------------
Runs the trained NER model on new job description text and displays
extracted entities in a structured format.

Usage
-----
    python inference.py                         # uses built-in demo texts
    python inference.py --text "Your text here" # single text from CLI
    python inference.py --csv path/to/file.csv  # batch inference on CSV
"""

import argparse
import json
import spacy
import pandas as pd

MODEL_PATH = "models/ner_model"


def extract_entities(text: str, nlp) -> dict:
    """Return a dict of label → list of extracted strings."""
    doc = nlp(text)
    result = {"SKILL": [], "DEGREE": [], "EXPERIENCE": []}
    for ent in doc.ents:
        if ent.label_ in result:
            val = ent.text.strip()
            if val not in result[ent.label_]:
                result[ent.label_].append(val)
    return result


def pretty_print(text: str, entities: dict):
    print("\n" + "═" * 60)
    print("TEXT (first 200 chars):")
    print(f"  {text[:200]}...")
    print("\nEXTRACTED ENTITIES:")
    for label, values in entities.items():
        if values:
            print(f"  {label:12s}: {', '.join(values)}")
        else:
            print(f"  {label:12s}: (none found)")
    print("═" * 60)


DEMO_TEXTS = [
    """
    Senior Data Scientist – Requirements:
    Bachelor's degree in Computer Science or Statistics (Master's preferred).
    Minimum 5 years of experience in machine learning and data analysis.
    Strong proficiency in Python, SQL, and TensorFlow. Experience with AWS or GCP.
    Excellent communication and leadership skills. Experience with NLP and deep learning.
    """,

    """
    Junior Software Engineer
    We are looking for a motivated junior developer with 1-2 years experience in
    web development. Required skills: JavaScript, React, Node.js, HTML, CSS.
    A BSc in Computer Science is required. Knowledge of Docker and Git is a plus.
    """,

    """
    HR Manager Position
    PhD or MBA preferred. At least 8 years of experience in human resources management.
    Expertise in project management, agile methodologies, and leadership.
    Strong teamwork and problem solving abilities required.
    """,
]


def main():
    parser = argparse.ArgumentParser(description="NER Inference for Job Descriptions")
    parser.add_argument("--text", type=str, default=None, help="Single text to process")
    parser.add_argument("--csv",  type=str, default=None, help="CSV file for batch inference")
    args = parser.parse_args()

    print(f"Loading model from: {MODEL_PATH}")
    nlp = spacy.load(MODEL_PATH)
    print("Model loaded.\n")

    if args.text:
        entities = extract_entities(args.text, nlp)
        pretty_print(args.text, entities)

    elif args.csv:
        df = pd.read_csv(args.csv)
        col = "Long Description" if "Long Description" in df.columns else df.columns[0]
        results = []
        for _, row in df.iterrows():
            ents = extract_entities(str(row[col]), nlp)
            results.append({
                "position": row.get("Position", ""),
                **{k: ", ".join(v) for k, v in ents.items()}
            })
        out = pd.DataFrame(results)
        out_path = args.csv.replace(".csv", "_extracted.csv")
        out.to_csv(out_path, index=False)
        print(f"Batch results saved → {out_path}")

    else:
        # Demo mode
        print("Running demo on sample job descriptions…\n")
        for text in DEMO_TEXTS:
            entities = extract_entities(text, nlp)
            pretty_print(text, entities)


if __name__ == "__main__":
    main()
