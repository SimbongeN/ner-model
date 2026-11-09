"""
run_pipeline.py
---------------
End-to-end pipeline for the COMP316 NER project.

Steps
-----
    1. Preprocess raw dataset     (data_preprocessing.py)
    2. Extract entities via NER   (ner_extractor.py)
    3. Evaluate on sample         (evaluate.py)
    4. Demo inference on examples (ner_extractor.py)

Usage
-----
    python run_pipeline.py                 # full pipeline
    python run_pipeline.py --skip-preprocess
"""

import argparse
import os


def header(msg):
    print("\n" + "=" * 60)
    print(f"  {msg}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-extract",    action="store_true")
    parser.add_argument("--skip-eval",       action="store_true")
    args = parser.parse_args()

    # ── Step 1: Preprocess ────────────────────────────────────────────────────
    if not args.skip_preprocess:
        header("STEP 1 – Data Preprocessing")
        if os.path.exists("dataset/job_descriptions.csv"):
            from data_preprocessing import preprocess
            preprocess()
        else:
            print("  dataset/job_descriptions.csv not found.")
            print("  Download from: https://huggingface.co/datasets/lang-uk/recruitment-dataset-job-descriptions-english")
            print("  Save as dataset/job_descriptions.csv and re-run.")
            # Create a small synthetic file for demo purposes
            _make_demo_dataset()

    # ── Step 2: Extract entities ──────────────────────────────────────────────
    if not args.skip_extract:
        header("STEP 2 – NER Entity Extraction")
        cleaned = "dataset/cleaned_job_descriptions.csv"
        if not os.path.exists(cleaned):
            print(f"  {cleaned} not found – run Step 1 first.")
        else:
            from ner_extractor import run
            run(cleaned_csv=cleaned, output_csv="dataset/ner_results.csv")

    # ── Step 3: Evaluate ──────────────────────────────────────────────────────
    if not args.skip_eval:
        header("STEP 3 – Evaluation (Precision / Recall / F1)")
        if not os.path.exists("dataset/ner_results.csv"):
            print("  dataset/ner_results.csv not found – run Step 2 first.")
        else:
            from evaluate import evaluate
            evaluate()

    # ── Step 4: Demo ──────────────────────────────────────────────────────────
    header("STEP 4 – Demo Inference")
    _run_demo()

    header("Pipeline complete")


def _run_demo():
    from ner_extractor import build_nlp_pipeline, extract_entities_from_df
    import pandas as pd

    demo_texts = [
        {
            "Position": "senior data scientist",
            "Long Description": (
                "bachelors degree in computer science or statistics masters preferred "
                "minimum 5 years of experience in machine learning and data analysis "
                "strong python sql and tensorflow skills aws or gcp experience required "
                "excellent communication and leadership skills nlp and deep learning a plus"
            ),
        },
        {
            "Position": "junior software engineer",
            "Long Description": (
                "bsc in computer science required 1 2 years experience in web development "
                "javascript react nodejs html css skills needed docker and git knowledge a plus "
                "teamwork and problem solving essential"
            ),
        },
        {
            "Position": "hr manager",
            "Long Description": (
                "mba preferred at least 8 years of experience in human resources management "
                "leadership project management and decision making skills agile and scrum knowledge beneficial"
            ),
        },
    ]

    df  = pd.DataFrame(demo_texts)
    nlp = build_nlp_pipeline()
    out = extract_entities_from_df(df, nlp=nlp)

    for _, row in out.iterrows():
        print(f"\n  Position  : {row['Position']}")
        print(f"  Skills    : {row['extracted_skills'] or '(none)'}")
        print(f"  Degrees   : {row['extracted_degrees'] or '(none)'}")
        print(f"  Experience: {row['extracted_experience'] or '(none)'}")


def _make_demo_dataset():
    """Creates a minimal CSV for demo when the real dataset is absent."""
    import pandas as pd, os
    rows = [
        {"Position": "data scientist",
         "Long Description": "bachelors degree in computer science required minimum 5 years of experience in machine learning strong python sql pandas tensorflow skills needed communication skills essential",
         "Long Description_lang": "en"},
        {"Position": "software engineer",
         "Long Description": "bsc computer science 3 years experience javascript react html css git docker teamwork problem solving required",
         "Long Description_lang": "en"},
        {"Position": "nlp engineer",
         "Long Description": "phd or masters degree computational linguistics 4 years experience natural language processing deep learning pytorch python leadership communication skills",
         "Long Description_lang": "en"},
        {"Position": "data analyst",
         "Long Description": "bachelors degree statistics 2 years experience data analysis python pandas numpy tableau sql communication teamwork",
         "Long Description_lang": "en"},
        {"Position": "ml engineer",
         "Long Description": "masters degree preferred at least 3 years of experience machine learning python tensorflow scikit learn docker agile",
         "Long Description_lang": "en"},
    ] * 30
    os.makedirs("dataset", exist_ok=True)
    pd.DataFrame(rows).to_csv("dataset/job_descriptions.csv", index=False)
    print("  Demo dataset created at dataset/job_descriptions.csv")


if __name__ == "__main__":
    main()
