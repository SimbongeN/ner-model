"""
data_preprocessing.py
---------------------
Cleans and prepares the raw job descriptions dataset for NER extraction.

Dataset : lang-uk/recruitment-dataset-job-descriptions-english
          (CSV with columns: Position, Long Description, Company Name,
           Exp Years, Primary Keyword, English Level, Published,
           Long Description_lang, id, __index_level_0__)

Steps
-----
1. Load CSV
2. Keep English-only rows  (Long Description_lang == 'en')
3. Drop rows with missing Position or Long Description
4. Remove duplicate descriptions
5. Clean text  (lowercase, remove markdown symbols, collapse whitespace)
6. Save cleaned dataset  →  dataset/cleaned_job_descriptions.csv
"""

import pandas as pd
import re
import os


# ── Configuration ───────────────────────────────────────────────────────────────
DATASET_PATH   = "dataset/job_descriptions.csv"
OUTPUT_CLEANED = "dataset/cleaned_job_descriptions.csv"


# ── Text cleaning ───────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Normalise raw job description text for NER processing."""
    if not isinstance(text, str):
        return ""

    # lowercase
    text = text.lower()

    # remove markdown bold/italic headers e.g. *Requirements*
    text = re.sub(r"\*.*?\*", "", text)

    # remove special characters; keep letters, digits, spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # collapse all whitespace (newlines, tabs, multiple spaces)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ── Main pipeline ───────────────────────────────────────────────────────────────
def preprocess(path: str = DATASET_PATH) -> pd.DataFrame:
    print(f"Loading dataset: {path}")
    df = pd.read_csv(path)
    print(f"  Raw rows             : {len(df):,}")

    # 1. Keep English only
    if "Long Description_lang" in df.columns:
        df = df[df["Long Description_lang"] == "en"].copy()
        print(f"  After language filter: {len(df):,}")

    # 2. Drop missing essential columns
    df.dropna(subset=["Long Description", "Position"], inplace=True)
    print(f"  After dropna         : {len(df):,}")

    # 3. Remove duplicate descriptions
    before = len(df)
    df.drop_duplicates(subset=["Long Description"], inplace=True)
    print(f"  After dedup          : {len(df):,}  (removed {before - len(df):,})")

    # 4. Clean text
    df["Long Description"] = df["Long Description"].apply(clean_text)
    df["Position"]         = df["Position"].apply(clean_text)

    # 5. Drop descriptions that are too short after cleaning
    df = df[df["Long Description"].str.len() > 50]
    print(f"  After length filter  : {len(df):,}")

    # 6. Save
    os.makedirs("dataset", exist_ok=True)
    df.to_csv(OUTPUT_CLEANED, index=False)
    print(f"\nSaved → {OUTPUT_CLEANED}")

    return df


if __name__ == "__main__":
    df = preprocess()
    print("\nSample output:")
    print(df[["Position", "Long Description"]].head(3).to_string())
