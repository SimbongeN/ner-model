"""
train_ner.py
------------
Trains a spaCy NER model to extract:
  SKILL       – technical / soft skills
  DEGREE      – education requirements
  EXPERIENCE  – years-of-experience mentions

Model: spaCy blank("en") + NER pipeline trained from scratch.
This mirrors the approach described in the project abstract (fine-tuning
a pre-trained NLP framework on a combined labeled dataset).

Usage
-----
    python train_ner.py

Outputs
-------
    models/ner_model/   – trained spaCy model directory
    results/training_log.json
"""

import json
import os
import random
import time
import spacy
from spacy.training import Example
from spacy.tokens import DocBin
from pathlib import Path

random.seed(42)


# ── Configuration ───────────────────────────────────────────────────────────────
TRAIN_DATA_PATH = "dataset/train.spacy"
TEST_DATA_PATH  = "dataset/test.spacy"
MODEL_OUTPUT    = "models/ner_model"
N_ITER          = 30      # training iterations
DROPOUT         = 0.35    # regularisation dropout
BATCH_SIZE      = 32


# ── Helpers ─────────────────────────────────────────────────────────────────────

def load_spacy_data(path: str, nlp):
    """Load a .spacy DocBin and return a list of Example objects."""
    db = DocBin().from_disk(path)
    examples = []
    for doc in db.get_docs(nlp.vocab):
        example = Example.from_dict(doc, {"entities": [(e.start_char, e.end_char, e.label_) for e in doc.ents]})
        examples.append(example)
    return examples


# ── Training ─────────────────────────────────────────────────────────────────────

def train(train_path: str = TRAIN_DATA_PATH,
          test_path:  str = TEST_DATA_PATH,
          output_dir: str = MODEL_OUTPUT,
          n_iter:     int = N_ITER):

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("results",  exist_ok=True)

    # ── Build pipeline ───────────────────────────────────────────────────────────
    nlp = spacy.blank("en")
    nlp.add_pipe("ner", last=True)
    ner = nlp.get_pipe("ner")

    # Add entity labels
    for label in ("SKILL", "DEGREE", "EXPERIENCE"):
        ner.add_label(label)

    # ── Load training examples ───────────────────────────────────────────────────
    print(f"Loading training data from: {train_path}")
    train_examples = load_spacy_data(train_path, nlp)
    print(f"  {len(train_examples)} training examples")

    # ── Initialise the model ─────────────────────────────────────────────────────
    nlp.initialize(lambda: train_examples)

    # ── Training loop ────────────────────────────────────────────────────────────
    optimizer = nlp.create_optimizer()
    log = []
    best_loss = float("inf")

    print(f"\nTraining for {n_iter} iterations  (dropout={DROPOUT})\n")
    print(f"{'Iter':>5}  {'Loss':>10}  {'Time(s)':>8}")
    print("-" * 30)

    for i in range(1, n_iter + 1):
        t0 = time.time()
        random.shuffle(train_examples)
        losses = {}

        # Mini-batch training
        batches = spacy.util.minibatch(train_examples, size=BATCH_SIZE)
        for batch in batches:
            nlp.update(batch, drop=DROPOUT, losses=losses, sgd=optimizer)

        elapsed = time.time() - t0
        loss_val = losses.get("ner", 0.0)
        print(f"{i:>5}  {loss_val:>10.4f}  {elapsed:>8.2f}")
        log.append({"iter": i, "loss": round(float(loss_val), 4), "time_s": round(elapsed, 2)})

        # Save best model
        if loss_val < best_loss:
            best_loss = loss_val
            nlp.to_disk(output_dir)

    # ── Save training log ────────────────────────────────────────────────────────
    log_path = "results/training_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nTraining log saved → {log_path}")
    print(f"Best model saved   → {output_dir}  (loss={best_loss:.4f})")

    return nlp


if __name__ == "__main__":
    train()
