import argparse
import re
from pathlib import Path
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
import warnings

# utišaj specifični FutureWarning iz sklearn.pipeline
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.pipeline")

# --- iste definicije kao u train_model.py ---

def clean_title(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    """Ekstrahuje numeričke karakteristike iz originalnog naslova (koristi se u sačuvanom pipeline-u)."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        rows = []
        for original in X:
            original = str(original)
            title_length = len(original)
            word_count = len(original.split())
            has_number = int(any(ch.isdigit() for ch in original))
            has_special = int(any((not ch.isalnum() and not ch.isspace()) for ch in original))
            words = re.sub(r"[^A-Za-z0-9 ]+", " ", original).split()
            max_word_length = max((len(w) for w in words), default=0)
            has_upper_acronym = int(any(len(w) >= 2 and w.isupper() for w in original.split()))
            rows.append([title_length, word_count, has_number, has_special, max_word_length, has_upper_acronym])
        return np.array(rows)

# --- ostatak skripte ---

def default_model_path() -> Path:
    home = Path.home()
    candidates = [
        Path.cwd() / "model.pkl",
        home / "model.pkl",
        home / "Desktop" / "model.pkl",
        home / "Radna površina" / "model.pkl",
        home / "OneDrive" / "Desktop" / "model.pkl",
        home / "OneDrive" / "Radna površina" / "model.pkl",
    ]
    for c in candidates:
        if c.exists():
            return c
    return Path.cwd() / "model.pkl"

def parse_args():
    p = argparse.ArgumentParser(description="Interaktivno testiranje modela klasifikacije kategorija proizvoda.")
    p.add_argument("--model", type=str, default=str(default_model_path()),
                   help="Putanja do model.pkl (podrazumevano: automatska detekcija).")
    return p.parse_args()

def main():
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"model.pkl nije pronađen na: {model_path}\n"
            f"Pokreni sa --model \"APSOLUTNA\\PUTANJA\\model.pkl\" ili premesti fajl."
        )
    print(f"Učitavam model: {model_path}")
    model = joblib.load(model_path)

    print("Interaktivni test — unesite naziv proizvoda (q za izlaz):")
    while True:
        try:
            s = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nKraj.")
            break
        if not s or s.lower() in {"q", "quit", "exit"}:
            print("Kraj.")
            break
        pred = model.predict([s])[0]
        print("Predviđena kategorija:", pred)

if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
