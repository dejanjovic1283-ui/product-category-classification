import argparse
import re
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression


# -------------------------------
# Helpers: locate CSV on Desktop
# -------------------------------
def default_csv_path() -> Path:
    """
    Pokušaj da pronađe products.csv na Desktop-u (eng) ili 'Radna površina' (sr).
    Takođe proverava OneDrive Desktop ako postoji.
    """
    home = Path.home()

    candidates = []

    # Standardni Desktop (eng)
    candidates.append(home / "Desktop" / "products.csv")

    # Windows SR: "Radna površina"
    candidates.append(home / "Radna površina" / "products.csv")

    # OneDrive Desktop
    one = Path.home() / "OneDrive"
    if one.exists():
        candidates.append(one / "Desktop" / "products.csv")
        candidates.append(one / "Radna površina" / "products.csv")
        # ponekad korisnici drže na Documents/Radna površina
        candidates.append(one / "Documents" / "Radna površina" / "products.csv")

    for c in candidates:
        if c.exists():
            return c

    # fallback: Desktop eng
    return home / "Desktop" / "products.csv"


def parse_args():
    p = argparse.ArgumentParser(description="Treniranje modela klasifikacije kategorija proizvoda.")
    p.add_argument(
        "--csv",
        type=str,
        default=str(default_csv_path()),
        help="Putanja do products.csv (podrazumevano: Desktop/products.csv, automatska detekcija).",
    )
    p.add_argument(
        "--out",
        type=str,
        default="model.pkl",
        help="Naziv izlaznog fajla za model (.pkl). Podrazumevano: model.pkl u tekućem folderu.",
    )
    return p.parse_args()


# -------------------------------
# Cleaning & features
# -------------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # ukloni NBSP/zero-width i viškove razmaka iz zaglavlja
    df.columns = (
        df.columns
        .str.replace("\u00A0", " ", regex=False)  # NBSP
        .str.replace("\u200B", "", regex=False)   # zero-width
        .str.replace(r"\s+", " ", regex=True)     # višestruki -> jedan
        .str.strip()
        .str.replace("_", " ", regex=False)
    )
    # par praktičnih preimenovanja
    if "product ID" in df.columns:
        df = df.rename(columns={"product ID": "Product ID"})
    if " Product Code" in df.columns:
        df = df.rename(columns={" Product Code": "Product Code"})
    return df


def clean_title(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    """Ekstrahuje numeričke karakteristike iz originalnog naslova (pre TF-IDF)."""

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


def main():
    args = parse_args()
    csv_path = Path(args.csv)

    # === 1) Učitaj CSV
    if not csv_path.exists():
        raise FileNotFoundError(
            f"products.csv nije pronađen na: {csv_path}\n"
            f"- Rešenja:\n"
            f"  A) Premesti CSV na Desktop i pokreni ponovo\n"
            f"  B) Pokreni sa argumentom --csv \"APSOLUTNA\\PUTANJA\\products.csv\""
        )

    print(f"Učitavam CSV: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8", engine="python", on_bad_lines="skip")
    print(f"Dimenzije skupa: {df.shape}")

    # === 2) Normalizuj nazive kolona
    df = normalize_columns(df)
    needed = ["Product Title", "Category Label"]
    for col in needed:
        if col not in df.columns:
            raise KeyError(f"Nedostaje obavezna kolona: '{col}'. Pronađene kolone: {list(df.columns)}")

    # === 3) Osnovno čišćenje
    df_clean = df.dropna(subset=["Product Title", "Category Label"]).copy()
    df_clean["Category Label"] = df_clean["Category Label"].astype(str).str.strip()
    # spajanje čestih varijanti u jedinstvene labele (po potrebi proširi)
    category_map = {
        "Mobile Phone": "Mobile Phones",
        "CPU": "CPUs",
        "Cpus": "CPUs",
        "cpus": "CPUs",
        "fridge": "Fridges",
        "tvs": "TVs",
        "Tvs": "TVs",
    }
    df_clean["Category Label"] = df_clean["Category Label"].replace(category_map)

    # === 4) Finalni pipeline: TF-IDF (sa preprocessor=clean_title) + dodatni feat + scaler + LR
    final_pipeline = Pipeline(
        steps=[
            (
                "features",
                FeatureUnion(
                    transformer_list=[
                        ("tfidf", Pipeline([
                            ("vectorizer", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), preprocessor=clean_title))
                        ])),
                        ("txtfeats", Pipeline([
                            ("extractor", TextFeatureExtractor())
                        ])),
                    ]
                ),
            ),
            ("scaler", MaxAbsScaler()),
            ("clf", LogisticRegression(max_iter=300)),
        ]
    )

    print("Treniram model (može da potraje minut-dva u zavisnosti od mašine)...")
    final_pipeline.fit(df_clean["Product Title"], df_clean["Category Label"])

    # === 5) Sačuvaj model
    out_path = Path(args.out).resolve()
    joblib.dump(final_pipeline, out_path)
    print(f"✅ Model sačuvan u: {out_path}")

    # kratka informacija o klasama
    classes = getattr(final_pipeline.named_steps["clf"], "classes_", None)
    if classes is not None:
        print("Klase:", list(classes))


if __name__ == "__main__":
    main()

