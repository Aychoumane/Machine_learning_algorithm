from pathlib import Path
from loguru import logger
import pandas as pd
import typer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

from reconnaissance_demotion.config import PROCESSED_DATA_DIR, RANDOM_STATE

app = typer.Typer()

# ------------------ Stopwords ------------------
def get_french_stopwords():
    nltk.download("stopwords")
    from nltk.corpus import stopwords
    return stopwords.words("french")

# ------------------ Préparation des features ------------------
def prepare_features(df: pd.DataFrame, french_stopwords):
    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    tfidf = TfidfVectorizer(stop_words=french_stopwords, max_features=20000)

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    return X_train_tfidf, X_test_tfidf, y_train, y_test, tfidf

# ------------------ Commande CLI ------------------
@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.npz",
):
    logger.info("Préparation des features avec TF-IDF...")

    # Charger dataset préparé
    df = pd.read_csv(input_path)

    # Stopwords français
    french_stopwords = get_french_stopwords()

    # Préparer features
    X_train_tfidf, X_test_tfidf, y_train, y_test, tfidf = prepare_features(df, french_stopwords)

    # Sauvegarde des features (format compressé numpy)
    from scipy import sparse
    sparse.save_npz(PROCESSED_DATA_DIR / "X_train_tfidf.npz", X_train_tfidf)
    sparse.save_npz(PROCESSED_DATA_DIR / "X_test_tfidf.npz", X_test_tfidf)
    y_train.to_csv(PROCESSED_DATA_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DATA_DIR / "y_test.csv", index=False)

    logger.success(f"Features sauvegardées dans {PROCESSED_DATA_DIR}")

if __name__ == "__main__":
    app()
