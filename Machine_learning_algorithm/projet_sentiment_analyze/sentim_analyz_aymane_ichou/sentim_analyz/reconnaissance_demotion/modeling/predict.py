from pathlib import Path
import typer
from loguru import logger
import joblib
import pandas as pd
from scipy import sparse
import numpy as np
from scipy.special import softmax

from reconnaissance_demotion.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

# ------------------ Prédiction Top3 ------------------
def predict_top3(model, vectorizer, text: str):
    """Retourne les 3 classes les plus probables pour un texte donné."""
    from reconnaissance_demotion.dataset import normalize_text

    txt = normalize_text(text)
    X_user = vectorizer.transform([txt])

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_user)[0] * 100
    else:
        dec = model.decision_function(X_user)[0]
        probs = softmax(dec) * 100

    classes = model.classes_
    pairs = list(zip(classes, probs))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:3]

# ------------------ Commande CLI ------------------
@app.command()
def main(
    X_test_path: Path = PROCESSED_DATA_DIR / "X_test_tfidf.npz",
    y_test_path: Path = PROCESSED_DATA_DIR / "y_test.csv",
    model_path: Path = MODELS_DIR / "naive_bayes.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
):
    logger.info(f"Chargement du modèle depuis {model_path}...")
    model = joblib.load(model_path)

    logger.info("Chargement des features de test...")
    X_test_tfidf = sparse.load_npz(X_test_path)
    y_test = pd.read_csv(y_test_path).squeeze()

    logger.info("Prédiction en cours...")
    preds = model.predict(X_test_tfidf)

    # Sauvegarde des prédictions
    pd.DataFrame({"y_true": y_test, "y_pred": preds}).to_csv(predictions_path, index=False)
    logger.success(f"Prédictions sauvegardées dans {predictions_path}")

if __name__ == "__main__":
    app()
