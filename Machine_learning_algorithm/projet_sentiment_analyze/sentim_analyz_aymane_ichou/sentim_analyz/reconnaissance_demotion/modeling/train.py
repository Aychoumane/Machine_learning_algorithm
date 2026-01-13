from pathlib import Path
import typer
from loguru import logger
import joblib
import pandas as pd
from scipy import sparse

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from reconnaissance_demotion.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

# ------------------ Entraînement ------------------
def train_models(X_train_tfidf, y_train):
    models = {
        "naive_bayes": MultinomialNB(),
        "svm_linear": LinearSVC(),
        "log_reg": LogisticRegression(max_iter=2000)
    }

    for name, model in models.items():
        logger.info(f"Entraînement du modèle {name}...")
        model.fit(X_train_tfidf, y_train)
        joblib.dump(model, MODELS_DIR / f"{name}.pkl")
        logger.success(f"Modèle {name} sauvegardé dans {MODELS_DIR}/{name}.pkl")

    return models

# ------------------ Évaluation ------------------
def evaluate_model(name, y_test, y_pred):
    logger.info(f"\n=== {name} ===")
    logger.info(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    logger.info(f"Precision : {precision_score(y_test, y_pred, average='macro'):.4f}")
    logger.info(f"Recall : {recall_score(y_test, y_pred, average='macro'):.4f}")
    logger.info(f"F1-score : {f1_score(y_test, y_pred, average='macro'):.4f}")
    logger.info("\nClassification Report :\n" + classification_report(y_test, y_pred))

def evaluate_all_models(models, X_test_tfidf, y_test):
    for name, model in models.items():
        preds = model.predict(X_test_tfidf)
        evaluate_model(name, y_test, preds)

# ------------------ Commande CLI ------------------
@app.command()
def main(
    X_train_path: Path = PROCESSED_DATA_DIR / "X_train_tfidf.npz",
    X_test_path: Path = PROCESSED_DATA_DIR / "X_test_tfidf.npz",
    y_train_path: Path = PROCESSED_DATA_DIR / "y_train.csv",
    y_test_path: Path = PROCESSED_DATA_DIR / "y_test.csv",
):
    logger.info("Chargement des features et labels...")

    # Charger données
    X_train_tfidf = sparse.load_npz(X_train_path)
    X_test_tfidf = sparse.load_npz(X_test_path)
    y_train = pd.read_csv(y_train_path).squeeze()
    y_test = pd.read_csv(y_test_path).squeeze()

    logger.info("Entraînement des modèles...")
    models = train_models(X_train_tfidf, y_train)

    logger.info("Évaluation des modèles...")
    evaluate_all_models(models, X_test_tfidf, y_test)

    logger.success("Entraînement et évaluation terminés.")

if __name__ == "__main__":
    app()
