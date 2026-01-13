from pathlib import Path
from loguru import logger
import pandas as pd
import re
from unidecode import unidecode
import typer

from reconnaissance_demotion.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

# ------------------ Normalisation ------------------
def normalize_text(text: str) -> str:
    text = text.lower()
    text = unidecode(text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ------------------ Chargement ------------------
def load_and_prepare_dataset(*csv_paths: Path) -> pd.DataFrame:
    dfs = [pd.read_csv(path) for path in csv_paths]
    df = pd.concat(dfs, ignore_index=True)

    logger.info(f"Taille dataset concaténé : {df.shape}")
    df["text"] = df["text"].apply(normalize_text)
    return df

# ------------------ Exploration ------------------
def explore_dataset(df: pd.DataFrame):
    logger.info("-------- Premières lignes du dataset -------------")
    logger.info(df.head())

    logger.info(f"=== Shape === {df.shape}")
    logger.info(f"=== Colonnes === {df.columns.tolist()}")
    logger.info("=== Info ===")
    logger.info(df.info())
    logger.info("=== Valeurs manquantes ===")
    logger.info(df.isnull().sum())

# ------------------ Commande CLI ------------------
@app.command()
def main(
    input_train: Path = RAW_DATA_DIR / "train.csv",
    input_test: Path = RAW_DATA_DIR / "test.csv",
    input_val: Path = RAW_DATA_DIR / "validation.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
):
    logger.info("Chargement et préparation du dataset...")
    df = load_and_prepare_dataset(input_train, input_test, input_val)
    explore_dataset(df)

    # Sauvegarde dataset préparé
    df.to_csv(output_path, index=False)
    logger.success(f"Dataset préparé sauvegardé dans {output_path}")

if __name__ == "__main__":
    app()
