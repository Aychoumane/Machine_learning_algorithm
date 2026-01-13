from reconnaissance_demotion.dataset import load_and_prepare_dataset, explore_dataset, normalize_text
from reconnaissance_demotion.features import get_french_stopwords, prepare_features
from reconnaissance_demotion.modeling.train import train_models, evaluate_all_models
from reconnaissance_demotion.plots import launch_gui
from reconnaissance_demotion.config import RAW_DATA_DIR, RANDOM_STATE

def main():
    print("=== Chargement Stopwords ===")
    french_stopwords = get_french_stopwords()

    print("=== Chargement Dataset ===")
    df = load_and_prepare_dataset(
        RAW_DATA_DIR / "train.csv",
        RAW_DATA_DIR / "test.csv",
        RAW_DATA_DIR / "validation.csv"
    )

    print("=== Exploration Dataset ===")
    explore_dataset(df)

    print("=== TF-IDF & Split ===")
    X_train_tfidf, X_test_tfidf, y_train, y_test, tfidf = prepare_features(df, french_stopwords)

    print("=== Entraînement modèles ===")
    models = train_models(X_train_tfidf, y_train)

    print("=== Évaluation ===")
    evaluate_all_models(models, X_test_tfidf, y_test)

    print("=== Lancement interface ===")
    launch_gui(models, tfidf)

if __name__ == "__main__":
    main()
