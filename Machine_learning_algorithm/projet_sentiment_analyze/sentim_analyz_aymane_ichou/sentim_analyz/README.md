# Reconnaissance_demotion

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Projet de technique d'apprentissage d'ia

## Project Organization

├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile avec commandes pratiques (`make data`, `make train`, etc.)
├── README.md          <- Documentation principale du projet
├── data
│   ├── external       <- Données externes (sources tierces)
│   ├── interim        <- Données intermédiaires transformées
│   ├── processed      <- Données finales prêtes pour le modeling
│   └── raw            <- Données brutes (train.csv, test.csv, validation.csv)
│
├── docs               <- Documentation mkdocs
│
├── models             <- Modèles entraînés et sérialisés (.pkl)
│
├── notebooks          <- Jupyter notebooks (exploration, prototypage)
│
├── pyproject.toml     <- Configuration du projet et outils (black, etc.)
├── references         <- Dictionnaires de données, manuels, docs explicatives
├── reports            <- Analyses générées (HTML, PDF, LaTeX, etc.)
│   └── figures        <- Graphiques et visualisations
│
├── requirements.txt   <- Dépendances Python
├── setup.cfg          <- Configuration flake8
│
└── reconnaissance_demotion   <- Code source du projet
    │
    ├── __init__.py             <- Rend reconnaissance_demotion un module Python
    ├── config.py               <- Variables globales et chemins
    ├── dataset.py              <- Chargement et préparation du dataset
    ├── features.py             <- Génération des features TF-IDF
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Inférence et prédictions
    │   └── train.py            <- Entraînement et évaluation des modèles
    └── plots.py                <- Interface Tkinter et visualisations
```

--------

# Reconnaissance_demotion

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Projet de reconnaissance d’émotions dans des textes en français, basé sur des modèles de machine learning (Naive Bayes, SVM linéaire, Régression Logistique).  
Ce projet suit la structure **cookiecutter data science**.

---

## 🚀 Installation

### 1. Ouvrir le projet (rester à l'endroit où se trouve le makefile).

###2. Créer l’environnement virtuel
make create_environment

et entrer cette commande pour activer l'environnement virtuel : 
-> source .venv/bin/activate (linux)

(Pour l'arrêter : deactivate )

###3. Installer les dépendances
make requirements

### Utilisation
Le pipeline complet peut être exécuté étape par étape avec Makefile 
(ou directement make gui  :)

###1. Préparer le dataset
make data

###2. Générer les features TF-IDF
make features

###3. Entraîner les modèles
make train

###4. Faire des prédictions sur le jeu de test
make predict

###5. Lancer l’interface graphique Tkinter
make gui