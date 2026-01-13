import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split


from sklearn.metrics import confusion_matrix, classification_report, precision_score, f1_score, accuracy_score

import warnings 
warnings.filterwarnings("ignore")

df = pd.read_csv("creditcard.csv")
print("Nombre de lignes et de colonnes :", df.shape)
print(df.head())

print("Informations générales :")
print(df.info())

print("\nDistribution des classes (0 = normal, 1 = fraude) :")
print(df["Class"].value_counts())

# Séparation des features et de la cible
X = df.drop("Class", axis=1) 
y = df["Class"]               

# Séparation en jeu d'entraînement et de test 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

print("Taille du jeu d'entraînement :", X_train.shape)
print("Taille du jeu de test :", X_test.shape)


#Standardisation des données 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialisation du modèle Isolation Forest
iso_forest = IsolationForest(n_estimators=100, 
                             contamination=0.0017,  
                             random_state=42)

# Entraînement sur les données d'entraînement
iso_forest.fit(X_train_scaled)

# Prédiction sur le jeu de test
# Isolation Forest renvoie -1 pour anomalies (fraudes) et 1 pour normal
y_pred = iso_forest.predict(X_test_scaled)

# Conversion pour correspondre aux classes du dataset (0 = normal, 1 = fraude)
y_pred = np.where(y_pred == -1, 1, 0)

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
print("Matrice de confusion :\n", cm)

# Rapport de classification
cr = classification_report(y_test, y_pred)
print("\nRapport de classification :\n", cr)

# Scores précis
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy : ", accuracy , " (non representaif dataset desequilibré)")

print("Précision :", precision)
print("F1-score :", f1)