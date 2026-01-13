import numpy as np
np.set_printoptions(threshold=np.inf)
import pandas as pd
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler, MinMaxScaler


k_voisin = 9

df = pd.read_csv("./credit_scoring.csv", sep=";")


# Aperçu des 5 premières lignes
print(df.head())

# Informations générales
print(df.info())

# Statistiques descriptives
print(df.describe())

#Conversion en numpy array 
data = df.values

#Séparation des features , et de la target

print(df.columns)

X = df.drop("Status", axis=1).values   
y = df["Status"].values    


print("Taille du dataset :", df.shape)


df["Status"].hist()


# Séparation en train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3,      # 30 % pour le test
    random_state=42,    # pour obtenir toujours le même split
    stratify=y          # pour garder la même proportion de classes
)

print("Taille jeu d'apprentissage :", X_train.shape)
print("Taille jeu de test :", X_test.shape)

knn = KNeighborsClassifier(n_neighbors=k_voisin)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("Accuracy w/ normalization:", round(accuracy_score(y_test, y_pred), 4))

# Création du scaler
scaler = StandardScaler()

# Ajustement sur le jeu d'entraînement, puis transformation
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entraînement du modèle kNN sur les données normalisées
knn_scaled = KNeighborsClassifier(n_neighbors = k_voisin)
knn_scaled.fit(X_train_scaled, y_train)

# Prédiction
y_pred_scaled = knn_scaled.predict(X_test_scaled)

# Évaluation
print("Accuracy with StandardScaler:", round(accuracy_score(y_test, y_pred_scaled), 4))
# print("Accuracy with normalization:", accuracy_score(y_test, y_pred_scaled))


scaler_minmax = MinMaxScaler()
X_train_mm = scaler_minmax.fit_transform(X_train)
X_test_mm = scaler_minmax.transform(X_test)

knn_mm = KNeighborsClassifier(n_neighbors=5)
knn_mm.fit(X_train_mm, y_train)
y_pred_mm = knn_mm.predict(X_test_mm)


print("Accuracy with MinMaxScaler:", round(accuracy_score(y_test, y_pred_mm),4))



plt.xlabel("Status")
plt.ylabel("Nombre d'exemples")
plt.title("Distribution des classes (positifs vs négatifs)")
plt.show()
