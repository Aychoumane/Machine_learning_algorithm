import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Lecture du fichier CSV
data = pd.read_csv('./crimes.csv', sep=';')

# Séparation des données
X = data.iloc[:, 1:].values   # Données quantitatives (types de crimes)
labels = data.iloc[:, 0].values # Noms des États

# Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Application de la PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Variances expliquées
explained_var = np.cumsum(pca.explained_variance_ratio_) * 100

# Affichage du pourcentage cumulé de variance expliquée
for i, var in enumerate(explained_var, start=1):
    print(f"Axe {i} : {var:.2f}% de variance cumulée")

# Trouver le nombre d’axes nécessaires pour atteindre 70 %
n_axes_70 = np.argmax(explained_var >= 70) + 1
print(f"\nNombre d'axes nécessaires pour conserver au moins 70% de l'information : {n_axes_70}")

# Visualisation du plan principal
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1])

for label, x, y in zip(labels, X_pca[:, 0], X_pca[:, 1]):
    plt.annotate(label, xy=(x, y), xytext=(3, 3), textcoords='offset points')

plt.xlabel('Axe 1 (%.2f%%)' % (pca.explained_variance_ratio_[0]*100))
plt.ylabel('Axe 2 (%.2f%%)' % (pca.explained_variance_ratio_[1]*100))
plt.title('Projection des États américains dans le plan principal (ACP - Crimes)')
plt.grid(True)
plt.show()
