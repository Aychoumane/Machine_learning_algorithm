import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import warnings
warnings.filterwarnings('ignore')

# Lecture du fichier CSV
data = pd.read_csv('./villes.csv', sep=';')

# Séparation des données
X = data.iloc[:, 1:13].values   # Données quantitatives
labels = data.iloc[:, 0].values # Noms des villes

# Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Classification hiérarchique (méthode Average)
#average : utilise la distance moyenne entre les points de deux clusters.
cah_avg = AgglomerativeClustering(n_clusters=3, linkage='average')
clustering_avg = cah_avg.fit_predict(X_scaled)

# Visualisation
colors = ['red', 'yellow', 'blue', 'pink']
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1],
            c=clustering_avg,
            cmap=matplotlib.colors.ListedColormap(colors))

for label, x, y in zip(labels, X_pca[:, 0], X_pca[:, 1]):
    plt.annotate(label, xy=(x, y), xytext=(-0.2, 0.2), textcoords='offset points')

plt.xlabel('Axe 1 (%.2f%%)' % (pca.explained_variance_ratio_[0]*100))
plt.ylabel('Axe 2 (%.2f%%)' % (pca.explained_variance_ratio_[1]*100))
plt.title('Classification hiérarchique (Average) - 3 clusters')
plt.show()
