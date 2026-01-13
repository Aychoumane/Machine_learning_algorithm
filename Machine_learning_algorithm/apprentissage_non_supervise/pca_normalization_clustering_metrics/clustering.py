import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
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

#PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

#Choix du meilleur de cluster a prendre, avec silhouette_score
bestk_score = 0 
nb_k = 3 
for k in range(2, 8):  # on teste entre 2 et 7 clusters
    kmeans = KMeans(n_clusters=k, random_state=0)
    clusters = kmeans.fit_predict(X_scaled)
    temp_score = silhouette_score(X_scaled, clusters)
    print(f"k = {k} --> Score de silhouette = {temp_score:.3f}")
    if(temp_score > bestk_score):
        bestk_score = temp_score
        nb_k = k

print(f"Meilleur nombre de clusters : {nb_k} avec un score de {bestk_score:.3f}")

#clustering K-Means avec le meilleur nb cluster
kmeans = KMeans(n_clusters=nb_k, random_state=0)
clustering = kmeans.fit_predict(X_scaled)

# Visualisation des clusters
colors = ['red', 'yellow', 'blue', 'pink']
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1],
            c=clustering,
            cmap=matplotlib.colors.ListedColormap(colors))

for label, x, y in zip(labels, X_pca[:, 0], X_pca[:, 1]):
    plt.annotate(label, xy=(x, y), xytext=(-0.2, 0.2), textcoords='offset points')

plt.xlabel('Axe 1 (%.2f%%)' % (pca.explained_variance_ratio_[0]*100))
plt.ylabel('Axe 2 (%.2f%%)' % (pca.explained_variance_ratio_[1]*100))
plt.title('Clustering K-Means (3 groupes) - Projection sur le plan principal')
plt.show()
