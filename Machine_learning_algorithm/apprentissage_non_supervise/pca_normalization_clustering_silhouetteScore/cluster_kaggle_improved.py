import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import matplotlib

# Chargement et nettoyage du dataset
df = sns.load_dataset('penguins').dropna()

# Transformation des variables catégorielles en variables numériques
# On encode 'island' et 'sex' avec One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=['island', 'sex'], drop_first=True)

# Visualisation
print("Colonnes après encodage :", df_encoded.columns.tolist())

# Sélection des colonnes numériques pour clustering
num_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g'] + \
           [col for col in df_encoded.columns if col.startswith('island_') or col.startswith('sex_')]
X = df_encoded[num_cols].values

# Standardisation
X_scaled = StandardScaler().fit_transform(X)

# PCA à 2 dimensions pour visualisation
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print("\nVariance expliquée par chaque composante :")
for i, ratio in enumerate(pca.explained_variance_ratio_, 1):
    print(f"PC{i} : {ratio:.2f}")

# Recherche du meilleur k avec silhouette score
best_score = 0
best_k = 3
for k in range(2, 8):
    clusters = KMeans(n_clusters=k, random_state=0).fit_predict(X_scaled)
    score = silhouette_score(X_scaled, clusters)
    print(f"k={k} --> silhouette score = {score:.3f}")
    if score > best_score:
        best_score = score
        best_k = k

print(f"\nMeilleur nombre de clusters : {best_k} avec un score = {best_score:.3f}")

# Clustering final
final_clusters = KMeans(n_clusters=best_k, random_state=0).fit_predict(X_scaled)

# DataFrame pour visualisation
df_vis = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_vis['Cluster'] = final_clusters
df_vis['Species'] = df['species'].values

print("\nAperçu des données PCA avec clusters :")
print(df_vis.head())

# Visualisation
colors = ['red', 'blue', 'yellow', 'pink']

plt.figure(figsize=(8,6))
plt.scatter(df_vis['PC1'], df_vis['PC2'],
            c=df_vis['Cluster'],
            cmap=matplotlib.colors.ListedColormap(colors[:best_k]),
            s=50)

for species, x, y in zip(df_vis['Species'], df_vis['PC1'], df_vis['PC2']):
    plt.annotate(species, xy=(x, y), xytext=(-0.2, 0.2), textcoords='offset points', fontsize=8)

plt.xlabel(f'Axe 1 ({pca.explained_variance_ratio_[0]*100:.2f}% variance)')
plt.ylabel(f'Axe 2 ({pca.explained_variance_ratio_[1]*100:.2f}% variance)')
plt.title(f'Clustering K-Means (k={best_k}) avec variables catégorielles encodées')
plt.show()


# 8. Choix du nombre optimal de clusters (k)
# --------------------------------------------------------
# On choisit k en testant plusieurs valeurs et en regardant la qualité du regroupement.
# Ici, on utilise le silhouette score : plus il est proche de 1, mieux c’est.
# On sélectionne le k qui donne le score le plus élevé.

# 9. Standardisation des caractéristiques
# --------------------------------------------------------
# On standardise les variables pour qu'elles soient toutes sur la même échelle.
# Sinon, les variables avec de grandes valeurs (comme body_mass_g) domineraient le calcul des distances.
# Cela permet au clustering et à la PCA de mieux fonctionner et d'être plus justes.

# 10. Avantages et limites de K-Means
# --------------------------------------------------------
# Avantages :
# - Simple et rapide
# - Facile à comprendre
# - Fonctionne bien sur de grands datasets
# Limites :
# - Il faut choisir k à l'avance
# - Suppose des clusters ronds et de taille similaire
# - Sensible aux valeurs extrêmes (outliers)
# - Résultat dépend de l'initialisation

# 11. Évaluation de la qualité des clusters
# --------------------------------------------------------
# On peut utiliser :
# - Silhouette score : indique la cohésion des clusters (1 = parfait)
# - Inertie/SSE : distance des points à leur centre (plus c’est faible, mieux c’est)
# - Autres indices : Dunn index, Davies-Bouldin, etc.
# - Visualisation (PCA, t-SNE) pour voir si les clusters sont séparés
# - Stabilité : vérifier si le clustering reste similaire avec des variations des données
