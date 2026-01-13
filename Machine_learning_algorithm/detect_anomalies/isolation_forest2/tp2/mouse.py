import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# Chargement et préparation des données
df = pd.read_csv("mouse.csv", header=None, comment='#', delim_whitespace=True)
df = df.iloc[:, :2]  # garde que les colonnes x1 et x2
df.columns = ["x1", "x2"]

# Standardisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# --- Isolation Forest ---
# On sait que les 10 dernières instances sont des outliers => contamination = 10/500 = 0.02
iso_forest = IsolationForest(contamination=0.02, random_state=42)
iso_forest.fit(X_scaled)

# Prédiction des anomalies
y_pred = iso_forest.predict(X_scaled)  # -1 = outlier, 1 = normal

# Extraire les outliers détectés
outliers = df[y_pred == -1]

print("Points détectés comme outliers :")
print(outliers)

# --- Visualisation ---
plt.figure(figsize=(6,6))
plt.scatter(df["x1"], df["x2"], label="Normal")
plt.scatter(outliers["x1"], outliers["x2"], color='red', label="Outlier")
plt.title("Détection des outliers avec Isolation Forest")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.show()
