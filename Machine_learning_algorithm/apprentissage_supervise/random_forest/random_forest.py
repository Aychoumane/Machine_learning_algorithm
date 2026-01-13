import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Charger le fichier CSV
df = pd.read_csv("iris.csv")

# Aperçu des 5 premières lignes
print(df.head())

# Variables explicatives et variable cible
target_column = 'variety'
X = df.drop(columns=[target_column])
y = df[target_column]

# Diviser les données en train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Création du modèle Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Prédictions
y_pred = rf.predict(X_test)

print("=== Performances du modèle sans paramètres ===")
print(classification_report(y_test, y_pred, digits=3))

# Expérimentation avec hyperparamètres
rf_param = RandomForestClassifier(n_estimators=500, max_depth=5, random_state=42)
rf_param.fit(X_train, y_train)
y_pred_tuned = rf_param.predict(X_test)

print("=== Performances du modèle avec paramètres ===")
print(classification_report(y_test, y_pred_tuned, digits=3))
