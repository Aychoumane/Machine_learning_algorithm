import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

#Charger le jeu de données CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),  # Convertit l'image en tenseur [0,1]
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

# Convertir les images en tableaux numpy et aplatir (32x32x3 -> vecteur 3072)
X = np.array([np.array(img).flatten() for img, label in trainset + testset])
y = np.array([label for img, label in trainset + testset])

#Prétraitement : normalisation avec StandardScaler et division en train/val/test
scaler = StandardScaler()
X = scaler.fit_transform(X)  # centrer et réduire

# Diviser en train+val et test (80/20)
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Diviser train+val en train et validation (75/25 -> 60/20/20 global)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval
)

print("Taille X_train :", X_train.shape)
print("Taille X_val   :", X_val.shape)
print("Taille X_test  :", X_test.shape)

# Random Forest

# === Modèle 1 : défaut ===
rf_default = RandomForestClassifier(random_state=42)
rf_default.fit(X_train, y_train)
y_val_pred1 = rf_default.predict(X_val)

acc1 = accuracy_score(y_val, y_val_pred1)

print("\nModèle 1 - Paramètres par défaut")
print("Exactitude :", round(acc1, 3))

# === Modèle 2 : plus d’arbres, profondeur limitée ===
rf_tuned1 = RandomForestClassifier(
    n_estimators=200,   # nombre d'arbres
    max_depth=10,       # profondeur maximale
    random_state=42
)

rf_tuned1.fit(X_train, y_train)
y_val_pred2 = rf_tuned1.predict(X_val)

acc2 = accuracy_score(y_val, y_val_pred2)

print("\nModèle 2 - n_estimators=200, max_depth=10")
print("Exactitude :", round(acc2, 3))

# === Modèle 3 : moins d’arbres, mais plus de feuilles aléatoires ===
rf_tuned2 = RandomForestClassifier(
    n_estimators=50,     # moins d’arbres
    max_depth=None,      # profondeur illimitée
    max_features='sqrt', # sous-échantillonnage des features
    random_state=42
)

rf_tuned2.fit(X_train, y_train)
y_val_pred3 = rf_tuned2.predict(X_val)

acc3 = accuracy_score(y_val, y_val_pred3)

print("\nModèle 3 - n_estimators=50, max_features='sqrt'")
print("Exactitude :", round(acc3, 3))

# === Résumé comparatif ===
print("\n=== Résultats ===")
print(f"Modèle 1 (défaut)                : {round(acc1, 3)}")
print(f"Modèle 2 (200 arbres, depth=10)   : {round(acc2, 3)}")
print(f"Modèle 3 (50 arbres, sqrt)        : {round(acc3, 3)}")


# Déterminer quel modèle a la meilleure performance sur la validation
val_accuracies = [acc1, acc2, acc3]
models = [rf_default, rf_tuned1, rf_tuned2]

best_index = val_accuracies.index(max(val_accuracies))
best_model = models[best_index]

print(f"\nLe meilleur modèle sur la validation est le modèle {best_index} avec une exactitude de {round(val_accuracies[best_index], 3)}")

# Évaluer ce modèle sur l'ensemble de test
y_test_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("=== Performance finale sur l'ensemble de test ===")
print("Exactitude :", round(test_accuracy, 3))