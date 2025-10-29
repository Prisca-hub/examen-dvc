#!/usr/bin/env python3
"""
Script de normalisation des données
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import os

print("🔄 Chargement des données d'entraînement et test...")

# Charger les données splitées
X_train = pd.read_csv('data/processed_data/X_train.csv')
X_test = pd.read_csv('data/processed_data/X_test.csv')

print(f"X_train shape : {X_train.shape}")
print(f"X_test shape : {X_test.shape}")

# Afficher les types de colonnes pour debug
print("\n📊 Types de colonnes :")
print(X_train.dtypes)

# Sélectionner seulement les colonnes numériques
numeric_columns = X_train.select_dtypes(include=['float64', 'int64']).columns
print(f"\n🔢 Colonnes numériques : {list(numeric_columns)}")

# Extraire seulement les colonnes numériques
X_train_numeric = X_train[numeric_columns]
X_test_numeric = X_test[numeric_columns]

print(f"X_train_numeric shape : {X_train_numeric.shape}")
print(f"X_test_numeric shape : {X_test_numeric.shape}")

# Initialiser le scaler
scaler = StandardScaler()

print("🔧 Normalisation des données numériques...")

# Fit sur train seulement, puis transform train et test
X_train_scaled = scaler.fit_transform(X_train_numeric)
X_test_scaled = scaler.transform(X_test_numeric)

# Convertir en DataFrame pour garder les noms de colonnes
X_train_scaled = pd.DataFrame(X_train_scaled, columns=numeric_columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=numeric_columns)

print(f"X_train_scaled shape : {X_train_scaled.shape}")
print(f"X_test_scaled shape : {X_test_scaled.shape}")

# Sauvegarder les données normalisées
print("💾 Sauvegarde des données normalisées...")

X_train_scaled.to_csv('data/processed_data/X_train_scaled.csv', index=False)
X_test_scaled.to_csv('data/processed_data/X_test_scaled.csv', index=False)

# Créer le dossier models si nécessaire
os.makedirs('models', exist_ok=True)

# Sauvegarder le scaler pour pouvoir l'utiliser plus tard
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("✅ Normalisation terminée !")
print("📁 Fichiers sauvés :")
print("   - data/processed_data/X_train_scaled.csv")
print("   - data/processed_data/X_test_scaled.csv") 
print("   - models/scaler.pkl")
print(f"📊 Colonnes normalisées : {len(numeric_columns)}")
