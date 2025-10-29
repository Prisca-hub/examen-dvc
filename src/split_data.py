#!/usr/bin/env python3
"""
Script de split des données
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# Charger les données
print("Chargement des données...")
df = pd.read_csv('data/raw_data/raw.csv')
print(f"Données chargées : {df.shape}")

# Séparer X (features) et y (target = dernière colonne)
X = df.iloc[:, :-1]  # Toutes les colonnes sauf la dernière
y = df.iloc[:, -1]   # Dernière colonne (silica_concentrate)

print(f"Features (X) : {X.shape}")
print(f"Target (y) : {y.shape}")

# Split train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"X_train : {X_train.shape}")
print(f"X_test : {X_test.shape}")

# Sauvegarder dans processed_data
X_train.to_csv('data/processed_data/X_train.csv', index=False)
X_test.to_csv('data/processed_data/X_test.csv', index=False)
y_train.to_csv('data/processed_data/y_train.csv', index=False)
y_test.to_csv('data/processed_data/y_test.csv', index=False)

print("✅ Split terminé ! Fichiers sauvés dans data/processed_data/")
