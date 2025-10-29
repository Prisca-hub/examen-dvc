#!/usr/bin/env python3
"""
Script d'entraînement du modèle avec les meilleurs paramètres
"""

import pandas as pd
import pickle
import os
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

print("�� Chargement des données et des meilleurs paramètres...")

# Charger les données
X_train_scaled = pd.read_csv('data/processed_data/X_train_scaled.csv')
X_test_scaled = pd.read_csv('data/processed_data/X_test_scaled.csv')
y_train = pd.read_csv('data/processed_data/y_train.csv').values.ravel()
y_test = pd.read_csv('data/processed_data/y_test.csv').values.ravel()

print(f"X_train_scaled shape : {X_train_scaled.shape}")
print(f"X_test_scaled shape : {X_test_scaled.shape}")
print(f"y_train shape : {y_train.shape}")
print(f"y_test shape : {y_test.shape}")

# Charger le meilleur modèle du GridSearch
try:
    with open('models/best_model.pkl', 'rb') as f:
        best_model = pickle.load(f)
    print(f"✅ Meilleur modèle chargé : {type(best_model).__name__}")
except FileNotFoundError:
    print("❌ Fichier best_model.pkl non trouvé. Lancez d'abord gridsearch.py")
    exit(1)

# Charger les meilleurs paramètres
try:
    with open('models/best_params.pkl', 'rb') as f:
        best_params = pickle.load(f)
    print(f"✅ Meilleurs paramètres : {best_params}")
except FileNotFoundError:
    print("⚠️ Fichier best_params.pkl non trouvé")
    best_params = {}

print("\n🚀 Entraînement du modèle final...")

# Le modèle est déjà entraîné depuis le GridSearch, mais on peut le réentraîner
# avec tous les données d'entraînement pour être sûr
final_model = best_model
final_model.fit(X_train_scaled, y_train)

print("✅ Modèle entraîné !")

print("\n📊 Évaluation du modèle...")

# Prédictions
y_pred_train = final_model.predict(X_train_scaled)
y_pred_test = final_model.predict(X_test_scaled)

# Métriques d'évaluation
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

# Afficher les résultats
print("📈 Métriques de performance :")
print(f"   Train RMSE : {train_rmse:.4f}")
print(f"   Test RMSE  : {test_rmse:.4f}")
print(f"   Train MAE  : {train_mae:.4f}")
print(f"   Test MAE   : {test_mae:.4f}")
print(f"   Train R²   : {train_r2:.4f}")
print(f"   Test R²    : {test_r2:.4f}")

# Créer un dictionnaire avec toutes les métriques
metrics = {
    'train_rmse': train_rmse,
    'test_rmse': test_rmse,
    'train_mae': train_mae,
    'test_mae': test_mae,
    'train_r2': train_r2,
    'test_r2': test_r2,
    'model_type': type(final_model).__name__,
    'best_params': best_params
}

print("\n💾 Sauvegarde du modèle final...")

# Sauvegarder le modèle final entraîné
with open('models/final_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)

# Sauvegarder les métriques
with open('models/model_metrics.pkl', 'wb') as f:
    pickle.dump(metrics, f)

# Sauvegarder les prédictions pour analyse
predictions = {
    'y_train_true': y_train,
    'y_train_pred': y_pred_train,
    'y_test_true': y_test,
    'y_test_pred': y_pred_test
}

with open('models/predictions.pkl', 'wb') as f:
    pickle.dump(predictions, f)

print("✅ Entraînement terminé !")
print("📁 Fichiers sauvés :")
print("   - models/final_model.pkl (modèle entraîné)")
print("   - models/model_metrics.pkl (métriques)")
print("   - models/predictions.pkl (prédictions)")

print(f"\n🎯 Résumé final :")
print(f"   Modèle : {type(final_model).__name__}")
print(f"   R² Test : {test_r2:.4f}")
print(f"   RMSE Test : {test_rmse:.4f}")

# Vérifier si le modèle a de bonnes performances
if test_r2 > 0.8:
    print("🎉 Excellent modèle ! (R² > 0.8)")
elif test_r2 > 0.6:
    print("👍 Bon modèle (R² > 0.6)")
else:
    print("⚠️ Modèle à améliorer (R² < 0.6)")
