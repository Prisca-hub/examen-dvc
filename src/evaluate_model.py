#!/usr/bin/env python3
"""
Script d'évaluation du modèle et génération des prédictions
"""

import pandas as pd
import pickle
import json
import os
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score

print("🔄 Chargement du modèle final et des données...")

# Charger le modèle final
try:
    with open('models/final_model.pkl', 'rb') as f:
        final_model = pickle.load(f)
    print(f"✅ Modèle chargé : {type(final_model).__name__}")
except FileNotFoundError:
    print("❌ Fichier final_model.pkl non trouvé. Lancez d'abord train_model.py")
    exit(1)

# Charger les données
X_train_scaled = pd.read_csv('data/processed_data/X_train_scaled.csv')
X_test_scaled = pd.read_csv('data/processed_data/X_test_scaled.csv')
y_train = pd.read_csv('data/processed_data/y_train.csv').values.ravel()
y_test = pd.read_csv('data/processed_data/y_test.csv').values.ravel()

print(f"Données chargées - Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")

print("\n🔮 Génération des prédictions...")

# Faire les prédictions
y_pred_train = final_model.predict(X_train_scaled)
y_pred_test = final_model.predict(X_test_scaled)

print(f"✅ Prédictions générées - Train: {len(y_pred_train)}, Test: {len(y_pred_test)}")

print("\n�� Calcul des métriques d'évaluation...")

# Calculer les métriques pour l'ensemble d'entraînement
train_mse = mean_squared_error(y_train, y_pred_train)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, y_pred_train)
train_r2 = r2_score(y_train, y_pred_train)
train_explained_var = explained_variance_score(y_train, y_pred_train)

# Calculer les métriques pour l'ensemble de test
test_mse = mean_squared_error(y_test, y_pred_test)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_pred_test)
test_r2 = r2_score(y_test, y_pred_test)
test_explained_var = explained_variance_score(y_test, y_pred_test)

# Métriques additionnelles
train_mean = np.mean(y_train)
test_mean = np.mean(y_test)
train_std = np.std(y_train)
test_std = np.std(y_test)

print("📈 Métriques calculées :")
print(f"   Test R² : {test_r2:.4f}")
print(f"   Test RMSE : {test_rmse:.4f}")
print(f"   Test MAE : {test_mae:.4f}")

print("\n💾 Sauvegarde des prédictions...")

# Créer le dataset avec les prédictions
predictions_data = {
    'y_train_true': y_train,
    'y_train_predicted': y_pred_train,
    'y_test_true': y_test,
    'y_test_predicted': y_pred_test
}

# Sauvegarder les prédictions sous forme de CSV
train_predictions = pd.DataFrame({
    'actual': y_train,
    'predicted': y_pred_train,
    'residual': y_train - y_pred_train,
    'dataset': 'train'
})

test_predictions = pd.DataFrame({
    'actual': y_test,
    'predicted': y_pred_test,
    'residual': y_test - y_pred_test,
    'dataset': 'test'
})

# Combiner train et test
all_predictions = pd.concat([train_predictions, test_predictions], ignore_index=True)

# Sauvegarder dans le dossier data
all_predictions.to_csv('data/predictions.csv', index=False)

print("✅ Prédictions sauvées dans data/predictions.csv")

print("\n📊 Sauvegarde des métriques...")

# Créer le dictionnaire des scores
scores = {
    "model_info": {
        "model_type": type(final_model).__name__,
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "features": X_train_scaled.shape[1]
    },
    "train_metrics": {
        "mse": float(train_mse),
        "rmse": float(train_rmse),
        "mae": float(train_mae),
        "r2": float(train_r2),
        "explained_variance": float(train_explained_var),
        "mean_target": float(train_mean),
        "std_target": float(train_std)
    },
    "test_metrics": {
        "mse": float(test_mse),
        "rmse": float(test_rmse),
        "mae": float(test_mae),
        "r2": float(test_r2),
        "explained_variance": float(test_explained_var),
        "mean_target": float(test_mean),
        "std_target": float(test_std)
    },
    "performance_summary": {
        "overfit_check": {
            "r2_difference": float(train_r2 - test_r2),
            "rmse_ratio": float(test_rmse / train_rmse),
            "is_overfitting": bool((train_r2 - test_r2) > 0.1)
        },
        "model_quality": {
            "excellent": bool(test_r2 > 0.8),
            "good": bool(0.6 < test_r2 <= 0.8),
            "fair": bool(0.4 < test_r2 <= 0.6),
            "poor": bool(test_r2 <= 0.4)
        }
    }
}

# Créer le dossier metrics si nécessaire
os.makedirs('metrics', exist_ok=True)

# Sauvegarder les scores en JSON
with open('metrics/scores.json', 'w') as f:
    json.dump(scores, f, indent=4)

print("✅ Métriques sauvées dans metrics/scores.json")

print("\n🎯 Résumé de l'évaluation :")
print("=" * 50)
print(f"Modèle : {type(final_model).__name__}")
print(f"R² Test : {test_r2:.4f}")
print(f"RMSE Test : {test_rmse:.4f}")
print(f"MAE Test : {test_mae:.4f}")
print(f"MSE Test : {test_mse:.4f}")

# Analyse du surapprentissage
overfitting = train_r2 - test_r2 > 0.1
print(f"\nSurapprentissage : {'🚨 Oui' if overfitting else '✅ Non'}")
print(f"Différence R² : {train_r2 - test_r2:.4f}")

# Qualité du modèle
if test_r2 > 0.8:
    quality = "🎉 Excellent"
elif test_r2 > 0.6:
    quality = "👍 Bon"
elif test_r2 > 0.4:
    quality = "⚠️ Moyen"
else:
    quality = "❌ Faible"

print(f"Qualité du modèle : {quality}")

print("\n📁 Fichiers générés :")
print("   - data/predictions.csv (prédictions)")
print("   - metrics/scores.json (métriques)")

print("\n✅ Évaluation terminée !")
