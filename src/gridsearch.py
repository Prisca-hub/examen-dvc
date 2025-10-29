#!/usr/bin/env python3
"""
Script de GridSearch pour trouver les meilleurs paramètres
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os
import numpy as np

print("🔄 Chargement des données normalisées...")

# Charger les données
X_train_scaled = pd.read_csv('data/processed_data/X_train_scaled.csv')
X_test_scaled = pd.read_csv('data/processed_data/X_test_scaled.csv')
y_train = pd.read_csv('data/processed_data/y_train.csv').values.ravel()
y_test = pd.read_csv('data/processed_data/y_test.csv').values.ravel()

print(f"X_train_scaled shape : {X_train_scaled.shape}")
print(f"y_train shape : {y_train.shape}")

# Créer le dossier models si nécessaire
os.makedirs('models', exist_ok=True)

print("\n🔍 GridSearch en cours...")

# Définir les modèles et leurs paramètres
models = {
    'RandomForest': {
        'model': RandomForestRegressor(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        }
    },
    'Ridge': {
        'model': Ridge(),
        'params': {
            'alpha': [0.1, 1.0, 10.0, 100.0],
            'solver': ['auto', 'svd', 'cholesky']
        }
    },
    'SVR': {
        'model': SVR(),
        'params': {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        }
    }
}

best_models = {}
results = {}

# Tester chaque modèle
for model_name, model_config in models.items():
    print(f"\n🤖 Test du modèle : {model_name}")
    
    # GridSearch
    grid_search = GridSearchCV(
        model_config['model'],
        model_config['params'],
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit
    grid_search.fit(X_train_scaled, y_train)
    
    # Meilleur modèle
    best_model = grid_search.best_estimator_
    
    # Prédictions
    y_pred_train = best_model.predict(X_train_scaled)
    y_pred_test = best_model.predict(X_test_scaled)
    
    # Métriques
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    # Sauvegarder les résultats
    results[model_name] = {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2
    }
    
    best_models[model_name] = best_model
    
    print(f"✅ Meilleurs paramètres : {grid_search.best_params_}")
    print(f"📊 Score CV : {grid_search.best_score_:.4f}")
    print(f"📈 RMSE Test : {test_rmse:.4f}")
    print(f"📈 R² Test : {test_r2:.4f}")

# Trouver le meilleur modèle global
best_model_name = max(results.keys(), key=lambda x: results[x]['test_r2'])
print(f"\n🏆 Meilleur modèle : {best_model_name}")
print(f"📊 R² Test : {results[best_model_name]['test_r2']:.4f}")

# Sauvegarder le meilleur modèle et ses paramètres
print("\n💾 Sauvegarde des résultats...")

# Sauvegarder le meilleur modèle
with open('models/best_model.pkl', 'wb') as f:
    pickle.dump(best_models[best_model_name], f)

# Sauvegarder les meilleurs paramètres
with open('models/best_params.pkl', 'wb') as f:
    pickle.dump(results[best_model_name]['best_params'], f)

# Sauvegarder tous les résultats
with open('models/gridsearch_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("✅ GridSearch terminé !")
print("📁 Fichiers sauvés :")
print("   - models/best_model.pkl")
print("   - models/best_params.pkl")
print("   - models/gridsearch_results.pkl")

# Afficher un résumé
print(f"\n📋 Résumé des résultats :")
for model, result in results.items():
    print(f"{model}: R²={result['test_r2']:.4f}, RMSE={result['test_rmse']:.4f}")
