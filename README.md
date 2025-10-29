# Projet Machine Learning - Concentration de Silice

Ce projet implémente un pipeline complet de machine learning pour prédire la concentration de silice en utilisant DVC (Data Version Control) et DagsHub.

## 📊 Vue d'ensemble

- **Objectif** : Prédire la variable `silica_concentrate` à partir de données de processus industriel
- **Dataset** : 1817 échantillons, 9 features numériques
- **Pipeline** : Préprocessing → GridSearch → Entraînement → Évaluation

## 🏗️ Structure du projet

├── data/
│ ├── raw_data/ # Données brutes
│ └── processed_data/ # Données traitées
├── src/
│ ├── split_data.py # Division train/test
│ ├── normalize_data.py # Normalisation des features
│ ├── gridsearch.py # Optimisation hyperparamètres
│ └── evaluate_model.py # Évaluation du modèle
├── models/
│ └── train_model.py # Entraînement final
├── metrics/ # Métriques d'évaluation
├── dvc.yaml # Pipeline DVC
├── dvc.lock # Verrouillage des versions
└── requirements.txt # Dépendances Python

## 🚀 Installation et utilisation

### Prérequis
```bash
pip install -r requirements.txt
