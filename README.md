# API de Prédiction USD/CAD avec LSTM & FinBERT  
# Analyse hybride des séries temporelles et du sentiment financier

## Description du projet

Ce projet propose une API capable de prédire les **tendances du taux de change USD/CAD** en combinant :

- un modèle **LSTM** entraîné sur les données historiques de Yahoo Finance  
- une analyse du **sentiment financier** à partir des actualités économiques grâce à **FinBERT**  
- un pipeline de **prétraitement Sklearn**  
- une API REST développée avec **FastAPI**

L’objectif est de fournir une prédiction plus robuste en intégrant à la fois les signaux quantitatifs (prix) et qualitatifs (news).

## Fonctionnalités principales

Téléchargement automatique des données USD/CAD depuis Yahoo Finance  
Analyse du sentiment des actualités financières via **FinBERT**  
Prétraitement des données avec **Scikit‑learn**  
Modèle LSTM pour la prédiction du taux de change  
Fusion des signaux prix + sentiment  
API REST (FastAPI) pour exposer les prédictions  
Visualisation et évaluation du modèle (MSE, RMSE, etc.)

# Données utilisées

# **1. Yahoo Finance**
- Taux USD/CAD (OHLCV)
- Historique configurable (1 jour, 1 heure, 5 minutes)
- Données utilisées pour entraîner le LSTM

# **2. Actualités financières**
- Titres et résumés d’articles économiques
- Analyse du sentiment via **FinBERT**
- Extraction des probabilités :  
  - négatif  
  - neutre  
  - positif  

Ces probabilités sont intégrées comme features supplémentaires dans le modèle.

## Modèle de prédiction

Le modèle repose sur une architecture hybride :

### Flux 1 — Séries temporelles (LSTM)
- Normalisation MinMax  
- Fenêtres glissantes  
- LSTM + Dense  

### Flux 2 — Sentiment (FinBERT)
- Tokenisation  
- Embeddings BERT  
- Probabilités de sentiment  

### Fusion
Les deux flux sont concaténés pour produire une prédiction finale du taux USD/CAD.

---

## API REST (FastAPI)

L’API expose plusieurs endpoints :

| Méthode | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Vérifie l’état de l’API |
| POST | `/predict` | Retourne une prédiction USD/CAD basée sur les prix + news |
| GET | `/history` | Retourne les données utilisées |

L’API est conçue pour être intégrée dans des systèmes de trading, dashboards ou applications financières.

---

## Installation

1. Cloner le projet  
2. Installer les dépendances  
3. Configurer les clés API (si nécessaire pour les news)  
4. Lancer l’API FastAPI  

---

## Entraînement du modèle

Le projet inclut un script d’entraînement permettant de :

- charger les données  
- prétraiter les séries temporelles  
- analyser les news avec FinBERT  
- entraîner le modèle LSTM  
- sauvegarder le modèle et le scaler  

---

## Exemple de sortie de l’API

```json
{
  "prediction": 1.3572,
  "sentiment": "positive",
  "confidence": 0.82
}
```
- un **résumé pour un rapport académique**  
- une **présentation PowerPoint**  
- ou un **poster scientifique** pour ton projet.
