# Oil Trading Bot

Système de **backtesting algorithmique** sur le pétrole WTI (ticker `CL=F`) : indicateurs techniques, stratégies de trading, gestion du risque, modèle de machine learning, analyse de sentiment sur des actualités, et tableau de bord interactif **Streamlit**.

> Projet académique : les résultats sont des simulations sur données historiques, pas des conseils d'investissement.

## Fonctionnalités

- **Indicateurs** : moyenne mobile, RSI, MACD, Bollinger, ATR
- **Stratégies** : croisement de moyennes mobiles, RSI (retour à la moyenne), stratégie combinée, stratégie ML, stratégie basée sur le sentiment
- **Moteur de backtesting** avec exécution simulée et gestion du risque à fraction fixe
- **Métriques de performance** calculées sur chaque simulation
- **Machine learning** : pipeline d'entraînement avec RandomForest et validation walk-forward
- **NLP** : score de sentiment des actualités avec TextBlob
- **Tableau de bord Streamlit** : Backtest, Comparaison de stratégies, ML Lab, NLP
- **Tests** : unitaires et d'intégration avec pytest

## Structure

```
src/oil_bot/
├── indicators/    # RSI, MACD, Bollinger, ATR, moyennes mobiles
├── strategies/    # stratégies de trading
├── backtesting/   # moteur de backtest
├── execution/     # exécution simulée
├── risk/          # gestion du risque
├── metrics/       # calcul des performances
├── features/ ml/  # features et pipeline de machine learning
├── nlp/           # analyse de sentiment
└── services/      # orchestration (backtest, ML, comparaison)
streamlit_app/     # interface (Home + 4 pages)
scripts/           # run_backtest.py, train_model.py
tests/             # unit/ et integration/
```

## Installation

Python 3.11 ou plus.

```bash
git clone https://github.com/hamzaballa05/oil-trading-bot
cd oil-trading-bot
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Utilisation

```bash
# Backtest en ligne de commande
python scripts/run_backtest.py

# Entraîner le modèle ML
python scripts/train_model.py

# Lancer le tableau de bord
streamlit run streamlit_app/Home.py

# Tests
pytest
```

## Limites connues

- L'analyse de sentiment utilise TextBlob, un outil généraliste (non spécialisé en finance).
- Les performances passées ne garantissent pas les performances futures.

## Auteur

**Hamza Balla** : étudiant ingénieur à l'INSEA (Data and Software Engineering).
