```markdown
# 🚀 Crypto-ML Portfolio

Une application de **machine learning** pour l'analyse et la gestion de portefeuille crypto, avec stratégie de rééquilibrage mensuel et dashboard interactif.

## 📊 Fonctionnalités

- **🤖 Prédictions ML** : Modèle LightGBM entraîné pour prédire les signaux d'achat/vente
- **📈 Gestion de portefeuille** : Rééquilibrage mensuel automatique avec stratégie top-2 crypto
- **💰 Investissement récurrent** : Dépôts mensuels automatiques (le 10 de chaque mois)
- **📊 Dashboard interactif** : Visualisation des performances avec Streamlit
- **⚡ Données temps réel** : Prix live via l'API publique Binance
- **📉 Analyse de risque** : Calculs de VaR historique et Expected Shortfall

## 🛠️ Installation

```bash
# Cloner le repository
git clone https://github.com/ahcene33/crypto_ml.git
cd crypto_ml

# Créer l'environnement virtuel
python -m venv .venv

# Activer l'environnement
# Sur Linux/Mac:
source .venv/bin/activate
# Sur Windows:
.venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt

# Préparer les artefacts (modèle, features, etc.)
python -m src.prepare
```

## 🎯 Utilisation

```bash
# Lancer l'application Streamlit
streamlit run src/streamlit_app.py
```

L'application sera accessible sur `http://localhost:8501`

## 📁 Structure du projet

```
crypto_ml/
├── src/                    # Code source
│   ├── streamlit_app.py    # Application principale
│   ├── portfolio.py        # Gestion du portefeuille
│   ├── portfolio_manager.py # Core du gestionnaire
│   ├── binance_price.py    # API prix temps réel
│   ├── features.py         # Feature engineering
│   ├── train.py            # Entraînement modèle
│   └── prepare.py          # Pipeline de préparation
├── data/
│   ├── raw/               # Données brutes (ignorées par git)
│   └── processed/         # Features transformées
├── models/                # Modèles entraînés
└── requirements.txt       # Dépendances Python
```

## ⚙️ Configuration

Les paramètres sont configurables dans `config.yaml` :

```yaml
coins:
  top_n: 100              # Nombre de cryptos à suivre
  days_history: 365       # Jours d'historique

training:
  horizon: 1              # Horizon de prédiction (jours)
  threshold_pct: 0.01     # Seuil de hausse pour signal BUY
  threshold: 0.5          # Seuil de probabilité
  optuna_trials: 10       # Nombre d'essais d'optimisation

risk:
  var_confidence: 0.99    # Niveau de confiance VaR
  es_confidence: 0.975    # Niveau de confiance ES
```

## 📈 Stratégie d'investissement

- **Capital initial** : 200 USDT
- **Dépôt mensuel** : 50 USDT le 10 de chaque mois
- **Sélection** : Top 2 cryptos par score ML
- **Rééquilibrage** : Mensuel le jour 10
- **Frais** : 0.1% par transaction

## 🌐 Déploiement

### Streamlit Cloud

L'application est déployée sur :  
🔗 **https://crypto-ml-portfolio.streamlit.app**

### Déploiement manuel

1. Forkez le repository GitHub
2. Connectez votre compte Streamlit Cloud
3. Déployez avec les paramètres :
   - Repository: `votre-username/crypto_ml`
   - Main file: `src/streamlit_app.py`
   - Python version: 3.9+

## 📊 Métriques calculées

- **ROI** : Return on Investment
- **Ratio de Sharpe** : Performance ajustée du risque
- **Max Drawdown** : Pire perte sur la période
- **Volatilité** : Écart-type annualisé des rendements
- **VaR 95%** : Valeur à risque historique
- **ES 95%** : Expected Shortfall historique

## 🤝 Contribution

Les contributions sont les bienvenues ! Pour contribuer :

1. Forkez le projet
2. Créez une branche feature (`git checkout -b feature/AmazingFeature`)
3. Committez vos changements (`git commit -m 'Add AmazingFeature'`)
4. Pushez la branche (`git push origin feature/AmazingFeature`)
5. Ouvrez une Pull Request

## 📝 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## ⚠️ Avertissement

**Ce projet est à but éducatif uniquement.**  
Les performances passées ne préjugent pas des résultats futurs.  
L'investissement dans les cryptomonnaies comporte des risques de perte en capital.

## 📞 Contact

Ahcene - [@ahcene33](https://github.com/ahcene33)

Lien du projet : [https://github.com/ahcene33/crypto_ml](https://github.com/ahcene33/crypto_ml)
```

Ce README comprend :
1. Une description claire du projet
2. Les instructions d'installation et d'utilisation
3. La structure du projet
4. La stratégie d'investissement
5. Les métriques calculées
6. Les instructions de déploiement
7. Les informations de contribution
8. Un avertissement sur les risques

Le lien Streamlit Cloud est indiqué comme étant à remplacer par l'URL réelle une fois le déploiement effectué.