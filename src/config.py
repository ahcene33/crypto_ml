# src/config.py
"""
Chargement du fichier de configuration YAML (optionnel).

Si *config.yaml* n’est pas présent, on utilise un jeu de valeurs par défaut
qui permettent à tout le pipeline (collecte, target, entraînement,
simulation…) de fonctionner sans lever d’exception.

Les clés attendues par le code :
    - coins.top_n               (ex. 100)
    - coins.days_history       (ex. 365)
    - training.horizon         (ex. 1)
    - training.threshold_pct  (ex. 0.01)
    - training.threshold      (ex. 0.5)   # décision BUY/SELL
    - training.optuna_trials  (ex. 10)
    - risk.var_confidence      (ex. 0.99)
    - risk.es_confidence       (ex. 0.975)
"""

import yaml
from pathlib import Path
import logging

log = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Chemin du fichier de configuration (situé à la racine du projet)
# ----------------------------------------------------------------------
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"

# ----------------------------------------------------------------------
# Valeurs par défaut – ces paramètres ont été choisis pour être
# raisonnables sur un jeu de données de taille moyenne.
# ----------------------------------------------------------------------
_DEFAULT_CFG = {
    "coins": {
        "top_n": 100,            # nombre maximal de cryptos à récupérer
        "days_history": 365,     # nombre de jours d’historique OHLCV
    },
    "training": {
        "horizon": 1,            # horizon (en jours) pour la cible
        "threshold_pct": 0.01,   # seuil de hausse pour créer la cible BUY
        "threshold": 0.5,       # seuil de probabilité pour le signal (BUY/SELL)
        "optuna_trials": 10,    # nombre de trials Optuna (petit pour les tests)
    },
    "risk": {
        "var_confidence": 0.99,   # VaR à 99 %
        "es_confidence": 0.975,   # Expected Shortfall à 97.5 %
    },
}

# ----------------------------------------------------------------------
# Chargement du fichier – s’il existe on le fusionne avec les defaults
# ----------------------------------------------------------------------
if CONFIG_PATH.is_file():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
    except Exception as exc:                     # pragma: no‑cover
        log.error(f"Erreur lors du parsing de {CONFIG_PATH}: {exc}")
        user_cfg = {}
    # Deep‑merge : on garde les valeurs du fichier lorsqu’elles existent,
    # sinon on retombe sur les valeurs par défaut.
    def _deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                _deep_update(d[k], v)
            else:
                d[k] = v
    cfg = _DEFAULT_CFG.copy()
    _deep_update(cfg, user_cfg)
    log.info(f"Configuration chargée depuis {CONFIG_PATH}")
else:  # Aucun fichier → on utilise uniquement les defaults
    cfg = _DEFAULT_CFG
    log.info("config.yaml introuvable – utilisation des paramètres par défaut.")

# ----------------------------------------------------------------------
# Helper – accès simple (conserve l’API précédente)
# ----------------------------------------------------------------------
def get(section: str, key: str, default=None):
    """Renvoie cfg[section][key] ou la valeur par défaut."""
    return cfg.get(section, {}).get(key, default)
