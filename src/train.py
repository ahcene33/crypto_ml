# src/train.py
# module d'entraînement du modèle LightGBM avec Optuna
# on utilise le principe du maximum de vraisemblance (log‑likelihood) et les conditions KKT du problème d'optimisation.

import logging
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
from .config import cfg

log = logging.getLogger(__name__)

def _objective(trial, X, y):
    """
    fonction objectif d'Optuna : on minimise 1 - AUC (équivaut à maximiser la vraisemblance).
    on protège le calcul d'AUC contre les splits où y_valid ne contient qu'une classe.
    """
    param = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 200),
        # Limiter la profondeur pour éviter le sur‑apprentissage
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 200),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-5, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-5, 10.0, log=True),
        "seed": 42,
        "verbosity": -1,
    }

    tscv = TimeSeriesSplit(n_splits=3)
    aucs = []
    for train_idx, valid_idx in tscv.split(X):
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        dtrain = lgb.Dataset(X_train, label=y_train)
        dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)

        callbacks = [lgb.early_stopping(stopping_rounds=30, verbose=False)]

        model = lgb.train(param, dtrain, valid_sets=[dvalid], callbacks=callbacks)

        pred = model.predict(X_valid)

        if len(np.unique(y_valid)) < 2:
            auc = 0.5
        else:
            auc = roc_auc_score(y_valid, pred)

        if np.isnan(auc):
            auc = 0.5
        aucs.append(auc)

    return 1.0 - np.mean(aucs)


class SimpleModel:
    """fallback model qui prédit la probabilité moyenne de la cible."""
    def __init__(self, prob: float):
        self._prob = prob

    def predict(self, X):
        if hasattr(X, "shape"):
            n = X.shape[0]
        else:
            n = len(X)
        return np.full(n, self._prob)


def train_best_model(df: pd.DataFrame,
                     feature_cols: list,
                     target_col: str):
    """
    pipeline complet d'entraînement :
    - sépare X / y
    - lance Optuna (nombre d'essais lu dans config.yaml) SI le jeu a >10 lignes
    - sinon retourne un modèle trivial qui prédit la moyenne de la cible
    - entraîne le modèle final sur l'ensemble des données
    - retourne le modèle, les meilleurs hyper‑paramètres et l'objet Optuna study
    """
    # -----------------------------------------------------------------
    # protections initiales
    # -----------------------------------------------------------------
    if not feature_cols:
        raise ValueError("feature_cols est vide – impossible d'entraîner le modèle.")
    if target_col not in df.columns:
        raise ValueError(f"target column '{target_col}' absent du DataFrame.")

    X = df[feature_cols].fillna(0).values
    y = df[target_col].values

    # -----------------------------------------------------------------
    # jeu de données trop petit → fallback immédiat
    # -----------------------------------------------------------------
    if X.shape[0] < 10:   # seuil arbitraire, suffisant pour la CV
        prob = float(np.mean(y)) if y.size > 0 else 0.0
        log.warning(
            f"jeu de données très petit ({X.shape[0]} lignes) – "
            "on utilise SimpleModel (probabilité moyenne)."
        )
        return SimpleModel(prob), {}, None

    # -----------------------------------------------------------------
    # Optuna – recherche d'hyper‑paramètres
    # -----------------------------------------------------------------
    n_trials = cfg.get("training", {}).get("optuna_trials", 10)
    # Fixer la graine pour rendre les runs reproductibles
    sampler = optuna.samplers.RandomSampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(lambda trial: _objective(trial, X, y),
                   n_trials=n_trials,
                   show_progress_bar=False)

    if not study.trials:
        raise RuntimeError(
            "aucune trial valide n'a été complétée – vérifiez que vos cibles "
            "contiennent les deux classes."
        )

    best_params = study.best_trial.params
    best_params.update({
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "seed": 42,
    })

    # -----------------------------------------------------------------
    # entraînement final sur l’ensemble des données
    # -----------------------------------------------------------------
    dfull = lgb.Dataset(X, label=y)
    model = lgb.train(best_params, dfull, num_boost_round=500)

    pred_all = model.predict(X)
    auc_all = roc_auc_score(y, pred_all) if len(np.unique(y)) > 1 else np.nan
    acc_all = accuracy_score(y, (pred_all > 0.5).astype(int))
    cm = confusion_matrix(y, (pred_all > 0.5).astype(int))

    log.info(f"entraînement terminé – auc:{auc_all:.4f} acc:{acc_all:.4f}")
    log.info(f"confusion matrix:\n{cm}")

    return model, best_params, study
