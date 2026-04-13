# src/target.py
# création de la cible à partir des prix

import pandas as pd
from config import cfg

def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    df doit contenir la colonne 'price'.
    ajoute deux colonnes :
      - future_ret_{horizon}   : rendement réel sur l’horizon h
      - target_buy_{horizon}   : 1 si le rendement > seuil, sinon 0
    """
    df = df.copy()
    horizon = cfg.get("training", {}).get("horizon", 1)
    thresh = cfg.get("training", {}).get("threshold_pct", 0.01)

    df[f"future_ret_{horizon}"] = df["price"].pct_change(horizon).shift(-horizon)
    df[f"target_buy_{horizon}"] = (df[f"future_ret_{horizon}"] > thresh).astype(int)

    df.dropna(subset=[f"future_ret_{horizon}", f"target_buy_{horizon}"], inplace=True)
    return df
