# src/prediction_utils.py
import numpy as np

def probability_to_category(proba: float) -> str:
    """
    Convertit la probabilité d’achat (0‑1) en catégorie de variation prévue sur 7 j.
    Mapping (modifiable) :
        p >= 0.90 → "+10 % (forte hausse)"
        0.70 ≤ p < 0.90 → "+5 % (hausse)"
        0.30 ≤ p < 0.70 → "0 % (stable)"
        0.10 ≤ p < 0.30 → "-5 % (baisse)"
        p < 0.10 → "-10 % (forte baisse)"
    """
    if proba >= 0.90:
        return "+10 % (forte hausse)"
    if 0.70 <= proba < 0.90:
        return "+5 % (hausse)"
    if 0.30 <= proba < 0.70:
        return "0 % (stable)"
    if 0.10 <= proba < 0.30:
        return "-5 % (baisse)"
    return "-10 % (forte baisse)"
