"""Shared helpers: autoencoder construction, reconstruction error, metric suite,
and threshold selection. Kept in one small module so the notebook stays readable
and preprocessing / modelling / evaluation logic is never duplicated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             fbeta_score, matthews_corrcoef,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)

RANDOM_SEED = 42


# --------------------------------------------------------------------------- #
# Autoencoder (same architecture as the source, reused for the fair rerun too) #
# --------------------------------------------------------------------------- #
def build_autoencoder(input_dim: int, encoding_dim: int = 14):
    """Recreate the source's 4-layer autoencoder (14 -> 7 -> 7 -> input_dim)."""
    from tensorflow.keras import regularizers
    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.models import Model

    inp = Input(shape=(input_dim,))
    x = Dense(encoding_dim, activation="tanh",
              activity_regularizer=regularizers.l1(1e-4))(inp)
    x = Dense(encoding_dim // 2, activation="relu")(x)
    x = Dense(encoding_dim // 2, activation="tanh")(x)
    out = Dense(input_dim, activation="relu")(x)
    model = Model(inp, out)
    # Loss only — no meaningless 'accuracy' metric here (unlike the original).
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def reconstruction_error(model, X: np.ndarray) -> np.ndarray:
    """Per-row mean squared reconstruction error used as the anomaly score."""
    recon = model.predict(X, verbose=0)
    return np.mean(np.square(X - recon), axis=1)


# --------------------------------------------------------------------------- #
# Evaluation                                                                   #
# --------------------------------------------------------------------------- #
def compute_metrics(y_true, y_pred, y_score, beta: float = 2.0) -> dict:
    """Return the metric suite appropriate for a highly imbalanced problem.

    Accuracy is deliberately omitted from decision-making (it is ~99.8% for the
    trivial 'always legitimate' classifier); PR-AUC / MCC / F-beta are reported
    instead because they reflect performance on the rare fraud class.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": fbeta_score(y_true, y_pred, beta=1, zero_division=0),
        f"f{beta:g}": fbeta_score(y_true, y_pred, beta=beta, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }


def best_threshold_by_fbeta(y_true, y_score, beta: float = 2.0):
    """Pick the score threshold that maximises F-beta on the GIVEN data.

    Call this on a validation split (never the test set) so threshold selection
    does not leak into the reported test performance — the exact mistake the
    original blog makes by eyeballing 2.9 on the test errors.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # precision/recall have one more element than thresholds; drop the last point.
    precision, recall = precision[:-1], recall[:-1]
    beta_sq = beta * beta
    denom = beta_sq * precision + recall
    fbeta = np.where(denom > 0, (1 + beta_sq) * precision * recall / denom, 0.0)
    best = int(np.argmax(fbeta))
    return float(thresholds[best]), float(fbeta[best])


def metrics_frame(rows: dict[str, dict]) -> pd.DataFrame:
    """Turn {model_name: metric_dict} into a tidy, rounded comparison table."""
    frame = pd.DataFrame(rows).T
    float_cols = [c for c in frame.columns if c not in {"TP", "FP", "FN", "TN"}]
    frame[float_cols] = frame[float_cols].astype(float).round(4)
    return frame
