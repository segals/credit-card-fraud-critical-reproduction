"""Faithful reproduction of the original curiousily autoencoder, ported to TF2/Keras3.

The ONLY changes vs. the 2017 original are the minimal edits required to run on a
modern stack (import paths, checkpoint format, non-deprecated pandas call). The
modelling choices are kept identical *on purpose* so we can measure the numbers the
author actually reported — including the methodological flaws (scaler fit on all
data, test set used as validation, threshold eyeballed at 2.9). Those flaws are then
dissected in the report; here we only reproduce.

Run:  python src/reproduce_original.py         (full 100 epochs)
      EPOCHS=2 python src/reproduce_original.py (fast smoke test)
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (auc, confusion_matrix, precision_recall_curve,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

ROOT = Path(__file__).resolve().parent.parent
RANDOM_SEED = 42
EPOCHS = int(os.environ.get("EPOCHS", "100"))
BATCH_SIZE = 32

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def build_autoencoder(input_dim: int, encoding_dim: int = 14) -> Model:
    """Recreate the exact 4-layer architecture from the original blog post."""
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(encoding_dim, activation="tanh",
                    activity_regularizer=regularizers.l1(10e-5))(input_layer)
    encoder = Dense(encoding_dim // 2, activation="relu")(encoder)
    decoder = Dense(encoding_dim // 2, activation="tanh")(encoder)
    decoder = Dense(input_dim, activation="relu")(decoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    # metrics=['accuracy'] is kept verbatim from the original even though it is
    # meaningless for an MSE autoencoder — this is one of the points we critique.
    autoencoder.compile(optimizer="adam", loss="mean_squared_error",
                        metrics=["accuracy"])
    return autoencoder


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "creditcard.csv")

    # --- original preprocessing (flaws preserved) ---
    data = df.drop(["Time"], axis=1)
    # LEAK #1: scaler is fit on the entire dataset before the train/test split.
    data["Amount"] = StandardScaler().fit_transform(data["Amount"].values.reshape(-1, 1))

    X_train, X_test = train_test_split(data, test_size=0.2, random_state=RANDOM_SEED)
    X_train = X_train[X_train.Class == 0].drop(["Class"], axis=1)
    y_test = X_test["Class"]
    X_test = X_test.drop(["Class"], axis=1)
    X_train, X_test = X_train.values, X_test.values

    model = build_autoencoder(X_train.shape[1])
    ckpt = ROOT / "artifacts" / "original_model.keras"
    ckpt.parent.mkdir(exist_ok=True)
    # LEAK #2: the test set is passed as validation_data and save_best_only picks
    # the checkpoint with the lowest *test* loss.
    checkpointer = ModelCheckpoint(filepath=str(ckpt), verbose=0, save_best_only=True)
    model.fit(X_train, X_train, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True,
              validation_data=(X_test, X_test), verbose=2, callbacks=[checkpointer])

    predictions = model.predict(X_test, verbose=0)
    mse = np.mean(np.power(X_test - predictions, 2), axis=1)

    roc_auc = roc_auc_score(y_test, mse)
    precision, recall, _ = precision_recall_curve(y_test, mse)
    pr_auc = auc(recall, precision)

    threshold = 2.9  # LEAK #3: hand-picked from the test-set error plots in the blog.
    y_pred = (mse > threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    results = {
        "epochs": EPOCHS,
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "threshold": threshold,
        "confusion_matrix": cm.tolist(),  # [[TN, FP], [FN, TP]]
        "n_test": int(len(y_test)),
        "n_fraud_test": int(y_test.sum()),
    }
    (ROOT / "artifacts" / "original_results.json").write_text(json.dumps(results, indent=2))
    # Cache per-row errors so the notebook can reuse them without retraining.
    pd.DataFrame({"reconstruction_error": mse, "true_class": y_test.values}).to_csv(
        ROOT / "artifacts" / "original_test_errors.csv", index=False)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
