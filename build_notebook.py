"""Assemble the project notebook from ordered (kind, source) cells.

Keeping the notebook in a builder script (rather than editing raw .ipynb JSON) makes
the content reviewable as plain Python and trivial to regenerate. Run:

    python build_notebook.py           # writes notebook.ipynb (no outputs)
    jupyter nbconvert --execute ...     # to populate outputs (see run in README)
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


# ===========================================================================
md(r"""
# Credit Card Fraud Detection: A Critical Reproduction Study

**Course:** Data Science in Cyber, Dr. Uri Itai. **Submitter:** Gilad Segal 216214353.

## Aim
This notebook reproduces and critically evaluates a published tutorial. The goal is not to build a new
detector, but to check whether the tutorial's claims hold once the analysis is done carefully.

- **Source reviewed:** V. Valkov, *"Credit Card Fraud Detection using Autoencoders in Keras"*
  ([blog](https://curiousily.com/posts/credit-card-fraud-detection-using-autoencoders-in-keras/),
  [GitHub](https://github.com/curiousily/Credit-Card-Fraud-Detection-using-Autoencoders-in-Keras), ~588 stars).
- **Dataset:** ULB credit-card fraud (Kaggle `mlg-ulb/creditcardfraud`), loaded from
  **OpenML id 42175** for reproducibility. 284,807 transactions, 492 frauds (0.17%), with features
  `Time`, `V1`..`V28` (PCA components), `Amount`, and `Class`.

## The claim under test
The source trains a semi-supervised autoencoder on legitimate transactions and flags transactions with a
high reconstruction error as fraud, reporting a ROC-AUC of about 0.95.

We argue two things. First, that figure is measured optimistically, because the source lets the test set
influence feature scaling, model selection, and the choice of threshold. Second, the approach is not
actually competitive: a plain Logistic Regression or Random Forest achieves a higher PR-AUC, which is the
metric that matters at 0.17% prevalence. The plan is to reproduce the original faithfully, show where the
leaks are, then rerun it correctly and compare against simple baselines.
""")

code(r"""
import sys, warnings
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pointbiserialr

from load_data import load_creditcard
from utils import (RANDOM_SEED, build_autoencoder, reconstruction_error,
                   compute_metrics, best_threshold_by_fbeta, metrics_frame)

np.random.seed(RANDOM_SEED)
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (11, 5)
ARTIFACTS = Path("artifacts")
FIGS = ARTIFACTS / "figs"; FIGS.mkdir(parents=True, exist_ok=True)  # report figures
def savefig(name): plt.savefig(FIGS / name, dpi=120, bbox_inches="tight")
BETA = 2.0  # F-beta with beta=2 weights recall higher: missing fraud costs more than a false alarm.
print("Setup complete.")
""")

# --------------------------------------------------------------------------- #
md(r"""
## 1. Data Loading & Inspection
""")

code(r"""
df = load_creditcard()               # downloads from OpenML on first run, then caches to data/
print("shape:", df.shape)
df.head()
""")

code(r"""
# Column types, memory, and a first look at the target.
df.info()
print("\nMissing values total:", int(df.isnull().sum().sum()))
print("Exact duplicate rows:", int(df.duplicated().sum()))
""")

code(r"""
# Column and index check. The columns are Time, 28 anonymised PCA components (V1..V28), Amount and Class.
# The index is a default RangeIndex with no meaningful key, which is appropriate for transaction records.
print("Columns:", list(df.columns))
print("Index:", df.index.name, "| is RangeIndex:", isinstance(df.index, pd.RangeIndex))
print("Fraud count:", int(df.Class.sum()), "| prevalence: {:.4%}".format(df.Class.mean()))
df[["Time", "Amount"]].describe()
""")

code(r"""
# Data-quality checks: constant (single-value) columns, duplicate columns, and duplicate rows.
constant_cols = [c for c in df.columns if df[c].nunique() == 1]
dup_cols = [c for c in df.columns if any(df[c].equals(df[o]) for o in df.columns[:list(df.columns).index(c)])]
n_dup_rows = int(df.duplicated().sum())
print("Constant (single-value) columns:", constant_cols or "none")
print("Duplicate columns:", dup_cols or "none")
print(f"Exact duplicate rows: {n_dup_rows} ({n_dup_rows / len(df):.2%} of the data)")
""")

md(r"""
There are no constant columns and no duplicate columns, so no feature is dropped on those grounds. There is
a small number of exact duplicate rows. The source keeps them, so the faithful reproduction in Section 3
keeps them as well, but we remove them before the corrected split in Section 4 to prevent identical
transactions from appearing in both the training and test sets.
""")

md(r"""
**Temporal feature.** `Time` records the seconds elapsed since the first transaction and covers roughly two
days. It is not a wall-clock timestamp, so we derive an hour-of-day value for the analysis below. It is the
only genuinely temporal column, since the `V*` features are static PCA projections.
""")

code(r"""
df["Hour"] = (df["Time"] // 3600) % 24          # engineered temporal feature (0..23)
print("Time span: {:.1f} hours ({:.1f} days)".format(df.Time.max()/3600, df.Time.max()/86400))
df["Hour"].describe()
""")

# --------------------------------------------------------------------------- #
md(r"""
## 2. Exploratory Data Analysis (EDA)
""")

code(r"""
# 2.1 Class imbalance / prevalence — the defining property of this problem.
counts = df["Class"].value_counts().sort_index()
ax = counts.plot(kind="bar", rot=0, color=["#4c72b0", "#c44e52"])
ax.set_xticklabels(["Legitimate (0)", "Fraud (1)"]); ax.set_ylabel("count"); ax.set_yscale("log")
ax.set_title("Class distribution (log scale): fraud is 0.17% of all transactions")
for i, v in enumerate(counts):
    ax.text(i, v, f"{v:,}", ha="center", va="bottom")
savefig("class_imbalance.png"); plt.show()
print("A trivial 'always legitimate' classifier already scores {:.3%} accuracy.".format(1 - df.Class.mean()))
""")

md(r"""
**Real-world meaning.** With 0.17% positives, accuracy is not a useful metric: predicting "never fraud"
already scores 99.83%. This is a genuine class-imbalance problem rather than a sampling artefact, because
fraud is rare in practice and the training prevalence is realistic. The source does not resample the data,
so our critique concerns how performance is measured rather than how imbalance is handled. We therefore
judge models with PR-AUC, MCC and F2.
""")

code(r"""
# 2.2 Amount distribution by class.
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
for ax, cls, title in [(a1, 0, "Legitimate"), (a2, 1, "Fraud")]:
    ax.hist(df.loc[df.Class == cls, "Amount"], bins=50, color="#4c72b0" if cls == 0 else "#c44e52")
    ax.set_title(f"{title}: Amount"); ax.set_xlabel("Amount ($)"); ax.set_yscale("log")
plt.tight_layout(); plt.show()
df.groupby("Class")["Amount"].describe()[["mean", "50%", "max"]]
""")

code(r"""
# 2.3 Outlier analysis for Amount (IQR rule), reported separately per class.
q1, q3 = df["Amount"].quantile([0.25, 0.75]); iqr = q3 - q1
upper = q3 + 1.5 * iqr
out_rate = df.assign(is_outlier=df["Amount"] > upper).groupby("Class")["is_outlier"].mean()
print("Amount upper IQR fence: ${:.2f}".format(upper))
print("Share of high-amount outliers by class:\n", (out_rate * 100).round(2))
print("Amount skewness: {:.1f} (before), {:.2f} (after log1p)".format(
      df["Amount"].skew(), np.log1p(df["Amount"]).skew()))
""")

md(r"""
`Amount` is strongly right-skewed with many high-value outliers under the IQR rule, which motivates the
robust scaling used later rather than removing these rows (a large transaction is not necessarily an error).
""")

code(r"""
# 2.4 Temporal pattern: fraud rate per hour-of-day.
rate = df.groupby("Hour")["Class"].mean() * 100
ax = rate.plot(marker="o", color="#c44e52")
ax.set_ylabel("fraud rate (%)"); ax.set_xlabel("hour of day (since first tx)")
ax.set_title("Fraud rate by hour of day")
plt.show()
""")

md(r"""
The fraud rate is higher during the low-volume night-time hours. This matches the expectation that
fraudulent activity concentrates when monitoring and legitimate traffic are low, and it is a signal the
source removes when it drops `Time`.
""")

code(r"""
# 2.5 Correlation analysis. The V* features are PCA components, so they are linearly uncorrelated with
# each other by construction. We verify this as a redundancy check, then measure the association of each
# feature with the binary target using point-biserial and Spearman coefficients.
corr = df[[f"V{i}" for i in range(1, 29)]].corr()  # Pearson among PCA components
off_diag = corr.where(~np.eye(len(corr), dtype=bool))
print("Max |off-diagonal| correlation among V1..V28: {:.4f}".format(off_diag.abs().max().max()))
plt.figure(figsize=(8, 6)); sns.heatmap(corr, cmap="coolwarm", center=0, cbar_kws={"label": "Pearson r"})
plt.title("Correlation among PCA components (near-diagonal, so no linear redundancy)")
savefig("corr_heatmap.png"); plt.show()
""")

code(r"""
# Association of each feature with the target: point-biserial (≈ Pearson for a binary target)
# vs Spearman (monotonic, robust to the heavy tails we saw in Amount).
feats = [f"V{i}" for i in range(1, 29)] + ["Amount", "Hour"]
assoc = pd.DataFrame({
    "point_biserial": [pointbiserialr(df[f], df.Class).statistic for f in feats],
    "spearman": [df[f].corr(df.Class, method="spearman") for f in feats],
}, index=feats)
assoc["abs_pb"] = assoc.point_biserial.abs()
top = assoc.sort_values("abs_pb", ascending=False).head(10)
print("Top features by |point-biserial| correlation with fraud:")
top[["point_biserial", "spearman"]].round(3)
""")

md(r"""
**Why these measures.** The target is binary, so Pearson reduces to the point-biserial coefficient, which
measures the linear association of each continuous feature with the 0/1 label. We report Spearman alongside
it because `Amount` is heavy-tailed with extreme outliers, and a monotonic rank correlation is more robust
to those few very large transactions. Kendall would give the same ordering but is more expensive to compute
on 285k rows and adds little here. The strongest associations are with `V14`, `V4`, `V12`, `V10` and `V17`.
At this sample size almost any association is statistically significant, so we rely on the effect size (the
magnitude of the coefficient) and, in the end, on model PR-AUC to judge practical significance.
""")

# --------------------------------------------------------------------------- #
md(r"""
## 3. Reproduction of the Original Method (flaws preserved)

We first reproduce the source as written, changing only the 2017 Keras-1/TF-1 code so that it runs on
TensorFlow 2. Three test-set leaks are kept in place deliberately, so that we measure what the author
actually reported:

1. The `StandardScaler` for `Amount` is fitted on the whole dataset before the split.
2. The test set is passed as `validation_data`, and `save_best_only` keeps the checkpoint with the lowest
   test loss, so model selection uses the test set.
3. The decision threshold of 2.9 is read off plots of the test-set reconstruction errors.

If the full 100-epoch run from `src/reproduce_original.py` has been cached under `artifacts/`, we load those
results; otherwise a shorter faithful version is trained inline.
""")

code(r"""
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
tf.random.set_seed(RANDOM_SEED)

res_path = ARTIFACTS / "original_results.json"
if res_path.exists():
    original = json.loads(res_path.read_text())
    err = pd.read_csv(ARTIFACTS / "original_test_errors.csv")
    orig_scores, orig_true = err.reconstruction_error.values, err.true_class.values
    print("Loaded cached 100-epoch reproduction:", {k: original[k] for k in ("epochs","roc_auc","pr_auc")})
else:
    data = df.drop(["Time", "Hour"], axis=1).copy()
    data["Amount"] = StandardScaler().fit_transform(data[["Amount"]])       # LEAK #1
    Xtr, Xte = train_test_split(data, test_size=0.2, random_state=RANDOM_SEED)
    Xtr = Xtr[Xtr.Class == 0].drop("Class", axis=1).values
    orig_true = Xte.Class.values; Xte = Xte.drop("Class", axis=1).values
    ae = build_autoencoder(Xtr.shape[1]); ae.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
    ae.fit(Xtr, Xtr, epochs=30, batch_size=32, shuffle=True,
           validation_data=(Xte, Xte), verbose=2)                            # LEAK #2
    orig_scores = reconstruction_error(ae, Xte)
    from sklearn.metrics import roc_auc_score, average_precision_score
    original = {"epochs": 30, "roc_auc": round(roc_auc_score(orig_true, orig_scores), 4),
                "pr_auc": round(average_precision_score(orig_true, orig_scores), 4)}
    print("Trained inline (30 epochs):", original)
""")

code(r"""
# Apply the author's hand-picked threshold of 2.9 (LEAK #3) and show the confusion matrix.
THRESH_ORIG = 2.9
orig_pred = (orig_scores > THRESH_ORIG).astype(int)
orig_metrics = compute_metrics(orig_true, orig_pred, orig_scores, beta=BETA)
print("Reported ROC-AUC: {roc_auc:.4f}   |   PR-AUC: {pr_auc:.4f}".format(**orig_metrics))
print("At threshold 2.9 -> precision {precision:.3f}, recall {recall:.3f}, "
      "F2 {f2:.3f}, MCC {mcc:.3f}".format(**orig_metrics))

from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay(np.array([[orig_metrics["TN"], orig_metrics["FP"]],
                                 [orig_metrics["FN"], orig_metrics["TP"]]]),
                       display_labels=["Legit", "Fraud"]).plot(cmap="Blues", values_format="d")
plt.title("Original autoencoder @ threshold 2.9"); savefig("original_confusion.png"); plt.show()
""")

md(r"""
**Reading the reproduction.** The ROC-AUC is high, around 0.95, which matches the blog. The headline hides
two things. First, the PR-AUC is much lower. Second, at the 2.9 threshold the model raises hundreds of false
alarms for each fraud it catches, so its precision is only a few percent. At 0.17% prevalence the ROC-AUC is
dominated by the many easily classified negatives, so it looks strong even though the operating point is
poor. We now remove the leaks and give simple baselines a fair comparison.
""")

# --------------------------------------------------------------------------- #
md(r"""
## 4. Feature Engineering (corrected pipeline)

For the fair rerun we correct the preprocessing:

- We split first into train, validation and test, stratified on `Class`. The test set is not touched until
  the final evaluation. The validation set is used for early stopping and threshold selection.
- `Amount` is heavy-tailed with extreme outliers, so we scale it with a `RobustScaler` (median and IQR)
  fitted on the training split only. The skew before and after is shown below.
- We keep the engineered `Hour` feature, scaled, for the supervised models.
- **Encoding:** all predictors are numeric (the PCA components, the amount, and the derived hour), so there
  are no categorical variables to encode.
- **Feature selection and dimensionality reduction:** the `V*` PCA components are already on a comparable
  scale and mutually uncorrelated (Section 2.5), so we keep all of them. There is no redundancy to prune and
  no need for a further reduction such as PCA, since the data is already a reduced representation. The tree
  models additionally perform implicit feature selection through their split choices.
- We remove the exact duplicate rows found in Section 1 before splitting, so that identical transactions do
  not leak across the training and test sets.
""")

code(r"""
from sklearn.preprocessing import RobustScaler

feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount", "Hour"]
# Remove exact duplicate rows before splitting so the same transaction cannot fall in both train and test.
model_df = df.drop_duplicates().reset_index(drop=True)
print("Dropped {} duplicate rows; {} remain.".format(len(df) - len(model_df), len(model_df)))
X = model_df[feature_cols].copy(); y = model_df["Class"].values

X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED)
X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.25, stratify=y_tmp,
                                                  random_state=RANDOM_SEED)  # 0.25*0.8 = 0.2 val

scaler = RobustScaler().fit(X_train[["Amount", "Hour"]])                     # fit on TRAIN ONLY
for part in (X_train, X_val, X_test):
    part[["Amount", "Hour"]] = scaler.transform(part[["Amount", "Hour"]])

print(f"train {X_train.shape}  val {X_val.shape}  test {X_test.shape}")
print("fraud per split -> train {}, val {}, test {}".format(y_train.sum(), y_val.sum(), y_test.sum()))
""")

code(r"""
# Skew of Amount before vs after RobustScaler (log1p only for the 'before' view).
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
a1.hist(np.log1p(df["Amount"]), bins=60, color="#4c72b0"); a1.set_title("log1p(Amount): heavy tail")
a2.hist(X_train["Amount"], bins=60, color="#55a868"); a2.set_title("Amount after RobustScaler (train)")
plt.tight_layout(); plt.show()
print("Amount raw skew: {:.2f}".format(df["Amount"].skew()))
""")

# --------------------------------------------------------------------------- #
md(r"""
## 5. Model Training: corrected autoencoder and baselines

We train four models. For each one the decision threshold is chosen to maximise F2 on the validation set,
not on the test set. Threshold-independent quality is compared with PR-AUC.
""")

code(r"""
# 5.1 Corrected autoencoder: train on legitimate TRAIN rows, early-stop on legitimate VAL rows,
#     choose threshold on the FULL validation set.
from tensorflow.keras.callbacks import EarlyStopping

Xtr_norm = X_train[y_train == 0].values
Xval_norm = X_val[y_val == 0].values
ae = build_autoencoder(X_train.shape[1])
ae.fit(Xtr_norm, Xtr_norm, epochs=60, batch_size=64, shuffle=True,
       validation_data=(Xval_norm, Xval_norm), verbose=0,
       callbacks=[EarlyStopping(patience=6, restore_best_weights=True)])

val_scores_ae = reconstruction_error(ae, X_val.values)
thr_ae, _ = best_threshold_by_fbeta(y_val, val_scores_ae, beta=BETA)
test_scores_ae = reconstruction_error(ae, X_test.values)
ae_metrics = compute_metrics(y_test, (test_scores_ae > thr_ae).astype(int), test_scores_ae, beta=BETA)
print("Corrected AE trained. Val-chosen threshold = {:.4f}".format(thr_ae))
ae_metrics
""")

code(r"""
# 5.2 Supervised baselines: Logistic Regression and Random Forest (both class_weight balanced).
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest

def eval_proba_model(model, name):
    model.fit(X_train.values, y_train)
    val_scores = model.predict_proba(X_val.values)[:, 1]
    thr, _ = best_threshold_by_fbeta(y_val, val_scores, beta=BETA)
    test_scores = model.predict_proba(X_test.values)[:, 1]
    return name, compute_metrics(y_test, (test_scores > thr).astype(int), test_scores, beta=BETA), test_scores

logreg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)
rf = RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",
                            n_jobs=-1, random_state=RANDOM_SEED)
_, lr_metrics, lr_scores = eval_proba_model(logreg, "LogReg")
_, rf_metrics, rf_scores = eval_proba_model(rf, "RandomForest")
print("Logistic Regression and Random Forest trained.")
""")

code(r"""
# 5.3 Isolation Forest: an unsupervised anomaly detector, the fair anomaly-detection baseline for the AE.
iso = IsolationForest(n_estimators=200, contamination=float(y_train.mean()),
                      n_jobs=-1, random_state=RANDOM_SEED).fit(X_train[y_train == 0].values)
iso_val = -iso.score_samples(X_val.values)      # higher = more anomalous
iso_test = -iso.score_samples(X_test.values)
thr_iso, _ = best_threshold_by_fbeta(y_val, iso_val, beta=BETA)
iso_metrics = compute_metrics(y_test, (iso_test > thr_iso).astype(int), iso_test, beta=BETA)
print("Isolation Forest trained.")
iso_metrics
""")

# --------------------------------------------------------------------------- #
md(r"""
## 6. Evaluation and Comparison

**Metrics and their meaning for fraud.** A false positive is a legitimate transaction that gets blocked,
which causes customer friction and review cost. A false negative is a fraud that is missed, which causes a
direct financial loss. A missed fraud is generally more costly than a false alarm, so we favour recall
through F2 and select thresholds accordingly.

- **Precision** = TP / (TP + FP): the share of flagged transactions that are truly fraud. Low precision means
  analysts spend time on false alarms.
- **Recall (TPR)** = TP / (TP + FN): the share of frauds that are caught. Low recall means fraud gets through.
- **F1** = 2PR / (P + R): the harmonic mean of precision and recall, weighting them equally.
- **F2** = 5PR / (4P + R): the F-beta score with beta = 2, weighting recall four times as much as precision.
- **MCC** = (TP·TN − FP·FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)): a balanced correlation between predictions
  and labels that uses all four confusion-matrix cells and behaves well under imbalance.
- **ROC-AUC**: ranking quality across all thresholds. It is optimistic here because the many true negatives
  dominate it, so we report it mainly to show why it is misleading.
- **PR-AUC**: the area under the precision-recall curve. This is our primary metric, because it ignores the
  abundant true negatives and focuses on the rare fraud class.
""")

code(r"""
table = metrics_frame({
    "Original AE (@2.9, leaked)": orig_metrics,
    "Corrected AE (fair)": ae_metrics,
    "Logistic Regression": lr_metrics,
    "Random Forest": rf_metrics,
    "Isolation Forest": iso_metrics,
})
cols = ["pr_auc", "roc_auc", "mcc", "f2", "precision", "recall", "TP", "FP", "FN", "TN"]
table[cols].sort_values("pr_auc", ascending=False)
""")

code(r"""
# Precision-Recall curves: the informative view under extreme imbalance.
from sklearn.metrics import precision_recall_curve, average_precision_score
plt.figure(figsize=(8, 6))
for name, sc in [("Corrected AE", test_scores_ae), ("LogReg", lr_scores),
                 ("RandomForest", rf_scores), ("IsolationForest", iso_test)]:
    p, r, _ = precision_recall_curve(y_test, sc)
    plt.plot(r, p, label=f"{name} (AP={average_precision_score(y_test, sc):.3f})")
plt.axhline(y_test.mean(), ls="--", c="grey", label=f"baseline={y_test.mean():.4f}")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall curves (test)")
plt.legend(); savefig("pr_curves.png"); plt.show()
""")

md(r"""
**Verdict on the claim.** The supervised baselines reach a substantially higher PR-AUC than the autoencoder,
at a fraction of the training cost. The autoencoder's headline ROC-AUC is genuine but misleading, and its
operating-point performance is the weakest of the models tried. The source's implicit recommendation, that an
autoencoder is a good way to detect this fraud, is not supported by the evidence.
""")

# --------------------------------------------------------------------------- #
md(r"""
## 7. Error Analysis
""")

code(r"""
# Threshold sensitivity for the best model (Random Forest): F2 across thresholds, val vs test.
from sklearn.metrics import fbeta_score
grid = np.linspace(0.01, 0.99, 99)
f2_val = [fbeta_score(y_val, (rf.predict_proba(X_val.values)[:, 1] > t).astype(int), beta=2, zero_division=0) for t in grid]
f2_test = [fbeta_score(y_test, (rf_scores > t).astype(int), beta=2, zero_division=0) for t in grid]
plt.plot(grid, f2_val, label="validation"); plt.plot(grid, f2_test, label="test", ls="--")
plt.xlabel("threshold"); plt.ylabel("F2"); plt.title("Random Forest: F2 vs threshold"); plt.legend()
savefig("f2_threshold.png"); plt.show()
""")

code(r"""
# Characterise the Random Forest's errors on the test set at its chosen threshold.
thr_rf, _ = best_threshold_by_fbeta(y_val, rf.predict_proba(X_val.values)[:, 1], beta=BETA)
rf_pred = (rf_scores > thr_rf).astype(int)
# X_test indexes into model_df (after de-duplication), so raw amounts are looked up there.
err_df = X_test.copy(); err_df["true"] = y_test; err_df["pred"] = rf_pred
err_df["Amount_raw"] = model_df.loc[X_test.index, "Amount"]
false_neg = err_df[(err_df.true == 1) & (err_df.pred == 0)]
false_pos = err_df[(err_df.true == 0) & (err_df.pred == 1)]
print(f"False negatives (missed fraud): {len(false_neg)}   |   False positives (blocked legit): {len(false_pos)}")
print("Median amount of missed fraud: ${:.2f}  vs  all fraud: ${:.2f}".format(
      false_neg.Amount_raw.median(), model_df.loc[model_df.Class == 1, "Amount"].median()))
false_neg[["Amount_raw", "V14", "V4", "V12", "V10"]].describe().round(2)
""")

md(r"""
**Cybersecurity reading of the errors.**

- False negatives, the missed frauds, are the expensive errors: money is lost and the attacker succeeds. The
  missed frauds tend to involve smaller amounts that look statistically ordinary, which is consistent with
  transactions kept deliberately low to avoid attention.
- False positives, the blocked legitimate customers, have a real but lower cost in the form of friction,
  support load and churn. Because we optimise F2, we accept more false positives in order to catch more fraud.
- The right operating point is a business decision. At 0.17% prevalence, even a model with 99.9% specificity
  produces many false alarms in absolute terms, so precision and recall should be tuned to the relative cost
  of a missed fraud against a blocked card rather than fixed at an arbitrary constant such as 2.9.
""")

# --------------------------------------------------------------------------- #
md(r"""
## 8. Conclusions

- **Reproducibility.** The work runs only after porting. The 2017 Keras-1/TF-1 code does not run on a current
  stack, and the data has to be fetched separately. We restored reproducibility with an OpenML loader and
  updated imports.
- **Were the claims supported?** The ROC-AUC of about 0.95 reproduces, but it is an optimistic figure that is
  partly the result of leakage, since the test set is used for scaling, model selection and the threshold. On
  PR-AUC, the appropriate metric for this problem, the autoencoder is beaten by a plain Logistic Regression
  and a Random Forest.
- **Recommendation.** We would not recommend the autoencoder for this problem. It is more complex, slower to
  train, and less effective than the supervised baselines whenever labels are available. It is reasonable only
  in the fully unlabelled setting, and even there an Isolation Forest is a stronger and cheaper first choice on
  this dataset.
- **Possible improvements.** Cost-sensitive thresholding tied to the money lost, calibrated probabilities,
  time-ordered validation instead of a random split, and use of the `Time` and `Hour` signal that the source
  discards.
""")

code(r"""
# Persist all headline numbers + the comparison table for the PDF report (no retraining needed).
report_data = {
    "prevalence": float(df.Class.mean()),
    "n_rows": int(len(df)), "n_fraud": int(df.Class.sum()),
    "original": {k: original[k] for k in ("epochs", "roc_auc", "pr_auc")},
    "thresholds": {"original": THRESH_ORIG, "corrected_ae": float(thr_ae),
                   "random_forest": float(thr_rf)},
    "metrics": {"Original AE (@2.9, leaked)": orig_metrics, "Corrected AE (fair)": ae_metrics,
                "Logistic Regression": lr_metrics, "Random Forest": rf_metrics,
                "Isolation Forest": iso_metrics},
    "max_offdiag_corr": float(off_diag.abs().max().max()),
    "error_analysis": {"false_neg": int(len(false_neg)), "false_pos": int(len(false_pos)),
                       "missed_fraud_median_amount": float(false_neg.Amount_raw.median()),
                       "all_fraud_median_amount": float(df.loc[df.Class == 1, "Amount"].median())},
}
Path("artifacts/report_data.json").write_text(json.dumps(report_data, indent=2, default=float))
table[cols].to_csv("artifacts/metrics_table.csv")
print("Saved artifacts/report_data.json and metrics_table.csv")
""")

# ===========================================================================
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
nbf.write(nb, "notebook.ipynb")
print(f"Wrote notebook.ipynb with {len(cells)} cells.")
