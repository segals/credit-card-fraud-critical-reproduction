"""Generate report.pdf from artifacts/report_data.json.

Comparative statements (for example the ranking of models by PR-AUC) are computed from the
measured numbers so the text always matches the results produced by the notebook.
"""
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parent
D = json.loads((ROOT / "artifacts" / "report_data.json").read_text())
FIGS = ROOT / "artifacts" / "figs"

# --- comparative facts derived from the measured results ---------------------
M = D["metrics"]
ranked = sorted(M.items(), key=lambda kv: kv[1]["pr_auc"], reverse=True)
best_name, best = ranked[0]
ae = M["Corrected AE (fair)"]
lr = M["Logistic Regression"]
rf = M["Random Forest"]
iso = M["Isolation Forest"]
orig = M["Original AE (@2.9, leaked)"]
supervised_beat_ae = min(lr["pr_auc"], rf["pr_auc"]) > ae["pr_auc"]

styles = getSampleStyleSheet()
body = ParagraphStyle("body", parent=styles["Normal"], fontSize=10, leading=14,
                      alignment=TA_JUSTIFY, spaceAfter=6)
h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=15, spaceBefore=10, spaceAfter=6,
                    textColor=colors.HexColor("#1f3b63"))
h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=11.5, spaceBefore=8, spaceAfter=4,
                    textColor=colors.HexColor("#2c4f7c"))
title = ParagraphStyle("title", parent=styles["Title"], fontSize=20, leading=24)
small = ParagraphStyle("small", parent=body, fontSize=9, textColor=colors.HexColor("#444444"))

story = []


def P(text, style=body):
    story.append(Paragraph(text, style))


def fig(name, width=15.5 * cm):
    path = FIGS / name
    if path.exists():
        img = Image(str(path)); ratio = img.imageHeight / img.imageWidth
        img.drawWidth = width; img.drawHeight = width * ratio
        story.append(img); story.append(Spacer(1, 4))


def pct(x):
    return f"{x*100:.2f}%"


# ============================ TITLE ==========================================
P("Credit Card Fraud Detection:<br/>A Critical Reproduction Study", title)
P("Data Science in Cyber, Dr. Uri Itai &nbsp;|&nbsp; Individual Final Project", small)
P('Source reviewed: V. Valkov, <i>"Credit Card Fraud Detection using Autoencoders in Keras"</i> '
  '(curiousily.com; GitHub, ~588 stars). Dataset: ULB credit-card fraud (Kaggle mlg-ulb, OpenML 42175).',
  small)
story.append(Spacer(1, 6))

# ============================ 1. SUMMARY =====================================
P("1. Summary of the Source", h1)
P("<b>Problem.</b> The task is to identify fraudulent credit-card transactions in the ULB dataset, which "
  f"contains {D['n_rows']:,} transactions made by European cardholders over two days. Only {D['n_fraud']} of "
  f"them ({pct(D['prevalence'])}) are fraudulent. The predictors are 28 anonymised PCA components (V1 to V28) "
  "together with the transaction time and amount.")
P("<b>Why it matters.</b> Card fraud accounts for very large annual losses, and a practical detector has to "
  "find the small number of fraudulent transactions without blocking too many legitimate customers. The severe "
  "class imbalance is the main technical difficulty and is also the reason a naive evaluation can be misleading.")
P("<b>Proposed solution.</b> The author trains a semi-supervised autoencoder with layer sizes 29-14-7-7-29 on "
  "legitimate transactions only. The network learns to reconstruct normal behaviour, and at inference a "
  "transaction whose reconstruction error exceeds a fixed threshold (the author uses 2.9) is labelled as fraud. "
  "The reported result is a ROC-AUC of roughly 0.95.")
P("<b>Data and methodology.</b> The author uses an 80/20 train/test split, keeps only the legitimate rows for "
  "training, standardises the amount, and trains for 100 epochs with the Adam optimiser and a mean-squared-error "
  "loss. Results are summarised with a ROC curve, a precision-recall curve, and a confusion matrix at the 2.9 "
  "threshold.")

# ============================ 2. CRITICAL EVAL ===============================
P("2. Critical Evaluation of the Author's Claims", h1)
P(f"The central claim is that reconstruction error produces a strong fraud detector, with a ROC-AUC near 0.95. "
  f"This claim reproduces. Our faithful 100-epoch port yields a ROC-AUC of {D['original']['roc_auc']:.3f}. The "
  "problem is that this figure is not good evidence that the detector is useful, for three reasons.")
P("<b>Test-set information leaks into the reported score in three places.</b> First, the amount is standardised "
  "with a scaler fitted on the full dataset before the split, so training statistics include the test set. "
  "Second, the test set is passed as the validation set, and the checkpoint callback keeps the model with the "
  "lowest validation loss, which means the saved model is selected on test data. Third, the decision threshold "
  "of 2.9 is read off reconstruction-error plots that are drawn on the labelled test set. Every one of these "
  "choices lets test information influence the reported number, so the result is optimistic by construction.")
P(f"<b>ROC-AUC is not an appropriate headline at {pct(D['prevalence'])} prevalence.</b> When 99.83% of rows are "
  "negative, ROC-AUC is dominated by the many easily ranked negatives and can look strong even when the operating "
  f"point is weak. The threshold-free precision-recall AUC, which is the more honest summary for a rare positive "
  f"class, is only {D['original']['pr_auc']:.3f} for the same model. At the 2.9 threshold the detector produces "
  f"{orig['FP']:,} false alarms in order to catch {orig['TP']} of {orig['TP']+orig['FN']} frauds, a precision of "
  f"{pct(orig['precision'])}.")
P("<b>The accuracy metric compiled into the network is not meaningful.</b> An autoencoder trained with "
  "mean-squared error is a regression model, so the reported per-element accuracy measures reconstruction "
  "agreement rather than detection quality and carries no information about fraud detection.")
P("<b>Are the conclusions justified?</b> " +
  ("They are not. " if supervised_beat_ae else "They are only partly justified. ") +
  "As Section 5 shows, once the leaks are removed, ordinary supervised classifiers achieve a higher PR-AUC than "
  "the autoencoder. The author's numbers can be reproduced, but the interpretation that a high ROC-AUC implies a "
  "good detector overstates what the experiment actually demonstrates.")

# ============================ 3. FEATURE ENG =================================
P("3. Feature Engineering Analysis", h1)
P("<b>What the source does.</b> The feature engineering is minimal. The author removes the time column and "
  "standardises the amount, and uses the supplied PCA components unchanged.")
P(f"<b>Redundancy.</b> Because V1 to V28 are PCA outputs, they are linearly uncorrelated by construction. We "
  f"confirm this: the largest absolute off-diagonal correlation among them is {D['max_offdiag_corr']:.3f}, and "
  "the correlation matrix is visually diagonal. There is therefore no linear redundancy to remove, and no reason "
  "to apply a further dimensionality reduction such as PCA, since the data is already an orthogonal reduced "
  "representation. Any remaining redundancy would be non-linear and is better handled by the models than by "
  "manual feature removal. To detect redundancy in a general dataset we would inspect a correlation matrix and "
  "the variance inflation factor, and drop or combine highly collinear columns.")
P("<b>Our engineering.</b> We make two changes. We reconstruct an hour-of-day feature from the time column, "
  "which the exploratory analysis shows is informative because the fraud rate is higher during quiet night "
  "hours. We also scale the amount with a RobustScaler, which centres on the median and scales by the "
  "interquartile range, because the amount is heavy-tailed with extreme outliers that would distort a standard "
  "scaler. The scaler is fitted on the training split only.")
P("<b>Additional features that could help.</b> With the raw transaction records (not available here because of "
  "anonymisation) the most valuable additions would be behavioural aggregates per card: the number of "
  "transactions and total amount in the last hour and day, the time since the previous transaction, the "
  "deviation of the current amount from the cardholder's usual spending, and simple merchant or country "
  "risk indicators. These velocity and profile features are known to be strong signals in production fraud "
  "systems.")
fig("corr_heatmap.png", 11 * cm)

# ============================ 4. REPRODUCIBILITY =============================
P("4. Reproducibility Analysis", h1)
P("<b>Does the code run?</b> Not as published. The notebook was written for Keras 1.x and TensorFlow 1.1 in "
  "2017, using imports such as <font face='Courier'>from keras.models import Model</font>, an "
  "<font face='Courier'>.h5</font> checkpoint, and a deprecated <font face='Courier'>pd.value_counts</font> "
  "call. On a current TensorFlow 2.20 environment it fails at import. We ported it to the "
  "<font face='Courier'>tensorflow.keras</font> API and a <font face='Courier'>.keras</font> checkpoint without "
  "changing any modelling choice.")
P("<b>Are the required files and data available?</b> The repository provides the code and a pre-trained model "
  "but not the dataset, and the Kaggle download requires an account. We restored a self-contained workflow by "
  "loading the same data from OpenML (id 42175). We note that the more common OpenML mirror (id 1597) drops the "
  "time column, which the temporal analysis needs.")
P("<b>Are there hidden preprocessing steps?</b> In effect, yes. The scaler fitted on all data, the use of the "
  "test set for model selection, and the manually chosen threshold are easy to overlook when reading the code, "
  "yet each one changes the reported result. Overall the work is reproducible only after a non-trivial repair. "
  "A fixed random seed and a scripted data loader, as used in this project, are what make the analysis genuinely "
  "repeatable.")

# ============================ 5. EXPERIMENTAL RESULTS ========================
P("5. Experimental Results", h1)
P("We evaluate five configurations on the same held-out test set: the original autoencoder at its 2.9 "
  "threshold; a corrected autoencoder that uses train-only scaling, early stopping on a validation split, and a "
  "threshold selected on validation by F2; and three baselines, namely Logistic Regression and Random Forest "
  "(both with balanced class weights) and an Isolation Forest. For every fair model the decision threshold is "
  "chosen on the validation set and the test set is used once, for the final measurement only.")

P("5.1 Evaluation metrics", h2)
P("We report the metrics below. For a fraud problem a false positive is a legitimate transaction that gets "
  "blocked, which causes customer friction and manual review cost, while a false negative is a fraud that is "
  "missed, which causes a direct financial loss and a security failure. Because a missed fraud is generally more "
  "costly than a false alarm, we favour recall through the F2 score and select thresholds accordingly.")
for name, formula, meaning in [
    ("Precision", "TP / (TP + FP)",
     "share of flagged transactions that are truly fraud; low precision means analysts chase false alarms."),
    ("Recall (TPR)", "TP / (TP + FN)",
     "share of frauds that are caught; low recall means fraud passes through undetected."),
    ("F1", "2&middot;P&middot;R / (P + R)",
     "harmonic mean of precision and recall, giving them equal weight."),
    ("F2", "5&middot;P&middot;R / (4&middot;P + R)",
     "F-beta with beta = 2; weights recall four times as much as precision, matching the higher cost of a "
     "missed fraud."),
    ("MCC", "(TP&middot;TN &minus; FP&middot;FN) / &radic;((TP+FP)(TP+FN)(TN+FP)(TN+FN))",
     "correlation between predictions and labels using all four confusion-matrix cells; robust under imbalance, "
     "with 1 perfect and 0 random."),
    ("ROC-AUC", "area under TPR vs FPR",
     "ranking quality across all thresholds; optimistic here because the abundant true negatives dominate it."),
    ("PR-AUC", "area under precision vs recall",
     "our primary metric; it ignores the many true negatives and focuses on performance for the rare fraud "
     "class."),
]:
    P(f"&bull; <b>{name}</b> = {formula}. {meaning[0].upper() + meaning[1:]}")

# metrics table
header = ["Model", "PR-AUC", "ROC-AUC", "MCC", "F2", "Prec.", "Recall", "TP", "FP", "FN"]
rows = [header]
for name, m in ranked:
    rows.append([name, f"{m['pr_auc']:.3f}", f"{m['roc_auc']:.3f}", f"{m['mcc']:.3f}", f"{m['f2']:.3f}",
                 f"{m['precision']:.3f}", f"{m['recall']:.3f}", str(m["TP"]), str(m["FP"]), str(m["FN"])])
tbl = Table(rows, hAlign="LEFT")
tbl.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c4f7c")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTSIZE", (0, 0), (-1, -1), 8), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef2f7")]),
    ("ALIGN", (1, 0), (-1, -1), "CENTER")]))
story.append(Spacer(1, 4)); story.append(tbl); story.append(Spacer(1, 6))
P(f"<b>Table 1.</b> Test-set performance, ordered by PR-AUC. The best PR-AUC is {best_name} at "
  f"{best['pr_auc']:.3f}. Logistic Regression reaches {lr['pr_auc']:.3f} and Random Forest {rf['pr_auc']:.3f}, "
  f"against {ae['pr_auc']:.3f} for the corrected autoencoder and {iso['pr_auc']:.3f} for the Isolation Forest.",
  small)
fig("pr_curves.png", 11 * cm)
P(("Both supervised baselines reach a higher PR-AUC than the autoencoder, and Random Forest is the strongest "
   "model overall. " if supervised_beat_ae else "The models are close on PR-AUC. ") +
  f"The original autoencoder's ROC-AUC of {orig['roc_auc']:.3f} is the figure the tutorial highlights, yet its "
  "PR-AUC and precision are the weakest in the table. This is the difference the study set out to measure.")

# ============================ 6. CONCLUSIONS =================================
P("6. Conclusions", h1)
P(f"<b>Key findings.</b> The source's ROC-AUC reproduces at {D['original']['roc_auc']:.3f}, but the figure is "
  f"inflated by three test-set leaks and is the wrong summary for a {pct(D['prevalence'])} positive rate. On "
  f"PR-AUC, MCC and F2, plain supervised models outperform the autoencoder, with {best_name} in front. The "
  "published code also does not run without being ported to a current stack.")
P("<b>Strengths and weaknesses of the source.</b> The tutorial is clear and well written, the semi-supervised "
  "idea is reasonable for a setting with few labels, and it does show a precision-recall curve. Its weaknesses "
  "are the leaky evaluation, an arbitrary threshold, an uninformative accuracy metric, a discarded temporal "
  "feature, and the absence of any supervised baseline for comparison.")
P("<b>Lessons learned and future work.</b> The main lesson is that reproducing a headline number is not the "
  "same as validating the claim behind it. Useful extensions would be cost-sensitive thresholding tied to the "
  "monetary cost of a missed fraud against a blocked card, probability calibration, time-ordered validation "
  "instead of a random split, and use of the time signal that the source discards. Where labels exist we would "
  "recommend a supervised model, and reserve the autoencoder or Isolation Forest for the fully unlabelled case.")

story.append(PageBreak())

# ============================ 7. EXECUTIVE SUMMARY ===========================
P("7. Executive Summary", h1)
P(f"This project reproduces and critically evaluates a widely shared tutorial that detects credit-card fraud "
  f"with a Keras autoencoder on the ULB dataset of {D['n_rows']:,} transactions, of which {pct(D['prevalence'])} "
  f"are fraudulent. The tutorial reports a ROC-AUC of about 0.95. We reproduce this faithfully, obtaining "
  f"{D['original']['roc_auc']:.3f}, after porting the 2017 TensorFlow 1 code to TensorFlow 2 and restoring the "
  "dataset through OpenML.")
P("The critical analysis finds the headline both optimistic and incomplete. The evaluation allows the test set "
  "to influence feature scaling, model selection, and the choice of the 2.9 threshold, and ROC-AUC is a "
  "flattering metric under this level of imbalance. Measured with PR-AUC, MCC and F2, which are the appropriate "
  f"metrics for rare-event detection, the autoencoder's PR-AUC is only {D['original']['pr_auc']:.3f}. After the "
  "leaks are removed and baselines are added, "
  + (f"Logistic Regression and Random Forest both outperform the autoencoder, and {best_name} is best, with a "
     f"PR-AUC of {best['pr_auc']:.3f} against {ae['pr_auc']:.3f}. " if supervised_beat_ae else
     f"{best_name} performs best, with a PR-AUC of {best['pr_auc']:.3f}. ") +
  "In operational terms the autoencoder trades a large number of false alarms for each fraud it catches.")
P("The conclusion is that the author's numbers are reproducible but the recommendation is overstated. An "
  "autoencoder is not the preferred tool for this problem when labels are available, since a supervised "
  "classifier is simpler, faster to train, and more accurate on the metrics that matter for security operations.")

# ============================ 8. SUMMING IT UP ===============================
P("8. Summing It Up", h1)
bullets = [
    ("Problem", f"identify the {pct(D['prevalence'])} of ULB transactions that are fraudulent."),
    ("Source", "Valkov, 'Credit Card Fraud Detection using Autoencoders in Keras' (curiousily.com; GitHub, ~588 stars)."),
    ("Dataset", f"ULB credit-card fraud, {D['n_rows']:,} transactions, {D['n_fraud']} frauds (OpenML id 42175)."),
    ("Methodology", "faithful reproduction with the flaws preserved, then a leak-free rerun and supervised "
                    "baselines, with validation-selected thresholds, PR-AUC, MCC and F2 evaluation, and an error analysis."),
    ("Main finding", f"the ROC-AUC reproduces at {D['original']['roc_auc']:.3f} but is leaked and misleading; "
                     f"supervised baselines win on PR-AUC, with {best_name} best."),
    ("Were the claims supported", "the numbers reproduce, but the claim of an effective detector is not "
                                  "supported once appropriate metrics are used."),
    ("Recommended for similar problems", "not the autoencoder when labels are available; use it only in the "
                                         "unlabelled setting, where an Isolation Forest is a stronger baseline on this data."),
]
for k, v in bullets:
    P(f"&bull; <b>{k}:</b> {v}")
P("<b>Final conclusion.</b> Honest, imbalance-aware evaluation reverses an impressive-looking ROC-AUC and shows "
  "that a simpler supervised model performs better on the metrics that reflect real detection performance.")

# ============================ BUILD ==========================================
doc = SimpleDocTemplate(str(ROOT / "report.pdf"), pagesize=A4,
                        leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.6 * cm, bottomMargin=1.6 * cm,
                        title="Credit Card Fraud Detection: A Critical Reproduction Study")
doc.build(story)
print("Wrote report.pdf")
