# Credit Card Fraud Detection: A Critical Reproduction Study

**Course:** Data Science in Cyber, Dr. Uri Itai. **Submission:** Gilad Segal 216214353
This project reproduces and critically evaluates a widely shared autoencoder-based credit-card-fraud
tutorial. The aim is not to build a detector from scratch, but to check whether the source's claims hold once
the analysis is repeated carefully.

## Source reviewed
- **Article / blog:** Venelin Valkov, "Credit Card Fraud Detection using Autoencoders in Keras
  (TensorFlow for Hackers, Part VII)".
  https://curiousily.com/posts/credit-card-fraud-detection-using-autoencoders-in-keras/
- **Original GitHub repository (about 588 stars):**
  https://github.com/curiousily/Credit-Card-Fraud-Detection-using-Autoencoders-in-Keras

## Dataset
- ULB Credit Card Fraud (Kaggle `mlg-ulb/creditcardfraud`; Dal Pozzolo et al., Université Libre de Bruxelles).
- Loaded from **OpenML id 42175** by `src/load_data.py`, so no Kaggle account is required.
- 284,807 transactions, 492 frauds (0.17% prevalence), with features `Time`, `V1`..`V28` (PCA components),
  `Amount` and `Class`. The raw CSV is about 148 MB and is not committed; it downloads on first run.

## Summary of findings
- The 2017 Keras-1/TensorFlow-1 code does not run on a current stack without porting, which is a
  reproducibility gap in itself.
- The reported ROC-AUC of about 0.95 reproduces (we obtain 0.961), but it is measured optimistically. The
  source lets the test set influence feature scaling, model selection (`validation_data=X_test` together with
  `save_best_only`), and the hand-picked decision threshold of 2.9.
- On PR-AUC, the appropriate metric under this level of imbalance, a plain Logistic Regression (0.72) and
  Random Forest (0.84) clearly beat the autoencoder (0.20). The implicit claim that an autoencoder is a good
  detector for this problem is not supported by the evidence.

The full write-up is in [`report.pdf`](report.pdf), and the executable analysis is in
[`notebook.ipynb`](notebook.ipynb).

## Repository layout
```
notebook.ipynb           complete, executable analysis (Sections 1-8)
report.pdf               8-section critical report
build_notebook.py        regenerates notebook.ipynb from source cells
make_report.py           regenerates report.pdf from the notebook's results
src/
  load_data.py           reproducible dataset loader (OpenML id 42175, cached to data/)
  reproduce_original.py  faithful TF2 port of the source, flaws preserved (writes artifacts/)
  utils.py               autoencoder, metrics suite, F-beta threshold selection
requirements.txt
```

## How to run
```bash
# 1. Environment (Python 3.10)
pip install -r requirements.txt

# 2. (Optional) reproduce the original method faithfully; this writes the artifacts/ the notebook reuses.
python src/reproduce_original.py            # about 15 min on CPU (100 epochs); EPOCHS=2 for a quick test

# 3. Execute the notebook end to end
python -m nbconvert --to notebook --execute notebook.ipynb --output notebook.ipynb \
       --ExecutePreprocessor.timeout=3600

# 4. (Optional) regenerate the PDF report from the notebook's saved results
python make_report.py
```
The dataset downloads automatically from OpenML on the first run and is cached under `data/`.

## Reproducibility notes
- A fixed random seed (42) is used throughout, and the train/validation/test split is stratified on `Class`.
- The test set is held out from scaling, model selection and threshold choice, unlike the source.
- The Keras-1/TF-1 imports were ported to `tensorflow.keras`, and the checkpoint format updated to `.keras`.
