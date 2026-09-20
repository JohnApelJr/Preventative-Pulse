# Preventative Pulse — Predictive Analytics for Heart Disease

A machine learning pipeline that predicts heart disease from CDC behavioral survey data, achieving **83.5% ROC-AUC** and **75.7% F1-score** using logistic regression with SMOTE oversampling. Built on 319K+ patient records from the 2020 BRFSS survey, then validated against labels on 440K unseen respondents from the 2022 survey (**ROC-AUC 0.836 out of sample**).

---

## Key Results

### Model Performance

Six classifiers were evaluated using 5-fold stratified cross-validation with SMOTE applied inside each fold to prevent data leakage. Logistic regression narrowly outperformed gradient boosting and XGBoost on F1-Macro, the lead metric chosen for its sensitivity to the minority class.

![Model Comparison](figures/model_comparison.png)

**Final validation metrics (tuned logistic regression, 20% hold-out set, undersampled to 50/50):**

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.835 |
| F1-Macro | 0.757 |
| Accuracy | 0.757 |
| Precision (Yes) | 0.75 |
| Recall (Yes) | 0.77 |

Precision and F1 here are measured on a balanced sample. At real-world prevalence they are much lower (see below).

### Out-of-Sample Validation (2022 BRFSS)

The trained pipeline was applied to the 2022 survey wave and scored against its labels. The 2020 target is ever-reported coronary heart disease or heart attack, so the matching 2022 label is `HadHeartAttack` or `HadAngina`.

| Metric | 2020 hold-out (50/50) | 2022 wave (9.0% prevalence) |
|--------|------|------|
| ROC-AUC | 0.835 | **0.836** |
| Average precision | 0.818 | 0.349 |
| Precision @ 0.5 | 0.75 | 0.23 |
| Recall @ 0.5 | 0.77 | 0.77 |
| Flagged positive @ 0.5 | 50% | 30% |

![2022 ROC](figures/roc_2022_validation.png)

Ranking held across survey waves. Calibration did not. The model was trained on undersampled 50/50 data, so a 0.5 cutoff assumes a 50% base rate, and on a population with 9% prevalence it flags 30% of people. Any screening use needs recalibration and a threshold chosen at real prevalence. Using `HadHeartAttack` alone as the label gives AUC 0.830.

Getting a valid score required mapping category values, not just column names. The 2022 file relabels age (`Age 18 to 24`), smoking (four levels instead of yes/no), race, and diabetes. An earlier version of this notebook skipped that step, which silently scored every 2022 respondent as age 18 to 24, and it reported only the distribution of predictions without checking labels. Both are fixed.

### Top Risk Factors

Logistic regression coefficients reveal the strongest predictors after scaling. The chart below shows the top 20 features by absolute coefficient magnitude — pink bars increase heart-disease risk, blue bars decrease it.

![Feature Importance](figures/feature_importance.png)

**Takeaways for clinicians:**
- Age dominates — the 80+ coefficient is 5× larger than any medical condition
- Stroke history and poor self-reported health are the next-strongest signals
- Kidney disease and diabetes show elevated risk but smaller model weight than expected, likely due to correlation with age
- Being in the 25–29 age bracket is the strongest *protective* factor

### ROC & Precision-Recall Curves

<p float="left">
  <img src="figures/roc_curve.png" width="48%" />
  <img src="figures/precision_recall_curve.png" width="48%" />
</p>

The precision-recall curve (AP = 0.818) shows the model maintains >80% precision up to ~60% recall — meaning it can correctly flag 6 in 10 actual heart-disease cases while keeping false alarms manageable.

---

## Dataset

**Source:** [CDC BRFSS 2020](https://www.cdc.gov/brfss/annual_data/annual_2020.html) via [Kaggle](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease)

| | Training (2020) | Test (2022) |
|--|----------------|-------------|
| Records | 319,795 | 445,132 |
| Features | 18 | 40 → aligned to 17 |
| Target | HeartDisease (Yes/No) | HadHeartAttack OR HadAngina |
| Class balance | 91.4% No / 8.6% Yes | 91.0% No / 9.0% Yes |

**Features span three domains:**
- Medical history — diabetes, kidney disease, stroke, asthma, skin cancer
- Demographics — age, sex, race
- Lifestyle — smoking, alcohol, physical activity, sleep, BMI, self-rated health

Full per-variable EDA with distributions, outlier detection, and heart-disease rate breakdowns is documented in the notebook.

---

## Methodology

```
Raw Data (319K records, 18 columns)
  │
  ├─ Quality Audit ──── 0 nulls, 18K duplicates flagged, IQR outlier detection
  │
  ├─ EDA ──────────────  Per-variable distributions, target-rate analysis,
  │                      correlation heatmap, before/after rebalancing metrics
  │
  ├─ Preprocessing ───── Median/mode imputation → one-hot encoding →
  │                      StandardScaler (fit on train only) → feature alignment
  │
  ├─ Class Balancing ─── Random undersampling to 50/50 for training;
  │                      SMOTE inside CV folds for evaluation
  │
  ├─ Model Selection ─── 6 classifiers × 5-fold CV × 3 metrics
  │                      (F1-Macro, Accuracy, ROC-AUC)
  │
  ├─ Tuning ──────────── RandomizedSearchCV on best model
  │                      (C=0.1, penalty=L2, solver=liblinear)
  │
  └─ 2022 Validation ─── Map 2022 column names AND category values →
                         score 440K labeled records → ROC-AUC 0.836
```

---

## Repo Structure

```
Preventative-Pulse/
├── README.md
├── Preventative_Pulse.ipynb     # Full pipeline: EDA → modeling → results
├── requirements.txt
├── .gitignore
├── data/
│   ├── heart_2020_cleaned.csv   # Training data (2020 BRFSS, 320K rows)
│   └── heart_2022_with_nans.csv # Test data — download separately (see below)
└── figures/                     # Auto-generated by notebook
```

## How to Run

```bash
git clone https://github.com/yourusername/Preventative-Pulse.git
cd Preventative-Pulse
pip install -r requirements.txt
```

The 2022 test dataset (139 MB) is too large for GitHub. Download it from [Kaggle](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease) and place the file in `data/heart_2022_with_nans.csv`. Sections 1–8 of the notebook run without it; only Section 9 (2022 out-of-sample validation) requires it.

```bash
jupyter notebook Preventative_Pulse.ipynb
```

All figures are saved to `figures/` on execution.

---

## Tech Stack

Python · pandas · NumPy · scikit-learn · imbalanced-learn (SMOTE) · XGBoost · Matplotlib · Seaborn

---

## License

Dataset published by [Kamil Pytlak](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease) under CC BY-SA 4.0. Code in this repository is MIT licensed.
