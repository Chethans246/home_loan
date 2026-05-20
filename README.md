# Home Loan Default Prediction

Predicting whether a home loan applicant will default on repayment using multi-table credit data — to help lenders reduce non-performing loans and make faster, data-driven credit decisions.

---

## Problem Statement

Home loan defaults are among the most costly events for a lending institution. With only ~8% of applicants defaulting, the dataset is severely imbalanced and a naive model that always predicts "no default" achieves 92% accuracy while being completely useless. The goal of this project is to build a model that reliably identifies high-risk applicants before disbursement, enabling the bank to prevent bad loans from entering the portfolio.

---

## Dataset

- **Source:** Provided by Rubixe AI Solutions as part of a structured analytics project program
- **Size:** 300,000+ records across 7 related tables (20% stratified sample used for development)
- **Target:** Binary — payment difficulty / default (1) or repaid on time (0)
- **Class distribution:** ~8% default rate (12:1 imbalance)

### Tables Used

| Table | Description |
|---|---|
| application_train | Main applicant demographics and loan details |
| bureau | Prior credit history from external credit bureau |
| bureau_balance | Monthly balance snapshots per bureau credit |
| previous_application | Past loan applications at Home Credit |
| POS_CASH_balance | Monthly POS and cash loan payment history |
| credit_card_balance | Monthly credit card usage and payment data |
| installments_payments | Installment payment records for prior loans |

---

## Repository Structure

```
home-loan-default-prediction/
│
└── home_loan_def.ipynb    # Full multi-table aggregation, EDA, modeling, and business insights notebook
```

---

## Approach

### 1. Multi-Table Feature Aggregation

Each supplementary table contains multiple rows per applicant (monthly snapshots or per-loan records) and cannot be directly merged. Custom aggregations were built for each table:

| Engineered Feature | Source Table | Description |
|---|---|---|
| bureau_loan_count | bureau | Total prior credits |
| bureau_total_debt | bureau | Total outstanding debt |
| bureau_max_day_overdue | bureau | Worst overdue days in credit history |
| prev_approval_rate | previous_application | Fraction of past applications approved |
| pos_max_dpd | POS_CASH_balance | Maximum days past due on POS loans |
| cc_avg_utilisation | credit_card_balance | Average credit card utilisation rate |
| inst_late_payment_rate | installments_payments | Fraction of installments paid late |
| inst_total_shortfall | installments_payments | Cumulative underpayment across all installments |

Bureau balance required a two-stage rollup — aggregated to bureau level first, then merged to applicant level.

### 2. Feature Engineering on Application Table

| Feature | Description |
|---|---|
| AGE_YEARS | Client age in years (converted from DAYS_BIRTH) |
| CREDIT_INCOME_RATIO | Debt burden — credit amount divided by annual income |
| ANNUITY_INCOME_RATIO | Affordability — annuity divided by income |
| EMPLOYED_YEARS | Employment duration in years |
| EXT_SOURCE_MEAN | Average of all three external credit bureau scores |
| INCOME_PER_PERSON | Income divided by family size |

### 3. Data Cleaning
- Replaced anomalous DAYS_EMPLOYED value of 365,243 (sentinel for unemployed) with NaN before imputation
- Numeric columns imputed with median; categorical columns imputed with mode
- High-null columns retained (tree-based models handle sparse features effectively)

### 4. Model Comparison

| Model | ROC-AUC | Notes |
|---|---|---|
| Logistic Regression | Baseline | Linear boundary, limited on complex credit patterns |
| Decision Tree | Lower | High overfitting risk |
| Random Forest | Strong | Ensemble, handles noise well |
| **XGBoost** | **~0.78** | Best — selected for deployment |

### 5. Class Imbalance Handling
- `class_weight='balanced'` applied to Logistic Regression, Decision Tree, and Random Forest
- `scale_pos_weight=12` applied to XGBoost (reflecting 92:8 class ratio)
- Primary evaluation metric: ROC-AUC
- Secondary metrics: Recall and F1-score on the default class

---

## Model Selection: Why XGBoost

- **ROC-AUC ~0.78** — highest among all models
- Iterative boosting corrects errors from previous trees, ideal for complex tabular credit patterns
- `scale_pos_weight` handles the 12:1 imbalance natively during training
- Captures non-linear relationships (e.g., interaction between age and debt burden) that logistic regression misses
- Native feature importance enables direct business communication of risk drivers
- Scalable to full 300k+ dataset and deployable via API for real-time scoring

---

## Key Features by Importance

1. **EXT_SOURCE_MEAN / EXT_SOURCE_2** — External credit bureau scores are the single strongest predictor; lower scores signal high default risk
2. **CREDIT_INCOME_RATIO** — Debt burden relative to income; applicants above a threshold show significantly higher default rates
3. **inst_late_payment_rate** — Payment behavior on prior loans is a critical behavioral signal
4. **AGE_YEARS** — Younger applicants (20 to 30) default significantly more than older applicants
5. **bureau_max_day_overdue** — Worst historical overdue days; a strong early warning from credit history

---

## Challenges and Solutions

| Challenge | Solution |
|---|---|
| Anomalous DAYS_EMPLOYED value of 365,243 | Replaced with NaN before median imputation |
| 92% vs 8% class imbalance | scale_pos_weight and class_weight; evaluated using ROC-AUC and Recall |
| 7 tables with hundreds of columns and tens of millions of rows | Stratified 20% sampling on main table; all supplementary tables filtered to matching IDs |
| Each supplementary table has multiple rows per applicant | Custom groupby aggregations per table; two-stage rollup for bureau balance |
| 122 columns with high null rates (some above 60%) | Median/mode imputation; high-null columns retained for tree-based models |
| Accuracy misleading at 92% on imbalanced data | Switched primary metric to ROC-AUC; tracked recall and F1 on default class |

---

## Business Recommendations

**Risk Stratification**
Use model probability scores to bucket applicants into low, medium, and high risk tiers. Apply different loan terms, interest rates, or collateral requirements per tier — pricing aligned with actual risk.

**Hard Policy Rules from Feature Importance**
- Set a CREDIT_INCOME_RATIO cap as a lending policy rule; applicants above the threshold require additional collateral
- Apply extra scrutiny to applicants aged 20 to 30 — request proof of stable employment or co-signers
- Prioritize bureau partnerships to improve external score coverage (EXT_SOURCE features are the strongest drivers)

**Automated Pre-Screening**
Use model scores as a first-pass filter before manual underwriting. Low-risk applicants get faster approvals; only borderline and high-risk applicants go through full manual review.

**Estimated Portfolio Impact**
- Prevents high-risk loans from entering the portfolio
- Reduces provision requirements and impairment charges
- Improves balance sheet health and regulatory capital efficiency

---

## Tech Stack

- **Language:** Python 3.13
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn
- **Environment:** Jupyter Notebook

---

## How to Run

1. Clone the repository
   ```bash
   git clone https://github.com/Chethans246/home-loan-default-prediction.git
   cd home-loan-default-prediction
   ```

2. Install dependencies
   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter
   ```

3. Place the dataset files in a `/dataset/` folder (application_train.csv, bureau.csv, bureau_balance.csv, credit_card_balance.csv, POS_CASH_balance.csv, previous_application.csv, installments_payments.csv)

4. Launch the notebook
   ```bash
   jupyter notebook home_loan_def.ipynb
   ```

---

## Limitations and Future Work

**Current Limitations**
- Trained on 20% sample; full dataset training may improve ROC-AUC further
- External macroeconomic factors (interest rate changes, job market conditions) are not included
- First-time borrowers with no credit history have many aggregated features set to zero, reducing accuracy for this segment
- Model probabilities are not calibrated — raw scores should not be used directly for risk-based pricing without calibration

**Planned Improvements**
- Apply SMOTE or advanced resampling for minority class handling
- Add alternative data sources (utility payments, mobile usage) for thin-file applicants
- Implement model monitoring and scheduled retraining as default patterns shift
- Calibrate model probabilities using Platt scaling for risk-based pricing use cases

---

## Author

**Chethan S** — Data Analyst  
[LinkedIn](https://linkedin.com/in/chethan-s-69b71b241) • [GitHub](https://github.com/Chethans246)
