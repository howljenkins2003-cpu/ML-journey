# Loan Prediction

## Problem
Predict loan approval (Y/N). Class imbalance ~69/31, and the costly 
real-world error is approving a bad loan — so precision(1) / recall(0) 
matter more than raw accuracy.

## Approach
- 3-way column split: binary (.map()), nominal (OneHotEncoder), numeric (median impute + scale)
- Pipeline + ColumnTransformer, compared LR/DT/RF via 5-fold CV

## Bugs found & fixed
1. train_test_split output was unpacked in wrong order (X_test, X_train swapped) 
   — model was training on 20% and "testing" on 80%
2. class_weight='balanced' was applied to DT/RF but missed on LR, the actual winning model

## Result
Logistic Regression — test acc 0.83, train 0.76 (no overfitting gap)
- recall(0) improved 0.41 → 0.71 after fixes
- Credit_History coefficient (3.09) dominates all other features 5-8x — 
  model is essentially a credit-history classifier with minor adjustments


## Feature Engineering — Income_to_Loan

**Idea:** ApplicantIncome / LoanAmount as an affordability signal.

**Leakage-safe implementation:** median from X_train only, applied to both train/test before dividing.

**Controlled test (fixed split, class_weight='balanced', LR, 5-fold CV F1):**
- Raw cols only: 0.8056 ± 0.0147
- + ratio (kept both): 0.8038 ± 0.0151
- Ratio only (dropped raw cols): 0.8038 ± 0.0205

**Result:** all three equivalent (within std). Ratio adds no new signal, but fully substitutes for the 2 raw columns — same performance, fewer features.

**Gotcha:** test-set accuracy looked better with the ratio (0.83→0.85), but that didn't hold under CV — small test set (123 rows), likely noise. Trust CV over one test split.

**Multicollinearity note:** with ratio + raw cols both present, ratio's coefficient went negative despite domain logic saying positive — derived features overlapping with their source columns make coefficients unreliable, even when predictions stay fine.

**Verdict:** no case for adding it on top; reasonable case for replacing the 2 raw cols with it if you want fewer features.

## Update — Pipeline is now ship-ready

Refactored the two remaining manual pandas steps into real custom transformers, closing the "half pipeline" gap.

**FeatureEngineer** (`BaseEstimator, TransformerMixin`)
- Replaces the manual `Income_to_Loan` computation that previously lived outside the pipeline
- `fit` learns `loan_median_` from whatever `X` it's given; `transform` builds the ratio using that frozen median
- Verified leakage-safe: `self.loan_median_` after fitting on `X_train` matches `X_train['LoanAmount'].median()` exactly — confirms `X_test` never influenced the learned value
- Results match the old manual approach almost exactly (LR CV: 0.8038 both ways) — moving the logic into a proper transformer didn't change performance, only made it portable 

**BinaryMapper** (`BaseEstimator, TransformerMixin`)
- Replaces manual `.map()` calls for `Gender`, `Married`, `Education`, `Self_Employed`
- No `fit` logic needed (fixed dictionaries, nothing learned from data) — `fit` is a no-op that just returns `self` to satisfy the pipeline contract
- `Loan_Status` (the target) deliberately stays **outside** the pipeline — mapped once on `df` before the split, since pipelines only ever process `X`, never `y`

**Why this matters:** the pipeline now takes raw, unprocessed input and handles feature engineering, encoding, imputation, and prediction entirely internally. No external caller needs private knowledge of the preprocessing steps — that's the actual bar for "production-ready," not just "high accuracy."

```python
Pipeline([
    ('feature_eng', FeatureEngineer()),
    ('binary_map', BinaryMapper()),
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
])
```