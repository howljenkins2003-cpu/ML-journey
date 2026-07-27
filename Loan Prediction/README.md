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

## Next
Test model with Credit_History dropped, to confirm its dominance