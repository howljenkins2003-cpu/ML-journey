# Loan Approval Model (learning project)

Predicts whether a loan application is approved, from the Loan Prediction dataset (614 rows).
Model: Logistic Regression inside one scikit-learn pipeline that takes raw input.

## Files
- `model.joblib` - the fitted pipeline
- `helpers.py` - `Binary_Mapper`, `FeatureEngineer`, `binary_mapper`. Must be importable (same folder) when loading
- `requirements.txt` - library versions used for training
- Python used for training: 3.14.4

## Use
```python
import joblib, pandas as pd

pipe = joblib.load("model.joblib")          # helpers.py must be in the same folder
row = pd.DataFrame([{
    "Gender": "Male", "Married": "Yes", "Dependents": "0", "Education": "Graduate",
    "Self_Employed": "No", "ApplicantIncome": 5000, "CoapplicantIncome": 1500,
    "LoanAmount": 128, "Loan_Amount_Term": 360, "Credit_History": 1.0,
    "Property_Area": "Urban",
}])
pipe.predict_proba(row)    # [[P(rejected), P(approved)]]
```
Only load `.joblib` files from sources you trust.

## Input (raw values, no manual preprocessing)
| Column | Allowed values |
|---|---|
| Gender | Male, Female |
| Married | Yes, No |
| Education | Graduate, Not Graduate |
| Self_Employed | Yes, No |
| Dependents | "0", "1", "2", "3+" (text) |
| Property_Area | Urban, Semiurban, Rural |
| Credit_History | 1.0 or 0.0 |
| ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term | numbers |

- Do not include `Loan_ID` or `Loan_Status`.
- Case and surrounding spaces are ignored for Gender, Married, Education, Self_Employed.
- **An unrecognised value in those four columns raises `ValueError`** instead of being guessed.
- Blank values are filled with the training-data most-common value (categories) or median (numbers).
  Education, ApplicantIncome, CoapplicantIncome and Property_Area were never blank in training, so avoid blanks there.
- A Dependents or Property_Area value not seen in training is ignored (encoded as all zeros), with no error.

## Output
- Column 0 = probability rejected, column 1 = probability approved. The default cutoff is 0.5.
- The model uses `class_weight='balanced'`, so the numbers rank applicants but are not calibrated real-world odds.

## Performance
- 5-fold cross-validation F1: 0.8038 +/- 0.0151
- Hold-out accuracy: 0.8537 (123 rows)
- Small dataset, one split: treat as indicative only.

## Limits
- For learning only. It uses Gender and Married as features and is not suitable for real lending decisions.
- Trained on 614 rows from one dataset, so it may not generalise to other populations.