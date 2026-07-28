Understanding the pipeline tree of the pipeline.(gets the scaler object from the leaf node.)
 ## Session 1 : Understanding the structure of a pipe and it's leaves and branches.
pipe                                    # this is a Pipeline
pipe.named_steps['preprocesor']         # ✓ Pipeline → use named_steps
                                         # → returns the ColumnTransformer object

preprocesor_obj = pipe.named_steps['preprocesor']   # now I'm holding a ColumnTransformer
preprocesor_obj.named_transformers_['num']          # ✓ ColumnTransformer → use named_transformers_
                                                      # → returns the numeric_pipeline (a Pipeline again!)

numeric_pipe_obj = preprocesor_obj.named_transformers_['num']   # now holding a Pipeline again
numeric_pipe_obj.named_steps['scaler']                          # ✓ Pipeline → back to named_steps


Actual code to implement:(example to get the median value from the leaf pipeline step)
lr_pipeline.named_steps['preprocesor'].named_transformers_['num'].named_steps['imputer'].statistics_




## Session 2 — Feature engineering + pipeline production gaps

### Key realization: feature engineering is the actual bottleneck
Every dataset so far (Adult Income, Pima, Loan Prediction) hit a ceiling where 
tuning C gave negligible gains — root cause: no feature engineering, not a 
model/tuning problem.

### Feature engineering process
1. Understand what a domain expert would look at
2. Clean first (impute/encode/scale) — already solid
3. Create ratios/combinations from existing columns (e.g. Income_to_Loan)
4. Extract structure from complex columns (dates, text, IDs)
5. Bin continuous variables when relationship isn't linear
6. Aggregate related columns (e.g. TotalServices)
7. Interaction terms, sparingly
8. Test one feature at a time — add, rerun CV, keep only if metric improves
9. Check for leakage in every new feature

### Leakage rule, extended beyond Pipeline internals
Not just Scaler/Imputer — ANY calculation that needs a cross-row statistic 
(mean, median, mode, std) to build a feature must be learned from train only, 
then applied to test. Row-wise operations (map, add, divide same-row columns) 
are safe anywhere. Caught this myself: filling LoanAmount NaN with .median() 
before creating an Income_to_Loan ratio, done pre-split, leaks — the median 
must come from train only.

### Production gap identified
Manual feature engineering (.map() calls, engineered ratios) lives OUTSIDE 
the saved Pipeline object — pickling only saves the Pipeline, not the manual 
pandas code before it. New raw data in production won't have these columns 
unless the same manual steps are reapplied by hand, using the SAME frozen 
statistics (not recomputed).

### Real fix: custom transformer class
```python
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.loan_median_ = X['LoanAmount'].median()  # learned from train only
        return self

    def transform(self, X):
        X = X.copy()
        X['Gender'] = X['Gender'].map({"Male": 0, "Female": 1})
        # ... other manual mappings
        X['Income_to_Loan'] = X['ApplicantIncome'] / X['LoanAmount'].fillna(self.loan_median_)
        return X
```
Same fit/transform contract as StandardScaler — just hand-written. Goes in 
as the FIRST step of the Pipeline, so saving the pipeline saves everything.

Decided: too advanced to implement today, filed for later — directly relevant 
to riding classifier (IMU feature engineering + ESP32 deployment will hit 
this exact gap, with no notebook safety net once it's embedded C code).

### Pipeline introspection (locked in this session)
- Object type decides the attribute: Pipeline → `.named_steps['name']`, 
  ColumnTransformer → `.named_transformers_['name']`
- Chain left to right based on current object type
- `model__C` in GridSearchCV is the same named-step addressing, just as a string
- Trailing underscore (`mean_`, `statistics_`, `coef_`) = learned only after fit

### Next
Test Income_to_Loan feature properly (post-split, train-only median), 
compare CV/coefficients before vs after.