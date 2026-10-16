from sklearn.base import TransformerMixin, BaseEstimator # forgot the base estimator


def binary_mapper(series, mapping , col):
    clean = series.str.lower().str.strip()
    mapped = clean.map(mapping)
    bad = clean.notna() & mapped.isna() # here notna() is opposite of isnull(), anding both of them gives the one which are bad values, i.e is there value? , is the result missing/ Nan? if both are true it's bad./// also & is for series anding normally 'and' is used.
    #also , bad is a column of true or false like a mask of binaries/series of binaries////// also item[mask] returns part of item where mask is true.
    if bad.any():  #at least one is true in the list/series
        raise ValueError(f"{col}: Unrecognized {clean[bad].unique().tolist()}; expected {list(mapping)}") #clean[bad] lines up 'clean' and 'bad' row by row and keep the true values only
    return mapped


class Binary_Mapper(BaseEstimator, TransformerMixin):
    def fit(self, X, y= None): #wrote small none
        return self
    def transform(self, X): #forgot the arguments/parameters here hehe ;p
        X = X.copy()
        X['Gender'] = binary_mapper(X['Gender'], {'male':1, 'female': 0}, 'Gender') # totally forgot how to write this line haha
        X['Married'] = binary_mapper(X['Married'], {'yes': 1, 'no': 0}, 'Married')
        X['Education'] = binary_mapper(X['Education'],{'graduate': 1, 'not graduate': 0}, 'Education')
        X['Self_Employed'] = binary_mapper(X['Self_Employed'],{'yes': 1, 'no': 0}, 'Self_Employed')
        return X

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y = None):
        self.median_ = X['LoanAmount'].median()
        return self
    def transform(self, X):
        X = X.copy()
        X['Income_to_Loan'] = X['ApplicantIncome']/X['LoanAmount'].fillna(self.median_)
        return X
