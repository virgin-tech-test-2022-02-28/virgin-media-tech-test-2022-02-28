"""This carries out:

## Task 1
3. Perform any feature engineering you might find useful to enable training. 
It's not required that you create new features.
"""
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

categorical_cols = [
    "Type",
    "Breed1",
    "Gender",
    "Color1",
    "Color2",
    "MaturitySize",
    "FurLength",
    "Vaccinated",
    "Sterilized",
    "Health",
]
continuous_cols = ["Age", "Fee", "PhotoAmt"]


class Preprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.onehot_encoder = OneHotEncoder(handle_unknown="ignore")
        self.onehot_encoder.fit(X[categorical_cols])
        self.columns = continuous_cols + list(
            self.onehot_encoder.get_feature_names_out()
        )
        return self

    def transform(self, X, y=None):
        onehots = self.onehot_encoder.transform(X[categorical_cols]).todense()
        return np.hstack([X[continuous_cols], onehots])
