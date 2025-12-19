
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class DropCols(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_drop=None):
        self.cols_to_drop = cols_to_drop or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Xc = X.copy()
        return Xc.drop(columns=self.cols_to_drop, errors='ignore')


class ImcCalculator(BaseEstimator, TransformerMixin):
    def __init__(self, weight_col='Weight', height_col='Height', imc_col='IMC'):
        self.weight_col = weight_col
        self.height_col = height_col
        self.imc_col = imc_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Xc = X.copy()
        if self.weight_col in Xc.columns and self.height_col in Xc.columns:
            w = pd.to_numeric(Xc[self.weight_col], errors='coerce')
            h = pd.to_numeric(Xc[self.height_col], errors='coerce')
            Xc[self.imc_col] = (w / (h ** 2)).round(2)
        return Xc


def safe_onehot_encoder(**kwargs):
    if sklearn.__version__ < "1.2":
        return OneHotEncoder(**{**kwargs, 'sparse': False})
    else:
        return OneHotEncoder(**{**kwargs, 'sparse_output': False})
