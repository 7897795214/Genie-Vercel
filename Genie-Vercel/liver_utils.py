"""Notebook-compatible liver preprocessing helpers."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline
from pipeline_utils import get_imputer, RoundBinary


class LogCols(BaseEstimator, TransformerMixin):
    def __init__(self, log_idx=None): self.log_idx = log_idx
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        if self.log_idx: X[:, self.log_idx] = np.log1p(np.clip(X[:, self.log_idx], 0, None))
        return X


def build_liver_pipeline(model, imputer_name, bin_idx, cont_idx, log_idx,
                         sampler=None, random_state=42):
    steps = [('log', LogCols(log_idx=list(log_idx))), ('imputer', get_imputer(imputer_name)),
             ('round', RoundBinary(bin_idx=list(bin_idx))),
             ('scale', ColumnTransformer([('num', StandardScaler(), list(cont_idx))], remainder='passthrough'))]
    if sampler == 'smote':
        n_c = len(cont_idx)
        steps.append(('smote', SMOTENC(categorical_features=list(range(n_c, n_c + len(bin_idx))), random_state=random_state)))
    steps.append(('model', model))
    return Pipeline(steps)
