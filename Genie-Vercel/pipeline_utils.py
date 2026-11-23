"""Notebook-compatible CKD preprocessing helpers."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTENC


class RoundBinary(BaseEstimator, TransformerMixin):
    def __init__(self, bin_idx=None): self.bin_idx = bin_idx
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        if self.bin_idx: X[:, self.bin_idx] = np.clip(np.round(X[:, self.bin_idx]), 0, 1)
        return X


def get_imputer(name):
    if name == 'median': return SimpleImputer(strategy='median')
    if name == 'knn': return KNNImputer(n_neighbors=5)
    if name == 'mice': return IterativeImputer(max_iter=10, random_state=42)
    raise ValueError(name)


def build_pipeline(model, imputer_name, bin_idx, cont_idx, sampler=None, random_state=42):
    steps = [('imputer', get_imputer(imputer_name)),
             ('round', RoundBinary(bin_idx=list(bin_idx))),
             ('scale', ColumnTransformer([('num', StandardScaler(), list(cont_idx))], remainder='passthrough'))]
    if sampler == 'smote':
        n_c = len(cont_idx)
        steps.append(('smote', SMOTENC(categorical_features=list(range(n_c, n_c + len(bin_idx))), random_state=random_state)))
    steps.append(('model', model))
    return Pipeline(steps)


def feature_order(columns, cont_idx, bin_idx):
    return [columns[i] for i in cont_idx] + [columns[i] for i in bin_idx]


def preprocess(pipe, X):
    for name, step in pipe.steps[:-1]:
        if name == 'smote': continue
        X = step.transform(X)
    return X
