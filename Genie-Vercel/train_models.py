"""Retrain Genie models following the final training cells in AI_H.ipynb.

The notebook's Colab .joblib files were not supplied with the .ipynb, so this
script rebuilds its final pipelines from the same UCI source CSVs, searches,
calibrates, thresholds, and held-out evaluation procedure.
"""
import json
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (RandomizedSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold, cross_val_predict, train_test_split)
from sklearn.base import clone
import shap

from pipeline_utils import build_pipeline, feature_order
from liver_utils import build_liver_pipeline

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'models'
OUT.mkdir(exist_ok=True)

def source_csv(dataset_id):
    with urlopen(f'https://archive.ics.uci.edu/static/public/{dataset_id}/data.csv', timeout=45) as response:
        return pd.read_csv(BytesIO(response.read()))

def load_ckd():
    raw = source_csv(336)
    X = raw.drop(columns=['class']).copy().rename(columns={'wbcc':'wc','rbcc':'rc'})
    binary_map = {'normal':0, 'abnormal':1, 'notpresent':0, 'present':1,
                  'no':0, 'yes':1, 'good':0, 'poor':1}
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            s = X[col].astype(str).str.strip().str.lower()
            s = s.replace({'?':np.nan, '':np.nan, 'nan':np.nan, 'none':np.nan})
            X[col] = s.replace(binary_map)
    X = X.apply(pd.to_numeric, errors='coerce')
    y = raw['class'].astype(str).str.strip().str.lower().map({'ckd':1, 'notckd':0}).astype(int)
    return X, y

def load_liver():
    raw = source_csv(225)
    X = raw.drop(columns=['Selector']).copy()
    X.columns = ['Age','Gender','TB','DB','Alkphos','Sgpt','Sgot','TP','ALB','AG_ratio']
    X['Gender'] = X['Gender'].astype(str).str.strip().str.lower().map({'male':1,'female':0})
    X = X.apply(pd.to_numeric, errors='coerce')
    y = (raw['Selector'].astype(int) == 1).astype(int)
    keep = ~X.duplicated()
    return X[keep], y[keep]

def model(class_weight=None):
    return RandomForestClassifier(n_estimators=200, class_weight=class_weight,
                                  random_state=42, n_jobs=-1)

def metric_row(y, p, pred):
    return {'accuracy':float(accuracy_score(y,pred)), 'sensitivity':float(recall_score(y,pred)),
            'specificity':float(recall_score(y,pred,pos_label=0)),
            'roc_auc':float(roc_auc_score(y,p)), 'pr_auc':float(average_precision_score(y,p)),
            'f1':float(f1_score(y,pred)), 'brier':float(brier_score_loss(y,p)),
            'n_test':int(len(y)), 'confusion_matrix':confusion_matrix(y,pred).tolist()}

def fit_one(name, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2,
        stratify=y, random_state=42)
    columns = list(X_train.columns)
    if name == 'ckd':
        bin_cols = [c for c in columns if set(X_train[c].dropna().unique()) <= {0,1}]
        log_cols = []
        imputer_name, strategy, class_weight = 'mice', 'smote', None
        target_sens = .97
        high_cut = .80
        grids = {'model__n_estimators':[200,300,500], 'model__max_depth':[None,3,5,8],
                 'model__min_samples_leaf':[1,2,4], 'model__max_features':['sqrt','log2']}
        def build(m, sampler=False):
            return build_pipeline(m, imputer_name, bin_idx, cont_idx,
                                  sampler='smote' if sampler else None)
    else:
        bin_cols = ['Gender']; log_cols = ['TB','DB','Alkphos','Sgpt','Sgot']
        imputer_name, strategy, class_weight = 'mice', 'class_weight', 'balanced'
        target_sens = .90
        grids = {'model__n_estimators':[200,300,500], 'model__max_depth':[None,3,5,8],
                 'model__min_samples_leaf':[1,2,4,8], 'model__max_features':['sqrt','log2']}
        def build(m, sampler=False):
            return build_liver_pipeline(m, imputer_name, bin_idx, cont_idx, log_idx,
                                        sampler='smote' if sampler else None)
    bin_idx = [columns.index(c) for c in bin_cols]
    cont_cols = [c for c in columns if c not in bin_cols]
    cont_idx = [columns.index(c) for c in cont_cols]
    log_idx = [columns.index(c) for c in log_cols]
    features_order = feature_order(columns, cont_idx, bin_idx)
    base = build(model(class_weight), sampler=(strategy=='smote'))
    n_iter = int(min(20, np.prod([len(v) for v in grids.values()])))
    search = RandomizedSearchCV(base, grids, n_iter=n_iter,
        scoring={'roc_auc':'roc_auc','sensitivity':'recall'}, refit='roc_auc',
        cv=RepeatedStratifiedKFold(n_splits=5,n_repeats=2,random_state=42),
        random_state=42,n_jobs=1)
    search.fit(X_train,y_train)
    tuned_pipe = search.best_estimator_

    outer = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    oof = {'uncalibrated':cross_val_predict(clone(tuned_pipe),X_train,y_train,cv=outer,
                                            method='predict_proba')[:,1]}
    for method in ('sigmoid','isotonic'):
        calibrated = CalibratedClassifierCV(clone(tuned_pipe),method=method,cv=5)
        oof[method] = cross_val_predict(calibrated,X_train,y_train,cv=outer,
                                       method='predict_proba')[:,1]
    briers = {key:brier_score_loss(y_train,prob) for key,prob in oof.items()}
    calibration = min(('sigmoid','isotonic'),key=lambda key:briers[key])
    probs = oof[calibration]
    fpr,tpr,thresholds = roc_curve(y_train,probs)
    eligible = np.where(tpr >= target_sens)[0]
    threshold = float(thresholds[eligible[np.argmin(fpr[eligible])]])
    if name == 'ckd': high_cut = .80
    else: high_cut = float(min(.95,max(.80,threshold+.15)))
    final_model = CalibratedClassifierCV(clone(tuned_pipe),method=calibration,cv=5).fit(X_train,y_train)

    probabilities = final_model.predict_proba(X_test)[:,1]
    predictions = (probabilities >= threshold).astype(int)
    metrics = metric_row(y_test,probabilities,predictions)
    metrics['threshold'] = threshold
    rng=np.random.default_rng(42); boot=[]
    y_array=np.asarray(y_test); pred_array=np.asarray(predictions); p_array=np.asarray(probabilities)
    for _ in range(1000):
        idx=rng.integers(0,len(y_array),len(y_array))
        if len(np.unique(y_array[idx]))<2: continue
        boot.append(metric_row(y_array[idx],p_array[idx],pred_array[idx]))
    for key in ('accuracy','sensitivity','specificity','roc_auc','pr_auc','f1','brier'):
        values=[row[key] for row in boot]
        metrics[key+'_ci']=[float(np.percentile(values,2.5)),float(np.percentile(values,97.5))]

    model_step=tuned_pipe.named_steps['model']
    Xt = X_train.copy()
    # SHAP is calculated lazily for individual uploaded reports by the app.
    artifact={'model':final_model,'explain_pipe':tuned_pipe,'model_name':'RandForest',
        'features':columns,'feature_order':features_order,'log_cols':log_cols,
        'threshold':threshold,'high_cut':high_cut,
        'train_min':X_train.min().to_dict(),'train_max':X_train.max().to_dict(),
        'train_median':X_train.median().to_dict(),
        'training':{'imputer':imputer_name,'strategy':strategy,'calibration':calibration,
                    'best_params':search.best_params_,'target_sensitivity':target_sens,
                    'cv_roc_auc':float(search.best_score_),'calibration_brier':{k:float(v) for k,v in briers.items()},
                    'train_rows':int(len(X_train))}}
    joblib.dump(artifact,OUT/f'{name}.joblib')
    info={'metrics':metrics,'threshold':threshold,'high_cut':high_cut,
          'features':columns,'feature_order':features_order,
          'medians':X_train.median().fillna(0).to_dict(),
          'training':artifact['training']}
    (OUT/f'{name}.json').write_text(json.dumps(info,indent=2))
    print(name,json.dumps({'training':artifact['training'],'metrics':metrics},indent=2))

if __name__ == '__main__':
    fit_one('ckd',*load_ckd())
    fit_one('liver',*load_liver())
