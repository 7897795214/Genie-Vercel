"""Load the notebook-trained Genie models and calculate per-report SHAP drivers."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from pipeline_utils import preprocess

ROOT = Path(__file__).resolve().parent
FEATURES = {
 'ckd':['age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc','htn','dm','cad','appet','pe','ane'],
 'liver':['Age','Gender','TB','DB','Alkphos','Sgpt','Sgot','TP','ALB','AG_ratio']}
LABELS = {'age':'Age (years)','bp':'Blood pressure','sg':'Specific gravity','al':'Urine albumin','su':'Urine sugar',
 'rbc':'Red blood cells','pc':'Pus cells','pcc':'Pus cell clumps','ba':'Bacteria','bgr':'Blood glucose (mg/dL)',
 'bu':'Blood urea (mg/dL)','sc':'Serum creatinine (mg/dL)','sod':'Sodium (mEq/L)','pot':'Potassium (mEq/L)',
 'hemo':'Hemoglobin (g/dL)','pcv':'Packed cell volume (%)','wc':'White blood cells (/cmm)','rc':'Red blood cells (millions/cmm)',
 'wbcc':'White blood cells (/cmm)','rbcc':'Red blood cells (millions/cmm)',
 'htn':'Hypertension','dm':'Diabetes','cad':'Coronary artery disease','appet':'Appetite','pe':'Pedal edema','ane':'Anemia',
 'Age':'Age (years)','Gender':'Sex','TB':'Total bilirubin (mg/dL)','DB':'Direct bilirubin (mg/dL)',
 'Alkphos':'Alkaline phosphatase (U/L)','Sgpt':'ALT / SGPT (U/L)','Sgot':'AST / SGOT (U/L)',
 'TP':'Total protein (g/dL)','ALB':'Albumin (g/dL)','AG_ratio':'Albumin/globulin ratio'}
BINARY={'rbc':{'normal':0,'abnormal':1},'pc':{'normal':0,'abnormal':1},'pcc':{'notpresent':0,'present':1},
 'ba':{'notpresent':0,'present':1},'htn':{'no':0,'yes':1},'dm':{'no':0,'yes':1,'\tno':0,'\tyes':1},
 'cad':{'no':0,'yes':1,'\tno':0},'appet':{'good':0,'poor':1},'pe':{'no':0,'yes':1},'ane':{'no':0,'yes':1}}
OPTIONS={'rbc':['Normal','Abnormal'],'pc':['Normal','Abnormal'],'pcc':['Not present','Present'],
 'ba':['Not present','Present'],'htn':['No','Yes'],'dm':['No','Yes'],'cad':['No','Yes'],
 'appet':['Good','Poor'],'pe':['No','Yes'],'ane':['No','Yes']}

class Predictor:
 def __init__(self):
  self.models={}; self.explain_pipes={}; self.info={}; self.ready=False
  self.message='Loading notebook-trained models…'
  try:
   model_dir=ROOT/'models'
   if not all((model_dir/f'{d}.joblib').exists() and (model_dir/f'{d}.json').exists() for d in ('ckd','liver')):
    raise FileNotFoundError('Model artifacts are missing. Include models/ckd.joblib, models/liver.joblib, and their JSON files in the deployment.')
   for disease in ('ckd','liver'):
    artifact=joblib.load(model_dir/f'{disease}.joblib')
    self.models[disease]=artifact['model']
    self.explain_pipes[disease]=artifact['explain_pipe']
    self.info[disease]=json.loads((model_dir/f'{disease}.json').read_text())
   self.ready=True; self.message='Models ready'
  except Exception as e:
   self.message=f'Model setup failed: {e}'

 def meta(self,disease):
  info=self.info[disease]
  return {'features':[{'key':k,'label':LABELS.get(k,k),'type':'select' if k in BINARY or k=='Gender' else 'number',
       'options':(['Female','Male'] if k=='Gender' else OPTIONS.get(k,[])), 'min':0,'step':'any'}
       for k in info['features']],
       'metrics':info['metrics'],'threshold':info['threshold'],'high_cut':info['high_cut'],
       'model':info['training'],'dataset':'UCI held-out split; trained with the AI_H.ipynb pipeline'}

 def predict(self,disease,values):
  info=self.info[disease]; row={}
  for key in info['features']:
   value=values.get(key)
   if value in ('',None): value=info['medians'].get(key,0)
   elif key in BINARY or key=='Gender':
    token=str(value).strip().lower().replace(' ','')
    mapping=BINARY.get(key,{'female':0,'male':1})
    if token not in mapping: raise ValueError(f'Invalid value for {LABELS.get(key,key)}')
    value=mapping[token]
   else:
    try:
     value=float(value)
     if not np.isfinite(value) or value<0: raise ValueError
    except (TypeError,ValueError): raise ValueError(f'Enter a non-negative number for {LABELS.get(key,key)}')
   row[key]=value
  frame=pd.DataFrame([row],columns=info['features'])
  risk=float(self.models[disease].predict_proba(frame)[0,1])
  pipe=self.explain_pipes[disease]
  transformed=preprocess(pipe,frame)
  values_shap=shap.TreeExplainer(pipe.named_steps['model']).shap_values(transformed,check_additivity=False)
  if isinstance(values_shap,list): values_shap=values_shap[1]
  elif np.ndim(values_shap)==3: values_shap=values_shap[:,:,1]
  contribution=np.asarray(values_shap)[0]
  drivers=[{'key':key,'label':LABELS.get(key,key),'value':row.get(key),
            'impact':round(float(impact),5)}
           for key,impact in zip(info['feature_order'],contribution)]
  drivers=sorted(drivers,key=lambda item:abs(item['impact']),reverse=True)[:6]
  threshold=float(info['threshold']); high_cut=float(info['high_cut'])
  band='High' if risk>=high_cut else ('Elevated' if risk>=threshold else 'Lower')
  return {'risk':round(risk,4),'band':band,'threshold':threshold,'drivers':drivers,'inputs':row,
          'disclaimer':'Educational screening estimate only. This is not a diagnosis or a substitute for clinical review.'}
