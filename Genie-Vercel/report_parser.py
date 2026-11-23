"""Map common lab report text to the selected notebook model's raw features."""
import re

NUMBER = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?")
PATTERNS = {
 'ckd': {
  'age': (['age'], 'number'), 'bp': (['blood pressure', 'bp'], 'number'),
  'sg': (['specific gravity', 'sp gravity', 'sg'], 'number'),
  'al': (['urine albumin', 'albumin in urine', 'albumin'], 'number'),
  'su': (['urine sugar', 'urine glucose', 'sugar'], 'number'),
  'rbc': (['urine red blood cells', 'urine rbc', 'red blood cells', 'rbc morphology', 'rbc'], 'rbc'),
  'pc': (['pus cells', 'urine pus cells'], 'rbc'),
  'pcc': (['pus cell clumps', 'pus cell clump'], 'presence'),
  'ba': (['bacteria', 'urine bacteria'], 'presence'),
  'bgr': (['blood glucose random', 'random blood glucose', 'blood glucose', 'glucose'], 'number'),
  'bu': (['blood urea', 'urea'], 'number'),
  'sc': (['serum creatinine', 'creatinine'], 'number'),
  'sod': (['serum sodium', 'sodium', 'na'], 'number'), 'pot': (['serum potassium', 'potassium', 'k'], 'number'),
  'hemo': (['hemoglobin', 'haemoglobin', 'hb'], 'number'),
  'pcv': (['packed cell volume', 'hematocrit', 'haematocrit', 'pcv'], 'number'),
  'wbcc': (['white blood cell count', 'white blood cells', 'total wbc count', 'leukocyte count', 'wbc'], 'number'),
  'rbcc': (['red blood cell count', 'red blood cells count', 'rbc count'], 'number'),
  'htn': (['hypertension', 'high blood pressure'], 'yesno'), 'dm': (['diabetes mellitus', 'diabetes'], 'yesno'),
  'cad': (['coronary artery disease', 'cad'], 'yesno'), 'appet': (['appetite'], 'appetite'),
  'pe': (['pedal edema', 'oedema', 'edema'], 'yesno'), 'ane': (['anemia', 'anaemia'], 'yesno'),
 },
 'liver': {
  'Age': (['age'], 'number'), 'Gender': (['gender', 'sex'], 'gender'),
  'TB': (['total bilirubin', 'bilirubin total'], 'number'), 'DB': (['direct bilirubin', 'conjugated bilirubin'], 'number'),
  'Alkphos': (['alkaline phosphatase', 'alk phosphatase', 'alkphos'], 'number'),
  'Sgpt': (['sgpt', 'alt'], 'number'), 'Sgot': (['sgot', 'ast'], 'number'),
  'TP': (['total protein'], 'number'), 'ALB': (['serum albumin', 'albumin'], 'number'),
  'AG_ratio': (['a/g ratio', 'a:g ratio', 'albumin globulin ratio', 'ag ratio'], 'number'),
 }
}

def _after_label(text, aliases, kind):
    for alias in sorted(aliases, key=len, reverse=True):
        escaped = re.escape(alias).replace(r'\ ', r'\s+')
        pattern = re.compile(r'(?<![a-z])' + escaped + r'(?![a-z])', re.I)
        for match in pattern.finditer(text):
            tail = text[match.end():match.end()+60]
            # Avoid using a result from the next row if this row has no value.
            tail = re.split(r'\n\s*[A-Za-z][A-Za-z /().:-]{2,30}\s*[:|=]', tail, maxsplit=1)[0]
            # A label can appear in report headings or notes before its result row.
            # Keep looking until this occurrence actually has a value of the right type.
            if _decode(tail, kind) is None:
                continue
            return tail
    return ''

def _decode(tail, kind):
    low = tail.lower()
    if kind == 'number':
        m = NUMBER.search(tail)
        if m: return float(m.group().replace(',', ''))
        if re.match(r'\s*(?:nil|negative|absent|none)\b', low): return 0.0
        return None
    if kind == 'rbc':
        for token, value in [('abnormal', 'Abnormal'), ('normal', 'Normal')]:
            if re.search(r'\b'+token+r'\b', low): return value
    if kind == 'presence':
        if re.search(r'\bnot\s+present\b|\babsent\b|\bnegative\b', low): return 'Not present'
        if re.search(r'\bpresent\b|\bpositive\b', low): return 'Present'
    if kind == 'yesno':
        if re.search(r'\bno\b|\bnegative\b|\babsent\b', low): return 'No'
        if re.search(r'\byes\b|\bpositive\b|\bpresent\b', low): return 'Yes'
    if kind == 'appetite':
        if re.search(r'\bpoor\b', low): return 'Poor'
        if re.search(r'\bgood\b', low): return 'Good'
    if kind == 'gender':
        if re.search(r'\bfemale\b', low): return 'Female'
        if re.search(r'\bmale\b', low): return 'Male'
    return None

def extract_values(text, disease):
    if disease not in PATTERNS: raise ValueError('Choose CKD or liver before uploading a report.')
    values = {}
    for key, (aliases, kind) in PATTERNS[disease].items():
        tail = _after_label(text, aliases, kind)
        value = _decode(tail, kind)
        if value is not None: values[key] = value
    return values
