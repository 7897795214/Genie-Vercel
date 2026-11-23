from pathlib import Path
from io import BytesIO

from flask import Flask, jsonify, render_template, request
from pypdf import PdfReader
from werkzeug.exceptions import RequestEntityTooLarge

from model_service import Predictor
from report_parser import extract_values

ROOT = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(ROOT / 'templates'), static_folder=None)
# Vercel's standard function request body limit is about 4.5 MB.
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024
predictor = Predictor()

@app.get('/')
def home():
    return render_template('index.html')

@app.get('/api/health')
def health():
    return jsonify({'ready': predictor.ready, 'message': predictor.message})

@app.get('/api/meta/<disease>')
def meta(disease):
    if disease not in ('ckd', 'liver'): return jsonify(error='Unknown condition'), 404
    return jsonify(predictor.meta(disease))

@app.post('/api/parse')
def parse_report():
    f = request.files.get('report')
    disease = request.form.get('disease', 'ckd')
    if disease not in ('ckd', 'liver'):
        return jsonify(error='Choose CKD or liver before uploading a report.'), 400
    if not f or not f.filename or not f.filename.lower().endswith('.pdf'):
        return jsonify(error='Please choose a PDF report.'), 400
    try:
        content = f.read()
        if not content.startswith(b'%PDF-'):
            return jsonify(error='This file does not look like a valid PDF. Please choose a PDF report.'), 400
        reader = PdfReader(BytesIO(content))
        if reader.is_encrypted:
            return jsonify(error='This PDF is password-protected. Please upload an unlocked copy.'), 400
        text = '\n'.join((p.extract_text() or '') for p in reader.pages)
    except Exception:
        return jsonify(error='This PDF could not be read. Try a text-based PDF or enter values manually.'), 400
    found = extract_values(text, disease)
    if found:
        note = 'Review extracted values against the report before predicting. Missing values remain blank.'
    elif not text.strip():
        note = 'This looks like a scanned PDF. Automatic OCR is unavailable on the hosted version; enter values manually or upload a searchable PDF.'
    else:
        note = 'No matching lab values were found in the PDF. Enter the values manually and check the report.'
    return jsonify(values=found, extracted=len(found), pages=len(reader.pages),
                   ocr=False, note=note)

@app.errorhandler(RequestEntityTooLarge)
def too_large(_error):
    return jsonify(error='PDF is too large for this hosted demo. Please upload a file under 4 MB.'), 413

@app.post('/api/predict')
def predict():
    body = request.get_json(force=True)
    disease = body.get('disease')
    if disease not in ('ckd', 'liver'): return jsonify(error='Choose CKD or liver.'), 400
    try:
        result = predictor.predict(disease, body.get('values', {}))
    except ValueError as e: return jsonify(error=str(e)), 400
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5050, debug=False)
