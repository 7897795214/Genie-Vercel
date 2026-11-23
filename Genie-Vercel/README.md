# Genie · Vercel deployment package

This folder is a Vercel-ready deployment copy of Genie. It keeps the existing notebook-trained `models/ckd.joblib` and `models/liver.joblib` artifacts unchanged. Prediction requests load those saved models; the Vercel app does not retrain them. The notebook is preserved under `reference/` as project documentation and is excluded from deployment uploads.

## Deploy using the GitHub import flow

1. Upload the **contents of this folder** to the root of a GitHub repository. The repository root should contain `app.py`, `requirements.txt`, `vercel.json`, `models/`, `templates/`, and `public/`.
2. In Vercel, select **Add New → Project**, then import that GitHub repository.
3. If Vercel asks for a root directory, choose the directory containing `app.py` (usually `.`).
4. Let Vercel detect the Python/Flask app. Leave the build command and output directory empty/default; Vercel installs packages from `requirements.txt` and discovers the Flask `app` object in `app.py`.
5. Click **Deploy**. When it finishes, open the generated `*.vercel.app` URL.
6. Check `https://YOUR-SITE.vercel.app/api/health`. It should return `{"ready":true,"message":"Models ready"}`.

## Files the deployment needs

- `app.py`, `model_service.py`, `pipeline_utils.py`, `report_parser.py`
- `requirements.txt`, `vercel.json`, `.python-version`
- `models/ckd.joblib`, `models/ckd.json`, `models/liver.joblib`, `models/liver.json`
- `templates/index.html`
- `public/static/app.js`, `public/static/style.css`

The `reference/` folder contains the original notebook and a macOS OCR helper for reference; the helper cannot run in Vercel's Linux functions, and both files are excluded from the deployed bundle. `train_models.py` is included in the source folder for reference but excluded from deployment. The saved models are all the runtime needs.

## Vercel-specific behavior

- **Model predictions:** The notebook-saved CKD and liver artifacts are loaded as-is. No training occurs during a prediction or Vercel function startup.
- **PDFs:** Upload searchable/text PDFs up to 4 MB. Scanned image-only PDFs cannot be OCR'd on Vercel because the existing OCR helper uses macOS Vision. Use OCR on the PDF before upload or enter values manually. The report is sent to the hosted function for extraction and is not written to persistent storage by this app.
- **History:** The last 30 results are saved in the current browser's local storage. History is specific to that browser and deployed domain; it is not shared across devices and clearing browser storage removes it.
- **Function limits:** Vercel executes Flask through serverless functions. Requests and execution are subject to your Vercel plan's current limits.

## Model context

The performance cards report the supplied models' held-out dataset results. They are not guarantees about an individual person and do not establish clinical performance. Genie is an educational screening demo, not a medical device or diagnosis.
