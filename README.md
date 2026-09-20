# AgroPredict

Crop recommendation for Indian agriculture, built end-to-end: EDA → model training →
FastAPI backend → web UI. Given soil test readings and climate conditions, it returns a
ranked shortlist of suitable crops — not a single answer, for reasons explained below.

**[Live demo](#) · [API docs](#) · [Notebook walkthrough](#)** — *(fill in once deployed, see "Deployment" below)*

---

## The interesting part: why this is a shortlist, not a single prediction

The first version of this project trained a standard 140-class classifier (Random Forest,
XGBoost, LightGBM, SVC, KNN, Logistic Regression, Decision Tree) and every single model —
regardless of algorithm family — plateaued around the same low accuracy. That pattern (identical
weakness across completely different model types) ruled out a bug and pointed at the data itself.

Digging into `crop_profiles.csv`, I quantified pairwise overlap between every pair of crops'
documented temperature, rainfall, and pH ranges. The result: **127 of 140 crops (91%) have at
least one agro-climatically near-identical "twin,"** clustering into 21 groups — one as large as
56 crops (mostly millets, pulses, and oilseeds with near-identical growing requirements). Several
pairs — beetroot/spinach, tomato/bitter gourd/bottle gourd, sweet orange/mandarin — have **100%
overlapping** stated requirement ranges. No feature set derived from this data can reliably tell
those crops apart, because the source data doesn't distinguish them either.

**The fix wasn't a better model — it was a better-matched task.** Instead of forcing an exact
single-crop prediction, the final model returns a **ranked top-5 shortlist with probabilities**,
evaluated with a cluster-aware metric (`top5_cluster_acc`: did the shortlist land in the right
agro-climatic neighborhood, even if not the exact crop). That number came out around **92%** on
the held-out test set, versus **~45% for exact-match top-5 accuracy** — evidence the model has
genuinely learned the real structure in the data, even though "guess the one exact crop out of
140" was never a fully answerable question from these features alone.

---

## Architecture

```
                     ┌────────────────────────────┐
                     │   Training (Colab)         │
                     │   AgroPredict_V4_final     │
                     │   .ipynb                   │
                     │                            │
                     │  EDA → feature engineering │
                     │  → crop-cluster analysis   │
                     │  → 7-model comparison      │
                     │  → XGBoost tuning          │
                     │  → SHAP explainability     │
                     └───────────┬────────────────┘
                                 │ exports
                 ┌───────────────┼────────────────┐
                 ▼               ▼                ▼
    model_final_pipeline   target_label      deploy_metadata
        .joblib             encoder.joblib      .json
                 │               │                │
                 └───────────────┼────────────────┘
                                 ▼
                     ┌───────────────────────────┐
                     │  FastAPI backend          │
                     │  /predict                 │
                     │  /predict/auto (weather)  │
                     │  /options, /health        │
                     └───────────┬───────────────┘
                                 │ calls
                                 ▼             ┌───────────────────┐
                     ┌─────────────────────────┤  OpenWeather API  │
                     │  fetch()                └───────────────────┘
                     ▼
                     ┌───────────────────────────┐
                     │  Frontend (static HTML)   │
                     │  frontend/index.html      │
                     │  soil form → shortlist UI │
                     └───────────────────────────┘
```

---

## Repo structure

```
.
├── README.md                     <- you are here
├── notebooks/
│   └── AgroPredict_V4_final.ipynb    <- training, from raw data to saved model
├── backend/
│   ├── app/
│   │   ├── main.py                <- FastAPI app + endpoints
│   │   ├── schemas.py             <- Pydantic request/response models
│   │   ├── model_registry.py      <- loads the trained pipeline + feature engineering
│   │   └── weather.py             <- OpenWeather integration
│   ├── artifacts/                 <- model_final_pipeline.joblib, label encoder, metadata
│   │                                  (not committed — see "Getting the model file" below)
│   ├── requirements.txt
│   └── README.md                  <- backend-specific setup detail
└── frontend/
    └── index.html                 <- single-file UI, calls the backend directly
```

---

## Results

*(Fill in with your actual final numbers from Section 10 of the notebook once the tuning run
finishes — these are the metrics the deploy_metadata.json also stores.)*

| Metric | Value |
|---|---|
| Held-out macro-F1 | `___` |
| Held-out top-3 accuracy (exact crop) | `___` |
| Held-out top-5 accuracy (exact crop) | `___` |
| Held-out top-5 cluster accuracy | `___` |

Winning model: **XGBoost**, tuned via checkpointed `RandomizedSearchCV` over 10 candidates,
3-fold CV. Full comparison against Random Forest, LightGBM, Decision Tree, Logistic Regression,
KNN, and SVC is in the notebook, Section 7–8.

---

## Running it locally

### 1. Train (or reuse the trained model)

Open `notebooks/AgroPredict_V4_final.ipynb` in Google Colab, point `DATA_DIR` at your dataset
folder, and run top to bottom. Every long-running cell (baseline comparison, hyperparameter
search) is checkpointed to `.jsonl` files, so a disconnect only costs you the in-progress step,
not the whole run. This produces `model_final_pipeline.joblib`, `target_label_encoder.joblib`,
and `deploy_metadata.json`.

### 2. Backend

```bash
cd backend
cp .env.example .env        # add your OpenWeather API key
pip install -r requirements.txt
# copy the 3 files from step 1 into backend/artifacts/
uvicorn app.main:app --reload
```

Visit `http://127.0.0.1:8000/docs` for interactive API docs.

### 3. Frontend

Open `frontend/index.html` directly in a browser (or serve it with `python -m http.server`
from the `frontend/` folder if you hit CORS issues with `file://`). It talks to the backend at
`http://127.0.0.1:8000` by default — change `API_BASE_URL` near the top of the `<script>` tag
if you deploy the backend elsewhere.

### Getting the model file

`model_final_pipeline.joblib` isn't committed to this repo (model files don't belong in git
history). Either:
- run the notebook yourself and copy the output into `backend/artifacts/`, or
- *(if you host it somewhere)* download it from `___` and place it in `backend/artifacts/`.

---

## Model limitations (read before trusting a recommendation)

- **This model supports a decision, it doesn't replace one.** It reflects patterns in a training
  dataset, not live soil testing, market prices, water availability, or local extension advice.
- **Crops in the same "cluster" are often close to interchangeable** given the model's inputs —
  the shortlist including several visually different crops isn't noise, it's an honest reflection
  of overlapping agro-climatic requirements in the source data (see the diagnosis above).
- **Exact single-crop accuracy is real but modest (~`___`%)** — trust the shortlist, not the #1
  answer in isolation, especially for crops known to sit in a large cluster.
- **Model A (this deployment) intentionally excludes geography** (state, lat/long) to avoid
  learning "this state grows this crop" as a shortcut instead of real agronomic relationships.
  A geography-aware variant (Model B) was tested and added a modest accuracy gain (~+1.6pp top-3
  accuracy in testing) but isn't the default here.
- **Not validated against real-world outcomes.** The `real_indian_crop_validation_dataset.csv`
  (ICRISAT) lacks environmental features and was only usable as a rough plausibility check
  (predicted crop vs. real state/season crop presence), not an accuracy benchmark.

---

## Tech stack

- **Modeling:** Python, pandas, scikit-learn, XGBoost, SHAP
- **Backend:** FastAPI, Pydantic, joblib, httpx (OpenWeather integration)
- **Frontend:** plain HTML/CSS/JS, no framework
- **Data:** 100,000-row synthetic Indian crop dataset, 140 crop classes, 10 raw features
  (N/P/K, temperature, humidity, rainfall, soil pH, soil moisture, soil type, season)

## Acknowledgments

Training dataset and crop-profile reference ranges are synthetic / aggregated from public
agricultural sources; treat absolute values as indicative, not authoritative agronomic guidance.
