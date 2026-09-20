# AgroPredict API

FastAPI backend serving the trained model from `AgroPredict_V4_final.ipynb`. Returns a
ranked top-5 crop shortlist (with probabilities and agro-climatic-cluster alternatives),
not a single answer -- see the training notebook for why.

## Setup

1. From your Colab `DATA_DIR`, download these three files:
   - `model_final_pipeline.joblib`
   - `target_label_encoder.joblib`
   - `deploy_metadata.json`

2. Put them in an `artifacts/` folder next to this README (or point
   `AGROPREDICT_ARTIFACT_DIR` at wherever you put them):
   ```
   agropredict_api/
     artifacts/
       model_final_pipeline.joblib
       target_label_encoder.joblib
       deploy_metadata.json
     app/
       ...
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Copy `.env.example` to `.env` and fill in your OpenWeather API key (only needed for
   `/predict/auto`; `/predict` works without it):
   ```bash
   cp .env.example .env
   # edit .env
   export $(cat .env | xargs)   # or use python-dotenv / your platform's env var settings
   ```

5. Run it:
   ```bash
   uvicorn app.main:app --reload
   ```

6. Open `http://127.0.0.1:8000/docs` for interactive Swagger docs, or check
   `http://127.0.0.1:8000/health` to confirm the model loaded correctly.

## Endpoints

- `POST /predict` -- all 10 features supplied manually. See `/docs` for the exact schema
  (soil_type/season are validated against whatever categories your model was trained on,
  read live from `deploy_metadata.json`).
- `POST /predict/auto` -- soil readings supplied manually, weather (`temperature_C`,
  `humidity_percent`, `rainfall_mm`) looked up automatically via OpenWeather using either
  a `city` name or `latitude`/`longitude`.
- `GET /health` -- readiness check, also returns the model's held-out test metrics.
- `GET /crop-clusters` -- the full crop -> agro-climatic-cluster map, useful if your
  frontend wants to explain *why* certain alternatives are suggested.

## Example request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "nitrogen_N_kg_ha": 90,
    "phosphorus_P_kg_ha": 42,
    "potassium_K_kg_ha": 43,
    "temperature_C": 26.5,
    "humidity_percent": 80,
    "rainfall_mm": 220,
    "soil_pH": 6.5,
    "soil_moisture_percent": 45,
    "soil_type": "loamy",
    "season": "kharif"
  }'
```

(Replace `soil_type`/`season` values with ones actually present in your trained model --
check `GET /health` or the Swagger docs for the valid list.)

## Notes for the frontend

- CORS is currently wide open (`allow_origins=["*"]`) so you can develop against it from
  anywhere. Tighten this to your actual frontend's origin before deploying publicly.
- The response's `shortlist` is already sorted by probability, descending -- render it
  as-is; no need to re-sort client-side.
- Each shortlist entry includes `cluster_alternatives` -- crops with near-identical
  agro-climatic requirements to that entry. Good candidates for a "similar options" UI
  element, directly reflecting what the training analysis found about this dataset.
