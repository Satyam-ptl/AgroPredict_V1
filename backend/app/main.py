"""
AgroPredict API

Endpoints:
  POST /predict        -- all features supplied manually
  POST /predict/auto    -- soil features manual, weather auto-filled via OpenWeather
  GET  /health           -- readiness check + model metrics
  GET  /crop-clusters     -- the full crop -> agro-climatic-cluster map

Run locally:
  uvicorn app.main:app --reload

The frontend (plain HTML/CSS/JS) calls this API directly -- see the `allow_origins`
CORS setting below, which you should tighten from "*" once you know your frontend's
real origin.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .model_registry import registry
from .schemas import CropConditionsInput, CropConditionsAutoInput, PredictionResponse, CropSuggestion
from .weather import resolve_weather
from dotenv import load_dotenv
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    registry.load()  # loads once, kept in memory for the life of the process
    yield


app = FastAPI(
    title="AgroPredict API",
    description="Recommends a ranked shortlist of suitable crops for given soil and climate conditions.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: replace with your actual frontend origin before going live
    allow_methods=["*"],
    allow_headers=["*"],
)


def _build_shortlist(row: dict) -> list[CropSuggestion]:
    top5 = registry.predict_shortlist(row, k=5)
    predicted_crops = {crop for crop, _ in top5}
    shortlist = []
    for crop, prob in top5:
        cluster_id = registry.crop_to_cluster.get(crop, "unknown")
        alternatives = registry.cluster_alternatives(crop, exclude=predicted_crops)
        shortlist.append(
            CropSuggestion(crop=crop, probability=round(prob, 4), cluster=cluster_id, cluster_alternatives=alternatives)
        )
    return shortlist


@app.get("/health")
def health():
    if not registry.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "ok",
        "n_classes": len(registry.label_encoder.classes_),
        "test_macro_f1": registry.metadata.get("test_macro_f1"),
        "test_top3_acc": registry.metadata.get("test_top3_acc"),
        "test_top5_acc": registry.metadata.get("test_top5_acc"),
        "test_top5_cluster_acc": registry.metadata.get("test_top5_cluster_acc"),
    }


@app.get("/options")
def options():
    """Valid soil_type/season values for this model -- the frontend builds its dropdowns from this."""
    return registry.metadata["categorical_options"]


@app.get("/crop-clusters")
def crop_clusters():
    """Full crop -> cluster map, e.g. for a frontend 'why these alternatives?' explainer."""
    return registry.crop_to_cluster


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: CropConditionsInput):
    row = payload.model_dump()
    shortlist = _build_shortlist(row)
    return PredictionResponse(shortlist=shortlist)


@app.post("/predict/auto", response_model=PredictionResponse)
async def predict_auto(payload: CropConditionsAutoInput):
    weather = await resolve_weather(payload.city, payload.latitude, payload.longitude)

    row = {
        "nitrogen_N_kg_ha": payload.nitrogen_N_kg_ha,
        "phosphorus_P_kg_ha": payload.phosphorus_P_kg_ha,
        "potassium_K_kg_ha": payload.potassium_K_kg_ha,
        "soil_pH": payload.soil_pH,
        "soil_moisture_percent": payload.soil_moisture_percent,
        "soil_type": payload.soil_type,
        "season": payload.season,
        "temperature_C": weather.temperature_C,
        "humidity_percent": weather.humidity_percent,
        "rainfall_mm": weather.rainfall_mm,
    }
    shortlist = _build_shortlist(row)
    return PredictionResponse(shortlist=shortlist, weather_source=weather.source)
