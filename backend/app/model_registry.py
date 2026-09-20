"""
Loads the artifacts produced by the training notebook (Section 12) once, at process
startup, and holds them in memory for the life of the app -- avoids re-reading the
.joblib files from disk on every request.
"""
import json
import os
from pathlib import Path

import joblib
import numpy as np

ARTIFACT_DIR = Path(
    os.getenv("AGROPREDICT_ARTIFACT_DIR", Path(__file__).resolve().parent.parent / "artifacts")
)


class ModelRegistry:
    """
    Call .load() once at app startup (see main.py's lifespan handler).
    All other modules import the shared `registry` instance below and read from it --
    never re-instantiate this class.
    """

    def __init__(self):
        self.pipeline = None
        self.label_encoder = None
        self.metadata = None
        self.crop_to_cluster = None

    def load(self):
        pipeline_path = ARTIFACT_DIR / "model_final_pipeline.joblib"
        encoder_path = ARTIFACT_DIR / "target_label_encoder.joblib"
        metadata_path = ARTIFACT_DIR / "deploy_metadata.json"

        for p in (pipeline_path, encoder_path, metadata_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"Expected artifact not found: {p}. "
                    f"Copy model_final_pipeline.joblib, target_label_encoder.joblib, and "
                    f"deploy_metadata.json from your Colab DATA_DIR into {ARTIFACT_DIR}/ "
                    f"(or set AGROPREDICT_ARTIFACT_DIR to point at them)."
                )

        self.pipeline = joblib.load(pipeline_path)
        self.label_encoder = joblib.load(encoder_path)
        with open(metadata_path) as f:
            self.metadata = json.load(f)
        self.crop_to_cluster = self.metadata.get("crop_to_cluster", {})

        print(
            f"[model_registry] Loaded model. Classes: {len(self.label_encoder.classes_)}. "
            f"Held-out top5_acc: {self.metadata.get('test_top5_acc')}, "
            f"top5_cluster_acc: {self.metadata.get('test_top5_cluster_acc')}"
        )

    def is_ready(self) -> bool:
        return self.pipeline is not None

    def _add_engineered_features(self, row: dict) -> dict:
        """
        Mirrors training notebook Section 2 EXACTLY. The trained pipeline expects these
        derived columns as real input features, not raw N/P/K/temperature/rainfall alone --
        if this drifts from the notebook's feature engineering, predictions will be wrong
        (or the ColumnTransformer will simply raise a missing-column error).
        """
        row = dict(row)  # don't mutate the caller's dict
        eps = 1e-3
        n, p, k = row["nitrogen_N_kg_ha"], row["phosphorus_P_kg_ha"], row["potassium_K_kg_ha"]
        row["npk_sum"] = n + p + k
        row["n_p_ratio"] = n / (p + eps)
        row["n_k_ratio"] = n / (k + eps)
        row["p_k_ratio"] = p / (k + eps)
        row["aridity_index"] = row["rainfall_mm"] / (row["temperature_C"] + eps)
        return row

    def predict_shortlist(self, row: dict, k: int = 5):
        """
        row: dict of the RAW feature columns (before engineered features are added) --
        i.e. exactly what the API's Pydantic schemas collect from the caller / weather API.
        Returns a list of (crop_name, probability) tuples, sorted by probability desc.
        """
        import pandas as pd

        row = self._add_engineered_features(row)
        features = self.metadata["features"]
        X = pd.DataFrame([{col: row[col] for col in features}])
        proba = self.pipeline.predict_proba(X)[0]

        top_idx = np.argsort(proba)[-k:][::-1]
        return [(self.label_encoder.classes_[i], float(proba[i])) for i in top_idx]

    def cluster_alternatives(self, crop: str, exclude: set) -> list:
        cluster_id = self.crop_to_cluster.get(crop)
        if cluster_id is None:
            return []
        return [
            c for c, cl in self.crop_to_cluster.items()
            if cl == cluster_id and c != crop and c not in exclude
        ][:5]


registry = ModelRegistry()
