"""
Pydantic schemas for AgroPredict.

Categorical fields (soil_type, season) are validated against deploy_metadata.json
(produced by the training notebook, Section 12) rather than hardcoded here -- so if
you retrain with different categories, the API picks them up without a code change.
"""
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator

from .model_registry import registry


class CropConditionsInput(BaseModel):
    """Manual input: every feature the model needs, provided directly by the caller."""

    nitrogen_N_kg_ha: float = Field(..., ge=0, le=500, description="Nitrogen, kg/ha")
    phosphorus_P_kg_ha: float = Field(..., ge=0, le=500, description="Phosphorus, kg/ha")
    potassium_K_kg_ha: float = Field(..., ge=0, le=500, description="Potassium, kg/ha")
    temperature_C: float = Field(..., ge=-10, le=60, description="Temperature, degrees C")
    humidity_percent: float = Field(..., ge=0, le=100, description="Relative humidity, %")
    rainfall_mm: float = Field(..., ge=0, le=5000, description="Rainfall, mm")
    soil_pH: float = Field(..., ge=0, le=14, description="Soil pH")
    soil_moisture_percent: float = Field(..., ge=0, le=100, description="Soil moisture, %")
    soil_type: str = Field(..., description="One of the trained soil types")
    season: str = Field(..., description="One of the trained seasons")

    @field_validator("soil_type")
    @classmethod
    def validate_soil_type(cls, v: str) -> str:
        valid = registry.metadata["categorical_options"]["soil_type"]
        if v not in valid:
            raise ValueError(f"soil_type must be one of {valid}, got {v!r}")
        return v

    @field_validator("season")
    @classmethod
    def validate_season(cls, v: str) -> str:
        valid = registry.metadata["categorical_options"]["season"]
        if v not in valid:
            raise ValueError(f"season must be one of {valid}, got {v!r}")
        return v


class CropConditionsAutoInput(BaseModel):
    """
    Auto-weather input: caller supplies soil readings + location; temperature/humidity/
    rainfall are looked up from OpenWeather instead of being typed in by hand.
    Provide EITHER city, OR both latitude and longitude.
    """

    nitrogen_N_kg_ha: float = Field(..., ge=0, le=500)
    phosphorus_P_kg_ha: float = Field(..., ge=0, le=500)
    potassium_K_kg_ha: float = Field(..., ge=0, le=500)
    soil_pH: float = Field(..., ge=0, le=14)
    soil_moisture_percent: float = Field(..., ge=0, le=100)
    soil_type: str
    season: str

    city: Optional[str] = Field(None, description="City name, e.g. 'Nagpur'")
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)

    @field_validator("soil_type")
    @classmethod
    def validate_soil_type(cls, v: str) -> str:
        valid = registry.metadata["categorical_options"]["soil_type"]
        if v not in valid:
            raise ValueError(f"soil_type must be one of {valid}, got {v!r}")
        return v

    @field_validator("season")
    @classmethod
    def validate_season(cls, v: str) -> str:
        valid = registry.metadata["categorical_options"]["season"]
        if v not in valid:
            raise ValueError(f"season must be one of {valid}, got {v!r}")
        return v


class CropSuggestion(BaseModel):
    crop: str
    probability: float
    cluster: str
    cluster_alternatives: List[str] = Field(
        default_factory=list,
        description="Other crops in this crop's agro-climatic cluster (near-identical growing needs)",
    )


class PredictionResponse(BaseModel):
    shortlist: List[CropSuggestion]
    weather_source: Optional[str] = Field(
        None, description="Present only for /predict/auto: 'openweather:<city or lat,lon>'"
    )
    note: str = (
        "This model returns a ranked shortlist, not a single answer -- crops sharing an "
        "agro-climatic cluster are often close to interchangeable given the input conditions."
    )
