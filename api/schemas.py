# ============================================================
# schemas.py
# Modèles Pydantic — Request/Response
# Predictive Maintenance AIoT Pipeline — Étape 6
# ============================================================

from typing import List, Optional
from pydantic import BaseModel, Field


# ── Auth ──────────────────────────────────────────────────────────────────────
class Token(BaseModel):
    access_token : str
    token_type   : str
    expires_in   : int


class TokenData(BaseModel):
    username : Optional[str] = None


# ── Prédiction ────────────────────────────────────────────────────────────────
class SensorInput(BaseModel):
    """Données capteurs pour une prédiction."""
    machine_id          : str   = Field(..., example="MACHINE_003")
    air_temperature     : float = Field(..., example=298.5,  description="Température air (Kelvin)")
    process_temperature : float = Field(..., example=308.2,  description="Température process (Kelvin)")
    rotational_speed    : float = Field(..., example=1408.0, description="Vitesse rotation (RPM)")
    torque              : float = Field(..., example=46.3,   description="Couple (Nm)")
    tool_wear           : float = Field(..., example=108.0,  description="Usure outil (min)")
    machine_type        : str   = Field("M", example="M",   description="Type machine : L, M ou H")
    criticality         : str   = Field("MEDIUM", example="MEDIUM", description="HIGH, MEDIUM ou LOW")


class PredictionFactor(BaseModel):
    feature    : str
    shap_value : float
    direction  : str


class PredictionResponse(BaseModel):
    """Résultat complet d'une prédiction."""
    machine_id               : str
    timestamp                : str
    xgboost_prediction       : str
    xgboost_confidence       : float
    isolation_forest_anomaly : bool
    anomaly_score            : float
    is_alert                 : bool
    recommendation           : str
    top_factors              : List[PredictionFactor]
    criticality              : str


# ── Batch ─────────────────────────────────────────────────────────────────────
class BatchSensorInput(BaseModel):
    """Batch de plusieurs machines."""
    machines : List[SensorInput]


class BatchPredictionResponse(BaseModel):
    total      : int
    alerts     : int
    predictions: List[PredictionResponse]


# ── Machine Status ────────────────────────────────────────────────────────────
class MachineStatus(BaseModel):
    machine_id          : str
    criticality         : str
    last_prediction     : Optional[str]
    last_confidence     : Optional[float]
    last_seen           : Optional[str]
    alert_count_1h      : int
    status              : str  # NORMAL, WARNING, CRITICAL


# ── Alerts ────────────────────────────────────────────────────────────────────
class AlertItem(BaseModel):
    timestamp   : str
    machine_id  : str
    criticality : str
    confidence  : float
    recommendation : str
    top_factor_1   : Optional[str]
    top_factor_2   : Optional[str]
    top_factor_3   : Optional[str]


class AlertsResponse(BaseModel):
    total  : int
    alerts : List[AlertItem]


# ── Model Health ──────────────────────────────────────────────────────────────
class ModelHealth(BaseModel):
    status           : str
    xgboost_f1       : Optional[float]
    xgboost_auc      : Optional[float]
    dataset_drift    : Optional[bool]
    drifted_features : Optional[List[str]]
    last_retrain     : Optional[str]
    model_version    : str


# ── Health Check ──────────────────────────────────────────────────────────────
class HealthCheck(BaseModel):
    status    : str
    version   : str
    timestamp : str
    services  : dict
