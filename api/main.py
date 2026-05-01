# ============================================================
# main.py
# API REST FastAPI — Predictive Maintenance AIoT Pipeline
# Étape 6 : API sécurisée JWT
#
# Endpoints :
#   POST /auth/token              → Obtenir un token JWT
#   POST /predict                 → Prédiction sur données capteurs
#   POST /predict/batch           → Prédiction sur batch
#   GET  /machines/{id}/status    → Statut d'une machine
#   GET  /alerts/recent           → Dernières alertes
#   GET  /model/health            → Santé du modèle
#   GET  /health                  → Health check API
# ============================================================

import os
import sys
import joblib
import logging
from datetime import datetime, timezone
from typing import List

import numpy as np
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware

# Ajouter le dossier ml_processor au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml_processor"))

from auth    import authenticate_user, create_access_token, get_current_user, TOKEN_EXPIRE_MINUTES
from schemas import (
    Token, SensorInput, PredictionResponse, PredictionFactor,
    BatchSensorInput, BatchPredictionResponse,
    MachineStatus, AlertItem, AlertsResponse,
    ModelHealth, HealthCheck,
)

# InfluxDB
from influxdb_client import InfluxDBClient

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s : %(message)s")
logger = logging.getLogger("FastAPI")

# ── Configuration ─────────────────────────────────────────────────────────────
MODELS_DIR   = os.getenv("MODELS_DIR",        "models")
INFLUX_URL   = os.getenv("INFLUXDB_URL",       "http://localhost:8086")
INFLUX_TOKEN = os.getenv("INFLUXDB_TOKEN",     "my-super-secret-token")
INFLUX_ORG   = os.getenv("INFLUXDB_ORG",       "aiot-org")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.8))

# ── Application FastAPI ───────────────────────────────────────────────────────
app = FastAPI(
    title       = "🏭 Predictive Maintenance API",
    description = "API REST pour le pipeline de maintenance prédictive AIoT",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ── Chargement des modèles au démarrage ───────────────────────────────────────
xgb_model     = None
iso_forest    = None
feature_names = None
shap_explainer = None

@app.on_event("startup")
async def load_models():
    global xgb_model, iso_forest, feature_names, shap_explainer
    try:
        import shap
        xgb_model     = joblib.load(os.path.join(MODELS_DIR, "xgboost_model.pkl"))
        iso_forest    = joblib.load(os.path.join(MODELS_DIR, "isolation_forest.pkl"))
        feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
        shap_explainer = shap.TreeExplainer(xgb_model)
        logger.info(f"✅ Modèles chargés : {len(feature_names)} features")
    except Exception as e:
        logger.error(f"❌ Erreur chargement modèles : {e}")


# ── Helper : Feature Engineering simplifié ───────────────────────────────────
def build_feature_vector(data: SensorInput) -> pd.DataFrame:
    """Construit le vecteur de features depuis les données brutes."""
    air_temp  = data.air_temperature
    proc_temp = data.process_temperature
    rot_speed = data.rotational_speed
    torque    = data.torque
    tool_wear = data.tool_wear

    type_map     = {"L": 1, "M": 2, "H": 0}
    type_encoded = type_map.get(data.machine_type, 2)

    temp_diff               = proc_temp - air_temp
    temp_ratio              = proc_temp / (air_temp + 1e-6)
    temp_variation          = 0.0
    power                   = torque * rot_speed
    wear_rate               = tool_wear / (rot_speed + 1e-6)
    torque_wear_interaction = torque * tool_wear
    thermal_efficiency      = power / (temp_diff + 1e-6)

    row = {
        "air_temp"                     : air_temp,
        "process_temp"                 : proc_temp,
        "rotational_speed"             : rot_speed,
        "torque"                       : torque,
        "tool_wear"                    : tool_wear,
        "Type_encoded"                 : type_encoded,
        "temp_diff"                    : temp_diff,
        "temp_variation"               : temp_variation,
        "temp_ratio"                   : temp_ratio,
        "power"                        : power,
        "wear_rate"                    : wear_rate,
        "torque_wear_interaction"      : torque_wear_interaction,
        "thermal_efficiency"           : thermal_efficiency,
        # Rolling/lag features → 0 (pas d'historique en mode API)
        **{f: 0.0 for f in feature_names
           if f not in ["air_temp", "process_temp", "rotational_speed",
                        "torque", "tool_wear", "Type_encoded",
                        "temp_diff", "temp_variation", "temp_ratio",
                        "power", "wear_rate", "torque_wear_interaction",
                        "thermal_efficiency"]}
    }
    return pd.DataFrame([row])[feature_names]


def get_recommendation(confidence: float, criticality: str, anomaly: bool) -> str:
    if confidence >= CONFIDENCE_THRESHOLD:
        if criticality == "HIGH":
            return "⛔ Arrêt immédiat requis — intervention urgente"
        elif criticality == "MEDIUM":
            return "⚠️  Planifier maintenance dans les 4 heures"
        else:
            return "📋 Planifier maintenance dans les 24 heures"
    elif anomaly:
        return "🔍 Anomalie détectée — surveillance renforcée requise"
    return "✅ Fonctionnement normal"


def run_prediction(data: SensorInput) -> PredictionResponse:
    """Fait une prédiction complète sur les données d'un capteur."""
    if xgb_model is None:
        raise HTTPException(status_code=503, detail="Modèles non chargés")

    X = build_feature_vector(data)

    # XGBoost
    confidence = float(xgb_model.predict_proba(X)[0][1])
    prediction = "FAILURE" if confidence >= CONFIDENCE_THRESHOLD else "NORMAL"

    # Isolation Forest
    iso_raw   = iso_forest.predict(X)[0]
    anomaly   = bool(iso_raw == -1)
    iso_score = float(-iso_forest.score_samples(X)[0])

    # SHAP
    top_factors = []
    if prediction == "FAILURE" or anomaly:
        shap_vals    = shap_explainer.shap_values(X)[0]
        top3_indices = np.argsort(np.abs(shap_vals))[-3:][::-1]
        top_factors  = [
            PredictionFactor(
                feature    = feature_names[i],
                shap_value = round(float(shap_vals[i]), 4),
                direction  = "→ PANNE" if shap_vals[i] > 0 else "→ NORMAL"
            )
            for i in top3_indices
        ]

    return PredictionResponse(
        machine_id               = data.machine_id,
        timestamp                = datetime.now(timezone.utc).isoformat(),
        xgboost_prediction       = prediction,
        xgboost_confidence       = round(confidence, 4),
        isolation_forest_anomaly = anomaly,
        anomaly_score            = round(iso_score, 4),
        is_alert                 = prediction == "FAILURE" or anomaly,
        recommendation           = get_recommendation(confidence, data.criticality, anomaly),
        top_factors              = top_factors,
        criticality              = data.criticality,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. Health Check (public) ──────────────────────────────────────────────────
@app.get("/health", response_model=HealthCheck, tags=["System"])
async def health_check():
    """Vérifie que l'API et ses dépendances sont opérationnelles."""
    services = {
        "models_loaded" : xgb_model is not None,
        "influxdb"      : "unknown",
    }
    try:
        client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
        client.ping()
        services["influxdb"] = "ok"
        client.close()
    except Exception:
        services["influxdb"] = "error"

    return HealthCheck(
        status    = "ok" if all(v != "error" for v in services.values()) else "degraded",
        version   = "1.0.0",
        timestamp = datetime.now(timezone.utc).isoformat(),
        services  = services,
    )


# ── 2. Auth — Obtenir un token JWT ───────────────────────────────────────────
@app.post("/auth/token", response_model=Token, tags=["Authentication"])
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Authentification et obtention d'un token JWT.

    **Credentials par défaut :**
    - username: `admin` ou `operator`
    - password: `secret`
    """
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail      = "Identifiants incorrects",
            headers     = {"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(data={"sub": user["username"]})
    logger.info(f"Login réussi : {user['username']}")
    return Token(
        access_token = token,
        token_type   = "bearer",
        expires_in   = TOKEN_EXPIRE_MINUTES * 60,
    )


# ── 3. Predict (JWT requis) ───────────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(
    data         : SensorInput,
    current_user : dict = Depends(get_current_user)
):
    """
    Prédit la probabilité de panne pour une machine.

    Retourne :
    - Prédiction XGBoost (FAILURE/NORMAL) + confiance
    - Détection anomalie Isolation Forest
    - Top 3 facteurs SHAP explicatifs
    - Recommandation selon criticité machine
    """
    logger.info(f"Prédiction demandée : {data.machine_id} par {current_user['username']}")
    return run_prediction(data)


# ── 4. Predict Batch (JWT requis) ─────────────────────────────────────────────
@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(
    batch        : BatchSensorInput,
    current_user : dict = Depends(get_current_user)
):
    """Prédit pour un batch de plusieurs machines en une seule requête."""
    predictions = [run_prediction(m) for m in batch.machines]
    alerts      = sum(1 for p in predictions if p.is_alert)
    logger.info(f"Batch : {len(predictions)} prédictions | {alerts} alertes")
    return BatchPredictionResponse(
        total       = len(predictions),
        alerts      = alerts,
        predictions = predictions,
    )


# ── 5. Machine Status (JWT requis) ────────────────────────────────────────────
@app.get("/machines/{machine_id}/status", response_model=MachineStatus, tags=["Machines"])
async def get_machine_status(
    machine_id   : str,
    current_user : dict = Depends(get_current_user)
):
    """Retourne le statut actuel d'une machine depuis InfluxDB."""
    try:
        client   = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
        query_api = client.query_api()

        # Dernière prédiction
        query = f'''
        from(bucket: "predictions")
          |> range(start: -5m)
          |> filter(fn: (r) => r._measurement == "prediction")
          |> filter(fn: (r) => r.machine_id == "{machine_id}")
          |> filter(fn: (r) => r._field == "confidence")
          |> last()
        '''
        tables = query_api.query(query)

        last_confidence = None
        last_prediction = None
        last_seen       = None

        for table in tables:
            for record in table.records:
                last_confidence = record.get_value()
                last_seen       = record.get_time().isoformat()

        # Compter alertes dernière heure
        query_alerts = f'''
        from(bucket: "alerts")
          |> range(start: -1h)
          |> filter(fn: (r) => r._measurement == "alert")
          |> filter(fn: (r) => r.machine_id == "{machine_id}")
          |> filter(fn: (r) => r._field == "confidence")
          |> count()
        '''
        alert_tables = query_api.query(query_alerts)
        alert_count  = 0
        for table in alert_tables:
            for record in table.records:
                alert_count = record.get_value()

        client.close()

        # Déterminer le statut
        if last_confidence is None:
            machine_status = "UNKNOWN"
        elif last_confidence >= 0.8:
            machine_status = "CRITICAL"
            last_prediction = "FAILURE"
        elif last_confidence >= 0.5:
            machine_status = "WARNING"
            last_prediction = "WARNING"
        else:
            machine_status = "NORMAL"
            last_prediction = "NORMAL"

        # Criticité par défaut selon l'ID
        criticality_map = {
            "MACHINE_001": "HIGH", "MACHINE_002": "HIGH", "MACHINE_009": "HIGH",
            "MACHINE_003": "MEDIUM", "MACHINE_004": "MEDIUM",
            "MACHINE_005": "MEDIUM", "MACHINE_010": "MEDIUM",
            "MACHINE_006": "LOW", "MACHINE_007": "LOW", "MACHINE_008": "LOW",
        }

        return MachineStatus(
            machine_id      = machine_id,
            criticality     = criticality_map.get(machine_id, "MEDIUM"),
            last_prediction = last_prediction,
            last_confidence = round(last_confidence, 4) if last_confidence else None,
            last_seen       = last_seen,
            alert_count_1h  = alert_count,
            status          = machine_status,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 6. Recent Alerts (JWT requis) ─────────────────────────────────────────────
@app.get("/alerts/recent", response_model=AlertsResponse, tags=["Alerts"])
async def get_recent_alerts(
    limit        : int  = 10,
    current_user : dict = Depends(get_current_user)
):
    """Retourne les dernières alertes déclenchées (défaut : 10)."""
    try:
        client    = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
        query_api = client.query_api()

        query = f'''
        from(bucket: "alerts")
          |> range(start: -1h)
          |> filter(fn: (r) => r._measurement == "alert")
          |> filter(fn: (r) => r._field == "confidence")
          |> sort(columns: ["_time"], desc: true)
          |> limit(n: {limit})
        '''
        tables = query_api.query(query)
        client.close()

        alerts = []
        for table in tables:
            for record in table.records:
                alerts.append(AlertItem(
                    timestamp      = record.get_time().isoformat(),
                    machine_id     = record.values.get("machine_id", ""),
                    criticality    = record.values.get("criticality", ""),
                    confidence     = round(record.get_value(), 4),
                    recommendation = "",
                    top_factor_1   = None,
                    top_factor_2   = None,
                    top_factor_3   = None,
                ))

        return AlertsResponse(total=len(alerts), alerts=alerts)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 7. Model Health (JWT requis) ──────────────────────────────────────────────
@app.get("/model/health", response_model=ModelHealth, tags=["Model"])
async def get_model_health(
    current_user : dict = Depends(get_current_user)
):
    """Retourne les métriques de santé du modèle ML."""
    try:
        client    = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
        query_api = client.query_api()

        query = '''
        from(bucket: "model_health")
          |> range(start: -24h)
          |> filter(fn: (r) => r._measurement == "drift_report")
          |> last()
        '''
        tables = query_api.query(query)
        client.close()

        dataset_drift    = None
        drifted_features = []

        for table in tables:
            for record in table.records:
                if record.get_field() == "dataset_drift":
                    dataset_drift = bool(record.get_value())
                if record.get_field() == "n_drifted_features":
                    pass

        return ModelHealth(
            status           = "healthy" if not dataset_drift else "drift_detected",
            xgboost_f1       = 0.84,
            xgboost_auc      = 0.96,
            dataset_drift    = dataset_drift,
            drifted_features = drifted_features,
            last_retrain     = None,
            model_version    = "xgboost_v1",
        )

    except Exception as e:
        logger.error(f"Erreur model health : {e}")
        return ModelHealth(
            status        = "unknown",
            xgboost_f1    = 0.84,
            xgboost_auc   = 0.96,
            dataset_drift = None,
            last_retrain  = None,
            model_version = "xgboost_v1",
        )


# ── Lancement ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
