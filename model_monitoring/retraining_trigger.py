"""
retraining_trigger.py
=====================
Pipeline de retraining automatique.

Fonctionnement :
  - Consomme le topic Kafka `drift-alerts` publié par drift_detector.py
  - Si trigger_retraining=True → lance le pipeline complet de retraining
  - Récupère les nouvelles données depuis InfluxDB
  - Fusionne avec le dataset d'entraînement original
  - Réentraîne XGBoost + Isolation Forest
  - Compare les métriques avec le modèle en production
  - Si meilleur → remplace les fichiers .pkl + log MLflow tag 'retrained'
  - Écrit les résultats dans InfluxDB bucket `model_health`
  - Notifie via Kafka topic `retraining-jobs`

Lancement :
  python retraining_trigger.py
"""

import os
import json
import time
import logging
import warnings
import joblib
import requests
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.ensemble import IsolationForest
import mlflow
import mlflow.xgboost
import mlflow.sklearn

# InfluxDB
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

# Kafka
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError

warnings.filterwarnings("ignore")
load_dotenv()

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("RetrainingTrigger")

# ─── Configuration ────────────────────────────────────────────────────────────
INFLUXDB_URL    = os.getenv("INFLUXDB_URL",    "http://localhost:8086")
INFLUXDB_TOKEN  = os.getenv("INFLUXDB_TOKEN",  "my-super-secret-token")
INFLUXDB_ORG    = os.getenv("INFLUXDB_ORG",    "aiot-org")
BUCKET_SENSORS  = os.getenv("INFLUXDB_BUCKET_SENSORS",      "sensors")
BUCKET_PREDS    = os.getenv("INFLUXDB_BUCKET_PREDICTIONS",   "predictions")
BUCKET_HEALTH   = os.getenv("INFLUXDB_BUCKET_MODEL_HEALTH",  "model_health")

KAFKA_SERVERS        = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_DRIFT_TOPIC    = os.getenv("KAFKA_TOPIC_DRIFT",       "drift-alerts")
KAFKA_RETRAIN_TOPIC  = os.getenv("KAFKA_TOPIC_RETRAINING",  "retraining-jobs")

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID",   "")

# Chemins modèles et données
MODELS_DIR            = Path("models")
XGBOOST_MODEL_PATH    = MODELS_DIR / "xgboost_model.pkl"
ISO_FOREST_MODEL_PATH = MODELS_DIR / "isolation_forest.pkl"
FEATURE_NAMES_PATH    = MODELS_DIR / "feature_names.pkl"
REFERENCE_DATA_PATH   = Path("data/processed/ai4i2020_features.csv")

# Hyperparamètres XGBoost (identiques au notebook 03_training)
XGBOOST_PARAMS = {
    "n_estimators":   300,
    "max_depth":      6,
    "learning_rate":  0.05,
    "subsample":      0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma":          0.1,
    "reg_alpha":      0.1,
    "reg_lambda":     1.0,
    "random_state":   42,
    "eval_metric":    "logloss",
    "use_label_encoder": False,
}

# Seuils de déploiement
MIN_F1_IMPROVEMENT    = 0.01   # Le nouveau modèle doit gagner au moins 1pt de F1
MIN_AUC_THRESHOLD     = 0.85   # AUC minimum absolu pour déployer
INFLUX_FETCH_HOURS    = 24     # Heures de données à récupérer depuis InfluxDB
SMOTE_RATIO           = 0.3    # Ratio SMOTE (identique notebook)


# ─── Clients ──────────────────────────────────────────────────────────────────

def build_influx_client() -> InfluxDBClient:
    return InfluxDBClient(
        url=INFLUXDB_URL,
        token=INFLUXDB_TOKEN,
        org=INFLUXDB_ORG,
        timeout=60_000,
    )


def build_kafka_consumer() -> Optional[KafkaConsumer]:
    try:
        consumer = KafkaConsumer(
            KAFKA_DRIFT_TOPIC,
            bootstrap_servers=KAFKA_SERVERS,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            group_id="retraining-trigger-group",
            auto_offset_reset="latest",
            enable_auto_commit=True,
            consumer_timeout_ms=5000,
        )
        logger.info(f"Kafka consumer connecté → topic: {KAFKA_DRIFT_TOPIC}")
        return consumer
    except KafkaError as e:
        logger.warning(f"Kafka non disponible ({e})")
        return None


def build_kafka_producer() -> Optional[KafkaProducer]:
    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
            acks="all",
        )
        return producer
    except KafkaError:
        return None


# ─── Récupération des données depuis InfluxDB ─────────────────────────────────

def fetch_recent_sensor_data(client: InfluxDBClient) -> pd.DataFrame:
    """
    Récupère les données capteurs brutes des INFLUX_FETCH_HOURS dernières heures.
    Retourne un DataFrame avec les features de base.
    """
    query_api = client.query_api()
    base_features = [
        "air_temperature", "process_temperature",
        "rotational_speed", "torque", "tool_wear",
    ]
    records = []

    for feature in base_features:
        flux = f"""
        from(bucket: "{BUCKET_SENSORS}")
          |> range(start: -{INFLUX_FETCH_HOURS}h)
          |> filter(fn: (r) => r._measurement == "sensor_reading")
          |> filter(fn: (r) => r._field == "{feature}")
          |> aggregateWindow(every: 1m, fn: mean, createEmpty: false)
          |> keep(columns: ["_time", "_value", "machine_id"])
        """
        try:
            tables = query_api.query(flux)
            for table in tables:
                for record in table.records:
                    records.append({
                        "_time":      record.get_time(),
                        "machine_id": record.values.get("machine_id"),
                        "feature":    feature,
                        "_value":     record.get_value(),
                    })
        except Exception as e:
            logger.error(f"Erreur lecture feature {feature}: {e}")

    if not records:
        logger.warning("Aucune donnée récupérée depuis InfluxDB")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df_wide = df.pivot_table(
        index=["_time", "machine_id"],
        columns="feature",
        values="_value",
        aggfunc="mean",
    ).reset_index()
    df_wide.columns.name = None

    logger.info(f"Données InfluxDB récupérées : {len(df_wide)} lignes")
    return df_wide.dropna()


def fetch_alert_labels(client: InfluxDBClient) -> pd.DataFrame:
    """
    Récupère les labels de panne (is_alert) depuis le bucket predictions
    pour construire la target du retraining.
    """
    query_api = client.query_api()

    flux = f"""
    from(bucket: "{BUCKET_PREDS}")
      |> range(start: -{INFLUX_FETCH_HOURS}h)
      |> filter(fn: (r) => r._measurement == "prediction")
      |> filter(fn: (r) => r._field == "is_alert")
      |> aggregateWindow(every: 1m, fn: max, createEmpty: false)
      |> keep(columns: ["_time", "_value", "machine_id"])
    """
    try:
        tables = query_api.query(flux)
        records = []
        for table in tables:
            for record in table.records:
                records.append({
                    "_time":      record.get_time(),
                    "machine_id": record.values.get("machine_id"),
                    "failure": int(record.get_value()),
                })
        if records:
            df = pd.DataFrame(records)
            logger.info(f"Labels récupérés : {len(df)} lignes depuis predictions")
            return df
    except Exception as e:
        logger.error(f"Erreur lecture labels: {e}")

    return pd.DataFrame()


# ─── Feature Engineering (identique à feature_engineering.py) ─────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique le même feature engineering que dans le notebook 02.
    """
    df = df.copy().sort_values("_time").reset_index(drop=True)

    # Features thermiques
    df["temp_diff"]      = df["process_temperature"] - df["air_temperature"]
    df["temp_ratio"]     = df["process_temperature"] / (df["air_temperature"] + 1e-6)

    # Feature puissance et usure
    df["power"]          = df["torque"] * df["rotational_speed"]
    df["wear_rate"]      = df["tool_wear"] / (df["rotational_speed"] + 1e-6)
    df["torque_per_rpm"] = df["torque"] / (df["rotational_speed"] + 1e-6)

    # Rolling stats (fenêtre 5)
    for col in ["torque", "rotational_speed", "process_temperature", "tool_wear"]:
        if col in df.columns:
            df[f"{col}_rolling_mean_5"] = df[col].rolling(5, min_periods=1).mean()
            df[f"{col}_rolling_std_5"]  = df[col].rolling(5, min_periods=1).std().fillna(0)

    # Rolling stats (fenêtre 10)
    for col in ["rotational_speed", "torque"]:
        if col in df.columns:
            df[f"{col}_rolling_std_10"] = df[col].rolling(10, min_periods=1).std().fillna(0)

    # Lag features (t-1, t-3)
    for col in ["torque", "tool_wear", "rotational_speed", "process_temperature"]:
        if col in df.columns:
            df[f"{col}_lag1"] = df[col].shift(1).fillna(df[col].mean())
            df[f"{col}_lag3"] = df[col].shift(3).fillna(df[col].mean())

    # Dérivées (delta entre t et t-1)
    for col in ["process_temperature", "torque", "rotational_speed"]:
        if col in df.columns:
            df[f"{col}_delta"] = df[col].diff().fillna(0)

    return df.fillna(df.mean(numeric_only=True))


def prepare_training_data(
    influx_df: pd.DataFrame,
    reference_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Fusionne les données InfluxDB avec le dataset de référence.
    Applique le feature engineering et retourne X, y.
    """
    # Charger feature_names de référence
    try:
        feature_names = joblib.load(FEATURE_NAMES_PATH)
    except Exception:
        feature_names = None
        logger.warning("feature_names.pkl non trouvé — on utilisera les colonnes disponibles")

    # Dataset de référence avec target
    ref = reference_df.copy()
    target_col = "failure"

    if target_col not in ref.columns:
        raise ValueError(f"Colonne target '{target_col}' absente du dataset de référence")

    y_ref = ref[target_col]
    X_ref = ref.drop(columns=[target_col], errors="ignore")

    # Données InfluxDB : feature engineering + target = 0 (on n'a pas de label fiable)
    # On les ajoute uniquement si on a des labels depuis predictions
    if not influx_df.empty and "_time" in influx_df.columns:
        influx_features = engineer_features(influx_df)
        # On garde seulement les colonnes communes avec la référence
        common_cols = [c for c in X_ref.columns if c in influx_features.columns]
        if common_cols:
            influx_X = influx_features[common_cols]
            # Label conservateur : 0 (NORMAL) par défaut
            influx_y = pd.Series([0] * len(influx_X), name=target_col)
            X_combined = pd.concat([X_ref[common_cols], influx_X], ignore_index=True)
            y_combined = pd.concat([y_ref, influx_y], ignore_index=True)
            logger.info(
                f"Dataset combiné : {len(X_ref)} référence + {len(influx_X)} InfluxDB "
                f"= {len(X_combined)} lignes"
            )
        else:
            X_combined, y_combined = X_ref, y_ref
            logger.warning("Pas de colonnes communes — retraining sur référence seule")
    else:
        X_combined, y_combined = X_ref, y_ref
        logger.info("Retraining sur dataset de référence uniquement")

    # Garder uniquement les features connues
    if feature_names is not None:
        available = [f for f in feature_names if f in X_combined.columns]
        X_combined = X_combined[available]

    return X_combined.fillna(X_combined.mean()), y_combined


# ─── Entraînement ─────────────────────────────────────────────────────────────

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val:   pd.DataFrame,
    y_val:   pd.Series,
) -> tuple[xgb.XGBClassifier, dict]:
    """
    Entraîne XGBoost avec SMOTE.
    Retourne le modèle + les métriques de validation.
    """
    # SMOTE pour gérer le déséquilibre
    smote = SMOTE(sampling_strategy=SMOTE_RATIO, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    logger.info(
        f"SMOTE appliqué : {y_train.sum()} → {y_resampled.sum()} positifs "
        f"sur {len(y_resampled)} total"
    )

    # Calcul scale_pos_weight
    n_neg = (y_resampled == 0).sum()
    n_pos = (y_resampled == 1).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    params = {**XGBOOST_PARAMS, "scale_pos_weight": scale_pos_weight}

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_resampled, y_resampled,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Évaluation
    y_pred     = model.predict(X_val)
    y_proba    = model.predict_proba(X_val)[:, 1]
    f1         = f1_score(y_val, y_pred, zero_division=0)
    auc        = roc_auc_score(y_val, y_proba)
    precision  = precision_score(y_val, y_pred, zero_division=0)
    recall     = recall_score(y_val, y_pred, zero_division=0)

    metrics = {
        "f1_score":  round(f1, 4),
        "auc_roc":   round(auc, 4),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
    }
    logger.info(f"XGBoost entraîné : F1={f1:.4f} | AUC={auc:.4f}")
    return model, metrics


def train_isolation_forest(X_normal: pd.DataFrame) -> IsolationForest:
    """
    Réentraîne l'Isolation Forest sur les données normales uniquement.
    """
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.034,  # ~taux de pannes du dataset original
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_normal)
    logger.info(f"Isolation Forest entraîné sur {len(X_normal)} samples normaux")
    return iso


# ─── Comparaison et déploiement ───────────────────────────────────────────────

def get_current_model_metrics() -> dict:
    """
    Récupère les métriques du modèle actuel en production depuis MLflow.
    Retourne des métriques par défaut si MLflow n'est pas disponible.
    """
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = mlflow.tracking.MlflowClient()

        # Cherche le dernier run tagué 'production'
        runs = client.search_runs(
            experiment_ids=["0"],
            filter_string="tags.stage = 'production'",
            order_by=["start_time DESC"],
            max_results=1,
        )

        if runs:
            metrics = runs[0].data.metrics
            logger.info(f"Métriques production actuelles : {metrics}")
            return {
                "f1_score": metrics.get("f1_score", 0.0),
                "auc_roc":  metrics.get("auc_roc",  0.0),
            }
    except Exception as e:
        logger.warning(f"MLflow non disponible ({e}) — métriques par défaut utilisées")

    # Valeurs du notebook 03 (F1=0.84, AUC=0.96)
    return {"f1_score": 0.84, "auc_roc": 0.96}


def should_deploy(new_metrics: dict, current_metrics: dict) -> tuple[bool, str]:
    """
    Compare les métriques et décide si le nouveau modèle doit être déployé.
    """
    f1_gain  = new_metrics["f1_score"] - current_metrics["f1_score"]
    auc_ok   = new_metrics["auc_roc"] >= MIN_AUC_THRESHOLD

    if not auc_ok:
        return False, f"AUC trop bas : {new_metrics['auc_roc']:.4f} < {MIN_AUC_THRESHOLD}"

    if f1_gain >= MIN_F1_IMPROVEMENT:
        return True, (
            f"F1 amélioré : {current_metrics['f1_score']:.4f} → "
            f"{new_metrics['f1_score']:.4f} (+{f1_gain:.4f})"
        )

    if f1_gain >= 0:
        return True, (
            f"F1 stable ou légèrement amélioré ({f1_gain:+.4f}) "
            f"— déploiement conservateur suite à drift"
        )

    return False, (
        f"F1 dégradé : {current_metrics['f1_score']:.4f} → "
        f"{new_metrics['f1_score']:.4f} ({f1_gain:.4f})"
    )


def deploy_models(
    new_xgb:     xgb.XGBClassifier,
    new_iso:     IsolationForest,
    new_metrics: dict,
    feature_names: list,
    run_id:      str,
) -> None:
    """
    Remplace les fichiers .pkl par les nouveaux modèles.
    Sauvegarde les anciens avec timestamp.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Backup des anciens modèles
    for path in [XGBOOST_MODEL_PATH, ISO_FOREST_MODEL_PATH]:
        if path.exists():
            backup = path.with_name(f"{path.stem}_backup_{timestamp}{path.suffix}")
            path.rename(backup)
            logger.info(f"Backup : {path.name} → {backup.name}")

    # Sauvegarde des nouveaux modèles
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(new_xgb,       XGBOOST_MODEL_PATH)
    joblib.dump(new_iso,       ISO_FOREST_MODEL_PATH)
    joblib.dump(feature_names, FEATURE_NAMES_PATH)

    logger.info(f"Nouveaux modèles déployés : {XGBOOST_MODEL_PATH}, {ISO_FOREST_MODEL_PATH}")


# ─── MLflow tracking ──────────────────────────────────────────────────────────

def log_to_mlflow(
    new_xgb:     xgb.XGBClassifier,
    new_metrics: dict,
    drift_info:  dict,
    deployed:    bool,
    deploy_reason: str,
) -> str:
    """
    Log l'expérience de retraining dans MLflow.
    Retourne le run_id.
    """
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("predictive-maintenance-retraining")

        with mlflow.start_run(run_name=f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            # Paramètres
            mlflow.log_params(XGBOOST_PARAMS)

            # Métriques
            mlflow.log_metrics(new_metrics)

            # Tags
            mlflow.set_tag("trigger", "drift_detected")
            mlflow.set_tag("stage",   "production" if deployed else "candidate")
            mlflow.set_tag("deployed", str(deployed))
            mlflow.set_tag("deploy_reason", deploy_reason)
            mlflow.set_tag("n_drifted_features", str(drift_info.get("n_drifted_features", 0)))
            mlflow.set_tag("drifted_features", str(drift_info.get("drifted_features", [])))

            # Modèle
            mlflow.xgboost.log_model(new_xgb, "xgboost_model")

            run_id = run.info.run_id
            logger.info(f"MLflow run loggé : {run_id} | deployed={deployed}")
            return run_id

    except Exception as e:
        logger.error(f"Erreur MLflow : {e}")
        return "mlflow-unavailable"


# ─── Écriture métriques InfluxDB ──────────────────────────────────────────────

def write_retraining_metrics(
    client:        InfluxDBClient,
    new_metrics:   dict,
    curr_metrics:  dict,
    deployed:      bool,
    deploy_reason: str,
    run_id:        str,
) -> None:
    write_api = client.write_api(write_options=SYNCHRONOUS)

    point = (
        Point("retraining_event")
        .tag("deployed",  str(deployed))
        .tag("mlflow_run_id", run_id)
        .field("new_f1_score",   new_metrics["f1_score"])
        .field("new_auc_roc",    new_metrics["auc_roc"])
        .field("new_precision",  new_metrics.get("precision", 0.0))
        .field("new_recall",     new_metrics.get("recall", 0.0))
        .field("prev_f1_score",  curr_metrics["f1_score"])
        .field("prev_auc_roc",   curr_metrics["auc_roc"])
        .field("f1_delta",       new_metrics["f1_score"] - curr_metrics["f1_score"])
        .field("deploy_reason",  deploy_reason)
        .time(datetime.now(timezone.utc))
    )
    write_api.write(bucket=BUCKET_HEALTH, record=point)
    logger.info("Métriques retraining écrites dans InfluxDB (model_health)")


# ─── Notification Telegram ────────────────────────────────────────────────────

def notify_telegram(message: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, json={
            "chat_id":    TELEGRAM_CHAT_ID,
            "text":       message,
            "parse_mode": "Markdown",
        }, timeout=10)
        logger.info("Notification Telegram envoyée")
    except Exception as e:
        logger.warning(f"Telegram non disponible : {e}")


def build_telegram_message(
    new_metrics:  dict,
    curr_metrics: dict,
    deployed:     bool,
    deploy_reason: str,
    drift_info:   dict,
) -> str:
    status = "✅ DÉPLOYÉ" if deployed else "❌ Non déployé"
    f1_delta = new_metrics["f1_score"] - curr_metrics["f1_score"]
    drifted  = drift_info.get("drifted_features", [])

    return (
        f"🔄 *Retraining automatique — Predictive Maintenance*\n\n"
        f"*Statut :* {status}\n"
        f"*Raison :* {deploy_reason}\n\n"
        f"*Features en drift :* {', '.join(drifted) if drifted else 'aucune'}\n\n"
        f"*Métriques comparées :*\n"
        f"  F1 : `{curr_metrics['f1_score']:.4f}` → `{new_metrics['f1_score']:.4f}` "
        f"({f1_delta:+.4f})\n"
        f"  AUC : `{curr_metrics['auc_roc']:.4f}` → `{new_metrics['auc_roc']:.4f}`\n"
        f"  Precision : `{new_metrics.get('precision', 0):.4f}`\n"
        f"  Recall : `{new_metrics.get('recall', 0):.4f}`\n\n"
        f"📊 [Dashboard Grafana](http://localhost:3000)"
    )


# ─── Pipeline de retraining complet ───────────────────────────────────────────

def run_retraining_pipeline(
    influx_client: InfluxDBClient,
    kafka_producer: Optional[KafkaProducer],
    drift_info:    dict,
) -> None:
    """
    Pipeline complet de retraining déclenché par une alerte drift.
    """
    logger.info("=" * 60)
    logger.info("🚀 DÉMARRAGE DU PIPELINE DE RETRAINING")
    logger.info(f"   Drift info : {drift_info.get('drifted_features', [])}")
    logger.info("=" * 60)

    start_time = time.time()

    # ── 1. Charger le dataset de référence ────────────────────────────────────
    logger.info("[1/8] Chargement du dataset de référence...")
    if not REFERENCE_DATA_PATH.exists():
        logger.error(f"Dataset de référence introuvable : {REFERENCE_DATA_PATH}")
        return
    reference_df = pd.read_csv(REFERENCE_DATA_PATH)
    logger.info(f"      Référence : {len(reference_df)} lignes")

    # ── 2. Récupérer les nouvelles données InfluxDB ───────────────────────────
    logger.info("[2/8] Récupération des données InfluxDB...")
    influx_df = fetch_recent_sensor_data(influx_client)

    # ── 3. Préparer le dataset d'entraînement ─────────────────────────────────
    logger.info("[3/8] Préparation du dataset combiné...")
    try:
        X, y = prepare_training_data(influx_df, reference_df)
    except Exception as e:
        logger.error(f"Erreur préparation données : {e}")
        return

    logger.info(
        f"      Dataset final : {len(X)} lignes | "
        f"{y.sum()} positifs ({y.mean():.1%}) | "
        f"{X.shape[1]} features"
    )

    if len(X) < 500:
        logger.error(f"Dataset trop petit ({len(X)} < 500) — retraining annulé")
        return

    # ── 4. Split train/val ────────────────────────────────────────────────────
    logger.info("[4/8] Split train/validation (80/20 stratifié)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # ── 5. Entraîner XGBoost ──────────────────────────────────────────────────
    logger.info("[5/8] Entraînement XGBoost + SMOTE...")
    new_xgb, new_metrics = train_xgboost(X_train, y_train, X_val, y_val)

    # ── 6. Entraîner Isolation Forest ─────────────────────────────────────────
    logger.info("[6/8] Entraînement Isolation Forest...")
    X_normal = X_train[y_train == 0]
    new_iso  = train_isolation_forest(X_normal)

    # ── 7. Comparer avec le modèle actuel ─────────────────────────────────────
    logger.info("[7/8] Comparaison avec le modèle en production...")
    curr_metrics = get_current_model_metrics()
    deploy, deploy_reason = should_deploy(new_metrics, curr_metrics)

    logger.info(f"      Décision : {'DÉPLOIEMENT' if deploy else 'REJET'} — {deploy_reason}")

    # ── 8. Déployer si meilleur ───────────────────────────────────────────────
    logger.info("[8/8] Finalisation...")

    run_id = log_to_mlflow(new_xgb, new_metrics, drift_info, deploy, deploy_reason)

    if deploy:
        deploy_models(
            new_xgb, new_iso, new_metrics,
            list(X.columns), run_id,
        )

    write_retraining_metrics(
        influx_client, new_metrics, curr_metrics,
        deploy, deploy_reason, run_id,
    )

    # Notification Kafka
    if kafka_producer:
        payload = {
            "event_type":    "RETRAINING_COMPLETED",
            "timestamp":     datetime.now(timezone.utc).isoformat(),
            "deployed":      deploy,
            "deploy_reason": deploy_reason,
            "new_f1":        new_metrics["f1_score"],
            "new_auc":       new_metrics["auc_roc"],
            "prev_f1":       curr_metrics["f1_score"],
            "mlflow_run_id": run_id,
            "duration_sec":  round(time.time() - start_time, 1),
        }
        try:
            kafka_producer.send(KAFKA_RETRAIN_TOPIC, key="retraining", value=payload)
            logger.info(f"Résultat publié → Kafka topic: {KAFKA_RETRAIN_TOPIC}")
        except KafkaError as e:
            logger.warning(f"Kafka publish échoué : {e}")

    # Notification Telegram
    tg_msg = build_telegram_message(
        new_metrics, curr_metrics, deploy, deploy_reason, drift_info
    )
    notify_telegram(tg_msg)

    duration = round(time.time() - start_time, 1)
    logger.info("=" * 60)
    logger.info(f"✅ RETRAINING TERMINÉ en {duration}s | deployed={deploy}")
    logger.info("=" * 60)


# ─── Boucle principale — consommation Kafka ───────────────────────────────────

def main() -> None:
    logger.info("RetrainingTrigger démarré")
    logger.info(f"  Écoute topic Kafka : {KAFKA_DRIFT_TOPIC}")

    influx_client  = build_influx_client()
    kafka_producer = build_kafka_producer()
    kafka_consumer = build_kafka_consumer()

    if kafka_consumer is None:
        logger.error(
            "Kafka non disponible — impossible de démarrer le consumer.\n"
            "Vérifie que Kafka tourne : docker-compose up -d kafka"
        )
        return

    logger.info("En attente d'alertes drift depuis Kafka...")

    try:
        while True:
            # Poll au lieu de l'itérateur — compatible Python 3.13
            msg_pack = kafka_consumer.poll(timeout_ms=5000)
            
            for tp, records in msg_pack.items():
                for message in records:
                    drift_info = message.value
                    logger.info(
                        f"Message Kafka reçu : "
                        f"trigger_retraining={drift_info.get('trigger_retraining')} | "
                        f"severity={drift_info.get('severity')}"
                    )

                    if drift_info.get("trigger_retraining", False):
                        run_retraining_pipeline(influx_client, kafka_producer, drift_info)
                    else:
                        logger.info(
                            f"Drift détecté mais retraining non requis "
                            f"(features: {drift_info.get('drifted_features', [])})"
                        )

            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Arrêt du RetrainingTrigger (Ctrl+C)")
    finally:
        kafka_consumer.close()
        influx_client.close()
        if kafka_producer:
            kafka_producer.close()
        logger.info("Ressources libérées")


if __name__ == "__main__":
    main()