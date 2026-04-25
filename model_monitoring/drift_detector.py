"""
drift_detector.py
=================
Détection de drift en temps réel avec Evidently AI.

Fonctionnement :
  - Toutes les N minutes (configurable), lit les dernières prédictions depuis InfluxDB
  - Compare avec les données de référence (dataset d'entraînement)
  - Détecte le drift sur chaque feature + sur les prédictions
  - Écrit les métriques dans le bucket InfluxDB `model_health`
  - Si drift critique détecté → publie dans le topic Kafka `drift-alerts`

Lancement :
  python drift_detector.py
"""

import os
import json
import time
import logging
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Evidently AI — API compatible 0.6.x
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
)
# Note: evidently 0.6.x garde la même API Report/metric_preset
# mais certains résultats dict ont une structure légèrement différente

# InfluxDB
from influxdb_client import InfluxDBClient, Point, WriteOptions
from influxdb_client.client.write_api import SYNCHRONOUS

# Kafka
from kafka import KafkaProducer
from kafka.errors import KafkaError

warnings.filterwarnings("ignore")
load_dotenv()

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DriftDetector")

# ─── Configuration ────────────────────────────────────────────────────────────
INFLUXDB_URL    = os.getenv("INFLUXDB_URL",    "http://localhost:8086")
INFLUXDB_TOKEN  = os.getenv("INFLUXDB_TOKEN",  "my-super-secret-token")
INFLUXDB_ORG    = os.getenv("INFLUXDB_ORG",    "aiot-org")
BUCKET_SENSORS  = os.getenv("INFLUXDB_BUCKET_SENSORS",     "sensors")
BUCKET_PREDS    = os.getenv("INFLUXDB_BUCKET_PREDICTIONS",  "predictions")
BUCKET_HEALTH   = os.getenv("INFLUXDB_BUCKET_MODEL_HEALTH", "model_health")

KAFKA_SERVERS   = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_DRIFT_TOPIC = os.getenv("KAFKA_TOPIC_DRIFT", "drift-alerts")

# Chemin vers les données d'entraînement (référence)
REFERENCE_DATA_PATH = Path("data/processed/ai4i2020_features.csv")

# Features surveillées (sous-ensemble des 42 features — les plus importantes)
MONITORED_FEATURES = [
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
    "temp_diff",
    "power",
]

# Seuils
DRIFT_CHECK_INTERVAL_SECONDS = 3600      # Vérification toutes les heures
CURRENT_WINDOW_MINUTES       = 60        # Fenêtre de données courantes
MIN_SAMPLES_REQUIRED         = 100       # Minimum de samples pour faire l'analyse
DRIFT_THRESHOLD_FEATURES     = 2         # Nb de features en drift pour déclencher retraining
PREDICTION_DRIFT_THRESHOLD   = 0.15      # Seuil de drift sur le ratio FAILURE/NORMAL


# ─── Clients ──────────────────────────────────────────────────────────────────

def build_influx_client() -> InfluxDBClient:
    return InfluxDBClient(
        url=INFLUXDB_URL,
        token=INFLUXDB_TOKEN,
        org=INFLUXDB_ORG,
        timeout=30_000,
    )


def build_kafka_producer() -> KafkaProducer | None:
    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
            acks="all",
            retries=3,
        )
        logger.info("Kafka producer connecté")
        return producer
    except KafkaError as e:
        logger.warning(f"Kafka non disponible ({e}) — alertes Kafka désactivées")
        return None


# ─── Chargement des données ───────────────────────────────────────────────────

def load_reference_data() -> pd.DataFrame:
    """
    Charge le dataset d'entraînement comme données de référence Evidently.
    Garde uniquement les features surveillées.
    """
    if not REFERENCE_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset de référence introuvable : {REFERENCE_DATA_PATH}\n"
            "Lance d'abord le notebook 02_feature_engineering.ipynb"
        )

    df = pd.read_csv(REFERENCE_DATA_PATH)
    logger.info(f"Référence chargée : {len(df)} lignes depuis {REFERENCE_DATA_PATH}")

    # Filtrer les colonnes disponibles
    available = [c for c in MONITORED_FEATURES if c in df.columns]
    missing   = [c for c in MONITORED_FEATURES if c not in df.columns]
    if missing:
        logger.warning(f"Features absentes du référentiel : {missing}")

    return df[available].dropna()


def fetch_current_data_from_influx(client: InfluxDBClient) -> pd.DataFrame:
    """
    Récupère les données capteurs des CURRENT_WINDOW_MINUTES dernières minutes
    depuis InfluxDB (bucket sensors).
    Retourne un DataFrame avec les features surveillées.
    """
    query_api = client.query_api()
    records = []

    for feature in MONITORED_FEATURES:
        flux = f"""
        from(bucket: "{BUCKET_SENSORS}")
          |> range(start: -{CURRENT_WINDOW_MINUTES}m)
          |> filter(fn: (r) => r._measurement == "sensor_reading")
          |> filter(fn: (r) => r._field == "{feature}")
          |> aggregateWindow(every: 10s, fn: mean, createEmpty: false)
          |> keep(columns: ["_time", "_value", "machine_id"])
        """
        try:
            tables = query_api.query(flux)
            for table in tables:
                for record in table.records:
                    records.append({
                        "_time":      record.get_time(),
                        "machine_id": record.values.get("machine_id", "unknown"),
                        "feature":    feature,
                        "_value":     record.get_value(),
                    })
        except Exception as e:
            logger.error(f"Erreur lecture InfluxDB pour {feature}: {e}")

    if not records:
        return pd.DataFrame(columns=MONITORED_FEATURES)

    df_long = pd.DataFrame(records)
    df_wide = df_long.pivot_table(
        index=["_time", "machine_id"],
        columns="feature",
        values="_value",
        aggfunc="mean",
    ).reset_index()
    df_wide.columns.name = None

    logger.info(f"Données courantes : {len(df_wide)} lignes sur {CURRENT_WINDOW_MINUTES}min")
    return df_wide[MONITORED_FEATURES].dropna()


def fetch_prediction_stats(client: InfluxDBClient) -> dict:
    """
    Calcule le ratio FAILURE/NORMAL sur la fenêtre courante.
    """
    query_api = client.query_api()

    flux_failure = f"""
    from(bucket: "{BUCKET_PREDS}")
      |> range(start: -{CURRENT_WINDOW_MINUTES}m)
      |> filter(fn: (r) => r._measurement == "prediction")
      |> filter(fn: (r) => r._field == "confidence")
      |> filter(fn: (r) => r.prediction == "FAILURE")
      |> group()
      |> count()
    """
    flux_total = f"""
    from(bucket: "{BUCKET_PREDS}")
      |> range(start: -{CURRENT_WINDOW_MINUTES}m)
      |> filter(fn: (r) => r._measurement == "prediction")
      |> filter(fn: (r) => r._field == "confidence")
      |> group()
      |> count()
    """

    def _scalar(flux: str) -> int:
        try:
            tables = query_api.query(flux)
            for table in tables:
                for record in table.records:
                    return int(record.get_value())
        except Exception:
            pass
        return 0

    n_failure = _scalar(flux_failure)
    n_total   = _scalar(flux_total)
    ratio     = n_failure / n_total if n_total > 0 else 0.0

    return {
        "n_failure": n_failure,
        "n_total":   n_total,
        "failure_ratio": ratio,
    }


# ─── Détection de drift Evidently ─────────────────────────────────────────────

def run_drift_analysis(
    reference: pd.DataFrame,
    current:   pd.DataFrame,
) -> dict:
    """
    Lance le rapport Evidently DataDrift.
    Compatible avec Evidently 0.6.x (pydantic v2).
    Retourne un dict avec les résultats structurés.
    """
    # Aligner les colonnes
    common_cols = [c for c in reference.columns if c in current.columns]
    ref = reference[common_cols].copy()
    cur = current[common_cols].copy()

    # Sous-échantillonnage de la référence pour équilibrer
    if len(ref) > len(cur) * 3:
        ref = ref.sample(n=min(len(cur) * 3, 3000), random_state=42)

    # Construction du rapport Evidently
    report = Report(metrics=[
        DataDriftPreset(),
        *[ColumnDriftMetric(column_name=col) for col in common_cols],
    ])

    report.run(reference_data=ref, current_data=cur)
    result_dict = report.as_dict()

    # ── Extraction compatible 0.6.x ──────────────────────────────────────────
    # Evidently 0.6.x : result_dict["metrics"] est une liste de dicts
    # Chaque dict a "metric" (nom de la classe) et "result" (les données)

    dataset_drift   = False
    n_drifted       = 0
    n_features      = len(common_cols)
    share_drifted   = 0.0
    feature_results = {}

    for metric_item in result_dict.get("metrics", []):
        metric_name = metric_item.get("metric", "")
        result      = metric_item.get("result", {})

        # DataDriftPreset ou DatasetDriftMetric
        if "DataDriftPreset" in metric_name or "DatasetDriftMetric" in metric_name:
            dataset_drift = result.get("dataset_drift", False)
            n_drifted     = result.get("number_of_drifted_columns", 0)
            n_features    = result.get("number_of_columns", n_features)
            share_drifted = result.get("share_of_drifted_columns", 0.0)

        # ColumnDriftMetric — une entrée par feature
        elif "ColumnDriftMetric" in metric_name:
            col   = result.get("column_name", "unknown")
            drift = result.get("drift_detected", False)
            score = result.get("drift_score",    1.0)
            stat  = result.get("stattest_name",  "unknown")
            feature_results[col] = {
                "drift_detected": drift,
                "drift_score":    round(float(score), 4),
                "stattest":       stat,
            }

    # Si DataDriftPreset n'a pas rempli dataset_drift, on le calcule
    if not dataset_drift and feature_results:
        n_drifted     = sum(1 for v in feature_results.values() if v["drift_detected"])
        n_features    = len(feature_results)
        share_drifted = n_drifted / n_features if n_features > 0 else 0.0
        dataset_drift = n_drifted > 0

    drifted_features = [f for f, v in feature_results.items() if v["drift_detected"]]

    return {
        "timestamp":           datetime.now(timezone.utc).isoformat(),
        "dataset_drift":       dataset_drift,
        "n_drifted_features":  n_drifted,
        "n_total_features":    n_features,
        "share_drifted":       round(float(share_drifted), 4),
        "drifted_features":    drifted_features,
        "feature_results":     feature_results,
        "n_current_samples":   len(cur),
        "n_reference_samples": len(ref),
    }


# ─── Écriture des métriques dans InfluxDB ─────────────────────────────────────

def write_drift_metrics(
    client:       InfluxDBClient,
    drift_result: dict,
    pred_stats:   dict,
) -> None:
    """
    Écrit les métriques de drift dans le bucket `model_health`.
    """
    write_api = client.write_api(write_options=SYNCHRONOUS)

    # Point global du dataset
    point_global = (
        Point("drift_report")
        .tag("source", "evidently")
        .field("dataset_drift",       int(drift_result["dataset_drift"]))
        .field("n_drifted_features",  drift_result["n_drifted_features"])
        .field("n_total_features",    drift_result["n_total_features"])
        .field("share_drifted",       drift_result["share_drifted"])
        .field("n_current_samples",   drift_result["n_current_samples"])
        .field("failure_ratio",       pred_stats["failure_ratio"])
        .field("n_alerts_window",     pred_stats["n_failure"])
        .field("n_total_preds",       pred_stats["n_total"])
        .time(datetime.now(timezone.utc))
    )
    write_api.write(bucket=BUCKET_HEALTH, record=point_global)

    # Point par feature
    for feature, stats in drift_result["feature_results"].items():
        point_feature = (
            Point("feature_drift")
            .tag("feature",  feature)
            .tag("stattest", stats["stattest"])
            .field("drift_detected", int(stats["drift_detected"]))
            .field("drift_score",    stats["drift_score"])
            .time(datetime.now(timezone.utc))
        )
        write_api.write(bucket=BUCKET_HEALTH, record=point_feature)

    logger.info(
        f"Métriques drift écrites → "
        f"dataset_drift={drift_result['dataset_drift']} | "
        f"drifted={drift_result['n_drifted_features']}/{drift_result['n_total_features']} features"
    )


# ─── Alerte Kafka ─────────────────────────────────────────────────────────────

def publish_drift_alert(
    producer:     KafkaProducer | None,
    drift_result: dict,
    pred_stats:   dict,
    trigger_retraining: bool,
) -> None:
    """
    Publie un événement de drift dans le topic Kafka `drift-alerts`.
    Ce message sera consommé par retraining_trigger.py si nécessaire.
    """
    if producer is None:
        logger.warning("Kafka non disponible — alerte drift non publiée")
        return

    payload = {
        "event_type":         "DRIFT_DETECTED",
        "timestamp":          drift_result["timestamp"],
        "dataset_drift":      drift_result["dataset_drift"],
        "n_drifted_features": drift_result["n_drifted_features"],
        "drifted_features":   drift_result["drifted_features"],
        "share_drifted":      drift_result["share_drifted"],
        "failure_ratio":      pred_stats["failure_ratio"],
        "n_current_samples":  drift_result["n_current_samples"],
        "trigger_retraining": trigger_retraining,
        "severity":           "CRITICAL" if trigger_retraining else "WARNING",
    }

    future = producer.send(
        KAFKA_DRIFT_TOPIC,
        key="drift-alert",
        value=payload,
    )
    try:
        future.get(timeout=10)
        logger.info(
            f"Alerte Kafka publiée → topic={KAFKA_DRIFT_TOPIC} | "
            f"retraining={trigger_retraining}"
        )
    except KafkaError as e:
        logger.error(f"Échec publication Kafka : {e}")


# ─── Décision de retraining ───────────────────────────────────────────────────

def should_trigger_retraining(
    drift_result: dict,
    pred_stats:   dict,
) -> tuple[bool, str]:
    """
    Évalue si le retraining doit être déclenché.
    Retourne (bool, raison).
    """
    reasons = []

    # Règle 1 : drift sur N features critiques
    if drift_result["n_drifted_features"] >= DRIFT_THRESHOLD_FEATURES:
        reasons.append(
            f"Drift sur {drift_result['n_drifted_features']} features "
            f"(seuil: {DRIFT_THRESHOLD_FEATURES})"
        )

    # Règle 2 : ratio FAILURE anormalement élevé
    if pred_stats["failure_ratio"] > 0.30:
        reasons.append(
            f"Ratio FAILURE anormal : {pred_stats['failure_ratio']:.1%} > 30%"
        )

    # Règle 3 : drift dataset global confirmé + features critiques affectées
    critical_features = {"torque", "tool_wear", "rotational_speed"}
    drifted_critical  = set(drift_result["drifted_features"]) & critical_features
    if drift_result["dataset_drift"] and len(drifted_critical) >= 2:
        reasons.append(
            f"Dataset drift global + features critiques affectées : "
            f"{list(drifted_critical)}"
        )

    trigger = len(reasons) > 0
    reason_str = " | ".join(reasons) if reasons else "Aucun critère de retraining atteint"
    return trigger, reason_str


# ─── Boucle principale ────────────────────────────────────────────────────────

def run_drift_check_once(
    influx_client: InfluxDBClient,
    kafka_producer: KafkaProducer | None,
    reference_data: pd.DataFrame,
) -> dict:
    """
    Exécute un cycle complet de détection de drift.
    Retourne le résultat du drift pour logging/monitoring.
    """
    logger.info("=" * 60)
    logger.info("Début du cycle de détection de drift")
    logger.info("=" * 60)

    # 1. Récupérer les données courantes
    current_data = fetch_current_data_from_influx(influx_client)

    if len(current_data) < MIN_SAMPLES_REQUIRED:
        logger.warning(
            f"Pas assez de données courantes ({len(current_data)} < {MIN_SAMPLES_REQUIRED}). "
            "Cycle ignoré."
        )
        return {}

    # 2. Récupérer les stats de prédiction
    pred_stats = fetch_prediction_stats(influx_client)
    logger.info(
        f"Stats prédictions : {pred_stats['n_failure']}/{pred_stats['n_total']} FAILURE "
        f"({pred_stats['failure_ratio']:.1%})"
    )

    # 3. Lancer l'analyse Evidently
    drift_result = run_drift_analysis(reference_data, current_data)

    logger.info(
        f"Résultat drift : dataset_drift={drift_result['dataset_drift']} | "
        f"features en drift : {drift_result['drifted_features']}"
    )

    # 4. Écrire les métriques dans InfluxDB
    write_drift_metrics(influx_client, drift_result, pred_stats)

    # 5. Décider si retraining nécessaire
    trigger_retraining, reason = should_trigger_retraining(drift_result, pred_stats)

    if trigger_retraining:
        logger.warning(f"⚠️  RETRAINING REQUIS : {reason}")
    else:
        logger.info(f"✅ Pas de retraining : {reason}")

    # 6. Publier alerte Kafka si drift détecté
    if drift_result["dataset_drift"] or trigger_retraining:
        publish_drift_alert(
            kafka_producer,
            drift_result,
            pred_stats,
            trigger_retraining,
        )

    logger.info(f"Cycle terminé — prochain dans {DRIFT_CHECK_INTERVAL_SECONDS}s")
    return drift_result


def main() -> None:
    logger.info("DriftDetector démarré")
    logger.info(f"  Intervalle de vérification : {DRIFT_CHECK_INTERVAL_SECONDS}s")
    logger.info(f"  Fenêtre de données courantes : {CURRENT_WINDOW_MINUTES}min")
    logger.info(f"  Features surveillées : {MONITORED_FEATURES}")

    # Initialisation
    reference_data = load_reference_data()
    influx_client  = build_influx_client()
    kafka_producer = build_kafka_producer()

    try:
        while True:
            try:
                run_drift_check_once(influx_client, kafka_producer, reference_data)
            except Exception as e:
                logger.error(f"Erreur dans le cycle de drift : {e}", exc_info=True)

            logger.info(f"Attente {DRIFT_CHECK_INTERVAL_SECONDS}s avant le prochain cycle...")
            time.sleep(DRIFT_CHECK_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        logger.info("Arrêt du DriftDetector (Ctrl+C)")
    finally:
        influx_client.close()
        if kafka_producer:
            kafka_producer.close()
        logger.info("Ressources libérées")


if __name__ == "__main__":
    main()
