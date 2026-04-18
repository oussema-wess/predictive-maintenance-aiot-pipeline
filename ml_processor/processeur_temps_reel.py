# ============================================================
# processeur_temps_reel.py
# Pipeline Principal Temps Réel
# Predictive Maintenance AIoT Pipeline — Étape 4
#
# Rôle : lit les messages Kafka, applique le feature engineering,
#        fait les prédictions ML, écrit dans InfluxDB
#        et envoie des alertes Telegram si nécessaire
# ============================================================

import json
import time
import logging
import os
import requests
from datetime import datetime, timezone

from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

from feature_engineering import RealTimeFeatureEngine
from predictor import MLPredictor

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s : %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("Pipeline")

# ── Configuration ─────────────────────────────────────────────────────────────
KAFKA_BOOTSTRAP     = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC         = os.getenv("KAFKA_TOPIC_SENSORS",     "sensor-data")
KAFKA_GROUP_ID      = os.getenv("KAFKA_GROUP_ID",          "ml-processor-group")

INFLUX_URL          = os.getenv("INFLUXDB_URL",    "http://localhost:8086")
INFLUX_TOKEN        = os.getenv("INFLUXDB_TOKEN",  "my-super-secret-token")
INFLUX_ORG          = os.getenv("INFLUXDB_ORG",    "aiot-org")
INFLUX_BUCKET_SENS  = os.getenv("INFLUXDB_BUCKET_SENSORS",     "sensors")
INFLUX_BUCKET_PRED  = os.getenv("INFLUXDB_BUCKET_PREDICTIONS", "predictions")
INFLUX_BUCKET_ALERT = os.getenv("INFLUXDB_BUCKET_ALERTS",      "alerts")

TELEGRAM_TOKEN      = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID    = os.getenv("TELEGRAM_CHAT_ID",   "")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.8))


# ── Connexion Kafka Consumer ──────────────────────────────────────────────────
def create_kafka_consumer(retries: int = 10) -> KafkaConsumer:
    """Crée un consommateur Kafka avec retry automatique."""
    for attempt in range(1, retries + 1):
        try:
            consumer = KafkaConsumer(
                KAFKA_TOPIC,
                bootstrap_servers = KAFKA_BOOTSTRAP,
                group_id          = KAFKA_GROUP_ID,
                auto_offset_reset = "latest",
                value_deserializer= lambda v: json.loads(v.decode("utf-8")),
                consumer_timeout_ms = 1000,
            )
            logger.info(f"✅ Connecté à Kafka : {KAFKA_BOOTSTRAP} | topic={KAFKA_TOPIC}")
            return consumer
        except NoBrokersAvailable:
            logger.warning(f"Kafka non disponible — tentative {attempt}/{retries}...")
            time.sleep(5)
    raise ConnectionError("❌ Impossible de se connecter à Kafka")


# ── Connexion InfluxDB ────────────────────────────────────────────────────────
def create_influx_client():
    """Crée un client InfluxDB."""
    client = InfluxDBClient(
        url   = INFLUX_URL,
        token = INFLUX_TOKEN,
        org   = INFLUX_ORG,
    )
    # Créer les buckets manquants
    buckets_api = client.buckets_api()
    for bucket_name in [INFLUX_BUCKET_SENS, INFLUX_BUCKET_PRED, INFLUX_BUCKET_ALERT]:
        existing = buckets_api.find_bucket_by_name(bucket_name)
        if not existing:
            buckets_api.create_bucket(bucket_name=bucket_name, org=INFLUX_ORG)
            logger.info(f"✅ Bucket InfluxDB créé : {bucket_name}")

    write_api = client.write_api(write_options=SYNCHRONOUS)
    logger.info(f"✅ Connecté à InfluxDB : {INFLUX_URL}")
    return client, write_api


# ── Écriture InfluxDB ─────────────────────────────────────────────────────────
def write_sensor_data(write_api, data: dict, features: dict):
    """Écrit les données capteurs brutes dans InfluxDB."""
    try:
        point = (
            Point("sensor_reading")
            .tag("machine_id",  data.get("machine_id"))
            .tag("criticality", data.get("criticality"))
            .tag("mode",        data.get("simulation_mode", "normal"))
            .field("air_temperature",      features.get("air_temp", 0))
            .field("process_temperature",  features.get("process_temp", 0))
            .field("rotational_speed",     features.get("rotational_speed", 0))
            .field("torque",               features.get("torque", 0))
            .field("tool_wear",            features.get("tool_wear", 0))
            .field("temp_diff",            features.get("temp_diff", 0))
            .field("power",                features.get("power", 0))
            .field("sensor_quality",       data.get("sensor_quality", 1.0))
            .time(datetime.now(timezone.utc), WritePrecision.NS)
        )
        write_api.write(bucket=INFLUX_BUCKET_SENS, org=INFLUX_ORG, record=point)
    except Exception as e:
        logger.error(f"Erreur écriture InfluxDB sensors : {e}")


def write_prediction(write_api, result: dict):
    """Écrit le résultat de prédiction dans InfluxDB."""
    try:
        point = (
            Point("prediction")
            .tag("machine_id",   result.get("machine_id"))
            .tag("criticality",  result.get("criticality"))
            .tag("prediction",   result.get("xgboost_prediction"))
            .field("confidence",      result.get("xgboost_confidence", 0))
            .field("anomaly_score",   result.get("anomaly_score", 0))
            .field("is_alert",        int(result.get("is_alert", False)))
            .field("iso_anomaly",     int(result.get("isolation_forest_anomaly", False)))
            .time(datetime.now(timezone.utc), WritePrecision.NS)
        )
        write_api.write(bucket=INFLUX_BUCKET_PRED, org=INFLUX_ORG, record=point)
    except Exception as e:
        logger.error(f"Erreur écriture InfluxDB predictions : {e}")


def write_alert(write_api, result: dict):
    """Écrit une alerte dans InfluxDB."""
    try:
        top_factors = result.get("top_factors", [])
        factor1 = top_factors[0]["feature"] if len(top_factors) > 0 else ""
        factor2 = top_factors[1]["feature"] if len(top_factors) > 1 else ""
        factor3 = top_factors[2]["feature"] if len(top_factors) > 2 else ""

        point = (
            Point("alert")
            .tag("machine_id",  result.get("machine_id"))
            .tag("criticality", result.get("criticality"))
            .field("confidence",     result.get("xgboost_confidence", 0))
            .field("anomaly_score",  result.get("anomaly_score", 0))
            .field("recommendation", result.get("recommendation", ""))
            .field("top_factor_1",   factor1)
            .field("top_factor_2",   factor2)
            .field("top_factor_3",   factor3)
            .time(datetime.now(timezone.utc), WritePrecision.NS)
        )
        write_api.write(bucket=INFLUX_BUCKET_ALERT, org=INFLUX_ORG, record=point)
    except Exception as e:
        logger.error(f"Erreur écriture InfluxDB alerts : {e}")


# ── Alertes Telegram ──────────────────────────────────────────────────────────
def send_telegram_alert(result: dict):
    """Envoie une alerte Telegram si configuré."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return

    try:
        top_factors = result.get("top_factors", [])
        factors_text = ""
        for i, f in enumerate(top_factors, 1):
            factors_text += f"\n  {i}. {f['feature']} (SHAP: {f['shap_value']:+.3f}) {f['direction']}"

        message = (
            f"🚨 *ALERTE — {result['machine_id']}*\n\n"
            f"📊 Prédiction : *PANNE IMMINENTE*\n"
            f"🎯 Confiance  : *{result['xgboost_confidence']:.0%}*\n"
            f"⚠️  Anomalie   : {'OUI' if result['isolation_forest_anomaly'] else 'NON'}\n\n"
            f"🔍 *Facteurs principaux :*{factors_text}\n\n"
            f"💡 *{result['recommendation']}*\n"
            f"🏭 Criticité : {result['criticality']}\n"
            f"🔗 Dashboard : http://localhost:3000"
        )

        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, json={
            "chat_id"    : TELEGRAM_CHAT_ID,
            "text"       : message,
            "parse_mode" : "Markdown",
        }, timeout=5)

    except Exception as e:
        logger.error(f"Erreur Telegram : {e}")


# ── Pipeline Principal ────────────────────────────────────────────────────────
class RealTimePipeline:
    """
    Pipeline principal qui orchestre :
    Kafka → Feature Engineering → ML → InfluxDB → Telegram
    """

    def __init__(self):
        self.consumer      = None
        self.write_api     = None
        self.influx_client = None
        self.feature_engine = RealTimeFeatureEngine(window_size=10)
        self.predictor      = MLPredictor()
        self.processed      = 0
        self.errors         = 0

    def start(self):
        logger.info("=" * 60)
        logger.info("  PIPELINE TEMPS RÉEL — DÉMARRAGE")
        logger.info(f"  Kafka   : {KAFKA_BOOTSTRAP} | topic={KAFKA_TOPIC}")
        logger.info(f"  InfluxDB: {INFLUX_URL}")
        logger.info("=" * 60)

        # Charger les modèles ML
        self.predictor.load_models()

        # Connexions
        self.consumer                        = create_kafka_consumer()
        self.influx_client, self.write_api   = create_influx_client()

        logger.info("✅ Pipeline démarré — en attente de messages Kafka...")
        logger.info("   Appuyez sur Ctrl+C pour arrêter")
        logger.info("=" * 60)

        try:
            self._processing_loop()
        except KeyboardInterrupt:
            self.stop()

    def _processing_loop(self):
        """Boucle principale de traitement des messages."""
        while True:
            try:
                for message in self.consumer:
                    self._process_message(message.value)

            except Exception as e:
                self.errors += 1
                logger.error(f"Erreur boucle principale : {e}")
                time.sleep(1)

    def _process_message(self, data: dict):
        """Traite un seul message capteur."""
        try:
            # 1. Feature Engineering temps réel
            features = self.feature_engine.process(data)
            if features is None:
                return

            # 2. Prédiction ML
            metadata = {
                "timestamp"      : data.get("timestamp"),
                "machine_id"     : data.get("machine_id"),
                "criticality"    : data.get("criticality", "MEDIUM"),
                "sequence_number": data.get("sequence_number"),
            }
            result = self.predictor.predict(features, metadata)

            # 3. Écriture InfluxDB — données capteurs
            write_sensor_data(self.write_api, data, features)

            # 4. Écriture InfluxDB — prédictions
            write_prediction(self.write_api, result)

            # 5. Si alerte → écriture + Telegram
            if result["is_alert"]:
                write_alert(self.write_api, result)
                send_telegram_alert(result)

            self.processed += 1

        except Exception as e:
            self.errors += 1
            logger.error(f"Erreur traitement message : {e}")

    def stop(self):
        logger.info(f"Arrêt — {self.processed} messages traités | {self.errors} erreurs")
        if self.consumer:
            self.consumer.close()
        if self.influx_client:
            self.influx_client.close()
        logger.info("✅ Pipeline arrêté proprement")


# ── Point d'entrée ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pipeline = RealTimePipeline()
    pipeline.start()
