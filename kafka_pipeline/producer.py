# ============================================================
# producer.py
# Bridge MQTT → Kafka
# Predictive Maintenance AIoT Pipeline — Étape 3
#
# Rôle : écoute les messages MQTT publiés par le simulateur IoT
#        et les transfère dans le topic Kafka "sensor-data"
# ============================================================

import json
import time
import logging
import os
import paho.mqtt.client as mqtt
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s : %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("MQTT-Kafka-Bridge")

# ── Configuration ─────────────────────────────────────────────────────────────
MQTT_BROKER          = os.getenv("MQTT_BROKER",          "localhost")
MQTT_PORT            = int(os.getenv("MQTT_PORT",         1883))
MQTT_TOPIC_SUBSCRIBE = os.getenv("MQTT_TOPIC",            "factory/machines/#")

KAFKA_BOOTSTRAP      = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC          = os.getenv("KAFKA_TOPIC_SENSORS",     "sensor-data")


# ── Connexion Kafka ───────────────────────────────────────────────────────────
def create_kafka_producer(retries: int = 10) -> KafkaProducer:
    """
    Crée un producteur Kafka avec retry automatique.
    Kafka peut prendre quelques secondes à démarrer.
    """
    for attempt in range(1, retries + 1):
        try:
            producer = KafkaProducer(
                bootstrap_servers = KAFKA_BOOTSTRAP,
                value_serializer  = lambda v: json.dumps(v).encode("utf-8"),
                key_serializer    = lambda k: k.encode("utf-8") if k else None,
                acks              = "all",        # attendre confirmation
                retries           = 3,
                linger_ms         = 10,           # regrouper les messages
                batch_size        = 16384,
            )
            logger.info(f"✅ Connecté à Kafka : {KAFKA_BOOTSTRAP}")
            return producer

        except NoBrokersAvailable:
            logger.warning(f"Kafka non disponible — tentative {attempt}/{retries} dans 5s...")
            time.sleep(5)

    raise ConnectionError(f"❌ Impossible de se connecter à Kafka après {retries} tentatives")


# ── Bridge MQTT → Kafka ───────────────────────────────────────────────────────
class MQTTKafkaBridge:
    """
    Écoute tous les topics MQTT factory/machines/#
    et transfère chaque message vers Kafka topic sensor-data.
    """

    def __init__(self):
        self.kafka_producer = None
        self.mqtt_client    = None
        self.message_count  = 0
        self.error_count    = 0

    def on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            logger.info(f"✅ Connecté au broker MQTT {MQTT_BROKER}:{MQTT_PORT}")
            # S'abonner à tous les topics machines
            client.subscribe(MQTT_TOPIC_SUBSCRIBE, qos=1)
            logger.info(f"   Abonné à : {MQTT_TOPIC_SUBSCRIBE}")
        else:
            logger.error(f"❌ Connexion MQTT échouée : rc={rc}")

    def on_message(self, client, userdata, msg):
        """
        Callback appelé à chaque message MQTT reçu.
        Parse le JSON et l'envoie dans Kafka.
        """
        try:
            # Décoder le payload JSON
            payload = json.loads(msg.payload.decode("utf-8"))

            # Utiliser machine_id comme clé Kafka
            # (garantit que les messages d'une même machine vont dans la même partition)
            machine_id = payload.get("machine_id", "UNKNOWN")

            # Envoyer dans Kafka
            future = self.kafka_producer.send(
                topic = KAFKA_TOPIC,
                key   = machine_id,
                value = payload,
            )

            self.message_count += 1

            # Log toutes les 50 messages
            if self.message_count % 50 == 0:
                logger.info(f"📨 {self.message_count} messages transférés MQTT→Kafka "
                           f"| Dernier : {machine_id} "
                           f"| Erreurs : {self.error_count}")

        except json.JSONDecodeError as e:
            self.error_count += 1
            logger.error(f"JSON invalide : {e} | payload={msg.payload[:100]}")

        except Exception as e:
            self.error_count += 1
            logger.error(f"Erreur transfert : {e}")

    def on_disconnect(self, client, userdata, disconnect_flags,rc, properties=None):
        logger.warning(f"Déconnecté du broker MQTT : rc={rc}")
        if rc != 0:
            logger.info("Tentative de reconnexion dans 5s...")
            time.sleep(5)

    def start(self):
        """Démarre le bridge MQTT → Kafka."""
        logger.info("=" * 60)
        logger.info("  MQTT → KAFKA BRIDGE")
        logger.info(f"  MQTT  : {MQTT_BROKER}:{MQTT_PORT} → {MQTT_TOPIC_SUBSCRIBE}")
        logger.info(f"  Kafka : {KAFKA_BOOTSTRAP} → topic:{KAFKA_TOPIC}")
        logger.info("=" * 60)

        # 1. Connexion Kafka
        logger.info("Connexion à Kafka...")
        self.kafka_producer = create_kafka_producer()

        # 2. Connexion MQTT
        logger.info("Connexion à MQTT...")
        self.mqtt_client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id="mqtt-kafka-bridge"
        )
        self.mqtt_client.on_connect    = self.on_connect
        self.mqtt_client.on_message    = self.on_message
        self.mqtt_client.on_disconnect = self.on_disconnect

        self.mqtt_client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)

        logger.info("✅ Bridge démarré — en attente de messages...")
        logger.info("   Appuyez sur Ctrl+C pour arrêter")
        logger.info("=" * 60)

        # Boucle bloquante — écoute MQTT en continu
        try:
            self.mqtt_client.loop_forever()
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """Arrête le bridge proprement."""
        logger.info(f"Arrêt du bridge — {self.message_count} messages transférés")
        if self.mqtt_client:
            self.mqtt_client.disconnect()
        if self.kafka_producer:
            self.kafka_producer.flush()
            self.kafka_producer.close()
        logger.info("✅ Bridge arrêté")


# ── Point d'entrée ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    bridge = MQTTKafkaBridge()
    bridge.start()
