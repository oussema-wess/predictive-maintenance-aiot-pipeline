# ============================================================
# simulateur_iot.py
# Simulateur IoT — Predictive Maintenance AIoT Pipeline
# Étape 3 : Simulation de 10 machines industrielles
#
# Fonctionnalités :
#   - 10 machines simulées en parallèle (threads)
#   - 4 modes : normal, degradation, failure, stress
#   - Bruit gaussien réaliste sur chaque capteur
#   - Perte de messages (5%) et capteurs défaillants (2%)
#   - Latence réseau variable
#   - Publication MQTT → Kafka
# ============================================================

import json
import time
import random
import threading
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import paho.mqtt.client as mqtt
import logging
import argparse
import os
from dataclasses import dataclass, field
from typing import Optional

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s : %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("IoT-Simulator")

# ── Configuration ────────────────────────────────────────────────────────────
MQTT_BROKER   = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT     = int(os.getenv("MQTT_PORT", 1883))
MQTT_TOPIC    = os.getenv("MQTT_TOPIC", "factory/machines")
DATASET_PATH  = os.getenv("DATASET_PATH", "data/raw/ai4i2020.csv")
# Profils de criticité des machines
MACHINE_PROFILES = {
    "MACHINE_001": {"criticality": "HIGH",   "type": "H"},
    "MACHINE_002": {"criticality": "HIGH",   "type": "H"},
    "MACHINE_003": {"criticality": "MEDIUM", "type": "M"},
    "MACHINE_004": {"criticality": "MEDIUM", "type": "M"},
    "MACHINE_005": {"criticality": "MEDIUM", "type": "M"},
    "MACHINE_006": {"criticality": "LOW",    "type": "L"},
    "MACHINE_007": {"criticality": "LOW",    "type": "L"},
    "MACHINE_008": {"criticality": "LOW",    "type": "L"},
    "MACHINE_009": {"criticality": "HIGH",   "type": "H"},
    "MACHINE_010": {"criticality": "MEDIUM", "type": "M"},
}

# ── Dataclass pour un message capteur ────────────────────────────────────────
@dataclass
class SensorMessage:
    timestamp        : str
    machine_id       : str
    criticality      : str
    machine_type     : str
    air_temperature  : Optional[float]
    process_temperature: Optional[float]
    rotational_speed : Optional[float]
    torque           : Optional[float]
    tool_wear        : Optional[float]
    simulation_mode  : str
    sensor_quality   : float
    sequence_number  : int

    def to_dict(self):
        return {
            "timestamp"           : self.timestamp,
            "machine_id"          : self.machine_id,
            "criticality"         : self.criticality,
            "machine_type"        : self.machine_type,
            "air_temperature"     : round(self.air_temperature, 2) if self.air_temperature else None,
            "process_temperature" : round(self.process_temperature, 2) if self.process_temperature else None,
            "rotational_speed"    : round(self.rotational_speed, 1) if self.rotational_speed else None,
            "torque"              : round(self.torque, 2) if self.torque else None,
            "tool_wear"           : round(self.tool_wear, 1) if self.tool_wear else None,
            "simulation_mode"     : self.simulation_mode,
            "sensor_quality"      : self.sensor_quality,
            "sequence_number"     : self.sequence_number,
        }

    def to_json(self):
        return json.dumps(self.to_dict())


# ── Classe principale : Machine simulée ──────────────────────────────────────
class SimulatedMachine:
    """
    Simule une machine industrielle qui envoie des données capteurs via MQTT.

    Modes de simulation :
    - normal      : données nominales avec bruit gaussien
    - degradation : drift progressif vers les valeurs de panne
    - failure     : injection de scénarios de panne connus
    - stress      : toutes les valeurs en limite haute
    """

    def __init__(self, machine_id: str, profile: dict, df: pd.DataFrame,
                 mqtt_client: mqtt.Client, mode: str = "normal"):
        self.machine_id   = machine_id
        self.criticality  = profile["criticality"]
        self.machine_type = profile["type"]
        self.df           = df.copy()
        self.mqtt_client  = mqtt_client
        self.mode         = mode
        self.seq_num      = 0
        self.running      = True
        self.drift_factor = 0.0  # pour le mode dégradation

        # Filtrer le dataset par type de machine si possible
        type_df = df[df['Type'] == self.machine_type]
        self.df = type_df if len(type_df) > 50 else df

        # Index courant dans le dataset
        self.current_idx = random.randint(0, len(self.df) - 1)

        self.logger = logging.getLogger(f"Machine-{machine_id}")
        self.logger.info(f"Initialisée — criticité={self.criticality} | mode={self.mode} | {len(self.df)} rows disponibles")

    def _add_noise(self, value: float, std: float) -> float:
        """Ajoute du bruit gaussien réaliste à une valeur capteur."""
        return value + np.random.normal(0, std)

    def _simulate_packet_loss(self) -> bool:
        """Simule une perte de message réseau (5% du temps)."""
        return random.random() < 0.05

    def _simulate_faulty_sensor(self, value: float) -> Optional[float]:
        """Simule un capteur défaillant (2% du temps → valeur None)."""
        if random.random() < 0.02:
            return None
        return value

    def _get_base_row(self) -> pd.Series:
        """Récupère la prochaine ligne du dataset en boucle."""
        row = self.df.iloc[self.current_idx]
        self.current_idx = (self.current_idx + 1) % len(self.df)
        return row

    def _apply_mode(self, row: pd.Series) -> dict:
        """Applique le mode de simulation sur les valeurs brutes."""

        air_temp   = row['Air temperature [K]']
        proc_temp  = row['Process temperature [K]']
        rot_speed  = row['Rotational speed [rpm]']
        torque     = row['Torque [Nm]']
        tool_wear  = row['Tool wear [min]']

        if self.mode == "normal":
            # Bruit gaussien réaliste
            air_temp  = self._add_noise(air_temp,  0.5)
            proc_temp = self._add_noise(proc_temp, 0.5)
            rot_speed = self._add_noise(rot_speed, 10)
            torque    = self._add_noise(torque,    0.5)
            tool_wear = self._add_noise(tool_wear, 1.0)
            sensor_quality = round(random.uniform(0.95, 1.0), 3)

        elif self.mode == "degradation":
            # Drift progressif — les valeurs dérivent lentement
            self.drift_factor = min(self.drift_factor + 0.001, 1.0)
            air_temp  = self._add_noise(air_temp  + self.drift_factor * 5,   0.5)
            proc_temp = self._add_noise(proc_temp + self.drift_factor * 8,   0.5)
            rot_speed = self._add_noise(rot_speed - self.drift_factor * 100, 10)
            torque    = self._add_noise(torque    + self.drift_factor * 15,  0.5)
            tool_wear = self._add_noise(tool_wear + self.drift_factor * 50,  1.0)
            sensor_quality = round(random.uniform(0.85, 0.95), 3)

        elif self.mode == "failure":
            # Valeurs typiques d'une panne imminente
            air_temp  = self._add_noise(302.0, 1.0)
            proc_temp = self._add_noise(315.0, 1.0)
            rot_speed = self._add_noise(1200,  20)
            torque    = self._add_noise(65.0,  2.0)
            tool_wear = self._add_noise(220,   5.0)
            sensor_quality = round(random.uniform(0.70, 0.85), 3)

        elif self.mode == "stress":
            # Toutes les machines en limite haute simultanément
            air_temp  = self._add_noise(304.0, 0.3)
            proc_temp = self._add_noise(313.0, 0.3)
            rot_speed = self._add_noise(2800,  15)
            torque    = self._add_noise(70.0,  1.0)
            tool_wear = self._add_noise(240,   3.0)
            sensor_quality = round(random.uniform(0.75, 0.90), 3)

        else:
            sensor_quality = 1.0

        return {
            "air_temperature"     : air_temp,
            "process_temperature" : proc_temp,
            "rotational_speed"    : max(0, rot_speed),
            "torque"              : max(0, torque),
            "tool_wear"           : max(0, tool_wear),
            "sensor_quality"      : sensor_quality,
        }

    def generate_message(self) -> Optional[SensorMessage]:
        """Génère un message capteur complet."""
        self.seq_num += 1
        row    = self._get_base_row()
        values = self._apply_mode(row)

        # Simuler capteurs défaillants
        values["air_temperature"]      = self._simulate_faulty_sensor(values["air_temperature"])
        values["process_temperature"]  = self._simulate_faulty_sensor(values["process_temperature"])
        values["torque"]               = self._simulate_faulty_sensor(values["torque"])

        return SensorMessage(
            timestamp           = datetime.now(timezone.utc).isoformat(),
            machine_id          = self.machine_id,
            criticality         = self.criticality,
            machine_type        = self.machine_type,
            air_temperature     = values["air_temperature"],
            process_temperature = values["process_temperature"],
            rotational_speed    = values["rotational_speed"],
            torque              = values["torque"],
            tool_wear           = values["tool_wear"],
            simulation_mode     = self.mode,
            sensor_quality      = values["sensor_quality"],
            sequence_number     = self.seq_num,
        )

    def publish(self, message: SensorMessage):
        """Publie le message sur le topic MQTT."""
        topic   = f"{MQTT_TOPIC}/{self.machine_id}"
        payload = message.to_json()

        result = self.mqtt_client.publish(topic, payload, qos=1)
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            self.logger.debug(f"[{self.seq_num}] Publié → {topic}")
        else:
            self.logger.warning(f"Échec publication MQTT : rc={result.rc}")

    def run(self, interval: float = 1.0):
        """
        Boucle principale de la machine.
        Envoie un message toutes les `interval` secondes.
        """
        self.logger.info(f"Démarrage — intervalle={interval}s")

        while self.running:
            try:
                # Perte de message simulée
                if self._simulate_packet_loss():
                    self.logger.debug("Paquet perdu (simulé)")
                    time.sleep(interval)
                    continue

                # Générer et publier
                message = self.generate_message()
                if message:
                    self.publish(message)

                # Latence réseau variable (±20%)
                jitter = random.uniform(0.8, 1.2)
                time.sleep(interval * jitter)

            except Exception as e:
                self.logger.error(f"Erreur : {e}")
                time.sleep(1)

    def stop(self):
        self.running = False
        self.logger.info("Arrêtée")


# ── Gestionnaire de la flotte ─────────────────────────────────────────────────
class FleetSimulator:
    """
    Gère la simulation de toute la flotte de 10 machines.
    Chaque machine tourne dans son propre thread.
    """

    def __init__(self, mode: str = "normal", interval: float = 1.0):
        self.mode      = mode
        self.interval  = interval
        self.machines  = []
        self.threads   = []
        self.mqtt_client = None

        logger.info(f"FleetSimulator initialisé — mode={mode} | intervalle={interval}s")

    def _connect_mqtt(self) -> mqtt.Client:
        """Établit la connexion MQTT avec Mosquitto."""
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="fleet-simulator")

        def on_connect(client, userdata, flags, rc, properties=None):
            if rc == 0:
                logger.info(f"✅ Connecté au broker MQTT {MQTT_BROKER}:{MQTT_PORT}")
            else:
                logger.error(f"❌ Connexion MQTT échouée : rc={rc}")

        def on_disconnect(client, userdata, disconnect_flags, rc, properties=None):
            logger.warning(f"Déconnecté du broker MQTT : rc={rc}")

        client.on_connect    = on_connect
        client.on_disconnect = on_disconnect

        try:
            client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
            client.loop_start()
            time.sleep(1)
            logger.info("Connexion MQTT établie")
        except Exception as e:
            logger.error(f"Impossible de se connecter au broker MQTT : {e}")
            logger.info("Vérifiez que Mosquitto tourne sur localhost:1883")
            raise

        return client

    def _load_dataset(self) -> pd.DataFrame:
        """Charge le dataset AI4I2020."""
        try:
            df = pd.read_csv(DATASET_PATH)
            logger.info(f"✅ Dataset chargé : {len(df):,} lignes depuis {DATASET_PATH}")
            return df
        except FileNotFoundError:
            logger.error(f"❌ Dataset non trouvé : {DATASET_PATH}")
            logger.info("Vérifiez que ai4i2020.csv est dans data/raw/")
            raise

    def start(self):
        """Démarre toutes les machines en parallèle."""
        logger.info("=" * 60)
        logger.info("  DÉMARRAGE DU SIMULATEUR IoT — FLOTTE DE 10 MACHINES")
        logger.info("=" * 60)

        # Connexion MQTT
        self.mqtt_client = self._connect_mqtt()

        # Chargement dataset
        df = self._load_dataset()

        # Modes spéciaux : certaines machines en mode différent
        machine_modes = {}
        for machine_id in MACHINE_PROFILES:
            if self.mode == "mixed":
                # Mode mixte : variété de modes pour plus de réalisme
                weights = ["normal"] * 6 + ["degradation"] * 3 + ["failure"] * 1
                machine_modes[machine_id] = random.choice(weights)
            else:
                machine_modes[machine_id] = self.mode

        # Créer et démarrer chaque machine dans un thread
        for machine_id, profile in MACHINE_PROFILES.items():
            mode    = machine_modes[machine_id]
            machine = SimulatedMachine(
                machine_id   = machine_id,
                profile      = profile,
                df           = df,
                mqtt_client  = self.mqtt_client,
                mode         = mode,
            )
            self.machines.append(machine)

            # Décalage de démarrage pour éviter que toutes publient en même temps
            start_delay = random.uniform(0, 0.5)
            thread = threading.Thread(
                target = self._delayed_start,
                args   = (machine, self.interval, start_delay),
                name   = f"Thread-{machine_id}",
                daemon = True,
            )
            self.threads.append(thread)
            thread.start()

        logger.info(f"✅ {len(self.machines)} machines démarrées en parallèle")
        logger.info(f"   Topics MQTT : {MQTT_TOPIC}/MACHINE_XXX")
        logger.info(f"   Broker      : {MQTT_BROKER}:{MQTT_PORT}")
        logger.info(f"   Mode        : {self.mode}")
        logger.info("")
        logger.info("Appuyez sur Ctrl+C pour arrêter le simulateur")
        logger.info("=" * 60)

        # Afficher les stats toutes les 10 secondes
        self._stats_loop()

    def _delayed_start(self, machine: SimulatedMachine,
                       interval: float, delay: float):
        """Démarre une machine avec un délai."""
        time.sleep(delay)
        machine.run(interval)

    def _stats_loop(self):
        """Affiche les statistiques de la flotte toutes les 10 secondes."""
        try:
            while True:
                time.sleep(10)
                total_messages = sum(m.seq_num for m in self.machines)
                logger.info(f"📊 Stats — Messages envoyés : {total_messages:,} | "
                           f"Machines actives : {len(self.machines)}")
                for machine in self.machines:
                    logger.info(f"   {machine.machine_id} [{machine.mode:<11}] "
                               f"seq={machine.seq_num:>4} | "
                               f"drift={machine.drift_factor:.3f}")

        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """Arrête toutes les machines proprement."""
        logger.info("Arrêt du simulateur...")
        for machine in self.machines:
            machine.stop()
        if self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
        logger.info("✅ Simulateur arrêté proprement")


# ── Point d'entrée ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Simulateur IoT — Flotte de 10 machines industrielles"
    )
    parser.add_argument(
        "--mode",
        choices=["normal", "degradation", "failure", "stress", "mixed"],
        default="normal",
        help="Mode de simulation (défaut: normal)"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Intervalle entre messages en secondes (défaut: 1.0)"
    )
    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════════╗
║       PREDICTIVE MAINTENANCE — IoT SIMULATOR             ║
║       Flotte : 10 machines industrielles                 ║
╚══════════════════════════════════════════════════════════╝
    """)

    fleet = FleetSimulator(mode=args.mode, interval=args.interval)

    try:
        fleet.start()
    except KeyboardInterrupt:
        fleet.stop()
    except Exception as e:
        logger.error(f"Erreur fatale : {e}")
        raise


if __name__ == "__main__":
    main()
