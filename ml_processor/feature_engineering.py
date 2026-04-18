# ============================================================
# feature_engineering.py
# Feature Engineering Temps Réel
# Predictive Maintenance AIoT Pipeline — Étape 4
#
# Rôle : applique le même feature engineering que le Notebook 02
#        mais en temps réel sur un flux de données Kafka
#        en maintenant un buffer par machine
# ============================================================

import numpy as np
from collections import deque
from typing import Optional
import logging

logger = logging.getLogger("FeatureEngine")


class RealTimeFeatureEngine:
    """
    Applique le feature engineering en temps réel.

    Pour chaque machine, maintient un buffer des N dernières valeurs
    et calcule rolling stats, lag features et features composites
    exactement comme dans le Notebook 02.
    """

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        # Un buffer par machine
        self.buffers = {}

    def _get_buffer(self, machine_id: str) -> dict:
        """Retourne le buffer d'une machine, le crée si inexistant."""
        if machine_id not in self.buffers:
            self.buffers[machine_id] = {
                "air_temp"    : deque(maxlen=self.window_size),
                "process_temp": deque(maxlen=self.window_size),
                "rot_speed"   : deque(maxlen=self.window_size),
                "torque"      : deque(maxlen=self.window_size),
                "tool_wear"   : deque(maxlen=self.window_size),
                "temp_diff"   : deque(maxlen=self.window_size),
                "power"       : deque(maxlen=self.window_size),
            }
        return self.buffers[machine_id]

    def _safe_mean(self, buf: deque) -> float:
        return float(np.mean(buf)) if len(buf) > 0 else 0.0

    def _safe_std(self, buf: deque) -> float:
        return float(np.std(buf)) if len(buf) > 1 else 0.0

    def _safe_lag(self, buf: deque, lag: int) -> float:
        """Retourne la valeur à t-lag depuis le buffer."""
        lst = list(buf)
        if len(lst) >= lag + 1:
            return lst[-(lag + 1)]
        return lst[0] if lst else 0.0

    def process(self, data: dict) -> Optional[dict]:
        """
        Enrichit un message capteur brut avec les features engineered.

        Paramètres :
            data : dict — message JSON du simulateur IoT

        Retourne :
            dict — données enrichies avec toutes les features
            None  — si les données sont invalides
        """
        machine_id = data.get("machine_id", "UNKNOWN")

        # ── Extraire les valeurs brutes ────────────────────────────────────
        air_temp   = data.get("air_temperature")
        proc_temp  = data.get("process_temperature")
        rot_speed  = data.get("rotational_speed")
        torque     = data.get("torque")
        tool_wear  = data.get("tool_wear")

        # Gérer les capteurs défaillants (valeur None)
        # On remplace par la dernière valeur connue du buffer
        buf = self._get_buffer(machine_id)

        if air_temp is None:
            air_temp = self._safe_mean(buf["air_temp"]) or 300.0
        if proc_temp is None:
            proc_temp = self._safe_mean(buf["process_temp"]) or 310.0
        if rot_speed is None:
            rot_speed = self._safe_mean(buf["rot_speed"]) or 1500.0
        if torque is None:
            torque = self._safe_mean(buf["torque"]) or 40.0
        if tool_wear is None:
            tool_wear = self._safe_mean(buf["tool_wear"]) or 100.0

        # ── Encodage type machine ──────────────────────────────────────────
        type_map = {"L": 1, "M": 2, "H": 0}
        machine_type    = data.get("machine_type", "M")
        type_encoded    = type_map.get(machine_type, 2)

        # ── Features thermiques ────────────────────────────────────────────
        temp_diff       = proc_temp - air_temp
        temp_ratio      = proc_temp / (air_temp + 1e-6)

        # ── Features mécaniques ────────────────────────────────────────────
        power                   = torque * rot_speed
        wear_rate               = tool_wear / (rot_speed + 1e-6)
        torque_wear_interaction = torque * tool_wear
        thermal_efficiency      = power / (temp_diff + 1e-6)

        # ── Mettre à jour les buffers ──────────────────────────────────────
        buf["air_temp"].append(air_temp)
        buf["process_temp"].append(proc_temp)
        buf["rot_speed"].append(rot_speed)
        buf["torque"].append(torque)
        buf["tool_wear"].append(tool_wear)
        buf["temp_diff"].append(temp_diff)
        buf["power"].append(power)

        # ── Rolling statistics (fenêtre 5) ─────────────────────────────────
        torque_roll_mean_5          = self._safe_mean(list(buf["torque"])[-5:])
        torque_roll_std_5           = self._safe_std(list(buf["torque"])[-5:])
        rot_speed_roll_mean_5       = self._safe_mean(list(buf["rot_speed"])[-5:])
        rot_speed_roll_std_5        = self._safe_std(list(buf["rot_speed"])[-5:])
        tool_wear_roll_mean_5       = self._safe_mean(list(buf["tool_wear"])[-5:])
        tool_wear_roll_std_5        = self._safe_std(list(buf["tool_wear"])[-5:])
        temp_diff_roll_mean_5       = self._safe_mean(list(buf["temp_diff"])[-5:])
        temp_diff_roll_std_5        = self._safe_std(list(buf["temp_diff"])[-5:])

        # ── Rolling statistics (fenêtre 10) ────────────────────────────────
        torque_roll_mean_10         = self._safe_mean(buf["torque"])
        torque_roll_std_10          = self._safe_std(buf["torque"])
        rot_speed_roll_mean_10      = self._safe_mean(buf["rot_speed"])
        rot_speed_roll_std_10       = self._safe_std(buf["rot_speed"])
        temp_diff_roll_mean_10      = self._safe_mean(buf["temp_diff"])
        temp_diff_roll_std_10       = self._safe_std(buf["temp_diff"])

        # ── Lag features ───────────────────────────────────────────────────
        tool_wear_lag1  = self._safe_lag(buf["tool_wear"], 1)
        tool_wear_lag3  = self._safe_lag(buf["tool_wear"], 3)
        tool_wear_lag5  = self._safe_lag(buf["tool_wear"], 5)
        torque_lag1     = self._safe_lag(buf["torque"],    1)
        torque_lag3     = self._safe_lag(buf["torque"],    3)
        torque_lag5     = self._safe_lag(buf["torque"],    5)
        temp_diff_lag1  = self._safe_lag(buf["temp_diff"], 1)
        temp_diff_lag3  = self._safe_lag(buf["temp_diff"], 3)
        temp_diff_lag5  = self._safe_lag(buf["temp_diff"], 5)
        power_lag1      = self._safe_lag(buf["power"],     1)
        power_lag3      = self._safe_lag(buf["power"],     3)
        power_lag5      = self._safe_lag(buf["power"],     5)

        # ── Delta features ─────────────────────────────────────────────────
        tool_wear_delta = tool_wear  - tool_wear_lag1
        torque_delta    = torque     - torque_lag1
        temp_diff_delta = temp_diff  - temp_diff_lag1

        # ── Variation température ──────────────────────────────────────────
        temp_variation  = proc_temp - (list(buf["process_temp"])[-2]
                          if len(buf["process_temp"]) >= 2
                          else proc_temp)

        # ── Assembler le vecteur de features final ─────────────────────────
        # Ordre identique au Notebook 02 et feature_names.pkl
        features = {
            # Originales
            "air_temp"                    : air_temp,
            "process_temp"                : proc_temp,
            "rotational_speed"            : rot_speed,
            "torque"                      : torque,
            "tool_wear"                   : tool_wear,
            "Type_encoded"                : type_encoded,
            # Thermiques
            "temp_diff"                   : temp_diff,
            "temp_variation"              : temp_variation,
            "temp_ratio"                  : temp_ratio,
            # Mécaniques
            "power"                       : power,
            "wear_rate"                   : wear_rate,
            "torque_wear_interaction"     : torque_wear_interaction,
            "thermal_efficiency"          : thermal_efficiency,
            # Rolling 5
            "torque_roll_mean_5"          : torque_roll_mean_5,
            "torque_roll_std_5"           : torque_roll_std_5,
            "rotational_speed_roll_mean_5": rot_speed_roll_mean_5,
            "rotational_speed_roll_std_5" : rot_speed_roll_std_5,
            "tool_wear_roll_mean_5"       : tool_wear_roll_mean_5,
            "tool_wear_roll_std_5"        : tool_wear_roll_std_5,
            "temp_diff_roll_mean_5"       : temp_diff_roll_mean_5,
            "temp_diff_roll_std_5"        : temp_diff_roll_std_5,
            # Rolling 10
            "torque_roll_mean_10"         : torque_roll_mean_10,
            "torque_roll_std_10"          : torque_roll_std_10,
            "rotational_speed_roll_mean_10": rot_speed_roll_mean_10,
            "rotational_speed_roll_std_10" : rot_speed_roll_std_10,
            "temp_diff_roll_mean_10"      : temp_diff_roll_mean_10,
            "temp_diff_roll_std_10"       : temp_diff_roll_std_10,
            # Lag
            "tool_wear_lag1"              : tool_wear_lag1,
            "tool_wear_lag3"              : tool_wear_lag3,
            "tool_wear_lag5"              : tool_wear_lag5,
            "torque_lag1"                 : torque_lag1,
            "torque_lag3"                 : torque_lag3,
            "torque_lag5"                 : torque_lag5,
            "temp_diff_lag1"              : temp_diff_lag1,
            "temp_diff_lag3"              : temp_diff_lag3,
            "temp_diff_lag5"              : temp_diff_lag5,
            "power_lag1"                  : power_lag1,
            "power_lag3"                  : power_lag3,
            "power_lag5"                  : power_lag5,
            # Delta
            "tool_wear_delta"             : tool_wear_delta,
            "torque_delta"                : torque_delta,
            "temp_diff_delta"             : temp_diff_delta,
        }

        return features
