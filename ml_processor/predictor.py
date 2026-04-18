# ============================================================
# predictor.py
# Inférence ML Temps Réel
# Predictive Maintenance AIoT Pipeline — Étape 4
#
# Rôle : charge XGBoost + Isolation Forest et fait des prédictions
#        sur chaque message capteur enrichi
# ============================================================

import numpy as np
import pandas as pd
import joblib
import shap
import logging
import os
from typing import Optional

logger = logging.getLogger("Predictor")

# Seuil de confiance — alerte uniquement si probabilité > 80%
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.8))

# Chemins des modèles
MODELS_DIR          = os.getenv("MODELS_DIR", "models")
XGB_MODEL_PATH      = os.path.join(MODELS_DIR, "xgboost_model.pkl")
ISO_MODEL_PATH      = os.path.join(MODELS_DIR, "isolation_forest.pkl")
FEATURES_PATH       = os.path.join(MODELS_DIR, "feature_names.pkl")


class MLPredictor:
    """
    Charge les modèles ML et fait des prédictions en temps réel.

    - XGBoost      : probabilité de panne (pannes connues)
    - Isolation Forest : score d'anomalie (pannes inconnues)
    - SHAP         : explication des prédictions
    """

    def __init__(self):
        self.xgb_model   = None
        self.iso_forest  = None
        self.feature_names = None
        self.shap_explainer = None
        self.prediction_count = 0
        self.alert_count      = 0

    def load_models(self):
        """Charge les modèles depuis les fichiers .pkl."""
        logger.info("Chargement des modèles ML...")

        # XGBoost
        if not os.path.exists(XGB_MODEL_PATH):
            raise FileNotFoundError(
                f"Modèle XGBoost non trouvé : {XGB_MODEL_PATH}\n"
                f"Exécutez d'abord le Notebook 03 pour entraîner les modèles."
            )
        self.xgb_model = joblib.load(XGB_MODEL_PATH)
        logger.info(f"✅ XGBoost chargé : {XGB_MODEL_PATH}")

        # Isolation Forest
        if not os.path.exists(ISO_MODEL_PATH):
            raise FileNotFoundError(f"Isolation Forest non trouvé : {ISO_MODEL_PATH}")
        self.iso_forest = joblib.load(ISO_MODEL_PATH)
        logger.info(f"✅ Isolation Forest chargé : {ISO_MODEL_PATH}")

        # Feature names
        if not os.path.exists(FEATURES_PATH):
            raise FileNotFoundError(f"Feature names non trouvé : {FEATURES_PATH}")
        self.feature_names = joblib.load(FEATURES_PATH)
        logger.info(f"✅ {len(self.feature_names)} features chargées")

        # SHAP Explainer
        self.shap_explainer = shap.TreeExplainer(self.xgb_model)
        logger.info("✅ SHAP Explainer initialisé")

        logger.info("✅ Tous les modèles chargés et prêts !")

    def _prepare_input(self, features: dict) -> pd.DataFrame:
        """
        Prépare le vecteur de features dans le bon ordre
        attendu par XGBoost.
        """
        row = {}
        for fname in self.feature_names:
            row[fname] = features.get(fname, 0.0)
        return pd.DataFrame([row])[self.feature_names]

    def _get_recommendation(self, confidence: float,
                             criticality: str,
                             anomaly: bool) -> str:
        """Génère une recommandation selon la criticité et la confiance."""
        if confidence >= CONFIDENCE_THRESHOLD:
            if criticality == "HIGH":
                return "⛔ Arrêt immédiat requis — intervention urgente"
            elif criticality == "MEDIUM":
                return "⚠️  Planifier maintenance dans les 4 heures"
            else:
                return "📋 Planifier maintenance dans les 24 heures"
        elif anomaly:
            return "🔍 Anomalie détectée — surveillance renforcée requise"
        else:
            return "✅ Fonctionnement normal"

    def _get_top_factors(self, shap_values: np.ndarray,
                          n: int = 3) -> list:
        """Retourne les N features avec le plus grand impact SHAP."""
        abs_shap = np.abs(shap_values)
        top_indices = np.argsort(abs_shap)[-n:][::-1]
        factors = []
        for i in top_indices:
            factors.append({
                "feature"      : self.feature_names[i],
                "shap_value"   : round(float(shap_values[i]), 4),
                "direction"    : "→ PANNE" if shap_values[i] > 0 else "→ NORMAL",
            })
        return factors

    def predict(self, features: dict, metadata: dict) -> dict:
        """
        Fait une prédiction complète sur un vecteur de features.

        Paramètres :
            features : dict — features engineered
            metadata : dict — infos machine (id, criticité, timestamp)

        Retourne :
            dict — résultat complet avec prédiction, confiance, SHAP
        """
        self.prediction_count += 1

        # Préparer l'input
        X = self._prepare_input(features)

        # ── XGBoost ────────────────────────────────────────────────────────
        xgb_proba      = float(self.xgb_model.predict_proba(X)[0][1])
        xgb_prediction = "FAILURE" if xgb_proba >= CONFIDENCE_THRESHOLD else "NORMAL"

        # ── Isolation Forest ───────────────────────────────────────────────
        iso_raw        = self.iso_forest.predict(X)[0]
        iso_anomaly    = bool(iso_raw == -1)
        iso_score      = float(-self.iso_forest.score_samples(X)[0])

        # ── SHAP (seulement si alerte pour économiser les ressources) ──────
        top_factors = []
        if xgb_prediction == "FAILURE" or iso_anomaly:
            shap_vals   = self.shap_explainer.shap_values(X)[0]
            top_factors = self._get_top_factors(shap_vals, n=3)

        # ── Alerte ─────────────────────────────────────────────────────────
        is_alert = xgb_prediction == "FAILURE" or iso_anomaly
        if is_alert:
            self.alert_count += 1

        # ── Recommandation ─────────────────────────────────────────────────
        criticality    = metadata.get("criticality", "MEDIUM")
        recommendation = self._get_recommendation(xgb_proba, criticality, iso_anomaly)

        # ── Résultat final ─────────────────────────────────────────────────
        result = {
            # Métadonnées
            "timestamp"              : metadata.get("timestamp"),
            "machine_id"             : metadata.get("machine_id"),
            "criticality"            : criticality,
            "sequence_number"        : metadata.get("sequence_number"),

            # Prédiction XGBoost
            "xgboost_prediction"     : xgb_prediction,
            "xgboost_confidence"     : round(xgb_proba, 4),

            # Isolation Forest
            "isolation_forest_anomaly": iso_anomaly,
            "anomaly_score"          : round(iso_score, 4),

            # Alerte globale
            "is_alert"               : is_alert,
            "recommendation"         : recommendation,

            # SHAP — top 3 facteurs
            "top_factors"            : top_factors,

            # Stats globales
            "total_predictions"      : self.prediction_count,
            "total_alerts"           : self.alert_count,
        }

        # Log si alerte
        if is_alert:
            logger.warning(
                f"🚨 ALERTE {metadata.get('machine_id')} | "
                f"XGB={xgb_proba:.2%} | "
                f"Anomalie={iso_anomaly} | "
                f"{recommendation}"
            )
        else:
            if self.prediction_count % 100 == 0:
                logger.info(
                    f"📊 {self.prediction_count} prédictions | "
                    f"{self.alert_count} alertes | "
                    f"Dernière machine : {metadata.get('machine_id')}"
                )

        return result
