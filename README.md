# 🏭 Predictive Maintenance AIoT Pipeline

<div align="center">

![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Apache Kafka](https://img.shields.io/badge/Apache_Kafka-231F20?style=for-the-badge&logo=apachekafka&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![InfluxDB](https://img.shields.io/badge/InfluxDB-22ADF6?style=for-the-badge&logo=influxdb&logoColor=white)
![Grafana](https://img.shields.io/badge/Grafana-F46800?style=for-the-badge&logo=grafana&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)

**End-to-end AIoT pipeline for industrial predictive maintenance**

*IoT Simulation → Kafka Streaming → XGBoost + Isolation Forest + SHAP → Evidently AI Drift Detection → InfluxDB → Grafana → FastAPI*

</div>

---

## 🎯 Problem Statement

In industrial environments, an **unplanned machine breakdown costs an average of €250,000 per hour** of downtime. Traditional maintenance approaches are either:

- **Reactive** — repair after failure → expensive, unsafe
- **Preventive** — fixed schedule replacements → wasteful, inaccurate

**Predictive Maintenance** solves this by continuously monitoring machine health and predicting failures **before they happen**, enabling timely interventions that minimize costs and maximize uptime.

---

## 🧠 Solution Overview

This project builds a **complete real-time AIoT pipeline** that:

1. **Simulates** a fleet of 10 industrial machines sending sensor data every second
2. **Transports** data via MQTT → Apache Kafka in real time
3. **Enriches** data with advanced feature engineering (42 features: rolling stats, lag features, derivatives)
4. **Predicts** failures using XGBoost (supervised) and detects unknown anomalies with Isolation Forest
5. **Explains** each prediction using SHAP (which sensors contributed most to the alert)
6. **Monitors** model health with Evidently AI (data drift detection)
7. **Retrains** the model automatically when drift is detected
8. **Alerts** maintenance teams with context and recommendations
9. **Exposes** a JWT-secured REST API for integration with external systems
10. **Visualizes** everything on a professional Grafana dashboard

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     IoT SIMULATION LAYER                        │
│  [10 Industrial Machines] → MQTT (noise + packet loss)          │
│         ↓                                                       │
│  [Eclipse Mosquitto] → MQTT-Kafka Bridge                        │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                     STREAMING LAYER                             │
│  [Apache Kafka] topic: sensor-data          [MLflow Tracking]   │
│         ↓                                   (experiments +      │
│  [Real-time Feature Engine]                  versioned models)  │
│  (42 features: rolling, lag, derivatives)                       │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                     AI/ML LAYER                                 │
│  [XGBoost]          → failure probability (F1=0.84, AUC=0.96)  │
│  [Isolation Forest] → unknown anomaly detection                 │
│  [SHAP]             → prediction explanation                    │
└──────────────┬──────────────────────────┬───────────────────────┘
               │                          │
┌──────────────▼──────────┐  ┌────────────▼───────────────────────┐
│     STORAGE LAYER       │  │        MLOPS LAYER                 │
│  [InfluxDB]             │  │  [Evidently AI] drift detection    │
│  ├── sensors            │  │         ↓                          │
│  ├── predictions        │  │  [Auto Retraining Pipeline]        │
│  ├── alerts             │  │  (8-step pipeline, 7.1s)           │
│  └── model_health       │  │         ↓                          │
└──────────────┬──────────┘  │  [MLflow] model versioning         │
               │             └────────────────────────────────────┘
┌──────────────▼──────────────────────────────────────────────────┐
│                  VISUALIZATION & API LAYER                      │
│  [Grafana Dashboard]     [FastAPI + JWT]                        │
│  ├── Fleet overview      ├── POST /predict                      │
│  ├── Machine details     ├── GET  /machines/{id}/status         │
│  ├── SHAP visualization  ├── GET  /alerts/recent                │
│  └── Model health        └── GET  /model/health                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **IoT / Simulation** | Python, Paho-MQTT, Eclipse Mosquitto |
| **Streaming** | Apache Kafka, Zookeeper |
| **ML / AI** | XGBoost, Isolation Forest, SHAP, Scikit-Learn, SMOTE |
| **MLOps** | MLflow, Evidently AI |
| **Storage** | InfluxDB 2.7 |
| **API** | FastAPI, JWT, Pydantic |
| **Visualization** | Grafana 10.2 |
| **Infrastructure** | Docker, Docker Compose |
| **Language** | Python 3.13 |

---

## 📊 ML Performance

| Model | Metric | Score |
|-------|--------|-------|
| XGBoost | F1-Score | **0.84** |
| XGBoost | AUC-ROC | **0.96** |
| XGBoost | Average Precision | **0.87** |
| Isolation Forest | AUC-ROC | **0.84** |

**Dataset:** AI4I 2020 Predictive Maintenance — 10,000 samples, 3.4% failure rate

**Feature Engineering:** 42 features from 6 raw sensors:
- Thermal: `temp_diff`, `temp_ratio`, `temp_variation`
- Mechanical: `power`, `wear_rate`, `torque_wear_interaction`
- Rolling statistics: mean/std over windows of 5 and 10
- Lag features: values at t-1, t-3, t-5
- Delta features: rate of change

---

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.10+
- Git

### 1. Clone the repository
```bash
git clone https://github.com/oussema-wess/predictive-maintenance-aiot-pipeline.git
cd predictive-maintenance-aiot-pipeline
```

### 2. Launch the infrastructure
```bash
docker-compose up -d
```

### 3. Set up Python environment
```bash
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 4. Train the models
```bash
jupyter notebook notebooks/03_training.ipynb
```

### 5. Launch the pipeline (4 terminals)
```bash
# Terminal 1 — IoT Simulator
python iot_simulator/simulateur_iot.py --mode mixed --interval 2

# Terminal 2 — MQTT→Kafka Bridge
python kafka_pipeline/producer.py

# Terminal 3 — Real-time ML Pipeline
python ml_processor/processeur_temps_reel.py

# Terminal 4 — REST API
python api/main.py
```

### 6. Access the services

| Service | URL | Credentials |
|---------|-----|-------------|
| 🎛️ Grafana Dashboard | http://localhost:3000 | admin / adminpassword |
| 📊 MLflow UI | http://localhost:5000 | — |
| 💾 InfluxDB UI | http://localhost:8086 | admin / adminpassword |
| 🔌 FastAPI Docs | http://localhost:8000/docs | — |
| 📨 Kafka UI | http://localhost:8080 | — |

---

## 📁 Project Structure

```
predictive-maintenance-aiot-pipeline/
│
├── data/
│   ├── raw/                        # AI4I 2020 dataset
│   └── processed/                  # Engineered features (42 cols)
│
├── notebooks/
│   ├── 01_EDA.ipynb                # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_training.ipynb           # XGBoost + SHAP + MLflow
│
├── models/
│   ├── xgboost_model.pkl           # Trained XGBoost (F1=0.84)
│   ├── isolation_forest.pkl        # Anomaly detector
│   └── feature_names.pkl           # 42 feature names
│
├── iot_simulator/
│   └── simulateur_iot.py           # 10 machines, 4 modes, noise
│
├── kafka_pipeline/
│   ├── producer.py                 # MQTT → Kafka bridge
│   └── consumer.py
│
├── ml_processor/
│   ├── feature_engineering.py      # Real-time 42-feature engine
│   ├── predictor.py                # XGBoost + IF + SHAP inference
│   └── processeur_temps_reel.py    # Main pipeline orchestrator
│
├── model_monitoring/
│   ├── drift_detector.py           # Evidently AI drift detection
│   └── retraining_trigger.py       # 8-step auto retraining pipeline
│
├── api/
│   ├── main.py                     # FastAPI (7 endpoints)
│   ├── auth.py                     # JWT authentication
│   └── schemas.py                  # Pydantic models
│
├── monitoring/
│   ├── grafana/datasources/
│   └── mosquitto/
│
├── docker-compose.yml              # 7 services in 1 command
├── requirements.txt
└── README.md
```

---

## 🔌 API Reference

```bash
# Health check (public)
GET  http://localhost:8000/health

# Authentication
POST http://localhost:8000/auth/token
     username=admin&password=secret

# Predict failure (JWT required)
POST http://localhost:8000/predict
     {"machine_id": "MACHINE_003", "air_temperature": 298.5,
      "process_temperature": 308.2, "rotational_speed": 1408,
      "torque": 46.3, "tool_wear": 108, "machine_type": "M",
      "criticality": "MEDIUM"}

# Machine status (JWT required)
GET  http://localhost:8000/machines/MACHINE_003/status

# Recent alerts (JWT required)
GET  http://localhost:8000/alerts/recent

# Model health (JWT required)
GET  http://localhost:8000/model/health
```

---

## 🗺️ Roadmap

- [x] Step 0 — Project structure & GitHub setup
- [x] Step 1 — EDA + Feature Engineering (42 features) + XGBoost + SHAP + MLflow
- [x] Step 2 — Docker infrastructure (Kafka, InfluxDB, Grafana, MLflow, Mosquitto)
- [x] Step 3 — IoT simulator (10 machines, 4 modes, noise, packet loss)
- [x] Step 4 — Real-time ML pipeline (Kafka → Features → XGBoost → InfluxDB)
- [x] Step 5 — Model monitoring (Evidently AI) + Auto retraining (8-step, 7.1s)
- [x] Step 6 — REST API (FastAPI + JWT, 7 endpoints)
- [x] Step 7 — Grafana dashboard (5 panels, machine filter, real-time)
- [ ] Step 8 — Telegram alerting + Demo video

---

## 📖 Dataset

**AI4I 2020 Predictive Maintenance Dataset**
- Source: [Kaggle](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020)
- 10,000 samples, 6 sensor features, 3.4% failure rate
- Imbalanced → handled with SMOTE (sampling_strategy=0.3)

---

## 👤 Author

**Oussema** — [GitHub](https://github.com/oussema-wess)

---

## 📄 License

MIT License
