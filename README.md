# 🏭 Predictive Maintenance AIoT Pipeline

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Apache Kafka](https://img.shields.io/badge/Apache_Kafka-231F20?style=flat&logo=apachekafka&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat&logo=mlflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![InfluxDB](https://img.shields.io/badge/InfluxDB-22ADF6?style=flat&logo=influxdb&logoColor=white)
![Grafana](https://img.shields.io/badge/Grafana-F46800?style=flat&logo=grafana&logoColor=white)
![Status](https://img.shields.io/badge/Status-In_Progress-orange?style=flat)

> **End-to-end AIoT pipeline for industrial predictive maintenance** — IoT simulation, Kafka streaming, XGBoost + Isolation Forest, Evidently AI drift detection, InfluxDB, Grafana alerting & FastAPI inference endpoint.

---

## 🎯 Problem Statement

In industrial environments, an unplanned machine breakdown costs an average of **€250,000 per hour** of downtime. Traditional maintenance approaches are either:
- **Reactive** — repair after failure (expensive, unsafe)
- **Preventive** — replace parts on a fixed schedule (wasteful, inaccurate)

**Predictive Maintenance** solves this by continuously monitoring machine health and predicting failures **before they happen**, enabling timely interventions that minimize costs and downtime.

---

## 🧠 Solution Overview

This project builds a complete real-time AIoT pipeline that:

1. **Simulates** a fleet of 10 industrial machines sending sensor data every second
2. **Transports** data via MQTT → Apache Kafka in real time
3. **Enriches** data with advanced feature engineering (rolling stats, lag features, derivatives)
4. **Predicts** failures using XGBoost (supervised) and detects unknown anomalies with Isolation Forest
5. **Explains** each prediction using SHAP (which sensors contributed most to the alert)
6. **Monitors** model health with Evidently AI (data drift detection)
7. **Retrains** the model automatically when drift is detected
8. **Alerts** maintenance teams via Telegram with context and recommendations
9. **Exposes** a JWT-secured REST API for integration with external systems
10. **Visualizes** everything on a professional Grafana dashboard

---

## 🏗️ Architecture

```
[10 IoT Machines] 
      │ MQTT (JSON + noise + packet loss)
      ▼
[Eclipse Mosquitto]
      │ MQTT-Kafka Bridge
      ▼
[Apache Kafka] ──────────────────────────── [MLflow Tracking]
  topic: sensor-data                         (experiments +
      │                                       versioned models)
      ▼
[Real-time Feature Engine]
  (rolling mean, lag features, derivatives)
      │
      ▼
[ML Processor]
  ├── XGBoost      → failure probability
  ├── Isolation Forest → unknown anomaly
  └── SHAP         → prediction explanation
      │
      ├────────────────────────────────────┐
      ▼                                    ▼
[InfluxDB]                         [Evidently AI]
  ├── sensors                        drift detection
  ├── predictions                          │
  ├── alerts                               ▼
  └── model_health                  [Auto Retraining]
      │
      ▼
[Grafana Dashboard] ── Alerts ──► [Telegram Bot]
  ├── Fleet overview
  ├── Machine details + SHAP
  └── Model health & drift

      +

[FastAPI] ◄── JWT Auth ── External Systems
  ├── POST /predict
  ├── GET  /machines/{id}/status
  └── GET  /model/health
```

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **IoT / Simulation** | Python, Paho-MQTT, Eclipse Mosquitto |
| **Streaming** | Apache Kafka, Zookeeper |
| **ML / AI** | XGBoost, Isolation Forest, SHAP, Scikit-Learn, SMOTE |
| **MLOps** | MLflow, Evidently AI |
| **Storage** | InfluxDB |
| **API** | FastAPI, JWT, Pydantic |
| **Visualization** | Grafana |
| **Infrastructure** | Docker, Docker Compose |
| **Language** | Python 3.10+ |

---

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose installed
- Python 3.10+
- Git

### Launch the full infrastructure

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/predictive-maintenance-aiot-pipeline.git
cd predictive-maintenance-aiot-pipeline

# 2. Copy and configure environment variables
cp .env.example .env

# 3. Launch all services
docker-compose up -d
```

### Set up Python environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

pip install -r requirements.txt
```

### Access the services

| Service | URL | Credentials |
|---------|-----|-------------|
| Grafana Dashboard | http://localhost:3000 | admin / admin |
| MLflow UI | http://localhost:5000 | — |
| InfluxDB UI | http://localhost:8086 | see .env |
| FastAPI Docs | http://localhost:8000/docs | — |

---

## 📁 Project Structure

```
predictive-maintenance-aiot-pipeline/
│
├── data/
│   ├── raw/                        # Original Kaggle dataset
│   └── processed/                  # Cleaned data + engineered features
│
├── notebooks/
│   ├── 01_EDA.ipynb                # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_training.ipynb           # Model training + evaluation + SHAP
│
├── models/                         # Saved .pkl models
│
├── iot_simulator/
│   └── simulateur_iot.py           # Multi-machine IoT simulator
│
├── kafka_pipeline/
│   ├── producer.py
│   └── consumer.py
│
├── ml_processor/
│   ├── feature_engineering.py      # Real-time feature engine
│   ├── predictor.py                # XGBoost + Isolation Forest inference
│   └── processeur_temps_reel.py    # Main real-time pipeline
│
├── model_monitoring/
│   ├── drift_detector.py           # Evidently AI drift detection
│   └── retraining_trigger.py       # Automatic retraining pipeline
│
├── api/
│   ├── main.py                     # FastAPI application
│   ├── auth.py                     # JWT authentication
│   └── schemas.py                  # Pydantic request/response models
│
├── monitoring/
│   ├── grafana/dashboards/         # Grafana dashboard JSON configs
│   └── influxdb/init.sh            # InfluxDB initialization
│
├── mlflow_tracking/
│   └── mlflow_config.py
│
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## 📊 ML Performance

> *Results will be updated after completing Step 1 — Model Training*

| Model | Metric | Score |
|-------|--------|-------|
| XGBoost | F1-Score | TBD |
| XGBoost | AUC-ROC | TBD |
| Isolation Forest | Anomaly Detection Rate | TBD |

---

## 🗺️ Roadmap

- [x] Step 0 — Project structure & documentation
- [ ] Step 1 — EDA, Feature Engineering, ML training + SHAP + MLflow
- [ ] Step 2 — Big Data infrastructure (Kafka + InfluxDB + Docker)
- [ ] Step 3 — Realistic IoT simulator (10 machines, noise, failures)
- [ ] Step 4 — Real-time processing pipeline
- [ ] Step 5 — Model monitoring + automatic retraining (Evidently AI)
- [ ] Step 6 — Secured REST API (FastAPI + JWT)
- [ ] Step 7 — Grafana dashboard + Telegram alerting
- [ ] Step 8 — Documentation + architecture diagram + demo video

---

## 📖 Dataset

**AI4I 2020 Predictive Maintenance Dataset**
- Source: [UCI Machine Learning Repository / Kaggle](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020)
- 10,000 data points with 6 sensor features
- Binary target: machine failure (1) / normal (0)
- Failure rate: ~3.4% (imbalanced dataset)

---

## 👤 Author

**[Your Name]**
- LinkedIn: [your-linkedin]
- GitHub: [your-github]

---

## 📄 License

This project is licensed under the MIT License.
