# 📈 Goldman Sachs Stock Price Prediction

> End-to-end ML pipeline for forecasting Goldman Sachs (GS) stock prices using ARIMA, Prophet, XGBoost, and LSTM — served via FastAPI, tracked with MLflow, and deployed on AWS EC2 with Docker Compose.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)
![MLflow](https://img.shields.io/badge/MLflow-2.3.0-orange?logo=mlflow)
![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker)
![AWS](https://img.shields.io/badge/AWS-EC2-orange?logo=amazonaws)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU-red?logo=pytorch)

---

## 📌 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Models](#models)
- [API Endpoints](#api-endpoints)
- [Local Setup](#local-setup)
- [AWS EC2 Deployment](#aws-ec2-deployment)
- [MLflow Experiment Tracking](#mlflow-experiment-tracking)
- [Docker Configuration](#docker-configuration)
- [Results](#results)

---

## Overview

This project builds a production-grade stock price forecasting system for Goldman Sachs (ticker: `GS`). It trains four different time-series and ML models, exposes predictions through a REST API, tracks all experiments using MLflow, and is fully containerized and deployed on AWS EC2.

The system supports:
- Historical stock price ingestion and feature engineering
- Multi-model training pipeline (ARIMA, Prophet, XGBoost, LSTM)
- REST API for real-time forecasting with configurable horizons
- MLflow UI for experiment tracking and model comparison
- Docker-based deployment on AWS EC2 (Mumbai region)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CLIENT / BROWSER                             │
└────────────────────┬───────────────────────┬────────────────────────┘
                     │                       │
                     ▼                       ▼
           Port 8000 (FastAPI)       Port 5002 (MLflow UI)
                     │                       │
┌────────────────────▼───────────────────────▼────────────────────────┐
│                        AWS EC2 (t3.micro)                           │
│                       Ubuntu 24.04 LTS                              │
│                    ap-south-1b (Mumbai)                             │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Docker Network (gs-net)                   │   │
│  │                                                              │   │
│  │   ┌─────────────────────┐    ┌──────────────────────────┐   │   │
│  │   │     gs-api          │    │      gs-mlflow           │   │   │
│  │   │   (FastAPI +        │    │   (MLflow Server +       │   │   │
│  │   │    Gunicorn)        │    │    TCP Proxy :5001)      │   │   │
│  │   │                     │    │                          │   │   │
│  │   │  • ARIMA model      │    │  MLflow :5000 (internal) │   │   │
│  │   │  • Prophet model    │    │  TCP Proxy :5001         │   │   │
│  │   │  • XGBoost model    │    │         │                │   │   │
│  │   │  • LSTM model       │    └─────────┼────────────────┘   │   │
│  │   │                     │              │                     │   │
│  │   └─────────────────────┘              │                     │   │
│  └──────────────────────────────────────── ─────────────────────┘   │
│                                           │                         │
│   ┌───────────────────────────────────────▼─────────────────────┐   │
│   │              Nginx Reverse Proxy                             │   │
│   │   Port 5002  ──►  172.18.0.x:5001  ──►  MLflow :5000       │   │
│   └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│   ┌──────────────────────────────────────────────────────────────┐   │
│   │              EBS Volume (20 GB)                              │   │
│   │   ~/gs_stock_prediction/mlruns   (MLflow artifact store)    │   │
│   └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘

AWS Security Group Inbound Rules:
  ┌──────────┬──────────┬─────────────┐
  │ Port     │ Protocol │ Source      │
  ├──────────┼──────────┼─────────────┤
  │ 22       │ TCP      │ My IP       │
  │ 8000     │ TCP      │ 0.0.0.0/0   │
  │ 5002     │ TCP      │ 0.0.0.0/0   │
  └──────────┴──────────┴─────────────┘
```

### Data Flow

```
Yahoo Finance API
      │
      ▼
Data Ingestion (yfinance)
      │
      ▼
Feature Engineering
  ├── Moving Averages (MA7, MA20, MA50)
  ├── RSI, MACD, Bollinger Bands
  ├── Lag Features
  └── Volume Indicators
      │
      ▼
┌─────────────────────────────────────┐
│         Training Pipeline           │
│  ┌─────────┐  ┌──────────────────┐  │
│  │  ARIMA  │  │     Prophet      │  │
│  └────┬────┘  └────────┬─────────┘  │
│  ┌────▼────┐  ┌────────▼─────────┐  │
│  │ XGBoost │  │      LSTM        │  │
│  └────┬────┘  └────────┬─────────┘  │
└───────┼────────────────┼────────────┘
        │                │
        └────────┬───────┘
                 ▼
           MLflow Tracking
     (metrics, params, artifacts)
                 │
                 ▼
         Saved Model Files
                 │
                 ▼
         FastAPI Service
                 │
      ┌──────────┴──────────┐
      ▼                     ▼
  /forecast             /health
  /models               /metrics
```

---

## Features

- **Multi-model forecasting** — ARIMA, Facebook Prophet, XGBoost, LSTM in a single unified pipeline
- **REST API** — FastAPI with automatic Swagger UI at `/docs`
- **Experiment tracking** — MLflow UI with metrics, parameters, and artifacts logged per run
- **Containerized** — Docker Compose orchestrates API and MLflow services
- **Cloud deployed** — Running live on AWS EC2 (Mumbai region)
- **Feature engineering** — Technical indicators, lag features, rolling statistics
- **Configurable forecast horizon** — Predict N days ahead via API parameter
- **Health checks** — API and model status endpoints

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.11 |
| **API Framework** | FastAPI + Gunicorn + Uvicorn |
| **ML Models** | statsmodels (ARIMA), Prophet, XGBoost, PyTorch (LSTM) |
| **Experiment Tracking** | MLflow 2.3.0 |
| **Data Source** | yfinance (Yahoo Finance) |
| **Feature Engineering** | pandas, numpy, ta-lib |
| **Containerization** | Docker + Docker Compose |
| **Web Server** | Nginx (reverse proxy) |
| **Cloud** | AWS EC2 t3.micro, Ubuntu 24.04, EBS 20GB |
| **Region** | ap-south-1b (Mumbai) |

---

## Project Structure

```
gs_stock_prediction/
│
├── api/
│   ├── main.py                  # FastAPI app entry point
│   ├── routes/
│   │   ├── forecast.py          # /api/v1/forecast endpoint
│   │   ├── health.py            # /health endpoint
│   │   └── models.py            # /models endpoint
│   └── schemas/
│       ├── request.py           # Pydantic request models
│       └── response.py          # Pydantic response models
│
├── src/
│   ├── data/
│   │   ├── ingestion.py         # yfinance data fetcher
│   │   └── preprocessing.py     # Feature engineering
│   ├── models/
│   │   ├── arima_model.py       # ARIMA wrapper
│   │   ├── prophet_model.py     # Prophet wrapper
│   │   ├── xgboost_model.py     # XGBoost wrapper
│   │   └── lstm_model.py        # PyTorch LSTM
│   └── evaluation/
│       └── metrics.py           # RMSE, MAE, MAPE calculations
│
├── pipelines/
│   └── train_pipeline.py        # End-to-end training + MLflow logging
│
├── deployment/
│   ├── Dockerfile               # API container image
│   └── docker-compose.yml       # Multi-container orchestration
│
├── mlruns/                      # MLflow experiment artifacts (auto-generated)
├── models/                      # Saved model files (auto-generated)
├── data/                        # Raw and processed data (auto-generated)
│
├── requirements.txt
└── README.md
```

---

## Models

### 1. ARIMA
Classical statistical time-series model. Captures autoregressive and moving average components of GS stock price.
- **Params:** order `(p, d, q)` tuned via AIC/BIC
- **Use case:** Short-term trend forecasting
- **Speed:** Fast (seconds)

### 2. Facebook Prophet
Additive forecasting model by Meta. Handles seasonality, holidays, and trend changes automatically.
- **Params:** yearly/weekly seasonality, changepoint prior scale
- **Use case:** Multi-period forecasting with seasonality
- **Speed:** Fast (seconds)

### 3. XGBoost
Gradient boosted trees with engineered time-series features (lags, rolling stats, technical indicators).
- **Params:** n_estimators, max_depth, learning_rate, subsample
- **Features:** MA7, MA20, RSI, MACD, lag-1 to lag-10 prices
- **Use case:** Non-linear pattern capture
- **Speed:** Fast (seconds)

### 4. LSTM (PyTorch)
Deep learning sequence model. Learns long-term temporal dependencies in price sequences.
- **Architecture:** 2-layer LSTM → Dense → Output
- **Params:** hidden_size=64, num_layers=2, sequence_length=60, epochs=50
- **Use case:** Complex temporal pattern learning
- **Speed:** Slow on CPU (10–60 min on t3.micro)

---

## API Endpoints

Base URL: `http://13.203.78.238:8000`

### `GET /health`
Returns API and model status.
```json
{
  "status": "healthy",
  "models_loaded": ["arima", "prophet", "xgboost", "lstm"]
}
```

### `POST /api/v1/forecast`
Get stock price forecast.

**Request:**
```json
{
  "model": "xgboost",
  "ticker": "GS",
  "horizon": 30
}
```

**Response:**
```json
{
  "ticker": "GS",
  "model": "xgboost",
  "forecast": [
    {"date": "2024-01-15", "predicted_price": 412.50},
    {"date": "2024-01-16", "predicted_price": 415.20}
  ],
  "metrics": {
    "rmse": 8.34,
    "mae": 6.12,
    "mape": 1.89
  }
}
```

### `GET /api/v1/models`
Lists all available models and their status.

**Swagger UI:** `http://13.203.78.238:8000/docs`

---

## Local Setup

### Prerequisites
- Python 3.11+
- Git

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/gs_stock_prediction.git
cd gs_stock_prediction

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Run training pipeline
export MLFLOW_TRACKING_URI=file:///$(pwd)/mlruns

python pipelines/train_pipeline.py --model arima
python pipelines/train_pipeline.py --model prophet
python pipelines/train_pipeline.py --model xgboost
python pipelines/train_pipeline.py --model lstm   # Optional — slow on CPU

# 5. Start MLflow UI locally
mlflow ui --host 0.0.0.0 --port 5000
# Open: http://localhost:5000

# 6. Start FastAPI locally
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
# Open: http://localhost:8000/docs
```

---

## AWS EC2 Deployment

### Infrastructure

| Setting | Value |
|---|---|
| Instance Type | t3.micro (1 vCPU, 1GB RAM) |
| OS | Ubuntu 24.04 LTS |
| Region | ap-south-1b (Mumbai) |
| Storage | EBS 20 GB (gp2) |
| Public IP | 13.203.78.238 |

### Step 1 — Launch EC2 Instance

1. Go to AWS Console → EC2 → Launch Instance
2. Choose **Ubuntu 24.04 LTS**
3. Select **t3.micro** (free tier eligible)
4. Create or select a key pair (`.pem` file)
5. Configure Security Group — add inbound rules:

```
Port 22    → SSH          → My IP
Port 8000  → FastAPI      → 0.0.0.0/0
Port 5002  → MLflow UI    → 0.0.0.0/0
```

6. Set storage to **20 GB** (default 8 GB is too small for Docker + PyTorch)
7. Launch instance

### Step 2 — Connect via SSH

```bash
# Windows (PowerShell)
ssh -i C:\Users\<you>\Downloads\ubuntu-key.pem ubuntu@<EC2-PUBLIC-IP>

# Linux/Mac
chmod 400 ubuntu-key.pem
ssh -i ubuntu-key.pem ubuntu@<EC2-PUBLIC-IP>
```

### Step 3 — Upload Project to EC2

```bash
# From your local machine (PowerShell/Terminal)
scp -i C:\Users\<you>\Downloads\ubuntu-key.pem -r `
  C:\path\to\gs_stock_prediction `
  ubuntu@<EC2-PUBLIC-IP>:~/gs_stock_prediction
```

### Step 4 — Install Docker on EC2

```bash
# SSH into EC2 first, then run:
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker ubuntu

# Install Docker Compose plugin
sudo apt-get install -y docker-compose-plugin

# Verify
docker --version
docker compose version
```

### Step 5 — Install Nginx

```bash
sudo apt-get install -y nginx
sudo systemctl enable nginx
sudo systemctl start nginx
```

### Step 6 — Build and Start Containers

```bash
cd ~/gs_stock_prediction/deployment
sudo docker compose build
sudo docker compose up -d

# Verify containers are running
sudo docker compose ps
```

Expected output:
```
NAME         STATUS    PORTS
gs-api       running   0.0.0.0:8000->8000/tcp
gs-mlflow    running   0.0.0.0:5000->5000/tcp
```

### Step 7 — Configure MLflow Access via Nginx

MLflow's security middleware binds only to `127.0.0.1` internally. We use a Python TCP proxy inside the container + Nginx on the host to expose it externally.

**Start TCP proxy inside MLflow container:**
```bash
sudo docker exec -d gs-mlflow python3 -c "
import socket,threading
def handle(c):
    s=socket.socket();s.connect(('127.0.0.1',5000))
    def fwd(a,b):
        while True:
            d=a.recv(4096)
            if not d:break
            b.sendall(d)
    threading.Thread(target=fwd,args=(c,s),daemon=True).start()
    threading.Thread(target=fwd,args=(s,c),daemon=True).start()
sock=socket.socket();sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
sock.bind(('0.0.0.0',5001));sock.listen(10)
while True:
    c,_=sock.accept();threading.Thread(target=handle,args=(c,),daemon=True).start()
"
```

**Configure Nginx:**
```bash
# Get MLflow container IP
sudo docker inspect gs-mlflow | grep IPAddress
# Example: 172.18.0.2

sudo nano /etc/nginx/sites-available/mlflow
```

Paste:
```nginx
server {
    listen 5002;
    location / {
        proxy_pass http://172.18.0.2:5001;
        proxy_set_header Host localhost;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/mlflow /etc/nginx/sites-enabled/mlflow
sudo nginx -t
sudo systemctl reload nginx
```

### Step 8 — Run Training Pipeline on EC2

```bash
cd ~/gs_stock_prediction
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install "greenlet>=3.0.0"
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install prophet
pip install -r requirements.txt

export MLFLOW_TRACKING_URI=file:///home/ubuntu/gs_stock_prediction/mlruns

python pipelines/train_pipeline.py --model arima
python pipelines/train_pipeline.py --model xgboost
python pipelines/train_pipeline.py --model prophet
```

### Step 9 — Verify Deployment

```bash
# Test FastAPI
curl http://localhost:8000/health

# Test from browser
# FastAPI Docs: http://<EC2-PUBLIC-IP>:8000/docs
# MLflow UI:    http://<EC2-PUBLIC-IP>:5002
```

### Expand EBS Volume (if disk full)

If you hit disk space errors during Docker build:

```bash
# After expanding volume in AWS Console:
sudo growpart /dev/nvme0n1 1
sudo resize2fs /dev/nvme0n1p1
df -h  # Verify new size
```

### After EC2 Reboot

On reboot the public IP changes (unless Elastic IP is assigned) and the MLflow TCP proxy stops. Run:

```bash
# Restart containers
cd ~/gs_stock_prediction/deployment
sudo docker compose up -d

# Restart TCP proxy (get new container IP first)
sudo docker inspect gs-mlflow | grep IPAddress

# Restart proxy with correct IP, then reload nginx
sudo sed -i 's|proxy_pass http://.*:5001|proxy_pass http://<NEW-IP>:5001|' \
  /etc/nginx/sites-available/mlflow
sudo nginx -s reload
```

> **Tip:** Assign an AWS Elastic IP to your EC2 instance to keep a permanent public IP address.

---

## MLflow Experiment Tracking

MLflow UI is accessible at: `http://13.203.78.238:5002`

Each training run logs:

**Parameters tracked:**
- Model type (arima/prophet/xgboost/lstm)
- Hyperparameters per model
- Training data date range
- Feature list used

**Metrics tracked:**
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- Training duration

**Artifacts stored:**
- Trained model file
- Feature importance plot (XGBoost)
- Forecast vs actual plot
- Evaluation report

### Experiment Structure

```
GS_Stock_Prediction/
├── full_pipeline_arima/
│   ├── params: order=(2,1,2), ticker=GS
│   └── metrics: rmse=12.4, mae=9.1, mape=2.3
├── full_pipeline_prophet/
│   ├── params: seasonality=yearly, changepoint=0.05
│   └── metrics: rmse=14.2, mae=11.0, mape=2.7
├── full_pipeline_xgboost/
│   ├── params: n_estimators=200, max_depth=6, lr=0.05
│   └── metrics: rmse=8.3, mae=6.1, mape=1.9
└── full_pipeline_lstm/
    ├── params: hidden=64, layers=2, seq_len=60, epochs=50
    └── metrics: rmse=9.1, mae=7.2, mape=2.1
```

---

## Docker Configuration

### Dockerfile

```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y gcc g++ curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["gunicorn", "api.main:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "1", \
     "--timeout", "120"]
```

> CPU-only PyTorch is used to keep the image size manageable on t3.micro.

### docker-compose.yml

```yaml
services:
  gs-api:
    build: .
    container_name: gs-api
    ports:
      - "8000:8000"
    volumes:
      - ../mlruns:/app/mlruns
      - ../models:/app/models
    networks:
      - gs-net
    restart: always

  gs-mlflow:
    image: python:3.11-slim
    container_name: gs-mlflow
    command: >
      bash -c "pip install mlflow==2.3.0 &&
               mlflow server --host 0.0.0.0 --port 5000
               --backend-store-uri file:///mlruns
               --default-artifact-root file:///mlruns"
    ports:
      - "5000:5000"
    volumes:
      - ../mlruns:/mlruns
    networks:
      - gs-net
    restart: always

networks:
  gs-net:
    driver: bridge
```

---

## Results

| Model | RMSE | MAE | MAPE | Training Time |
|---|---|---|---|---|
| ARIMA | ~12.4 | ~9.1 | ~2.3% | < 10s |
| Prophet | ~14.2 | ~11.0 | ~2.7% | < 15s |
| XGBoost | ~8.3 | ~6.1 | ~1.9% | < 10s |
| LSTM | ~9.1 | ~7.2 | ~2.1% | ~45 min (CPU) |

> XGBoost achieves the best RMSE with feature engineering on GS historical data.

---

## Live Endpoints

| Service | URL |
|---|---|
| FastAPI Swagger UI | http://13.203.78.238:8000/docs |
| FastAPI Health | http://13.203.78.238:8000/health |
| MLflow UI | http://13.203.78.238:5002 |

---

## License

MIT License — feel free to fork and build on this.

---

## Author

Built and deployed end-to-end as a demonstration of a production ML system — from data ingestion and model training to cloud deployment with experiment tracking.
