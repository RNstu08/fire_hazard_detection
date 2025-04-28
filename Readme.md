
# 🚀 Fire Hazard Detection System

---

## 📁 Project Structure

```plaintext
fire_hazard_detection/
├── api/                  # Flask API for real-time inference
│   └── app.py
├── deployment/            # Docker and Kubernetes files
│   ├── Dockerfile
│   ├── fire-deployment.yaml
│   ├── fire-service.yaml
│   └── prometheus.yml
├── mlops/                 # MLflow setup, GitHub Actions (CI/CD)
├── preprocessing/         # Data cleaning, feature engineering
│   ├── data_generator.py
│   ├── preprocess.py
│   └── feature_engineering.py
├── training/              # Model training and evaluation
│   ├── train.py
│   ├── evaluate.py
│   └── models/
│       ├── logistic_regression.py
│       ├── random_forest.py
│       └── xgboost_model.py
├── data/                  # Generated and processed datasets
├── models/                # Saved machine learning models
├── requirements.txt       # Python dependencies
├── docker-compose.yml     # Local orchestration (Flask + Prometheus)
├── README.md              # 📑 You are here
└── .gitignore             # Ignore unnecessary files
```

---

#  Full Project Pipeline

We will **build, train, deploy, and monitor** a **Fire Hazard Detection** ML application, ready for **production** and **interviews**. Every phase is organized with **real-world standards**.

---

## Phase 1: Data Generation

---

### Goals for Phase 1:

- Generate **synthetic**, yet **statistically realistic** fire hazard data.
- Simulate fire events (5% fire label = 1, 95% normal = 0).
- Save the dataset as **CSV** files (no database used).

---

### Explanation:

- We simulate sensor data like **temperature**, **humidity**, **CO2 levels**, etc.
- **5%** of the samples are labeled as **fire event** (`fire_label = 1`).
- **95%** are **normal** (`fire_label = 0`).

### Script: `preprocessing/data_generator.py`

```bash
cd preprocessing/
python data_generator.py
```
Output:  
Creates `fire_hazard_data.csv` inside the `data/` folder.

---

## Phase 2: Preprocessing Module

---

### Goals for Phase 2:

| Step        | Description |
| ----------- | ----------- |
| 2.1 | Handle missing values gracefully |
| 2.2 | Smooth signals (rolling averages) to remove noise |
| 2.3 | Create new features: derivatives, rolling std, risk score |
| 2.4 | Normalize features (StandardScaler) |
| 2.5 | Split into train/test sets **chronologically** |
| 2.6 | Save clean datasets |

---

### Scripts:

- `preprocessing/preprocess.py`
- `preprocessing/feature_engineering.py`

### Commands:

```bash
cd preprocessing/
python preprocess.py
```

 Output:  
Creates `X_train_scaled.csv` and `X_test_scaled.csv` in the `data/` folder.

---

## Phase 3: Model Training Module

---

### Goals for Phase 3:

| Step | Description |
|------|-------------|
| 3.1 | Train Logistic Regression, Random Forest, XGBoost |
| 3.2 | Properly handle **imbalanced** classes (scale_pos_weight) |
| 3.3 | Evaluate using Precision, Recall, F1, AUC-ROC |
| 3.4 | Track experiments using MLflow |
| 3.5 | Save best model (`best_model_xgboost.joblib`) |

---

### Scripts:

- `training/train.py`
- `training/models/xgboost_model.py`
- `training/evaluate.py`

### Commands:

```bash
cd training/
python train.py
```
 Output:  
Trained models saved in `models/` directory.

---

## Phase 4: API Module (Flask Application)

---

### Goals for Phase 4:

| Step | Description |
|------|-------------|
| 4.1 | Create Flask app (`app.py`) |
| 4.2 | Load `best_model_xgboost.joblib` |
| 4.3 | Implement `/predict` endpoint (POST) |
| 4.4 | Validate requests, handle errors |
| 4.5 | Return predictions as JSON |
| 4.6 | Prepare Flask app for Dockerization |

---

### Script:

- `api/app.py`

### Run Locally:

```bash
cd api/
python app.py
```

 Output:  
Flask server running at `http://localhost:5000/predict`.

Example request:

```bash
curl -X POST http://localhost:5000/predict \
    -H "Content-Type: application/json" \
    -d '{"temperature":34, "humidity":40, "aqi":250, "co2_levels":700, "wind_speed":10}'
```

---

## Phase 5: Dockerization

---

### Goals for Phase 5:

| Step | Description |
|------|-------------|
| 5.1 | Containerize Flask API app using Docker |
| 5.2 | Use lightweight image (python:3.9-slim) |
| 5.3 | Expose port 5000 |
| 5.4 | Prepare for Kubernetes deployment |
| 5.5 | Optimize Dockerfile for small image size |
| 5.6 | Test Docker image locally |

---

### File:

- `deployment/Dockerfile`

### Commands:

```bash
cd deployment/
docker build -t fire-hazard-api .
docker run -p 5000:5000 fire-hazard-api
```

 Output:  
Flask app running inside a Docker container.

---

## Phase 6: Kubernetes Deployment

---

### Goals for Phase 6:

| Step | Description |
|------|-------------|
| 6.1 | Write Deployment YAML (`fire-deployment.yaml`) |
| 6.2 | Write Service YAML (`fire-service.yaml`) |
| 6.3 | Apply YAMLs to Kubernetes |
| 6.4 | Access API locally |

---

### Files:

- `deployment/fire-deployment.yaml`
- `deployment/fire-service.yaml`

### Commands:

```bash
kubectl apply -f deployment/fire-deployment.yaml
kubectl apply -f deployment/fire-service.yaml
kubectl get pods
kubectl get services
```
 Output:  
API exposed via Kubernetes LoadBalancer.

---

## Phase 7: MLOps & Monitoring

---

### Goals for Phase 7:

| Step | Description |
|------|-------------|
| 7.1 | Use MLflow for model tracking and experiment logging |
| 7.2 | Monitor API health using Prometheus |
| 7.3 | Visualize metrics using Grafana |
| 7.4 | Set up GitHub Actions for CI/CD |

---

### Files:

- `mlops/mlflow_setup.py`
- `deployment/prometheus.yml`
- `docker-compose.yml` (for local testing Prometheus + Flask)

### Local Run:

```bash
docker-compose up
```
 Output:  
- Prometheus scrapes API metrics.
- Access API at `localhost:5000`.
- Access Prometheus at `localhost:9090`.

---

# 📌 Requirements

Install Python dependencies:

```bash
pip install -r requirements.txt
```

---

# 📑 Additional Notes

- **save_data()** function: saves data in CSV format, no database needed.
- **Prometheus + Flask** integrated through a custom metrics endpoint.
- Best practices followed for:
  - **MLOps**
  - **Docker & K8s Deployment**
  - **Monitoring**
  - **API error handling**

---

# 🏁 Project Ready!

---