# Parkinson ML API

A production-oriented machine learning API for Parkinson's disease prediction.

This project demonstrates how to take a trained scikit-learn model and turn it into a structured, validated, tested, and containerized inference service using **Litestar, Pydantic, pytest, and Docker**.

> **Disclaimer:** This project is for educational purposes only and is not a medical diagnostic tool.

## Features

- Pre-trained scikit-learn Random Forest model
- Joblib model serialization
- Litestar REST API
- Pydantic request validation
- Dedicated prediction service
- Structured application logging
- Automated API, schema, and model tests
- Dockerized deployment
- Non-root Docker container
- Health check endpoint

## Architecture

```text
Client
  |
  | POST /predict
  v
Litestar API
  |
  v
Pydantic Validation
  |
  v
Prediction Service
  |
  v
scikit-learn Model
  |
  v
Prediction + Probability
Tech Stack
Component	Technology
Language	Python 3.10
API	Litestar
Server	Uvicorn
Validation	Pydantic
ML	scikit-learn
Data	pandas
Serialization	Joblib
Testing	pytest
Containerization	Docker
Project Structure
Parkinson_MachineModel/
│
├── app/
│   ├── api/
│   │   └── prediction.py
│   ├── core/
│   │   └── config.py
│   ├── schemas/
│   │   └── prediction.py
│   ├── services/
│   │   └── prediction_service.py
│   ├── logging_config.py
│   └── main.py
│
├── artifacts/
│   └── parkinson_model.joblib
│
├── notebooks/
│   └── Last_Parkinson_Model.ipynb
│
├── scripts/
│   └── train.py
│
├── tests/
│   ├── test_api.py
│   ├── test_model.py
│   └── test_schema.py
│
├── Dockerfile
├── requirements.txt
└── pytest.ini
API
Health Check
GET /health

Example:

curl http://127.0.0.1:8000/health

Response:

{
  "status": "healthy",
  "service": "parkinson-ml-api"
}
Prediction
POST /predict

The endpoint accepts the 22 numerical features used by the trained model.

Example:

curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "MDVP:Fo(Hz)": 119.992,
    "MDVP:Fhi(Hz)": 157.302,
    "MDVP:Flo(Hz)": 74.997,
    "MDVP:Jitter(%)": 0.00784,
    "MDVP:Jitter(Abs)": 0.00007,
    "MDVP:RAP": 0.00370,
    "MDVP:PPQ": 0.00554,
    "Jitter:DDP": 0.01109,
    "MDVP:Shimmer": 0.04374,
    "MDVP:Shimmer(dB)": 0.426,
    "Shimmer:APQ3": 0.02182,
    "Shimmer:APQ5": 0.03130,
    "MDVP:APQ": 0.02971,
    "Shimmer:DDA": 0.06545,
    "NHR": 0.02211,
    "HNR": 21.033,
    "RPDE": 0.414783,
    "DFA": 0.815285,
    "spread1": -4.813031,
    "spread2": -0.137177,
    "D2": 3.413649,
    "PPE": 0.284654
  }'

Response:

{
  "prediction": 1,
  "probability": 0.9999960635246605,
  "label": "Parkinson"
}

Where:

0 → Healthy
1 → Parkinson
Testing

The project contains API, schema, and model tests.

Run:

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -v

Current test result:

14 passed

Tests cover:

Health endpoint
Valid prediction requests
Invalid requests
Missing features
Feature type validation
Model loading
Model predictions
Prediction probabilities
Schema validation
Docker

Build the image:

docker build -t parkinson-ml-api .

Run the container:

docker run --rm -p 8000:8000 parkinson-ml-api

Test:

curl http://127.0.0.1:8000/health

The Docker image uses a slim Python base image and runs the application as a non-root user.

Local Development

Create a virtual environment:

python3 -m venv .venv
source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Run the API:

uvicorn app.main:app --host 127.0.0.1 --port 8000

The API will be available at:

http://127.0.0.1:8000
ML Workflow

The original training workflow is available in:

notebooks/Last_Parkinson_Model.ipynb

The training script is available in:

scripts/train.py

The trained model is stored as:

artifacts/parkinson_model.joblib

The deployment workflow is:

Dataset
   ↓
Training
   ↓
Model Artifact
   ↓
Prediction Service
   ↓
API Validation
   ↓
Inference
   ↓
Docker Deployment
Engineering Focus

This project focuses on the transition from a machine learning notebook to a deployable ML service.

It demonstrates:

Separation of API and inference logic
Input validation
Model serialization
Automated testing
Application logging
Containerization
Non-root container execution
Reproducible local deployment
Disclaimer

This project is intended for educational and engineering demonstration purposes only.

It is not a medical device and predictions should not be used for medical diagnosis or treatment decisions.



**This is the version I'd use.** It gives a recruiter/engineer enough information to understand the project in ~1–2 minutes without making them scroll through a wall of documentation.
