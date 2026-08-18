# Parkinson ML API


A production-oriented machine learning API for Parkinson's disease prediction.


This project demonstrates how to take a trained scikit-learn model and turn it into a structured, validated, tested, and containerized inference service using **Litestar, Pydantic, pytest, and Docker**.


> **Disclaimer:** This project is for educational and engineering purposes only. It is not a medical diagnostic tool and must not be used for clinical decisions.


## Features


- Trained scikit-learn SVM model
- Model serialization with Joblib
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
Serialized ML Pipeline
  |
  +--> StandardScaler
  |
  +--> SVM Classifier
  |
  v
Prediction + Probability

The API layer is separated from the prediction logic. Incoming requests are validated using Pydantic before being passed to the prediction service, which loads the serialized model and performs inference.

Tech Stack
Component	Technology
Language	Python 3.10
API Framework	Litestar
ASGI Server	Uvicorn
Validation	Pydantic
Machine Learning	scikit-learn
Data Processing	pandas
Model Serialization	Joblib
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
Machine Learning Model

The project uses the Parkinson's voice dataset to classify samples into two classes:

0 → Healthy
1 → Parkinson

The training workflow includes:

Dataset loading
Feature/target separation
Train/test split
Feature standardization
SVM model training
Hyperparameter search using GridSearchCV
Model evaluation
Serialization using Joblib

The resulting trained pipeline is stored as:

artifacts/parkinson_model.joblib

The model artifact contains the preprocessing and classifier required for inference.

Input Features

The prediction endpoint expects 22 numerical voice features:

MDVP:Fo(Hz)
MDVP:Fhi(Hz)
MDVP:Flo(Hz)
MDVP:Jitter(%)
MDVP:Jitter(Abs)
MDVP:RAP
MDVP:PPQ
Jitter:DDP
MDVP:Shimmer
MDVP:Shimmer(dB)
Shimmer:APQ3
Shimmer:APQ5
MDVP:APQ
Shimmer:DDA
NHR
HNR
RPDE
DFA
spread1
spread2
D2
PPE

These features describe different characteristics of recorded voice signals, including frequency variation, jitter, shimmer, harmonicity, and nonlinear measures.

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

Example response:

{
  "prediction": 1,
  "probability": 0.9999960635246605,
  "label": "Parkinson"
}

The response contains:

prediction — predicted class
probability — model probability for the Parkinson class
label — human-readable prediction
Validation

The API uses Pydantic schemas to validate incoming requests.

Invalid requests are rejected when:

Required features are missing
Feature values cannot be converted to numbers
The request structure does not match the expected schema

This keeps validation at the API boundary before the data reaches the ML inference layer.

Testing

The project includes API, schema, and model tests.

Run the test suite with:

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -v

Current test suite:

14 passed

Tests cover:

Health endpoint
Valid prediction requests
Prediction values
Prediction probabilities
Prediction labels
Missing features
Invalid feature types
Model artifact existence
Model loading
Model inference
Schema validation
Numeric type conversion

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 is used to prevent unrelated globally installed pytest plugins from interfering with the project's test environment.

Local Development

Create a virtual environment:

python3 -m venv .venv
source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Run the tests:

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -v

Start the API:

uvicorn app.main:app --host 127.0.0.1 --port 8000

The API will be available at:

http://127.0.0.1:8000
Docker

Build the image:

docker build -t parkinson-ml-api .

Run the container:

docker run --rm -p 8000:8000 parkinson-ml-api

Test the container:

curl http://127.0.0.1:8000/health

The Docker image:

Uses a slim Python base image
Installs only the required dependencies
Copies the application and model artifact
Exposes port 8000
Runs Uvicorn
Runs the application as a non-root user
Training

The original experimentation and training workflow is available in:

notebooks/Last_Parkinson_Model.ipynb

The training script is available at:

scripts/train.py

The overall workflow is:

Dataset
   ↓
Data Preparation
   ↓
Feature Scaling
   ↓
SVM Training
   ↓
Hyperparameter Search
   ↓
Model Evaluation
   ↓
Joblib Artifact
   ↓
Prediction Service
   ↓
Litestar API
   ↓
Docker Deployment
Engineering Focus

The main goal of this project is not only model training, but demonstrating the transition from an ML experiment into a deployable inference service.

The project applies several software and MLOps practices:

Separation of API and inference logic
Explicit request schemas
Model serialization
Centralized configuration
Application logging
Automated testing
Containerization
Non-root container execution
Reproducible local deployment
Clear separation between training and inference
Limitations

This project is an engineering demonstration and has several limitations.

The model should not be interpreted as a clinically validated diagnostic system. Model performance depends on the training dataset, preprocessing pipeline, feature quality, and distribution of incoming data.

No clinical validation, prospective testing, or production monitoring is implemented in this project.

Disclaimer

This project is intended for educational and software engineering demonstration purposes only.

It is not a medical device, clinical decision-support system, or substitute for professional medical evaluation.

Predictions generated by this application should not be used for diagnosis, treatment, or other medical decisions.
