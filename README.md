# Parkinson ML API

A containerized machine learning API for Parkinson's disease prediction.

The project demonstrates how to take a trained ML model and expose it as a reliable API using Litestar, Pydantic, pytest, structured logging, and Docker.

## Features

- Pre-trained scikit-learn model serialized with Joblib
- Litestar REST API
- Pydantic request validation
- Prediction service using OOP and composition
- Structured application logging
- Unit and API tests with pytest
- Dockerized deployment
- Health check endpoint

## API Endpoints

### Health

```http
GET /health
