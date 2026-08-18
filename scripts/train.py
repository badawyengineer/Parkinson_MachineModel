from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = Path(
    "/home/badawy/.cache/kagglehub/datasets/"
    "debasisdotcom/parkinson-disease-detection/"
    "versions/2/Parkinsson disease.csv"
)

MODEL_PATH = PROJECT_ROOT / "artifacts" / "parkinson_model.joblib"


def load_data(path: Path) -> pd.DataFrame:
    """Load the Parkinson dataset."""
    return pd.read_csv(path)


def prepare_data(df: pd.DataFrame):
    """Prepare features and target."""
    df = df.drop(columns=["name"], errors="ignore")

    X = df.drop(columns=["status"])
    y = df["status"]

    return X, y


def build_pipeline() -> Pipeline:
    """Build the ML preprocessing and model pipeline."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", SVC(probability=True)),
        ]
    )


def train_model(X_train, y_train):
    """Train and tune the SVM pipeline."""
    pipeline = build_pipeline()

    param_grid = {
        "model__C": [0.1, 1, 10],
        "model__kernel": ["linear", "rbf"],
    }

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
    )

    grid.fit(X_train, y_train)

    return grid


def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model."""
    y_pred = model.predict(X_test)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")


def save_model(model, path: Path):
    """Save the trained model artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, path)

    print(f"\nModel saved to: {path}")


def main():
    print("Loading dataset...")

    df = load_data(DATA_PATH)

    print(f"Dataset shape: {df.shape}")

    X, y = prepare_data(df)

    print(f"Features: {X.shape[1]}")
    print(f"Samples: {X.shape[0]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    print("\nTraining model...")

    grid = train_model(X_train, y_train)

    print(f"\nBest parameters: {grid.best_params_}")
    print(f"Best CV F1: {grid.best_score_:.4f}")

    print("\nEvaluating model...")

    evaluate_model(
        grid.best_estimator_,
        X_test,
        y_test,
    )

    save_model(
        grid.best_estimator_,
        MODEL_PATH,
    )


if __name__ == "__main__":
    main()
