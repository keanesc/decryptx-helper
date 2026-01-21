"""
Evaluation module for DecryptX Round 3.

Computes RMSE scores and generates secure run metadata.
"""

import hashlib
import time
import uuid
from typing import Any, Protocol, Tuple, runtime_checkable

import numpy as np
from sklearn.metrics import mean_squared_error


@runtime_checkable
class Predictor(Protocol):
    """Protocol for objects that can make predictions."""

    def predict(self, X: Any) -> np.ndarray: ...


class EvaluationError(Exception):
    """Raised when evaluation fails."""

    pass


def _generate_run_id() -> str:
    """Generate a unique run ID."""
    return str(uuid.uuid4())


def _compute_metadata_hash(
    run_id: str,
    score: float,
    n_samples: int,
    timestamp: float,
) -> str:
    """
    Compute a hash of the evaluation metadata.

    This hash is used for tamper detection on the server.
    """
    # Create a deterministic string from metadata
    metadata_str = f"{run_id}:{score:.10f}:{n_samples}:{timestamp:.0f}"

    # Compute SHA-256 hash
    return hashlib.sha256(metadata_str.encode()).hexdigest()


def evaluate(
    model: Predictor,
    X_test: Any,
    y_test: Any,
) -> Tuple[float, str, dict]:
    """
    Evaluate your trained model and compute the RMSE score.

    This function:
    1. Makes predictions using your model
    2. Computes the Root Mean Square Error (RMSE)
    3. Generates a unique run ID
    4. Creates metadata for secure submission

    Args:
        model: A trained model with a `predict` method.
        X_test: Test features (from get_train_test_split).
        y_test: True target values (from get_train_test_split).

    Returns:
        Tuple of (score, run_id, metadata):
        - score: RMSE value (lower is better)
        - run_id: Unique identifier for this evaluation run
        - metadata: Dictionary with evaluation details for submission

    Example:
        >>> from decryptx import get_train_test_split, evaluate
        >>> from sklearn.ensemble import RandomForestRegressor
        >>>
        >>> X_train, X_test, y_train, y_test = get_train_test_split(df_clean)
        >>> model = RandomForestRegressor(random_state=42)
        >>> model.fit(X_train, y_train)
        >>> score, run_id, metadata = evaluate(model, X_test, y_test)
        >>> print(f"RMSE: {score:.4f} (lower is better)")

    Raises:
        EvaluationError: If model prediction fails or data is invalid.
    """
    timestamp = time.time() * 1000  # milliseconds

    # Validate model
    if not isinstance(model, Predictor):
        if not hasattr(model, "predict"):
            raise EvaluationError(
                "Model must have a 'predict' method. "
                "Common models from sklearn, xgboost, lightgbm, etc. are supported."
            )

    # Convert to numpy arrays if needed
    y_true = np.asarray(y_test).flatten()

    # Get array length for X_test
    if hasattr(X_test, "__len__"):
        n_samples = len(X_test)
    else:
        n_samples = X_test.shape[0] if hasattr(X_test, "shape") else -1

    # Make predictions
    try:
        y_pred = model.predict(X_test)
        y_pred = np.asarray(y_pred).flatten()
    except Exception as e:
        raise EvaluationError(f"Model prediction failed: {e}") from e

    # Validate prediction shape
    if len(y_pred) != len(y_true):
        raise EvaluationError(
            f"Prediction shape mismatch: got {len(y_pred)} predictions "
            f"for {len(y_true)} samples"
        )

    # Check for invalid predictions
    if np.isnan(y_pred).any():
        n_nan = np.isnan(y_pred).sum()
        raise EvaluationError(
            f"Model produced {n_nan} NaN predictions. "
            "Check your model and data preprocessing."
        )

    if np.isinf(y_pred).any():
        n_inf = np.isinf(y_pred).sum()
        raise EvaluationError(
            f"Model produced {n_inf} infinite predictions. "
            "Check your model and data preprocessing."
        )

    # Compute RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Generate run ID
    run_id = _generate_run_id()

    # Compute metadata hash
    metadata_hash = _compute_metadata_hash(run_id, rmse, n_samples, timestamp)

    # Build metadata
    metadata = {
        "run_id": run_id,
        "score": rmse,
        "n_samples": n_samples,
        "timestamp": timestamp,
        "metadata_hash": metadata_hash,
    }

    # Print results
    print("📈 Evaluation Complete")
    print(f"   RMSE Score: {rmse:.4f} (lower is better)")
    print(f"   Test Samples: {n_samples:,}")
    print(f"   Run ID: {run_id}")

    return rmse, run_id, metadata


def compute_rmse(y_true: Any, y_pred: Any) -> float:
    """
    Compute RMSE between true and predicted values.

    This is a utility function if you want to compute RMSE manually.

    Args:
        y_true: True target values.
        y_pred: Predicted values.

    Returns:
        RMSE value (lower is better).
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    return np.sqrt(mean_squared_error(y_true, y_pred))
