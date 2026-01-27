"""
Data loading and splitting module for DecryptX Round 3.
"""

import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# Fixed parameters for fairness
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = "OVA"

# Cache directory
CACHE_DIR = Path.home() / ".cache" / "decryptx"
CACHE_FILE = CACHE_DIR / "fifa_raw_data.csv"

# Cache validity duration (24 hours)
CACHE_MAX_AGE_SECONDS = 24 * 60 * 60


def _download_dataset(url: str, dest: Path) -> None:
    """
    Download dataset from URL with progress bar.

    Args:
        url: URL to download from.
        dest: Destination file path.

    Raises:
        Exception: If download fails.
    """
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"📥 Downloading dataset from: {url}")

    try:
        if tqdm is not None:
            # Download with progress bar
            with tqdm(
                unit="B", unit_scale=True, unit_divisor=1024, miniters=1
            ) as pbar:

                def update_progress(block_num, block_size, total_size):
                    if pbar.total is None and total_size > 0:
                        pbar.total = total_size
                    pbar.update(block_size)

                urllib.request.urlretrieve(url, dest, reporthook=update_progress)
        else:
            # Download without progress bar
            urllib.request.urlretrieve(url, dest)

        print(f"✅ Dataset downloaded successfully to: {dest}")

    except Exception as e:
        # Clean up partial download
        if dest.exists():
            dest.unlink()
        raise Exception(
            f"Failed to download dataset from {url}.\n"
            f"Error: {e}\n"
            f"Please check your internet connection or try setting DECRYPTX_DATA_PATH "
            f"to a local file path."
        ) from e


def _is_cache_valid(cache_file: Path) -> bool:
    """
    Check if cached file is still valid (exists and not too old).

    Args:
        cache_file: Path to cached file.

    Returns:
        True if cache is valid, False otherwise.
    """
    if not cache_file.exists():
        return False

    # Check file age
    file_age = time.time() - cache_file.stat().st_mtime
    return file_age < CACHE_MAX_AGE_SECONDS


def _find_data_file() -> Path:
    """
    Find the FIFA dataset file.

    Searches in multiple locations:
    1. DECRYPTX_DATA_PATH environment variable (if set)
    2. Package data directory
    3. Current working directory
    4. Common data directories (Colab)
    5. Downloads from Convex API dataset URL to cache

    Returns:
        Path to the data file.

    Raises:
        FileNotFoundError: If the data file cannot be found or downloaded.
    """
    # Possible local locations
    search_paths = [
        # Package data directory
        Path(__file__).parent.parent.parent.parent / "data" / "fifa_raw_data.csv",
        # Current directory
        Path.cwd() / "data" / "fifa_raw_data.csv",
        Path.cwd() / "fifa_raw_data.csv",
        # Common notebook locations (Colab)
        Path("/content/data/fifa_raw_data.csv"),
        Path("/content/fifa_raw_data.csv"),
    ]

    # Check environment variable for explicit path
    env_path = os.environ.get("DECRYPTX_DATA_PATH")
    if env_path:
        search_paths.insert(0, Path(env_path))

    # Check local paths first
    for path in search_paths:
        if path.exists():
            return path

    # If not found locally, try to download from URL
    from .config import get_dataset_url

    dataset_url = get_dataset_url()

    # Check if cache is valid
    if _is_cache_valid(CACHE_FILE):
        print(f"📦 Using cached dataset from: {CACHE_FILE}")
        return CACHE_FILE

    # Download to cache
    try:
        _download_dataset(dataset_url, CACHE_FILE)
        return CACHE_FILE
    except Exception as e:
        raise FileNotFoundError(
            f"FIFA dataset not found locally and download failed.\n\n"
            f"Attempted to download from: {dataset_url}\n"
            f"Error: {e}\n\n"
            f"Options to resolve:\n"
            f"1. Check your internet connection and try again\n"
            f"2. Set DECRYPTX_DATA_PATH environment variable to a local file:\n"
            f"   export DECRYPTX_DATA_PATH=/path/to/fifa_raw_data.csv\n\n"
            f"Local paths searched: {[str(p) for p in search_paths]}"
        ) from e


def load_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load the raw FIFA dataset for cleaning.

    This function loads the dataset that needs to be cleaned as part of the
    competition. The data contains various quality issues that you need to
    address before training your model.

    Args:
        filepath: Optional path to the data file. If not provided, the function
                 will search in common locations.

    Returns:
        pandas DataFrame with the raw FIFA player data.

    Example:
        >>> from decryptx import load_data
        >>> df = load_data()
        >>> print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        >>> print(df.columns.tolist())

    Data Quality Issues to Address:
        - Multi-line cell values (club names with embedded newlines)
        - Mixed formats (Height: "170cm", Weight: "72kg")
        - Currency values (€103.5M, €43K)
        - Missing values in various columns
        - Special characters and star ratings (4 ★)
        - URLs that may not be useful for prediction
    """
    if filepath:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
    else:
        path = _find_data_file()

    print(f"📊 Loading dataset from: {path}")

    # Read CSV with appropriate settings
    df = pd.read_csv(
        path,
        encoding="utf-8",
        on_bad_lines="warn",
    )

    print(f"   Loaded {len(df):,} rows × {len(df.columns)} columns")
    print(f"   Target column: '{TARGET_COLUMN}' (Overall Rating)")

    # Check if target column exists
    if TARGET_COLUMN not in df.columns:
        # Try alternative names
        alt_names = ["↓OVA", "Overall", "overall", "OVA"]
        for alt in alt_names:
            if alt in df.columns:
                print(f"   Note: Using '{alt}' as target column")
                df = df.rename(columns={alt: TARGET_COLUMN})
                break

    return df


def _get_train_test_split(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Internal function to split cleaned data into training and test sets.

    This function uses FIXED parameters to ensure fairness across all teams:
    - random_state=42 (fixed seed for reproducibility)
    - test_size=0.2 (20% held out for testing)

    DO NOT modify these parameters or create your own split, as your submission
    will be invalid.

    Args:
        df: Your cleaned DataFrame.
        target_col: The target column name. Defaults to "OVA" (Overall Rating).
        feature_cols: Optional list of feature columns to use. If not provided,
                     all columns except the target will be used as features.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test):
        - X_train: Training features
        - X_test: Test features
        - y_train: Training target values
        - y_test: Test target values

    Example:
        >>> from decryptx import load_data, get_train_test_split
        >>> df = load_data()
        >>> df_clean = my_cleaning_function(df)
        >>> X_train, X_test, y_train, y_test = get_train_test_split(df_clean)
        >>> print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    Raises:
        ValueError: If the target column is not found or data is invalid.
    """
    # Validate target column
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in DataFrame. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Extract target
    y = df[target_col].copy()

    # Determine feature columns
    if feature_cols is not None:
        # Validate feature columns
        missing = set(feature_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Feature columns not found: {missing}")
        X = df[feature_cols].copy()
    else:
        # Use all columns except target
        X = df.drop(columns=[target_col]).copy()

    # Check for valid data
    if len(X) == 0:
        raise ValueError("DataFrame is empty")

    if y.isna().all():
        raise ValueError(f"Target column '{target_col}' contains all NaN values")

    # Remove rows with NaN target (required for evaluation)
    valid_mask = ~y.isna()
    if not valid_mask.all():
        n_dropped = (~valid_mask).sum()
        print(f"⚠️  Dropping {n_dropped} rows with missing target values")
        X = X[valid_mask]
        y = y[valid_mask]

    # Perform the split with FIXED parameters
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print(
        f"✅ Data split complete (random_state={RANDOM_STATE}, test_size={TEST_SIZE})"
    )
    print(f"   Training set: {len(X_train):,} samples")
    print(f"   Test set: {len(X_test):,} samples")

    return X_train, X_test, y_train, y_test


def validate_data(df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> dict:
    """
    Validate your cleaned DataFrame before splitting.

    This function checks common data quality issues and returns a report.

    Args:
        df: Your cleaned DataFrame.
        target_col: The target column name.

    Returns:
        Dictionary with validation results.
    """
    report = {
        "valid": True,
        "warnings": [],
        "errors": [],
        "stats": {},
    }

    # Check target column
    if target_col not in df.columns:
        report["valid"] = False
        report["errors"].append(f"Target column '{target_col}' not found")
        return report

    # Basic stats
    report["stats"]["rows"] = len(df)
    report["stats"]["columns"] = len(df.columns)
    report["stats"]["target_missing"] = df[target_col].isna().sum()
    report["stats"]["target_dtype"] = str(df[target_col].dtype)

    # Check target is numeric
    if not np.issubdtype(df[target_col].dtype, np.number):
        report["valid"] = False
        report["errors"].append(
            f"Target column must be numeric, got {df[target_col].dtype}"
        )

    # Check for missing values in target
    missing_pct = df[target_col].isna().sum() / len(df) * 100
    if missing_pct > 0:
        report["warnings"].append(
            f"Target column has {missing_pct:.1f}% missing values"
        )

    # Check for non-numeric features
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if target_col in non_numeric:
        non_numeric.remove(target_col)
    if non_numeric:
        report["warnings"].append(
            f"Non-numeric columns that may need encoding: {non_numeric[:5]}"
            + (f" (+{len(non_numeric) - 5} more)" if len(non_numeric) > 5 else "")
        )

    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_cols = [col for col in numeric_cols if np.isinf(df[col]).any()]
    if inf_cols:
        report["warnings"].append(f"Columns with infinite values: {inf_cols}")

    return report
