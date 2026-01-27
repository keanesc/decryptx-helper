"""
Configuration management for the DecryptX library.
"""

import os
from typing import Optional

# Default configuration
_config = {
    "api_base_url": "https://unique-manatee-458.convex.site",
    "timeout": 30,
}


def configure(
    api_base_url: Optional[str] = None,
    timeout: Optional[int] = None,
) -> None:
    """
    Configure the DecryptX library settings.

    Args:
        api_base_url: Base URL for the DecryptX API. Defaults to production URL.
        timeout: Request timeout in seconds. Defaults to 30.

    Example:
        >>> from decryptx import configure
        >>> configure(api_base_url="http://localhost:3000")
    """
    global _config

    if api_base_url is not None:
        _config["api_base_url"] = api_base_url.rstrip("/")

    if timeout is not None:
        _config["timeout"] = timeout

    # Allow environment variable override
    env_url = os.environ.get("DECRYPTX_API_URL")
    if env_url:
        _config["api_base_url"] = env_url.rstrip("/")


def get_config() -> dict:
    """
    Get the current configuration.

    Returns:
        Dictionary with current configuration values.
    """
    return _config.copy()


def _get_api_url(endpoint: str) -> str:
    """
    Get the full API URL for an endpoint.

    Args:
        endpoint: API endpoint path (e.g., "/api/round3/auth")

    Returns:
        Full URL string.
    """
    return f"{_config['api_base_url']}{endpoint}"


def _get_timeout() -> int:
    """Get the configured timeout value."""
    return _config["timeout"]


def get_dataset_url() -> str:
    """
    Get the dataset URL from the Convex API.

    Returns:
        Dataset URL string.

    Raises:
        Exception: If unable to fetch the dataset URL from the API.
    """
    import json
    import urllib.request

    try:
        api_url = _get_api_url("/api/round3/dataset-url")
        with urllib.request.urlopen(api_url, timeout=_get_timeout()) as response:
            data = json.loads(response.read().decode())
            return data.get(
                "datasetUrl", "https://www.dataai.club/dataset/fifa_raw_data.csv"
            )
    except Exception as e:
        # Fallback to default URL if API is unavailable
        print(f"⚠️  Unable to fetch dataset URL from API: {e}")
        print(
            "📌 Using fallback URL: https://www.dataai.club/dataset/fifa_raw_data.csv"
        )
        return "https://www.dataai.club/dataset/fifa_raw_data.csv"
