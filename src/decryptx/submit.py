"""
Submission module for DecryptX Round 3.

Handles secure score submission with HMAC signing.
"""

import hashlib
import hmac
import time
from typing import Optional, TypedDict

import requests

from decryptx.auth import Session
from decryptx.config import _get_api_url, _get_timeout


class SubmissionResult(TypedDict):
    """Type definition for submission result."""

    success: bool
    message: str
    submissionId: str
    remainingAttempts: int


class SubmissionError(Exception):
    """Raised when submission fails."""

    pass


class CooldownError(SubmissionError):
    """Raised when submission is on cooldown."""

    def __init__(self, message: str, wait_seconds: int):
        super().__init__(message)
        self.wait_seconds = wait_seconds


class QuotaExceededError(SubmissionError):
    """Raised when submission quota is exceeded."""

    pass


def _sign_submission(
    team_id: str,
    run_id: str,
    score: float,
    timestamp: float,
    team_secret: str,
) -> str:
    """
    Create HMAC signature for the submission.

    The signature is computed as:
    HMAC-SHA256(team_secret, "{team_id}:{run_id}:{score}:{timestamp}")
    """
    payload = f"{team_id}:{run_id}:{score}:{timestamp}"
    signature = hmac.new(
        team_secret.encode(),
        payload.encode(),
        hashlib.sha256,
    ).hexdigest()
    return signature


def _get_team_secret(session: Session) -> str:
    """
    Get the team-specific secret for signing.

    The secret is provided by the server during login and is derived from:
    team_secret = HMAC-SHA256(master_secret, team_id)

    This ensures that only the server can generate valid secrets, and
    each team has a unique signing key.
    """
    team_secret = session.get("teamSecret")
    if not team_secret:
        raise SubmissionError(
            "Team secret not found in session. Please login again."
        )
    return team_secret


def submit(
    session: Session,
    score: float,
    run_id: str,
    metadata: Optional[dict] = None,
) -> SubmissionResult:
    """
    Submit your score to the DecryptX leaderboard.

    This function securely submits your evaluation score to the server.
    The submission is signed with HMAC to prevent tampering.

    Args:
        session: Session dictionary from login().
        score: Your RMSE score from evaluate().
        run_id: The run ID from evaluate().
        metadata: Optional metadata dictionary from evaluate().

    Returns:
        SubmissionResult dictionary containing:
        - success: Whether submission was accepted
        - message: Status message
        - submissionId: Unique ID for this submission
        - remainingAttempts: Number of attempts left (out of 5)

    Raises:
        SubmissionError: If submission fails.
        CooldownError: If you need to wait before submitting again.
        QuotaExceededError: If you've used all 5 submission attempts.

    Example:
        >>> from decryptx import login, evaluate, submit
        >>> session = login("MyTeam", "password")
        >>> score, run_id, metadata = evaluate(model, X_test, y_test)
        >>> result = submit(session, score, run_id, metadata)
        >>> print(f"Remaining attempts: {result['remainingAttempts']}/5")

    Limits:
        - Maximum 5 submissions total (lifetime)
        - 1 minute cooldown between submissions
    """
    # Validate session
    if not session or not session.get("sessionId"):
        raise SubmissionError("Invalid session. Please login again.")

    if not session.get("qualified"):
        raise SubmissionError(
            "Your team is not qualified for Round 3. "
            "Only top 10 teams from Round 2 can submit."
        )

    # Check remaining attempts
    if session.get("remainingAttempts", 0) <= 0:
        raise QuotaExceededError(
            "You have used all 5 submission attempts. "
            "No more submissions are allowed."
        )

    # Prepare submission data
    timestamp = time.time() * 1000  # milliseconds
    team_id = session["teamId"]
    session_id = session["sessionId"]

    # Compute HMAC signature
    team_secret = _get_team_secret(session)
    signature = _sign_submission(team_id, run_id, score, timestamp, team_secret)

    # Get metadata hash
    metadata_hash = ""
    if metadata and "metadata_hash" in metadata:
        metadata_hash = metadata["metadata_hash"]
    else:
        # Generate a simple hash if metadata not provided
        metadata_str = f"{run_id}:{score}:{timestamp}"
        metadata_hash = hashlib.sha256(metadata_str.encode()).hexdigest()

    # Build request payload
    payload = {
        "teamId": team_id,
        "sessionId": session_id,
        "runId": run_id,
        "score": score,
        "timestamp": timestamp,
        "signature": signature,
        "metadataHash": metadata_hash,
    }

    url = _get_api_url("/api/round3/submit")

    print("📤 Submitting score to leaderboard...")
    print(f"   RMSE: {score:.4f}")
    print(f"   Run ID: {run_id}")

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=_get_timeout(),
            headers={"Content-Type": "application/json"},
        )

        data = response.json()

        # Handle specific error cases
        if response.status_code == 400:
            error = data.get("error", "Submission failed")

            # Check for cooldown
            if "wait" in error.lower() or "cooldown" in error.lower():
                # Extract wait time if present
                import re

                match = re.search(r"(\d+)\s*seconds?", error)
                wait_seconds = int(match.group(1)) if match else 60
                raise CooldownError(error, wait_seconds)

            # Check for quota exceeded
            if "maximum" in error.lower() or "limit" in error.lower():
                raise QuotaExceededError(error)

            raise SubmissionError(error)

        if response.status_code == 401:
            raise SubmissionError(
                data.get("error", "Authentication failed. Please login again.")
            )

        if response.status_code != 200:
            raise SubmissionError(
                data.get(
                    "error", f"Submission failed with status {response.status_code}"
                )
            )

        if not data.get("success"):
            raise SubmissionError(data.get("error", "Submission failed"))

        # Success!
        remaining = data.get("remainingAttempts", 0)

        # Update session with new remaining attempts
        session["remainingAttempts"] = remaining

        result: SubmissionResult = {
            "success": True,
            "message": data.get("message", "Score submitted successfully"),
            "submissionId": data.get("submissionId", ""),
            "remainingAttempts": remaining,
        }

        print("✅ Submission successful!")
        print(f"   Submission ID: {result['submissionId']}")
        print(f"   Remaining attempts: {remaining}/5")

        if remaining == 0:
            print("   ⚠️  This was your last submission!")
        elif remaining == 1:
            print("   ⚠️  Only 1 attempt remaining!")

        return result

    except requests.RequestException as e:
        raise SubmissionError(f"Network error: {e}") from e


def get_submission_status(session: Session) -> dict:
    """
    Get your current submission status.

    Args:
        session: Session dictionary from login().

    Returns:
        Dictionary with submission history and remaining attempts.
    """
    # This would call an API endpoint to get status
    # For now, return what we have in the session
    return {
        "qualified": session.get("qualified", False),
        "remainingAttempts": session.get("remainingAttempts", 0),
        "round3Status": session.get("round3Status", "unknown"),
    }
