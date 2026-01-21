"""
Authentication module for DecryptX Round 3.
"""

from typing import TypedDict

import requests

from decryptx.config import _get_api_url, _get_timeout


class Session(TypedDict):
    """Type definition for session data."""

    teamId: str
    teamName: str
    sessionId: str
    qualified: bool
    round3Status: str
    remainingAttempts: int
    teamSecret: str


class LoginError(Exception):
    """Raised when login fails."""

    pass


class NotQualifiedError(Exception):
    """Raised when team is not qualified for Round 3."""

    pass


def login(team_name: str, password: str) -> Session:
    """
    Authenticate with the DecryptX server.

    This function logs in your team and returns a session that can be used
    for submitting scores. The session is valid for 24 hours.

    Args:
        team_name: Your registered team name.
        password: Your team password.

    Returns:
        Session dictionary containing:
        - teamId: Your unique team identifier
        - teamName: Your team name
        - sessionId: Authentication token
        - qualified: Whether you're qualified for Round 3
        - round3Status: Current status of Round 3
        - remainingAttempts: Number of submissions remaining

    Raises:
        LoginError: If authentication fails.
        NotQualifiedError: If team is not qualified for Round 3.
        requests.RequestException: If there's a network error.

    Example:
        >>> from decryptx import login
        >>> session = login("MyTeam", "secret123")
        >>> print(f"Logged in as {session['teamName']}")
        >>> print(f"Qualified: {session['qualified']}")
    """
    url = _get_api_url("/api/round3/auth")

    try:
        response = requests.post(
            url,
            json={"teamName": team_name, "password": password},
            timeout=_get_timeout(),
            headers={"Content-Type": "application/json"},
        )

        data = response.json()

        if response.status_code == 401:
            raise LoginError(data.get("error", "Invalid credentials"))

        if response.status_code != 200:
            raise LoginError(
                data.get("error", f"Login failed with status {response.status_code}")
            )

        if not data.get("success"):
            raise LoginError(data.get("error", "Login failed"))

        # Check qualification
        if not data.get("qualified"):
            print("⚠️  Warning: Your team is not qualified for Round 3.")
            print("   Only the top 10 teams from Round 2 can participate.")

        # Build session object
        session: Session = {
            "teamId": data["teamId"],
            "teamName": team_name,
            "sessionId": data["sessionId"],
            "qualified": data.get("qualified", False),
            "round3Status": data.get("round3Status", "unknown"),
            "remainingAttempts": data.get("remainingAttempts", 0),
            "teamSecret": data.get("teamSecret", ""),
        }

        # Print status info
        print(f"✅ Logged in successfully as '{team_name}'")
        if session["qualified"]:
            print(f"   Round 3 Status: {session['round3Status']}")
            print(f"   Remaining Attempts: {session['remainingAttempts']}/5")

        return session

    except requests.RequestException as e:
        raise LoginError(f"Network error: {e}") from e


def verify_session(session: Session) -> bool:
    """
    Verify that a session is still valid.

    Args:
        session: Session dictionary from login().

    Returns:
        True if session is valid, False otherwise.
    """
    if not session or not session.get("sessionId"):
        return False

    # For now, we trust the session locally
    # The server will validate on submission
    return True
