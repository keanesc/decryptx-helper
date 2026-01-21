# DecryptX Helper Library

A Python library for the DecryptX Round 3 Data Cleaning Contest.

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/keanesc/decryptx-helper.git
```

Or for development:

```bash
git clone https://github.com/keanesc/decryptx-helper.git
cd decryptx-helper
pip install -e .
```

## Quick Start

```python
from decryptx import login, load_data, get_train_test_split, evaluate, submit

# 1. Login with your team credentials
session = login(team_name="YourTeamName", password="your_password")

# 2. Load the raw FIFA21 dataset
df = load_data()

# 3. Clean your data (YOUR WORK GOES HERE)
df_clean = your_cleaning_function(df)

# 4. Get train/test split (fixed seed for fairness)
X_train, X_test, y_train, y_test = get_train_test_split(df_clean)

# 5. Train your model
model = YourModel()
model.fit(X_train, y_train)

# 6. Evaluate your model (computes RMSE)
score, run_id = evaluate(model, X_test, y_test)
print(f"Your RMSE score: {score:.4f} (lower is better)")

# 7. Submit your score
result = submit(session, score, run_id)
print(f"Remaining attempts: {result['remainingAttempts']}/5")
```

## API Reference

### `login(team_name: str, password: str) -> dict`

Authenticate with the DecryptX server.

**Returns:** Session dictionary containing `teamId`, `sessionId`, and qualification status.

### `load_data() -> pd.DataFrame`

Load the raw FIFA21 dataset that needs cleaning.

**Returns:** pandas DataFrame with the raw data.

### `get_train_test_split(df: pd.DataFrame, target_col: str = "OVA") -> tuple`

Split your cleaned data into training and test sets using fixed parameters for fairness.

- `random_state=42` (fixed)
- `test_size=0.2` (fixed)
- Target column: `OVA` (Overall Rating)

**Returns:** `(X_train, X_test, y_train, y_test)`

### `evaluate(model, X_test, y_test) -> tuple[float, str]`

Evaluate your trained model and compute the RMSE score.

**Returns:** `(rmse_score, run_id)`

### `submit(session: dict, score: float, run_id: str) -> dict`

Submit your score to the leaderboard.

**Returns:** Submission result with `remainingAttempts` and status.

## Rules

1. **Fixed Parameters**: The train/test split uses `random_state=42` and `test_size=0.2`. Do not modify these.

2. **5 Submission Limit**: You have exactly 5 submission attempts total (lifetime limit).

3. **1 Minute Cooldown**: Wait at least 1 minute between submissions.

4. **RMSE Scoring**: Lower RMSE is better. The target is the player's Overall Rating (OVA).

5. **Data Cleaning Focus**: The competition is about data cleaning, not model architecture. A simple model on well-cleaned data often beats a complex model on dirty data.

## Tips

- Handle missing values appropriately
- Parse numeric values from strings (e.g., "€103.5M" → 103500000)
- Handle height/weight formats (e.g., "170cm" → 170)
- Remove or encode special characters
- Consider feature engineering from the available columns

## License

MIT License
