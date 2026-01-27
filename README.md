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

## Dataset Access

The library automatically downloads the FIFA dataset from the DecryptX server when you call `load_data()`. The dataset is cached locally in `~/.cache/decryptx/` and refreshed every 24 hours.

### Environment Variables (Optional)

You can customize the dataset location using an environment variable:

- **`DECRYPTX_DATA_PATH`**: Direct path to a local dataset file (bypasses download)

**Example:**

```bash
# Use a local file (no download)
export DECRYPTX_DATA_PATH="/path/to/your/fifa_raw_data.csv"
```

**Google Colab Note:** The dataset will be automatically downloaded on first use. No manual upload required!

## Quick Start

```python
from decryptx import login, load_data, submit

# 1. Login with your team credentials
session = login(team_name="YourTeamName", password="your_password")

# 2. Load the raw FIFA dataset
df = load_data()

# 3. Clean your data (YOUR WORK GOES HERE)
df_clean = your_cleaning_function(df)

# 4. Submit your cleaned dataset
# This will automatically split data, train a fixed model, evaluate performance, and submit the score.
result = submit(session, df_clean)
print(f"Remaining attempts: {result['remainingAttempts']}/5")
```

## API Reference

### `login(team_name: str, password: str) -> dict`

Authenticate with the DecryptX server.

**Returns:** Session dictionary containing `teamId`, `sessionId`, and qualification status.

### `load_data() -> pd.DataFrame`

Load the raw FIFA dataset that needs cleaning.

**Returns:** pandas DataFrame with the raw data.

### `submit(session: dict, df: pd.DataFrame) -> dict`

Submit your cleaned dataset for evaluation.

This function:

1. Splits your data into train/test sets (fixed random seed)
2. Trains a standardized Random Forest model
3. Evaluates RMSE on the test set
4. Submits the score to the leaderboard

**Returns:** Submission result dictionary.

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
