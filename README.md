# Steelers Sports Analytics Project

## Steelers Touchdown Rate Analyzer

This Streamlit app analyzes real NFL play-by-play data from nflverse, loaded with
[`nflreadpy`](https://github.com/nflverse/nflreadpy). It groups Steelers pass and
rush plays by **yards from the opponent's goal line** and classifies them as
inside/outside runs, middle/sideline passes, or charted screen passes. It reports
plays, touchdowns, and touchdown rate for each category and distance band.
Touchdown rate means touchdowns divided by qualifying plays in that band.

The dashboard also compares Steelers win rates across turnover-margin bands
(from losing the turnover battle by 3+ to winning it by 3+). Turnovers here are
interceptions and lost fumbles on pass/run plays. That comparison uses all downs
in the selected seasons and reports wins divided by games, with ties counted as
non-wins; it is descriptive, not evidence that turnovers caused the result.
The dashboard presents the touchdown, turnover-margin, and defensive takeaway
summaries as tables.

The defensive takeaways table compares interceptions and sack-fumbles forced
per opposing pass play by FTN's charted blitzer count (none, one, or two-plus).
It also reports whether the defense recovered the sack-fumble. FTN charting
starts in 2022 and does not provide man/zone coverage, blitzer position, or
blitz direction, so the app does not infer those details. Earlier or unmatched
plays are listed as unavailable rather than assigned to a blitz group.

### Run the app on Windows

Open PowerShell in this project folder and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If the workspace already has a `.venv` environment, skip the first command.
The first app run downloads the selected seasons from nflverse; subsequent loads
are read from local Parquet files in `data/raw/`. To pre-download the default
data set before opening the app, run:

```powershell
python download_nfl_data.py
```

The downloader stores full play-by-play files for the five most recently
completed seasons and available FTN charting files for 2022 onward. The app
automatically downloads and saves other selected seasons when their local files
are missing. Raw data files are excluded from Git because of their size; they
remain in your local project folder. Choose seasons and a down in the sidebar.
By default, the app selects the five most recently completed seasons. The data's
`yardline_100` field measures distance from the opponent's goal line. The `50+`
bucket means more than 50 yards from that goal line; exactly 50 belongs in `41-50`.

Touchdown rate is passing/rushing touchdowns divided by pass/rush plays from that
distance. The tables are descriptive, not causal or predictive. They exclude
special-teams plays, penalties/no-plays, and plays missing field position.
Opponent, score, clock, and personnel are not controlled for; small play counts
can make rates volatile.

Run direction comes from nflverse `run_location`: `middle` is an inside run and
`left`/`right` are outside runs. Pass direction comes from `pass_location`;
screens use FTN charting's `is_screen_pass` flag, joined by game and play ID.
FTN screen labels are available starting in 2022. Unmatched or missing labels are
shown as unclassified rather than guessed, so early-season pass categories may
have reduced coverage.

Run the summary-logic tests with:

```powershell
python -m unittest discover -s tests -v
```

### Save summaries to TiDB

Click **Save all summaries to TiDB** in the app to write all three dashboard
summaries for the selected seasons. The touchdown table also records the selected
down filter. One shared run ID links the rows saved by a single click:

- `steelers_analytics.touchdown_summaries` stores distance bands, play
  categories, plays, touchdowns, and touchdown rates.
- `steelers_analytics.turnover_margin_outcomes` stores game outcomes by
  Pittsburgh's turnover margin.
- `steelers_analytics.defensive_turnovers_by_blitz` stores opposing pass plays,
  interceptions, sack-fumbles, recoveries, and rates by charted blitz count.

The app creates these tables if they do not exist. Saving does not upload raw
play-by-play, and each save adds a new run rather than replacing earlier runs.
You can inspect the most recent run in TiDB with:

```sql
SELECT * FROM steelers_analytics.turnover_margin_outcomes
WHERE run_id = '<run-id-from-the-app>'
ORDER BY turnover_margin;

SELECT * FROM steelers_analytics.defensive_turnovers_by_blitz
WHERE run_id = '<run-id-from-the-app>'
ORDER BY blitz_pressure;
```

The app connects over TLS and creates the dedicated
`steelers_analytics` database if it does not exist; the TiDB user needs permission
to create that database the first time, then create and write the summary table.

Connection secrets are read locally and are excluded from Git:

- Set `TIDB_DATABASE_URL` in `.env` to a MySQL URL, or
- Set `TIDB_USER`, `TIDB_PASSWORD`, and `TIDB_HOST` in `.env`. `TIDB_PORT`
  defaults to `4000`; the existing `data_base_password` variable is also accepted
  as the password if `TIDB_PASSWORD` is unset. TiDB Cloud Starter/Essential
  usernames must include the cluster prefix and user suffix (for example,
  `<cluster-prefix>.root`); the supplied cluster-prefix default uses `.root`.

Do not commit `.env` or paste its password into source code.

## NFL offense mismatch notebook

This file is a step-by-step notebook-style guide for using **NFL Data Py** to explore how the Pittsburgh Steelers can create mismatches on offense. Because the reminder limits the change to this one file, the complete notebook is written here in the README instead of creating separate files.

> The notebook is designed for beginners. Each section explains what it does before the code runs.

## 1. What is the source, and how do we get it?

The source is the **NFL Data Py** public database. It contains play-by-play data, player statistics, and roster information for NFL seasons. The data is downloaded from the package's public data files when the code calls the package.

The Steelers' offensive mismatch question is broad, so this first version looks at **offensive play outcomes by down and distance**. That lets us see whether the Steelers are successful when they have to convert short-yardage situations, move the ball on third down, or create a high-value chance after a turnover.

Before running the notebook, open a terminal in this project folder and run:

```powershell
py -m pip install nfl-data-py pandas matplotlib
```

If the `py` command is not available, use `python` instead:

```powershell
python -m pip install nfl-data-py pandas matplotlib
```
The package name is `nfl-data-py`, while the Python import name is `nfl_data_py`. Verify the installation before continuing:

```powershell
python -c "import nfl_data_py, pandas, matplotlib; print('Dependencies loaded successfully')"
```

If the verification command fails, the install did not complete successfully. Re-run the installation command and include the full error message when asking for help.

Then run the cells below in order. The code uses only the data needed for this analysis.

## 2. Load the NFL data

### Why this step exists

This step downloads the NFL play-by-play data for the selected seasons and turns it into a table-like Python object. The workbook is called a DataFrame, which is similar to a spreadsheet.

### Code

```python
import nfl_data_py as nfl
import pandas as pd
import matplotlib.pyplot as plt

# Change this to the seasons you want to examine.
SEASONS = [2022, 2023, 2024]

# Download one season at a time so the result is easy to inspect.
raw_data = []
for season in SEASONS:
    print(f"Loading {season}...")
    frame = nfl.import_pandas_data("play_by_play", years=[season])
    frame = frame.copy()
    frame["season"] = season
    raw_data.append(frame)

raw = pd.concat(raw_data, ignore_index=True)
```

### What the output should look like

The code prints one loading message for each season. After the last line, `raw` is a large table containing every play in the selected seasons. It is not yet cleaned, so column names and values may be unfamiliar.

## 3. See the raw data and learn its structure

### Why this step exists

Raw data is the uncleaned result from NFL Data Py. Looking at it before changing it helps us identify the correct columns and prevents mistakes caused by a column name or value format.

### Code

```python
# Show the number of rows and columns.
print(f"Rows: {raw.shape[0]}; columns: {raw.shape[1]}")

# Show the column names in order.
print("Columns:")
print(raw.columns.tolist())

# Show the first five rows. This is the raw data view.
print(raw.head().to_string())

# Show the data types for each column.
print(raw.dtypes)
```

### How to read the raw data

- Each row is one NFL play.
- Each column is a measure or description of that play.
- `season` is added by this notebook.
- `home_team` and `away_team` identify the teams playing that game.
- `possession` identifies which team had the ball at the start of the play.
- `offense`, `defense`, and `special_teams` identify the teams involved.
- `down` and `distance` describe the football situation before the play.
- `play_type` describes the play, such as a pass or rush.
- `yards` and `yards_per_play` describe the result.
- `pass_yards`, `rush_yards`, `pass_attempts`, and `rush_attempts` contain play-level statistics when available.

The exact set of columns can vary slightly between NFL Data Py releases. The next code uses the common columns but also checks that they exist before it continues.

## 4. Clean the data into a simple table

### Why this step exists

The raw table has more information than we need and some fields are blank or use inconsistent labels. Cleaning gives us one compact table for the Steelers' offensive analysis.

### Code

```python
# Required fields. If a field is missing, stop with a useful message.
required = [
    "season", "week", "home_team", "away_team", "possession",
    "offense", "defense", "down", "distance", "play_type",
    "yards", "yards_per_play", "pass_attempts", "rush_attempts",
    "pass_yards", "rush_yards"
]
missing = [column for column in required if column not in raw.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Make a copy so the raw data is not changed.
clean = raw[required].copy()

# Normalize the team names so the Steelers are easy to identify.
clean["possession"] = clean["possession"].astype("string").str.strip()
clean["offense"] = clean["offense"].astype("string").str.strip()
clean["defense"] = clean["defense"].astype("string").str.strip()
clean["play_type"] = clean["play_type"].astype("string").str.strip().str.lower()

# Treat blank values as missing numbers.
number_columns = [
    "week", "down", "distance", "yards", "yards_per_play",
    "pass_attempts", "rush_attempts", "pass_yards", "rush_yards"
]
for column in number_columns:
    clean[column] = pd.to_numeric(clean[column], errors="coerce")

# Keep only plays called by the Steelers offense.
clean = clean[clean["offense"] == "PIT"]

# Create a few useful short names for the analysis.
clean["is_pass"] = clean["play_type"].eq("pass")
clean["is_rush"] = clean["play_type"].eq("rush")
clean["is_third_down"] = clean["down"].eq(3)
clean["is_short_yardage"] = clean["distance"].le(3)

# Show the cleaned table.
print(clean.head().to_string(index=False))
```

### What this step does

- It removes columns we do not need.
- It converts text and numbers into consistent values.
- It keeps only Pittsburgh's offensive plays.
- It creates flags for passes, rushes, third down, and short-yardage situations.
- It converts blank values to missing values instead of treating them as zero.

## 5. Check that the numbers look right

### Why this step exists

A clean table can still contain bad values. These checks confirm that the rows are plausible before we draw conclusions.

### Code

```python
# Check the number of Steelers offensive plays.
print(f"Steelers offensive plays: {len(clean)}")

# Check the season totals.
print(clean.groupby("season").size().to_string())

# Check the number of pass and rush plays.
print(clean.groupby("play_type").size().to_string())

# Check basic ranges for the values that matter.
for column in ["yards", "yards_per_play", "pass_attempts", "rush_attempts"]:
    print(f"{column}: min={clean[column].min()}, max={clean[column].max()}")

# Check that every play has a down and distance.
print("Missing down or distance:", clean[["down", "distance"]].isna().any().to_dict())

# Check whether the expected play types are present.
print("Play types:", clean["play_type"].value_counts(dropna=False).to_dict())
```

### What to look for

- The season totals should be roughly similar across seasons, but they do not need to be identical.
- The number of pass and rush plays should be positive and should not contain unexpected values.
- `yards` should not be negative or larger than the largest possible gain on one play.
- `down` should be between 1 and 4.
- `distance` should be positive.
- A missing value is not the same as zero; it may mean that the source did not record a statistic for that play.

If a check fails, do not continue to the chart until you understand the cause. Common causes are an incorrect team code, a missing source field, or a season that uses a different column name.

## 6. Create a simple mismatch chart

### Why this step exists

This chart compares the Steelers' average yards per play on third down by play type. The goal is to reveal whether the Steelers are getting more value from passes or rushes when the offense needs to convert the first down.

### Code

```python
# Use only complete third-down rows with a play type and yards result.
third_down = clean[
    clean["is_third_down"]
    & clean["play_type"].isin(["pass", "rush"])
    & clean["yards"].notna()
]

# Calculate average yards per play by type and season.
summary = (
    third_down.groupby(["season", "play_type"], as_index=False)["yards"]
    .mean()
    .rename(columns={"yards": "average_yards_per_play"})
)

# Make the table easy to read.
summary["play_type"] = summary["play_type"].str.upper()
print(summary.to_string(index=False))

# Draw one chart.
fig, ax = plt.subplots(figsize=(9, 5))
for season, group in summary.groupby("season"):
    ax.plot(
        group["play_type"],
        group["average_yards_per_play"],
        marker="o",
        label=str(season),
    )

ax.set_title("Steelers Third-Down Average Yards per Play")
ax.set_xlabel("Play type")
ax.set_ylabel("Average yards per play")
ax.grid(axis="y", alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()
```

### How to read the chart

- A higher average means that play type produced more yards on average.
- A large gap between pass and rush can be a mismatch if the Steelers use one type much more often than the other.
- A high average does not automatically mean a play is better: it also needs to be considered with play frequency, down, distance, and the opponent's defensive strengths.
- This chart is a starting point, not a final answer.

## 7. Short summary of the full process

1. **Find the source:** NFL Data Py provides public NFL play-by-play data.
2. **Install and import it:** Python loads the package and the plotting tools.
3. **Load the data:** The notebook downloads selected seasons and combines them into one table.
4. **Inspect the raw data:** Rows are plays, columns are fields, and the fields describe the game situation and result.
5. **Clean the table:** Keep Steelers offensive plays, normalize values, and create useful flags.
6. **Check the numbers:** Confirm season totals, play types, ranges, and missing values.
7. **Create one chart:** Compare third-down yards per play by pass and rush.
8. **Interpret cautiously:** Use the chart with frequency, down and distance, and defensive context before recommending a strategy.

### Final idea

The most useful mismatch is not simply “pass more” or “rush more.” It is a difference between the play the Steelers can execute consistently and the defensive look they are likely to face. A good next step would be to compare the Steelers' pass and rush rates, average yards, and success rates against the opponent's defensive tendencies for the same down-and-distance situations.

```
