"""
Predict the outcome of the 2025/26 LaLiga season using a random
forest classifier.

This script takes a series of historical LaLiga seasons in CSV
format, builds summary statistics for each club (points, wins, draws,
losses, goals for/against and goal difference) and then trains a
RandomForestClassifier from scikit‑learn to predict the final league
position of each team in a subsequent season.  The 2025/26 LaLiga
will include 20 clubs – the 17 sides that remained in the
division in 2024/25 and three promoted clubs (Levante, Elche
and Real Oviedo).  Since results from the 2024/25 campaign are not
yet freely available, the model uses the 2023/24 season as the most
recent set of training features.  New clubs that did not compete in
2023/24 are assigned average feature values based on the bottom three
sides from that season.

The historical match files used by this script can be downloaded from
the open‑source football csv mirror hosted on GitHub.  Each file
(`season-1819.csv`, `season-1920.csv`, … `season-2324.csv`) lists
every LaLiga match in the given season with columns for the
date, home side, final score (FT), half‑time score (HT) and
away side.  An example of the first few rows of the
2019/20 file is shown below:

```
              Date      HomeTeam      FTHG  FTAG   AwayTeam
0   Fri Aug 9 2019       Athletic       1     0     Barcelona
1  Sat Aug 10 2019       Celta Vigo     1     3     Real Madrid
2  Sat Aug 10 2019       Valencia       1     1     Real Sociedad
3  Sat Aug 10 2019       Mallorca       2     1     Eibar
4  Sat Aug 10 2019       Villarreal     4     4     Granada
```

Each CSV contains 380 matches (20 clubs playing 38 games each).  The
script parses the final score to determine home and away goals and
computes win/draw/loss outcomes accordingly.  After summarising the
season, the teams are sorted by points, goal difference and goals
scored to derive a final ranking.  For training data, each team’s
performance statistics from season `n` are used to predict its
position in season `n+1`.  Teams that enter the league via promotion
are assigned default feature values representing the average of the
bottom three clubs from the previous season.

The RandomForestClassifier hyperparameters can be adjusted via the
constants at the bottom of the script.  By default the model uses
100 trees, a maximum depth of 8 and a random seed for reproducible
results.  After training, the script prints the predicted league
table for 2025/26 along with a comparison to the training periods.

Usage
-----
Run the script from a terminal with Python 3.  Ensure that
`pandas`, `numpy` and `scikit‑learn` are installed.  All required
CSV files should reside in the same directory as this script or an
alternate path may be provided via the `season_files` list.

Example:

```
python predictor.py
```

The script outputs a predicted ranking of the 20 clubs for the
2025/26 LaLiga season.
"""



import os
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# This function remains unchanged from your original script
def summarise_season(matches: pd.DataFrame) -> pd.DataFrame:
    """Summarise a LaLiga season into per‑team statistics and final ranking.

    Given a DataFrame of matches with columns `HomeTeam`, `AwayTeam`,
    `FTHG` (full‑time home goals) and `FTAG` (full‑time away goals),
    compute the total points, wins, draws, losses, goals for and against
    and goal difference for each team.  After accumulating statistics,
    the teams are sorted by points (descending), goal difference (descending)
    and goals for (descending) to determine the final ranking.

    Parameters
    ----------
    matches : DataFrame
        DataFrame of parsed match results.

    Returns
    -------
    DataFrame
        Summary of the season with one row per team and columns:
        [`team`, `points`, `wins`, `draws`, `losses`, `goals_for`,
        `goals_against`, `goal_diff`, `position`].
    """


    teams: Dict[str, Dict[str, int]] = defaultdict(lambda: {
        "points": 0, "wins": 0, "draws": 0, "losses": 0,
        "goals_for": 0, "goals_against": 0,
    })
    for _, row in matches.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]
        hg, ag = row["FTHG"], row["FTAG"]
        teams[home]["goals_for"] += hg
        teams[home]["goals_against"] += ag
        teams[away]["goals_for"] += ag
        teams[away]["goals_against"] += hg
        if hg > ag:
            teams[home]["points"] += 3
            teams[home]["wins"] += 1
            teams[away]["losses"] += 1
        elif hg < ag:
            teams[away]["points"] += 3
            teams[away]["wins"] += 1
            teams[home]["losses"] += 1
        else:
            teams[home]["points"] += 1
            teams[away]["points"] += 1
            teams[home]["draws"] += 1
            teams[away]["draws"] += 1

    data = []
    for team, stats in teams.items():
        goal_diff = stats["goals_for"] - stats["goals_against"]
        data.append({"team": team, **stats, "goal_diff": goal_diff})

    summary = pd.DataFrame(data)
    summary = summary.sort_values(
        ["points", "goal_diff", "goals_for"], ascending=[False, False, False]
    ).reset_index(drop=True)
    summary["position"] = summary.index + 1
    return summary

# This function is modified to accept promoted teams
def prepare_training_data(season_files: List[str], promoted_teams: List[str]) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Prepare training features and labels from a list of LaLiga seasons.

    Given a list of file paths ordered chronologically, compute per‑team
    statistics for each season and build a dataset where the feature
    vector for season `n+1` comes from the statistics of season `n`.
    Teams promoted into LaLiga without previous season statistics are
    assigned default feature values equal to the average of the bottom
    three clubs in the prior season.

    Parameters
    ----------
    season_files : list of str
        Paths to season CSV files ordered from oldest to newest.

    Returns
    -------
    X_train : DataFrame
        Feature matrix (numeric) for training.
    y_train : Series
        Target series containing league positions (1–20).
    latest_features : DataFrame
        Feature matrix for the most recent season in the list (used
        for prediction).
    """

    season_summaries: Dict[str, pd.DataFrame] = {}
    # Inside the prepare_training_data function

    for file_path in season_files:
        try:
            # Define only the columns we need for the model
            columns_to_use = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
            
            # Use both encoding='latin1' and usecols to robustly read the CSV
            raw = pd.read_csv(
                file_path, 
                encoding='latin1', 
                usecols=columns_to_use
            )
            
            summary = summarise_season(raw)
            season_summaries[file_path] = summary
        except FileNotFoundError:
            print(f"Warning: Could not find season file {file_path}. Skipping.")
            continue
        except Exception as e:
            print(f"Warning: Error processing {file_path}: {e}. Skipping.")
            continue

    feature_rows, target_rows = [], []
    files_sorted = sorted(season_summaries.keys())

    for i in range(len(files_sorted) - 1):
        prev_summary = season_summaries[files_sorted[i]].copy().set_index("team")
        curr_summary = season_summaries[files_sorted[i+1]].copy().set_index("team")
        bottom_three = prev_summary.tail(3)
        default_features = bottom_three.mean(numeric_only=True).to_dict()

        for team, row in curr_summary.iterrows():
            if team in prev_summary.index:
                feats = prev_summary.loc[team].to_dict()
            else: # Promoted team
                feats = default_features
            feature_rows.append(feats)
            target_rows.append(row["position"])

    X_train = pd.DataFrame(feature_rows)[["points", "wins", "draws", "losses", "goals_for", "goals_against", "goal_diff"]]
    y_train = pd.Series(target_rows)

    # Prepare features for the final prediction
    last_season_path = files_sorted[-1]
    last_summary = season_summaries[last_season_path].copy().set_index("team")
    bottom_three_last = last_summary.tail(3)
    default_features_last = bottom_three_last.mean(numeric_only=True).to_dict()

    prediction_features = {}
    for team in last_summary.index:
        prediction_features[team] = last_summary.loc[team].to_dict()

    for team in promoted_teams:
        if team not in prediction_features:
            prediction_features[team] = default_features_last

    latest_features_df = pd.DataFrame.from_dict(prediction_features, orient='index')
    latest_features_df = latest_features_df[["points", "wins", "draws", "losses", "goals_for", "goals_against", "goal_diff"]]

    return X_train, y_train, latest_features_df

# This function remains unchanged
def build_and_train_model(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Create a pipeline that scales features and trains a RandomForest.

    Parameters
    ----------
    X : DataFrame
        Training features.
    y : Series
        Target positions (1–20).

    Returns
    -------
    Pipeline
        Scikit‑learn pipeline with StandardScaler and RandomForestClassifier.
    """
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=100, max_depth=8, random_state=42, class_weight="balanced"
        ))
    ])
    model.fit(X, y)
    return model

# This function remains unchanged
def predict_league_table(model: Pipeline, features: pd.DataFrame) -> pd.DataFrame:
    """Predict the LaLiga table ordering for the given features.

    Parameters
    ----------
    model : Pipeline
        Trained scikit‑learn pipeline.
    features : DataFrame
        Feature rows indexed by team name.

    Returns
    -------
    DataFrame
        Predicted positions sorted from 1 to 20.
    """
    # use predicted probabilities to compute an expected finishing
    # position.  RandomForestClassifier returns a probability
    # distribution over the 20 possible finishing positions.  By
    # multiplying each probability by its corresponding class index
    # (1–20) we obtain an expected (fractional) finishing position.
    probas = model.predict_proba(features)
    classes = model.named_steps["rf"].classes_
    exp_positions = probas.dot(classes)
    prediction_df = pd.DataFrame({
        "team": features.index,
        "expected_position": exp_positions
    })
    prediction_df = prediction_df.sort_values("expected_position").reset_index(drop=True)
    prediction_df["predicted_rank"] = prediction_df.index + 1
    return prediction_df[["predicted_rank", "team", "expected_position"]]

# New high-level API function
def predict_season(data_dir: str, league: str, prediction_year: int, promoted_teams: List[str]) -> pd.DataFrame:
    """
    Trains a model on historical data and predicts the outcome of a given season.

    Args:
        data_dir: Path to the directory containing season CSV files.
        prediction_year: The starting year of the season to predict (e.g., 2025 for 2025-26 season).
        promoted_teams: A list of team names promoted for the prediction season.

    Returns:
        A DataFrame with the predicted league table.
    """
    # Generate file paths for all seasons up to the one before the prediction year
    # Example: for prediction_year=2025, we need data up to season-2425.csv

    league_path = os.path.join(data_dir, league)
    season_files = []
    for year in range(2000, prediction_year):
        start_year_short = str(year)[-2:]
        end_year_short = str(year + 1)[-2:]
        filename = f"season-{start_year_short}{end_year_short}.csv"
        season_files.append(os.path.join(league_path, filename))

    X_train, y_train, latest_features = prepare_training_data(season_files, promoted_teams)
    model = build_and_train_model(X_train, y_train)
    predictions = predict_league_table(model, latest_features)
    
    # Ensure we only return 20 teams for a standard league
    return predictions.iloc[:20].copy()
