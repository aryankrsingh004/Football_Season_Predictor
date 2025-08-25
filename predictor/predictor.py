# laliga_predictor/predictor.py

import os
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# This function remains unchanged from your original script
def summarise_season(matches: pd.DataFrame) -> pd.DataFrame:
    """Summarise a LaLiga season into per-team statistics and final ranking."""
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
    """Prepare training features and labels from a list of LaLiga seasons."""
    season_summaries: Dict[str, pd.DataFrame] = {}
    for file_path in season_files:
        try:
            raw = pd.read_csv(file_path)
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
    """Create a pipeline that scales features and trains a RandomForest."""
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
    """Predict the LaLiga table ordering for the given features."""
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
def predict_season(data_dir: str, prediction_year: int, promoted_teams: List[str]) -> pd.DataFrame:
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
    season_files = []
    for year in range(2000, prediction_year):
        start_year_short = str(year)[-2:]
        end_year_short = str(year + 1)[-2:]
        filename = f"season-{start_year_short}{end_year_short}.csv"
        season_files.append(os.path.join(data_dir, filename))

    X_train, y_train, latest_features = prepare_training_data(season_files, promoted_teams)
    model = build_and_train_model(X_train, y_train)
    predictions = predict_league_table(model, latest_features)
    
    # Ensure we only return 20 teams for a standard league
    return predictions.iloc[:20].copy()