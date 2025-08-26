import streamlit as st
import pandas as pd
import os
from datetime import datetime

# Make sure your backend package is installed ('pip install -e .')
# and that its predict_season function accepts a 'league' argument.
from predictor import predict_season
from predictor.predictor import summarise_season

st.set_page_config(page_title="Football League Predictor", layout="wide")

# --- DATA DIRECTORY ---
DATA_DIR = "data"

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Controls")

# 1. LEAGUE SELECTOR 
league_display_name = st.sidebar.selectbox(
    "Choose a league",
    ("LaLiga", "Premier League")
)
# Convert display name to the directory name format (e.g., "Premier League" -> "premier_league")
league_dir_name = league_display_name.lower().replace(" ", "_")


# 2. DYNAMIC TITLE 
st.title(f"🏆 {league_display_name} Season Predictor")
st.write(f"This app uses a Random Forest model to predict the final standings of a {league_display_name} season based on historical data.")


# --- HELPER FUNCTION TO EVALUATE PAST PERFORMANCE ---
@st.cache_data # Caches results per league
def evaluate_historical_performance(league: str):
    """Evaluate the model's performance on past seasons for a specific league."""
    errors = {}
    predictions_log = {}
    league_path = os.path.join(DATA_DIR, league)
    columns_to_use = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']

    # We can predict seasons for which we have prior data
    for year in range(2005, 2024):
        try:
            # Construct file paths for the specific league
            start_yr, end_yr = str(year)[-2:], str(year + 1)[-2:]
            actual_df = pd.read_csv(
                os.path.join(league_path, f"season-{start_yr}{end_yr}.csv"),
                encoding='latin1',
                usecols=columns_to_use
            )
            actual_teams = set(actual_df['HomeTeam'].unique())
            
            prev_start_yr, prev_end_yr = str(year - 1)[-2:], str(year)[-2:]
            prev_df = pd.read_csv(
                os.path.join(league_path, f"season-{prev_start_yr}{prev_end_yr}.csv"),
                encoding='latin1',
                usecols=columns_to_use
            )
            prev_teams = set(prev_df['HomeTeam'].unique())
            promoted = list(actual_teams - prev_teams)

            # Pass the league parameter to the prediction function
            predicted_table = predict_season(DATA_DIR, league, year, promoted)
            actual_table = summarise_season(actual_df)[['team', 'position']].set_index('team')
            comparison = predicted_table.set_index('team').join(actual_table, rsuffix='_actual')
            comparison['error'] = abs(comparison['predicted_rank'] - comparison['position'])
            errors[f"{year}-{year+1}"] = comparison['error'].mean()
            predictions_log[f"{year}-{year+1}"] = comparison.reset_index()

        except FileNotFoundError:
            continue
            
    return errors, predictions_log


# --- SIDEBAR FOR USER INPUT ---
st.sidebar.header("Make a Prediction for the Upcoming Season (2025-2026)")
year_str = st.sidebar.text_input("Enter Prediction Year (e.g., 2025)", "2025")

default_promoted_teams = {
    "laliga": "Levante, Elche, Real Oviedo",
    "premier_league": "Leicester, Ipswich, Southampton"
}
promoted_teams_str = st.sidebar.text_area(
    "Enter Promoted Teams (comma-separated)",
    default_promoted_teams.get(league_dir_name, "")
)

if st.sidebar.button("Predict"):
    try:
        prediction_year = int(year_str)
        promoted_teams = [team.strip() for team in promoted_teams_str.split(',')]
        
        if not promoted_teams or not all(promoted_teams):
            st.sidebar.error("Please enter valid team names.")
        else:
            with st.spinner(f"Predicting the {prediction_year}-{prediction_year+1} {league_display_name} season..."):
                predictions = predict_season(DATA_DIR, league_dir_name, prediction_year, promoted_teams)
                
                st.subheader(f"Predicted Table for {prediction_year}-{prediction_year+1} Season")
                predictions.index = predictions.index + 1
                st.dataframe(
                    predictions.style.format({'expected_position': "{:.2f}"}),
                    use_container_width=True
                )

    except ValueError:
        st.sidebar.error("Please enter a valid year.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
st.write("Right now, the model can only predict seasons up to 2025-2026 due to data limitations. Additional data will be added as it becomes available.")


# --- MAIN PANEL FOR HISTORICAL PERFORMANCE ---
st.header("Model Performance on Past Seasons")

try:
    with st.spinner(f"Evaluating model performance for {league_display_name}..."):
        historical_errors, predictions_log = evaluate_historical_performance(league_dir_name)

    if historical_errors:
        error_df = pd.DataFrame.from_dict(historical_errors, orient='index', columns=['Mean Absolute Error (Position)'])
        
        st.write("The chart below shows the model's average prediction error for each past season.")
        st.line_chart(error_df)
        
        st.write("Here's a detailed look at the prediction for the last available season versus the actual outcome:")
        last_season = list(predictions_log.keys())[-1]
        st.subheader(f"Prediction vs. Actual for {last_season} Season")
        
        # Reset index to start from 1 instead of 0 for display
        display_df = predictions_log[last_season].copy()
        display_df.index = display_df.index + 1
        st.dataframe(
            display_df.style.format({'expected_position': "{:.2f}", 'error': "{:.0f}", 'position': "{:.0f}"}),
            use_container_width=True
        )
    else:
        st.warning(f"Could not load historical data for {league_display_name}. Make sure your `data/{league_dir_name}` directory is populated.")
except Exception as e:
    st.error(f"Could not process data for {league_display_name}. Please check the data files. Error: {e}")