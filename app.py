import streamlit as st
import pandas as pd
import os
from datetime import datetime

# Import the main function from your new package
from predictor import predict_season
from predictor.predictor import summarise_season

st.set_page_config(page_title="LaLiga Season Predictor", layout="wide")

st.title("⚽ LaLiga Season Predictor")
st.write("This app uses a Random Forest model to predict the final standings of a LaLiga season based on historical data.")

# --- DATA DIRECTORY ---
DATA_DIR = "data"

# --- HELPER FUNCTION TO EVALUATE PAST PERFORMANCE ---
@st.cache_data # Cache the result to avoid re-running on every interaction
def evaluate_historical_performance():
    """Evaluate the model's performance on past seasons."""
    errors = {}
    predictions_log = {}

    # We can predict seasons for which we have prior data, e.g., from 2005-06 onwards
    for year in range(2005, datetime.now().year -1): 
        # Get the actual results for the season we are "predicting"
        try:
            start_yr, end_yr = str(year)[-2:], str(year + 1)[-2:]
            actual_df = pd.read_csv(os.path.join(DATA_DIR, f"season-{start_yr}{end_yr}.csv"))
            actual_teams = set(actual_df['HomeTeam'].unique())
            
            # Find promoted teams (teams in this season that were not in the last)
            prev_start_yr, prev_end_yr = str(year-1)[-2:], str(year)[-2:]
            prev_df = pd.read_csv(os.path.join(DATA_DIR, f"season-{prev_start_yr}{prev_end_yr}.csv"))
            prev_teams = set(prev_df['HomeTeam'].unique())
            promoted = list(actual_teams - prev_teams)

            # Predict the season
            predicted_table = predict_season(DATA_DIR, year, promoted)
            
            # Get actual final standings
            actual_table = summarise_season(actual_df)[['team', 'position']].set_index('team')
            
            # Merge and calculate error
            comparison = predicted_table.set_index('team').join(actual_table, rsuffix='_actual')
            comparison['error'] = abs(comparison['predicted_rank'] - comparison['position'])
            errors[f"{year}-{year+1}"] = comparison['error'].mean()
            predictions_log[f"{year}-{year+1}"] = comparison.reset_index()

        except FileNotFoundError:
            continue # Skip if a season file is missing
            
    return errors, predictions_log


# --- SIDEBAR FOR USER INPUT ---
st.sidebar.header("Make a New Prediction")
year_str = st.sidebar.text_input("Enter Prediction Year (e.g., 2025 for 2025-26 season)", "2025")
promoted_teams_str = st.sidebar.text_area(
    "Enter Promoted Teams (comma-separated)", "Levante, Elche, Real Oviedo"
)

if st.sidebar.button("Predict"):
    try:
        prediction_year = int(year_str)
        promoted_teams = [team.strip() for team in promoted_teams_str.split(',')]
        
        if not promoted_teams or not all(promoted_teams):
             st.sidebar.error("Please enter valid team names.")
        else:
            with st.spinner(f"Predicting the {prediction_year}-{prediction_year+1} season..."):
                predictions = predict_season(DATA_DIR, prediction_year, promoted_teams)
                
                st.subheader(f"Predicted Table for {prediction_year}-{prediction_year+1} Season")
                st.dataframe(
                    predictions.style.format({'expected_position': "{:.2f}"}),
                    use_container_width=True
                )

    except ValueError:
        st.sidebar.error("Please enter a valid year.")
    except Exception as e:
        st.sidebar.error(f"An error occurred: {e}")


# --- MAIN PANEL FOR HISTORICAL PERFORMANCE ---
st.header("Model Performance on Past Seasons")

with st.spinner("Evaluating model performance on historical data..."):
    historical_errors, predictions_log = evaluate_historical_performance()

if historical_errors:
    error_df = pd.DataFrame.from_dict(historical_errors, orient='index', columns=['Mean Absolute Error (Position)'])
    
    st.write("The chart below shows the model's average prediction error for each past season. The error is the average absolute difference between a team's predicted rank and its actual final rank.")
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
    st.warning("Could not load historical data to evaluate performance. Make sure your `data` directory is populated.")