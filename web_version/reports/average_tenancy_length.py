import streamlit as st
import pandas as pd
import logging
from sqlalchemy import text
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
import plotly.graph_objects as go

# Import shared functions from utils
from web_version.utils import get_engine

# Configure logging for this report module
logger = logging.getLogger(__name__)

@st.cache_data(ttl=600)
def calculate_rolling_avg_tenancy(start_date, end_date, window_months=12):
    """
    Calculates the rolling average length of completed tenancies.

    For each month between start_date and end_date, it calculates the average
    tenancy duration (in days) for all leases that ended within the preceding
    'window_months' period.

    Args:
        start_date (date): The beginning of the reporting period.
        end_date (date): The end of the reporting period.
        window_months (int): The number of months to include in the rolling window.

    Returns:
        pd.DataFrame: DataFrame with columns ['month_end', 'avg_tenancy_months', 'leases_in_window']
                      or an empty DataFrame if an error occurs or no data is found.
    """
    logger.info(f"Calculating rolling average tenancy length in months ({window_months}-month window) from {start_date} to {end_date}")
    engine = get_engine()
    if engine is None:
        logger.error("Failed to get DB engine for tenancy length data.")
        st.error("Database connection failed.")
        return pd.DataFrame()

    # --- Step 1: Query Completed Leases ---
    # IMPORTANT: Adjust the WHERE clause based on how you identify completed/ended leases.
    # Common ways: lease_status = 'Expired', 'Terminated', 'Ended', or lease_to_date IS NOT NULL AND lease_to_date <= today
    # Using lease_to_date is generally more reliable if statuses aren't consistently updated.
    query_leases = """
        SELECT
            id AS lease_id,
            lease_from_date,
            lease_to_date
        FROM leases
        WHERE lease_to_date IS NOT NULL
          AND lease_from_date IS NOT NULL
          AND lease_to_date >= lease_from_date -- Basic sanity check
          -- Add more specific status checks if needed, e.g.:
          -- AND lease_status IN ('Expired', 'Terminated', 'Past')
    """
    try:
        with engine.connect() as conn:
            leases_df = pd.read_sql_query(text(query_leases), conn)
        logger.info(f"Loaded {leases_df.shape[0]} leases with start and end dates.")
    except Exception as e:
        logger.exception("Failed to load lease data.")
        st.error(f"DB Error loading leases: {e}")
        return pd.DataFrame()

    if leases_df.empty:
        logger.warning("No leases with valid start and end dates found.")
        st.warning("No completed lease data found to analyze.")
        return pd.DataFrame()

    # --- Step 2: Calculate Tenancy Duration ---
    # Convert to datetime objects first, coercing errors to NaT
    leases_df['lease_from_dt'] = pd.to_datetime(leases_df['lease_from_date'], errors='coerce')
    leases_df['lease_to_dt'] = pd.to_datetime(leases_df['lease_to_date'], errors='coerce')

    # Drop rows where either date conversion failed
    leases_df.dropna(subset=['lease_from_dt', 'lease_to_dt'], inplace=True)

    # Calculate duration in days first
    leases_df['tenancy_duration_days'] = (leases_df['lease_to_dt'] - leases_df['lease_from_dt']).dt.days
    # Approximate duration in months (average days per month ~ 30.4375)
    leases_df['tenancy_duration_months'] = leases_df['tenancy_duration_days'] / 30.4375

    # Filter out negative or zero durations which indicate data issues or same-day leases
    leases_df = leases_df[leases_df['tenancy_duration_days'] > 0].copy() # Use .copy() to avoid SettingWithCopyWarning

    # Keep original date columns if needed elsewhere, or drop the intermediate dt columns
    # leases_df.drop(columns=['lease_from_dt', 'lease_to_dt'], inplace=True)
    # Convert back to date objects for filtering logic later (optional, depends on comparison needs)
    leases_df['lease_from_date'] = leases_df['lease_from_dt'].dt.date
    leases_df['lease_to_date'] = leases_df['lease_to_dt'].dt.date


    if leases_df.empty:
        logger.warning("No leases with positive tenancy duration found after calculation and filtering.")
        st.warning("No valid tenancy durations could be calculated.")
        return pd.DataFrame()

    logger.info(f"Calculated duration in months for {leases_df.shape[0]} leases.")

    # --- Step 3: Calculate Rolling Average ---
    results = []
    current_month_start = date(start_date.year, start_date.month, 1)

    while current_month_start <= end_date:
        month_end = (current_month_start + relativedelta(months=1)) - relativedelta(days=1)
        window_start = month_end - relativedelta(months=window_months) + relativedelta(days=1)

        # Filter leases that ended within the rolling window [window_start, month_end]
        leases_in_window_df = leases_df[
            (leases_df['lease_to_date'] >= window_start) &
            (leases_df['lease_to_date'] <= month_end)
        ]

        avg_duration_months = leases_in_window_df['tenancy_duration_months'].mean()
        count = leases_in_window_df.shape[0]

        results.append({
            'month_end': month_end,
            'avg_tenancy_months': avg_duration_months if pd.notna(avg_duration_months) else 0,
            'leases_in_window': count
        })

        # Move to the next month
        current_month_start += relativedelta(months=1)

    if not results:
        logger.warning("No results generated in the rolling average calculation.")
        return pd.DataFrame()

    results_df = pd.DataFrame(results)
    # Ensure 'month_end' is a datetime type before using .dt accessor
    results_df['month_end'] = pd.to_datetime(results_df['month_end'], errors='coerce')
    results_df.dropna(subset=['month_end'], inplace=True) # Drop if conversion failed

    if results_df.empty:
        logger.warning("No valid month_end dates after conversion in results.")
        return pd.DataFrame()

    results_df['month_str'] = results_df['month_end'].dt.strftime('%Y-%m') # For display

    logger.info(f"Finished calculating rolling average tenancy length. Generated {results_df.shape[0]} monthly data points.")
    return results_df


def main(start_date, end_date):
    """
    Main function to display the Average Tenancy Length report in Streamlit.
    """
    # st.header("Average Tenancy Length (Rolling 12-Month Window)") # Removed - Redundant
    # st.caption("Calculates the average length (in months) of tenancies that ended within the previous 12 months, updated monthly.") # Removed - Redundant
    # st.write(f"Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}") # Removed - Redundant
    logger.info(f"Running Average Tenancy Length report (in months) for {start_date} to {end_date}")

    # --- Configuration ---
    window_months = st.sidebar.slider("Rolling Window Size (Months)", min_value=3, max_value=24, value=12, step=1)
    st.header(f"Average Tenancy Length (Rolling {window_months}-Month Window)") # Keep dynamic header

    # --- Load Data ---
    avg_tenancy_df = calculate_rolling_avg_tenancy(start_date, end_date, window_months)

    if avg_tenancy_df.empty:
        st.warning("No data available to calculate average tenancy length for the selected period.")
        logger.warning("No tenancy data loaded or processed.")
        return

    # --- Display Chart ---
    logger.info("Generating plot...")
    try:
        fig = go.Figure()

        # Trace for average tenancy length in months
        fig.add_trace(go.Scatter(
            x=avg_tenancy_df['month_end'],
            y=avg_tenancy_df['avg_tenancy_months'],
            mode='lines+markers',
            name=f'Avg. Tenancy Length ({window_months}mo Rolling)',
            line=dict(color='purple'),
            fill='tozeroy', # Add fill
            fillcolor='rgba(128, 0, 128, 0.2)', # Semi-transparent purple
            hovertemplate =
                '<b>Month End:</b> %{x|%Y-%m-%d}<br>' +
                '<b>Avg. Length:</b> %{y:.1f} months<br>' + # Show one decimal place for months
                '<b>Leases in Window:</b> %{customdata[0]}<extra></extra>',
            customdata=avg_tenancy_df[['leases_in_window']]
        ))

        fig.update_layout(
            title=f"Rolling {window_months}-Month Average Tenancy Length",
            xaxis_title="Month",
            yaxis_title="Average Tenancy Length (Months)",
            # template="plotly_white", # Removed template
            hovermode="x unified"
        )
        fig.update_yaxes(rangemode='tozero')

        st.plotly_chart(fig, use_container_width=True)

        # --- Add plain English description ---
        st.markdown(f"""
        **How this graph is created:**
        *   This graph shows the average length (in months) of completed tenancies.
        *   For each month shown on the graph, the average is calculated based on all leases that *ended* within the previous **{window_months} months** (the rolling window size selected in the sidebar).
        *   This helps visualize the trend in how long tenants are staying over time.
        """)
        # --- End description ---

    except Exception as e:
        logger.exception("Failed to generate plot.")
        st.error(f"Error generating plot: {e}")

    # --- Display Data Table ---
    st.subheader("Data Table")
    try:
        display_df = avg_tenancy_df[['month_str', 'avg_tenancy_months', 'leases_in_window']].copy()
        display_df.rename(columns={
            'month_str': 'Month End',
            'avg_tenancy_months': 'Avg. Length (Months)',
            'leases_in_window': f'# Leases Ended in Prior {window_months} Mos.'
        }, inplace=True)
        # Format average months to one decimal place for display
        display_df['Avg. Length (Months)'] = display_df['Avg. Length (Months)'].round(1)
        st.dataframe(display_df.sort_values(by='Month End', ascending=False), hide_index=True)
    except Exception as e:
        logger.exception("Failed to display data table.")
        st.error(f"Error displaying data table: {e}")

    logger.info("Average Tenancy Length report execution finished.")

# Note: This report won't run directly via `python average_tenancy_length.py`
# It needs to be integrated into your Streamlit app structure (e.g., called from app.py)
# where start_date and end_date are provided, likely via user input widgets.
