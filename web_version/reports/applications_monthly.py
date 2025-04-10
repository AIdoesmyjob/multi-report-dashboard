# reports/applications_monthly.py
# (Refactored to use shared utils)

import streamlit as st
import pandas as pd
from sqlalchemy import text, exc as sqlalchemy_exc # Removed create_engine
from datetime import datetime, timedelta
import plotly.graph_objects as go
import os
import logging # <-- Added
# from dotenv import load_dotenv # <-- Removed

# Import shared functions from utils
from web_version.utils import get_engine # <-- Import shared engine function

# Configure logging for this report module
logger = logging.getLogger(__name__)


# --- Data Loading Function (with JOIN) ---
@st.cache_data(ttl=600)
def load_applicant_status_data(_engine, start_date, end_date):
    """
    Retrieves application data, joining with applicants to count total and
    applications where the applicant status is 'AddedToLease'.

    Args:
        _engine: SQLAlchemy database engine instance.
        start_date: The start date of the analysis period (for application submission).
        end_date: The end date of the analysis period (for application submission).

    Returns:
        pandas.DataFrame: DataFrame with columns
            ['month_start', 'total_applications', 'applicant_added_count', 'month_start_dt'],
            or an empty DataFrame on error/no data.
    """
    logger.info(f"Loading applicant status data from {start_date} to {end_date}")
    # --- Target status is 'AddedToLease' in the APPLICANTS table ---
    target_applicant_status_value = 'AddedToLease'

    # Adjust end_date to be exclusive for the query
    end_date_adjusted = end_date + timedelta(days=1)

    # --- Query JOINS applications (a) with applicants (p) ---
    query = text(f"""
        SELECT
            DATE_TRUNC('month', a.application_submitted_datetime)::date AS month_start_dt,
            COUNT(a.application_id) AS total_applications,
            -- Count applications where the corresponding applicant has the target status
            COUNT(a.application_id) FILTER (WHERE p.status ILIKE :target_status) AS applicant_added_count
        FROM
            applications a
        LEFT JOIN -- Use LEFT JOIN in case an applicant record is missing
            applicants p ON a.applicant_id = p.applicant_id
        WHERE
            a.application_submitted_datetime >= :start_date
            AND a.application_submitted_datetime < :end_date_adj
            AND a.application_submitted_datetime IS NOT NULL
        GROUP BY
            month_start_dt
        ORDER BY
            month_start_dt ASC;
    """)

    try:
        with _engine.connect() as conn:
            params = {
                "start_date": start_date,
                "end_date_adj": end_date_adjusted,
                "target_status": target_applicant_status_value # Check applicants.status
            }
            db_counts = pd.read_sql_query(query, conn, params=params, parse_dates=['month_start_dt'])
        logger.info(f"Loaded {len(db_counts)} monthly application counts from DB.")

        # --- Create complete monthly range ---
        start_date_dt = pd.to_datetime(start_date)
        end_date_dt = pd.to_datetime(end_date)
        # Ensure start_month is not after end_month
        start_month = start_date_dt.replace(day=1)
        end_month = end_date_dt.replace(day=1)
        if start_month > end_month:
             logger.warning("Start date is after end date, resulting in empty month range.")
             all_months = pd.DatetimeIndex([])
        else:
             all_months = pd.date_range(start=start_month, end=end_month, freq='MS')

        df_months = pd.DataFrame({'month_start_dt': all_months})

        # Merge and clean
        if not df_months.empty:
            merged_df = pd.merge(df_months, db_counts, on='month_start_dt', how='left')
        else:
            # If no months in range, use only db results (should be empty anyway)
            merged_df = db_counts

        merged_df['total_applications'] = merged_df['total_applications'].fillna(0).astype(int)
        merged_df['applicant_added_count'] = merged_df['applicant_added_count'].fillna(0).astype(int)
        # Ensure month_start_dt is datetime before formatting
        merged_df['month_start_dt'] = pd.to_datetime(merged_df['month_start_dt'])
        merged_df['month_start'] = merged_df['month_start_dt'].dt.strftime('%Y-%m-%d')


        # Select and order final columns
        final_df = merged_df[['month_start', 'total_applications', 'applicant_added_count', 'month_start_dt']]
        logger.info(f"Processed applicant status data, shape: {final_df.shape}")
        return final_df

    except Exception as e:
        error_msg = f"Error querying or processing applicant status data: {e}"
        logger.exception(error_msg)
        st.error(error_msg)
        return pd.DataFrame(columns=['month_start', 'total_applications', 'applicant_added_count', 'month_start_dt'])

# --- Main Report Function ---
def main(start_date, end_date): # Takes only start_date, end_date
    """
    Main function for the Monthly Applications vs Applicant 'AddedToLease' Status report.
    """
    # st.header("Monthly Applications vs. Applicants 'AddedToLease'") # Removed - Redundant with app.py subheader
    # st.write(f"Displaying applications submitted between {start_date.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}") # Removed - Redundant with app.py caption
    st.caption(f"Note: Counts applications where the applicant's status in the 'applicants' table is 'AddedToLease'.")
    logger.info(f"Running Applications Monthly report for {start_date} to {end_date}")

    engine = get_engine() # <-- Use shared engine function
    if engine is None:
        logger.error("Failed to get DB engine. Aborting report.")
        # Error already shown by get_engine in utils
        return

    status_counts_df = load_applicant_status_data(engine, start_date, end_date)

    if status_counts_df.empty:
        st.warning("No application data found for the selected date range, or an error occurred during loading.")
        logger.warning("No application data loaded or processed.")
        return

    # --- Plotting ---
    logger.info("Generating plot...")
    try:
        fig = go.Figure()

        # --- Trace 1: Total Applications ---
        fig.add_trace(go.Scatter(
            x=status_counts_df['month_start_dt'],
            y=status_counts_df['total_applications'],
            mode='lines+markers',
            name='Total Applications',
            line=dict(color='royalblue'),
            fill='tozeroy', # Add fill to x-axis
            fillcolor='rgba(65, 105, 225, 0.2)' # Semi-transparent royal blue
        ))

        # --- Trace 2: Applicant AddedToLease ---
        fig.add_trace(go.Scatter(
            x=status_counts_df['month_start_dt'],
            y=status_counts_df['applicant_added_count'], # Use correct column
            mode='lines+markers',
            name='Applicant AddedToLease',
            line=dict(color='limegreen'), # Correct label
            fill='tozeroy', # Add fill to x-axis
            fillcolor='rgba(50, 205, 50, 0.2)' # Semi-transparent lime green
        ))

        fig.update_layout(
            title="Total Applications vs. Applicants 'AddedToLease' per Month", # Correct title
            xaxis_title="Month", yaxis_title="Number of Applications",
            xaxis_tickformat="%b %Y", hovermode="x unified", # Removed template="plotly_white"
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        fig.update_yaxes(rangemode='tozero')

        st.plotly_chart(fig, use_container_width=True)

        # --- Add plain English description ---
        st.markdown("""
        **How this graph is created:**
        *   The **blue line** shows the total number of new rental applications received each month within the selected date range.
        *   The **green line** shows how many of those monthly applications came from applicants who were eventually approved and marked as 'AddedToLease' in the system.
        """)
        # --- End description ---

    except Exception as e:
        error_msg = f"Failed to generate plot: {e}"
        logger.exception(error_msg)
        st.error(error_msg)


    # --- Data Table ---
    st.subheader("Data Table")
    try:
        display_df = status_counts_df[['month_start_dt', 'total_applications', 'applicant_added_count']].copy()
        # Ensure month_start_dt is datetime before formatting
        display_df['month_start_dt'] = pd.to_datetime(display_df['month_start_dt'])
        display_df['Month'] = display_df['month_start_dt'].dt.strftime('%B %Y')
        display_df = display_df[['Month', 'total_applications', 'applicant_added_count']]
        display_df.rename(columns={
            'total_applications': 'Total Applications',
            'applicant_added_count': 'Applicant AddedToLease' # Correct header
            }, inplace=True)

        # Sort by Month (handle potential errors during conversion)
        try:
             display_df_sorted = display_df.sort_values(
                 by='Month',
                 ascending=False,
                 key=lambda s: pd.to_datetime(s, format='%B %Y', errors='coerce'),
                 na_position='last' # Keep rows with unparseable months at the end
             )
        except Exception as sort_e:
             logger.warning(f"Could not sort dataframe by month string: {sort_e}. Displaying unsorted.")
             display_df_sorted = display_df

        st.dataframe(display_df_sorted, use_container_width=True, hide_index=True)
    except Exception as e:
        error_msg = f"Failed to display data table: {e}"
        logger.exception(error_msg)
        st.error(error_msg)

    logger.info("Report execution finished.")

# Note: No __main__ block needed if only run via app.py
