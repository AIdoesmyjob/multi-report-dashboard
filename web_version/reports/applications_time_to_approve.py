# reports/applications_time_to_approve.py
# (Refactored to use shared utils)

import streamlit as st
import pandas as pd
from sqlalchemy import text, exc as sqlalchemy_exc # Removed create_engine
from datetime import datetime, timedelta
import plotly.graph_objects as go
import os
import logging # <-- Added
import numpy as np # For handling potential NaN in formatting
# from dotenv import load_dotenv # <-- Removed

# Import shared functions from utils
from web_version.utils import get_engine # <-- Import shared engine function

# Configure logging for this report module
logger = logging.getLogger(__name__)

# --- Data Loading Function ---
@st.cache_data(ttl=600)
def load_avg_approval_time_data(_engine, start_date, end_date):
    """
    Calculates the average time (in days) from application submission to
    applicant status becoming 'AddedToLease', grouped by submission month.
    Uses logic to link the approval date to the most recent application
    submitted by the applicant *before* the approval date.

    Args:
        _engine: SQLAlchemy database engine instance.
        start_date: The start date of the analysis period (for application submission).
        end_date: The end date of the analysis period (for application submission).

    Returns:
        pandas.DataFrame: DataFrame with columns
            ['month_start', 'avg_approval_days', 'month_start_dt'],
            or an empty DataFrame on error/no data.
    """
    logger.info(f"Loading average approval time data from {start_date} to {end_date}")
    target_applicant_status_value = 'AddedToLease'
    # Adjust end_date to be exclusive for the query
    end_date_adjusted = end_date + timedelta(days=1)

    # Query uses ROW_NUMBER() to find the most recent application submitted
    # *before* the applicant's status was updated to AddedToLease.
    query = text("""
        WITH RankedApplications AS (
            SELECT
                a.application_id,
                a.applicant_id,
                a.application_submitted_datetime,
                p.last_updated_datetime AS applicant_approval_datetime,
                -- Rank applications per applicant, latest first, only considering those submitted <= approval date
                ROW_NUMBER() OVER(
                    PARTITION BY a.applicant_id
                    ORDER BY a.application_submitted_datetime DESC
                ) as rn
            FROM
                applications a
            INNER JOIN
                applicants p ON a.applicant_id = p.applicant_id
            WHERE
                -- Filter for applicants who reached the target status
                p.status ILIKE :target_status
                -- Filter by application submission date range (optional optimization, can also filter later)
                AND a.application_submitted_datetime >= :start_date
                AND a.application_submitted_datetime < :end_date_adj
                -- Ensure timestamps are valid for calculation and comparison
                AND a.application_submitted_datetime IS NOT NULL
                AND p.last_updated_datetime IS NOT NULL
                -- Crucial: Only consider applications submitted *before* or *on* the approval date
                AND a.application_submitted_datetime <= p.last_updated_datetime
        )
        SELECT
            DATE_TRUNC('month', ra.application_submitted_datetime)::date AS month_start_dt,
            -- Calculate average difference in days using the *latest relevant* application's submission date
            AVG(
                EXTRACT(EPOCH FROM (ra.applicant_approval_datetime - ra.application_submitted_datetime))
            ) / 86400.0 AS avg_approval_days
        FROM
            RankedApplications ra
        WHERE
            ra.rn = 1 -- Select only the most recent application (rn=1) before approval for each applicant
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
                "target_status": target_applicant_status_value
            }
            db_results = pd.read_sql_query(query, conn, params=params, parse_dates=['month_start_dt'])
        logger.info(f"Loaded {len(db_results)} monthly average approval times from DB.")

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

        # Merge results with full month range
        if not df_months.empty:
             merged_df = pd.merge(df_months, db_results, on='month_start_dt', how='left')
        else:
             merged_df = db_results # Use only db results if month range is empty

        # No need to fillna(0) for average - NaN is more appropriate if no data exists
        # Ensure month_start_dt is datetime before formatting
        merged_df['month_start_dt'] = pd.to_datetime(merged_df['month_start_dt'])
        merged_df['month_start'] = merged_df['month_start_dt'].dt.strftime('%Y-%m-%d')


        # Select and order final columns
        final_df = merged_df[['month_start', 'avg_approval_days', 'month_start_dt']]
        logger.info(f"Processed average approval time data, shape: {final_df.shape}")
        return final_df

    except Exception as e:
        error_msg = f"Error querying or processing average approval time data: {e}"
        logger.exception(error_msg)
        st.error(error_msg)
        return pd.DataFrame(columns=['month_start', 'avg_approval_days', 'month_start_dt'])

# --- Main Report Function ---
def main(start_date, end_date): # Takes only start_date, end_date
    """
    Main function for the Average Lease Approval Time report.
    """
    # st.header("Average Time to Lease Approval (Days)") # Removed - Redundant
    # st.write(f"Showing average days from application submission to applicant status 'AddedToLease'") # Removed - Redundant
    # st.write(f"Based on applications submitted between {start_date.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}") # Removed - Redundant
    st.info("Note: Calculation attempts to link approval date to the most recent application submitted beforehand.") # Updated info message
    logger.info(f"Running Average Approval Time report for {start_date} to {end_date}")

    engine = get_engine() # <-- Use shared engine function
    if engine is None:
        logger.error("Failed to get DB engine. Aborting report.")
        return

    approval_time_df = load_avg_approval_time_data(engine, start_date, end_date)

    # Check if DataFrame is completely empty or only contains NaN averages
    if approval_time_df.empty or approval_time_df['avg_approval_days'].isnull().all():
        st.warning("No data found for average approval time in the selected date range, or an error occurred.")
        logger.warning("No average approval time data loaded or processed.")
        return

    # --- Plotting ---
    logger.info("Generating plot...")
    try:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=approval_time_df['month_start_dt'],
            y=approval_time_df['avg_approval_days'],
            mode='lines+markers',
            name='Avg Approval Time',
            line=dict(color='orange'),
            fill='tozeroy', # Add fill
            fillcolor='rgba(255, 165, 0, 0.2)', # Semi-transparent orange
            hovertemplate='<b>%{x|%b %Y}</b><br>Avg Days: %{y:.1f}<extra></extra>'
        ))

        fig.update_layout(
            title="Average Time from Application Submission to Lease Approval",
            xaxis_title="Application Submission Month",
            yaxis_title="Average Days to Approval",
            xaxis_tickformat="%b %Y",
            # template="plotly_white", # Removed template
            hovermode="x unified"
        )
        fig.update_yaxes(rangemode='tozero')

        st.plotly_chart(fig, use_container_width=True)

        # --- Add plain English description ---
        st.markdown("""
        **How this graph is created:**
        *   This graph shows the average number of days it took from when an application was submitted to when the applicant's status was marked as 'AddedToLease'.
        *   **Data Note:** The large spike observed roughly between July 2023 and June 2024 is likely due to incomplete or inaccurate historical data entry for approval dates during that period and should be interpreted with caution.
        """)
        # --- End description ---

    except Exception as e:
        error_msg = f"Failed to generate plot: {e}"
        logger.exception(error_msg)
        st.error(error_msg)

    # --- Data Table ---
    st.subheader("Data Table")
    try:
        display_df = approval_time_df[['month_start_dt', 'avg_approval_days']].copy()
        # Ensure month_start_dt is datetime before formatting
        display_df['month_start_dt'] = pd.to_datetime(display_df['month_start_dt'])
        display_df['Month'] = display_df['month_start_dt'].dt.strftime('%B %Y')
        # Format average days to 1 decimal place, handle potential NaN
        display_df['Avg Days to Approval'] = display_df['avg_approval_days'].apply(lambda x: f'{x:.1f}' if pd.notna(x) else 'N/A')
        display_df = display_df[['Month', 'Avg Days to Approval']] # Select and order columns

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

# --- Removed Example Usage (if running standalone) ---
# if __name__ == "__main__":
#     ...
