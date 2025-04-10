# reports/showing_status_trends.py

import streamlit as st
import pandas as pd
from sqlalchemy import text, exc as sqlalchemy_exc
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import os
import logging

# Import shared functions from utils
from web_version.utils import get_engine

# Configure logging for this report module
logger = logging.getLogger(__name__)

# --- Data Loading Function ---
@st.cache_data(ttl=600)
def load_showing_status_data(_engine, start_date, end_date):
    """
    Loads showing data and aggregates counts by status per month for the entire portfolio.

    Args:
        _engine: SQLAlchemy database engine instance.
        start_date: The start date of the analysis period (showtime).
        end_date: The end date of the analysis period (showtime).

    Returns:
        pandas.DataFrame: DataFrame with columns like
            ['month_start_dt', 'showing_status', 'count'],
            or an empty DataFrame on error/no data.
    """
    logger.info(f"Loading overall showing status trends data from {start_date} to {end_date}")
    end_date_adjusted = end_date + timedelta(days=1)

    # Query counts showings by status per month
    query = text("""
        SELECT
            DATE_TRUNC('month', showtime)::date AS month_start_dt,
            COALESCE(showing_status, 'Unknown') AS showing_status,
            COUNT(showing_id) AS count
        FROM
            showmojo_prospect_showing_data
        WHERE
            showtime >= :start_date
            AND showtime < :end_date_adj -- Use adjusted end date
            AND showtime IS NOT NULL
        GROUP BY
            month_start_dt,
            showing_status
        ORDER BY
            month_start_dt ASC,
            showing_status ASC;
    """)

    try:
        with _engine.connect() as conn:
            params = {
                "start_date": start_date,
                "end_date_adj": end_date_adjusted
            }
            db_results = pd.read_sql_query(query, conn, params=params, parse_dates=['month_start_dt'])
        logger.info(f"Loaded {len(db_results)} showing status records from DB.")

        # --- Create complete monthly range for each status ---
        start_date_dt = pd.to_datetime(start_date)
        end_date_dt = pd.to_datetime(end_date)
        start_month = start_date_dt.replace(day=1)
        end_month = end_date_dt.replace(day=1)

        if start_month > end_month:
            logger.warning("Start date is after end date, resulting in empty month range.")
            all_months = pd.DatetimeIndex([])
            all_statuses = []
        else:
            all_months = pd.date_range(start=start_month, end=end_month, freq='MS')
            all_statuses = db_results['showing_status'].unique() if not db_results.empty else []

        # Create a full grid of months and statuses
        if len(all_months) > 0 and len(all_statuses) > 0:
            idx = pd.MultiIndex.from_product([all_months, all_statuses], names=['month_start_dt', 'showing_status'])
            df_full_grid = pd.DataFrame(index=idx).reset_index()
        else:
             df_full_grid = db_results if not db_results.empty else pd.DataFrame(columns=['month_start_dt', 'showing_status'])

        # Merge results with the full grid
        if not df_full_grid.empty:
            df_full_grid['month_start_dt'] = pd.to_datetime(df_full_grid['month_start_dt'])
            if not db_results.empty:
                 db_results['month_start_dt'] = pd.to_datetime(db_results['month_start_dt'])
                 merged_df = pd.merge(df_full_grid, db_results, on=['month_start_dt', 'showing_status'], how='left')
            else:
                 merged_df = df_full_grid
                 if 'count' not in merged_df.columns: merged_df['count'] = 0

            # Fill NaN counts with 0 after merge
            merged_df['count'] = merged_df['count'].fillna(0).astype(int)
        else:
             merged_df = db_results # Use only db results if grid creation failed

        # Clean up status names (remove 'STATUS_')
        merged_df['showing_status'] = merged_df['showing_status'].str.replace('STATUS_', '', regex=False).str.title()

        logger.info(f"Processed showing status data, final shape: {merged_df.shape}")
        return merged_df.sort_values(by=['month_start_dt', 'showing_status'])

    except Exception as e:
        error_msg = f"Error querying or processing showing status data: {e}"
        logger.exception(error_msg)
        st.error(error_msg)
        return pd.DataFrame(columns=['month_start_dt', 'showing_status', 'count'])

# --- Main Report Function ---
def main(start_date, end_date):
    """
    Main function for the Overall Showing Status Trends report.
    """
    st.info("Analyzes the trends of different showing statuses (Confirmed, Cancelled, etc.) across the portfolio over time.")
    logger.info(f"Running Overall Showing Status Trends report for {start_date} to {end_date}")

    engine = get_engine()
    if engine is None:
        logger.error("Failed to get DB engine. Aborting report.")
        return

    status_data = load_showing_status_data(engine, start_date, end_date)

    if status_data.empty:
        st.warning("No showing status data found for the selected date range, or an error occurred.")
        logger.warning("No showing status data loaded or processed.")
        return

    # --- Plotting ---
    st.subheader("Monthly Showing Counts by Status")
    logger.info("Generating showing status plot...")
    try:
        fig = px.line(status_data, x='month_start_dt', y='count', color='showing_status',
                      title="Monthly Showing Counts by Status (Portfolio-wide)",
                      labels={'month_start_dt': 'Month', 'count': 'Number of Showings', 'showing_status': 'Showing Status'},
                      markers=True)

        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Number of Showings",
            xaxis_tickformat="%b %Y",
            hovermode="x unified"
        )
        fig.update_yaxes(rangemode='tozero')
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        error_msg = f"Failed to generate plot: {e}"
        logger.exception(error_msg)
        st.error(error_msg)

    # --- Data Table ---
    st.subheader("Showing Status Data Table")
    try:
        # Pivot table for better readability
        pivot_df = status_data.pivot(index='month_start_dt', columns='showing_status', values='count').fillna(0).astype(int)
        pivot_df.index = pivot_df.index.strftime('%B %Y')
        pivot_df = pivot_df.reset_index().rename(columns={'month_start_dt': 'Month'})

        # Sort by Month (desc)
        try:
             pivot_df_sorted = pivot_df.sort_values(
                 by='Month',
                 ascending=False,
                 key=lambda s: pd.to_datetime(s, format='%B %Y', errors='coerce'),
                 na_position='last'
             )
        except Exception as sort_e:
             logger.warning(f"Could not sort status pivot table: {sort_e}. Displaying unsorted.")
             pivot_df_sorted = pivot_df

        st.dataframe(pivot_df_sorted, use_container_width=True, hide_index=True)
    except Exception as e:
        error_msg = f"Failed to display status data table: {e}"
        logger.exception(error_msg)
        st.error(error_msg)

    # --- Add plain English description ---
    st.markdown("""
    ---
    **How this report works:**
    *   This report tracks the total number of showings across all listings, categorized by their status (e.g., Confirmed, Cancelled).
    *   It uses the `showmojo_prospect_showing_data` table and groups the counts by month based on the `showtime`. Statuses like 'STATUS_CONFIRMED' are cleaned up for display (e.g., 'Confirmed').
    *   The line chart shows the monthly trend for each showing status.
    *   The table provides the detailed monthly counts for each status in a pivoted format for easy comparison.
    """)
    # --- End description ---

    logger.info("Report execution finished.")
