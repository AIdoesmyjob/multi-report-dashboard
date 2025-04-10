# reports/total_showing_volume.py

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
def load_total_showing_volume_data(_engine, start_date, end_date):
    """
    Loads showing data and calculates the total showing volume per month for the entire portfolio.

    Args:
        _engine: SQLAlchemy database engine instance.
        start_date: The start date of the analysis period (showtime).
        end_date: The end date of the analysis period (showtime).

    Returns:
        pandas.DataFrame: DataFrame with columns like
            ['month_start_dt', 'Total Showings'],
            or an empty DataFrame on error/no data.
    """
    logger.info(f"Loading total showing volume data from {start_date} to {end_date}")
    end_date_adjusted = end_date + timedelta(days=1)

    # Query counts total showings per month
    query = text("""
        SELECT
            DATE_TRUNC('month', showtime)::date AS month_start_dt,
            COUNT(showing_id) AS "Total Showings"
        FROM
            showmojo_prospect_showing_data
        WHERE
            showtime >= :start_date
            AND showtime < :end_date_adj -- Use adjusted end date
            AND showtime IS NOT NULL
            -- Consider if any showing_status filter is needed? e.g., exclude cancelled?
            -- For now, counting all records with a showtime in the range.
        GROUP BY
            month_start_dt
        ORDER BY
            month_start_dt ASC;
    """)

    try:
        with _engine.connect() as conn:
            params = {
                "start_date": start_date,
                "end_date_adj": end_date_adjusted
            }
            db_results = pd.read_sql_query(query, conn, params=params, parse_dates=['month_start_dt'])
        logger.info(f"Loaded {len(db_results)} monthly total showing volume records from DB.")

        # --- Create complete monthly range ---
        start_date_dt = pd.to_datetime(start_date)
        end_date_dt = pd.to_datetime(end_date)
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
             merged_df['Total Showings'] = merged_df['Total Showings'].fillna(0).astype(int)
        else:
             merged_df = db_results
             if not merged_df.empty:
                 merged_df['Total Showings'] = merged_df['Total Showings'].fillna(0).astype(int)

        logger.info(f"Processed total showing volume data, final shape: {merged_df.shape}")
        return merged_df.sort_values(by='month_start_dt')

    except Exception as e:
        error_msg = f"Error querying or processing total showing volume data: {e}"
        logger.exception(error_msg)
        st.error(error_msg)
        return pd.DataFrame(columns=['month_start_dt', 'Total Showings'])

# --- Main Report Function ---
def main(start_date, end_date):
    """
    Main function for the Total Showing Volume report.
    """
    st.info("Analyzes the total number of showings scheduled across the portfolio over time.")
    logger.info(f"Running Total Showing Volume report for {start_date} to {end_date}")

    engine = get_engine()
    if engine is None:
        logger.error("Failed to get DB engine. Aborting report.")
        return

    volume_data = load_total_showing_volume_data(engine, start_date, end_date)

    if volume_data.empty:
        st.warning("No showing volume data found for the selected date range, or an error occurred.")
        logger.warning("No showing volume data loaded or processed.")
        return

    # --- Plotting ---
    st.subheader("Total Monthly Showing Volume")
    logger.info("Generating total showing volume plot...")
    try:
        fig = px.line(volume_data, x='month_start_dt', y='Total Showings',
                      title="Total Monthly Showing Volume (Portfolio-wide)",
                      labels={'month_start_dt': 'Month', 'Total Showings': 'Number of Showings'},
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
    st.subheader("Total Showing Volume Data Table")
    try:
        display_df = volume_data.copy()
        display_df['Month'] = display_df['month_start_dt'].dt.strftime('%B %Y')
        display_df = display_df[['Month', 'Total Showings']]

        # Sort by Month (desc)
        try:
             display_df_sorted = display_df.sort_values(
                 by='Month',
                 ascending=False,
                 key=lambda s: pd.to_datetime(s, format='%B %Y', errors='coerce'),
                 na_position='last'
             )
        except Exception as sort_e:
             logger.warning(f"Could not sort total showing volume table: {sort_e}. Displaying unsorted.")
             display_df_sorted = display_df

        st.dataframe(display_df_sorted, use_container_width=True, hide_index=True)
    except Exception as e:
        error_msg = f"Failed to display total showing volume data table: {e}"
        logger.exception(error_msg)
        st.error(error_msg)

    # --- Add plain English description ---
    st.markdown("""
    ---
    **How this report works:**
    *   This report tracks the total number of showings scheduled across all listings in the portfolio each month.
    *   It uses the `showmojo_prospect_showing_data` table and counts all showing records based on their `showtime`.
    *   *Note: This currently includes showings of all statuses (e.g., Confirmed, Cancelled). It represents the total volume scheduled.*
    *   The line chart shows the trend of total monthly showing volume over time.
    *   The table provides the detailed monthly total counts.
    """)
    # --- End description ---

    logger.info("Report execution finished.")
