# reports/portfolio_no_show_rate.py

import streamlit as st
import pandas as pd
from sqlalchemy import text, exc as sqlalchemy_exc
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import os
import logging
import numpy as np # For NaN handling

# Import shared functions from utils
from web_version.utils import get_engine

# Configure logging for this report module
logger = logging.getLogger(__name__)

# --- Data Loading Function ---
@st.cache_data(ttl=600)
def load_no_show_rate_data(_engine, start_date, end_date):
    """
    Loads showing data and calculates the portfolio-wide no-show rate per month.
    No-show rate is calculated as (No-Shows / Confirmed Showings).

    Args:
        _engine: SQLAlchemy database engine instance.
        start_date: The start date of the analysis period (showtime).
        end_date: The end date of the analysis period (showtime).

    Returns:
        pandas.DataFrame: DataFrame with columns like
            ['month_start_dt', 'Confirmed Showings', 'No Shows', 'No-Show Rate (%)'],
            or an empty DataFrame on error/no data.
    """
    logger.info(f"Loading portfolio no-show rate data from {start_date} to {end_date}")
    end_date_adjusted = end_date + timedelta(days=1)

    # Query counts confirmed showings and no-shows per month
    query = text("""
        SELECT
            DATE_TRUNC('month', showtime)::date AS month_start_dt,
            -- Count confirmed showings (including those that became no-shows)
            SUM(CASE WHEN showing_status = 'STATUS_CONFIRMED' THEN 1 ELSE 0 END) AS "Confirmed Showings",
            -- Count no-shows (which should also be confirmed)
            SUM(CASE WHEN no_show = TRUE THEN 1 ELSE 0 END) AS "No Shows"
        FROM
            showmojo_prospect_showing_data
        WHERE
            showtime >= :start_date
            AND showtime < :end_date_adj -- Use adjusted end date
            AND showtime IS NOT NULL
            -- We only care about confirmed showings for the rate calculation base
            AND showing_status = 'STATUS_CONFIRMED'
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
        logger.info(f"Loaded {len(db_results)} monthly no-show count records from DB.")

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
             merged_df = merged_df.fillna(0)
             for col in ["Confirmed Showings", "No Shows"]:
                 merged_df[col] = merged_df[col].astype(int)
        else:
             merged_df = db_results
             if not merged_df.empty:
                 merged_df = merged_df.fillna(0)
                 for col in ["Confirmed Showings", "No Shows"]:
                     merged_df[col] = merged_df[col].astype(int)

        # Calculate No-Show Rate
        # Use pd.NA for division by zero cases to represent undefined rate
        merged_df['No-Show Rate (%)'] = (merged_df['No Shows'] / merged_df['Confirmed Showings'].replace(0, np.nan)) * 100

        logger.info(f"Processed no-show rate data, final shape: {merged_df.shape}")
        return merged_df.sort_values(by='month_start_dt')

    except Exception as e:
        error_msg = f"Error querying or processing no-show rate data: {e}"
        logger.exception(error_msg)
        st.error(error_msg)
        return pd.DataFrame(columns=['month_start_dt', 'Confirmed Showings', 'No Shows', 'No-Show Rate (%)'])

# --- Main Report Function ---
def main(start_date, end_date):
    """
    Main function for the Portfolio No-Show Rate report.
    """
    st.info("Analyzes the portfolio-wide no-show rate (No-Shows / Confirmed Showings) over time.")
    logger.info(f"Running Portfolio No-Show Rate report for {start_date} to {end_date}")

    engine = get_engine()
    if engine is None:
        logger.error("Failed to get DB engine. Aborting report.")
        return

    no_show_data = load_no_show_rate_data(engine, start_date, end_date)

    if no_show_data.empty:
        st.warning("No showing data found to calculate no-show rate for the selected date range, or an error occurred.")
        logger.warning("No no-show rate data loaded or processed.")
        return

    # --- Plotting ---
    st.subheader("Monthly Portfolio No-Show Rate")
    logger.info("Generating no-show rate plot...")
    try:
        fig = px.line(no_show_data, x='month_start_dt', y='No-Show Rate (%)',
                      title="Monthly No-Show Rate (Portfolio-wide)",
                      labels={'month_start_dt': 'Month', 'No-Show Rate (%)': 'No-Show Rate (%)'},
                      markers=True)

        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="No-Show Rate (%)",
            xaxis_tickformat="%b %Y",
            yaxis_ticksuffix="%",
            hovermode="x unified"
        )
        fig.update_yaxes(rangemode='tozero')
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        error_msg = f"Failed to generate plot: {e}"
        logger.exception(error_msg)
        st.error(error_msg)

    # --- Data Table ---
    st.subheader("No-Show Rate Data Table")
    try:
        display_df = no_show_data.copy()
        display_df['Month'] = display_df['month_start_dt'].dt.strftime('%B %Y')
        # Format rate, handle NaN
        display_df['No-Show Rate (%)'] = display_df['No-Show Rate (%)'].apply(lambda x: f'{x:.1f}%' if pd.notna(x) else 'N/A')
        display_df = display_df[['Month', 'Confirmed Showings', 'No Shows', 'No-Show Rate (%)']]

        # Sort by Month (desc)
        try:
             display_df_sorted = display_df.sort_values(
                 by='Month',
                 ascending=False,
                 key=lambda s: pd.to_datetime(s, format='%B %Y', errors='coerce'),
                 na_position='last'
             )
        except Exception as sort_e:
             logger.warning(f"Could not sort no-show rate table: {sort_e}. Displaying unsorted.")
             display_df_sorted = display_df

        st.dataframe(display_df_sorted, use_container_width=True, hide_index=True)
    except Exception as e:
        error_msg = f"Failed to display no-show rate data table: {e}"
        logger.exception(error_msg)
        st.error(error_msg)

    # --- Add plain English description ---
    st.markdown("""
    ---
    **How this report works:**
    *   This report calculates the monthly no-show rate for showings across the entire portfolio.
    *   It uses the `showmojo_prospect_showing_data` table, focusing on showings with `showing_status = 'STATUS_CONFIRMED'`.
    *   The **No-Show Rate** is calculated as: `(Total No Shows / Total Confirmed Showings) * 100` for each month based on the `showtime`.
    *   The line chart shows the trend of this no-show rate over time.
    *   The table provides the monthly counts of confirmed showings, no-shows, and the calculated no-show rate. 'N/A' indicates months with zero confirmed showings.
    """)
    # --- End description ---

    logger.info("Report execution finished.")
