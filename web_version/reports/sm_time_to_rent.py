# reports/time_to_rent.py

import streamlit as st
import pandas as pd
from sqlalchemy import text, exc as sqlalchemy_exc
from datetime import datetime, timedelta
import plotly.graph_objects as go
import os
import logging
import numpy as np

# Import shared functions from utils
from web_version.utils import get_engine

# Configure logging for this report module
logger = logging.getLogger(__name__)

# --- Data Loading Function ---
@st.cache_data(ttl=600)
def load_time_to_rent_data(_engine, start_date, end_date):
    """
    Calculates the average time (in days) a listing was on the market,
    grouped by the month it was last on the market.

    Args:
        _engine: SQLAlchemy database engine instance.
        start_date: The start date of the analysis period (for last_on_market).
        end_date: The end date of the analysis period (for last_on_market).

    Returns:
        pandas.DataFrame: DataFrame with columns
            ['month_start', 'avg_days_on_market', 'month_start_dt'],
            or an empty DataFrame on error/no data.
    """
    logger.info(f"Loading average time to rent data from {start_date} to {end_date}")
    # Adjust end_date to be exclusive for the query
    end_date_adjusted = end_date + timedelta(days=1)

    query = text("""
        SELECT
            DATE_TRUNC('month', last_on_market)::date AS month_start_dt,
            AVG(days_on_market) AS avg_days_on_market
        FROM
            showmojo_detailed_listing_data
        WHERE
            last_on_market >= :start_date
            AND last_on_market < :end_date_adj -- Use adjusted end date for exclusive range
            AND days_on_market IS NOT NULL -- Ensure we have data to average
            AND last_on_market IS NOT NULL -- Ensure the date is valid for grouping
            -- Consider adding a filter for market_status if needed, e.g., AND market_status = 'Rented'
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
        logger.info(f"Loaded {len(db_results)} monthly average days on market from DB.")

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
        else:
             merged_df = db_results # Use only db results if month range is empty

        # Ensure month_start_dt is datetime before formatting
        merged_df['month_start_dt'] = pd.to_datetime(merged_df['month_start_dt'])
        merged_df['month_start'] = merged_df['month_start_dt'].dt.strftime('%Y-%m-%d')

        # Select and order final columns
        final_df = merged_df[['month_start', 'avg_days_on_market', 'month_start_dt']]
        logger.info(f"Processed average time to rent data, shape: {final_df.shape}")
        return final_df

    except Exception as e:
        error_msg = f"Error querying or processing average time to rent data: {e}"
        logger.exception(error_msg)
        st.error(error_msg)
        return pd.DataFrame(columns=['month_start', 'avg_days_on_market', 'month_start_dt'])

# --- Main Report Function ---
def main(start_date, end_date):
    """
    Main function for the Average Time to Rent report.
    """
    st.info("Shows the average number of days listings were on the market, grouped by the month they were last listed.")
    logger.info(f"Running Average Time to Rent report for {start_date} to {end_date}")

    engine = get_engine()
    if engine is None:
        logger.error("Failed to get DB engine. Aborting report.")
        return

    time_to_rent_df = load_time_to_rent_data(engine, start_date, end_date)

    # Check if DataFrame is completely empty or only contains NaN averages
    if time_to_rent_df.empty or time_to_rent_df['avg_days_on_market'].isnull().all():
        st.warning("No data found for average time to rent in the selected date range, or an error occurred.")
        logger.warning("No average time to rent data loaded or processed.")
        return

    # --- Plotting ---
    logger.info("Generating plot...")
    try:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=time_to_rent_df['month_start_dt'],
            y=time_to_rent_df['avg_days_on_market'],
            mode='lines+markers',
            name='Avg Days on Market',
            line=dict(color='teal'), # Changed color
            fill='tozeroy',
            fillcolor='rgba(0, 128, 128, 0.2)', # Semi-transparent teal
            hovertemplate='<b>%{x|%b %Y}</b><br>Avg Days: %{y:.1f}<extra></extra>'
        ))

        fig.update_layout(
            title="Average Days on Market by Month (Last On Market)",
            xaxis_title="Month Last On Market",
            yaxis_title="Average Days on Market",
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
    st.subheader("Data Table")
    try:
        display_df = time_to_rent_df[['month_start_dt', 'avg_days_on_market']].copy()
        # Ensure month_start_dt is datetime before formatting
        display_df['month_start_dt'] = pd.to_datetime(display_df['month_start_dt'])
        display_df['Month'] = display_df['month_start_dt'].dt.strftime('%B %Y')
        # Format average days to 1 decimal place, handle potential NaN
        display_df['Avg Days on Market'] = display_df['avg_days_on_market'].apply(lambda x: f'{x:.1f}' if pd.notna(x) else 'N/A')
        display_df = display_df[['Month', 'Avg Days on Market']] # Select and order columns

        # Sort by Month
        try:
             display_df_sorted = display_df.sort_values(
                 by='Month',
                 ascending=False,
                 key=lambda s: pd.to_datetime(s, format='%B %Y', errors='coerce'),
                 na_position='last'
             )
        except Exception as sort_e:
             logger.warning(f"Could not sort dataframe by month string: {sort_e}. Displaying unsorted.")
             display_df_sorted = display_df

        st.dataframe(display_df_sorted, use_container_width=True, hide_index=True)
    except Exception as e:
        error_msg = f"Failed to display data table: {e}"
        logger.exception(error_msg)
        st.error(error_msg)

    # --- Add plain English description ---
    st.markdown("""
    ---
    **How this report works:**
    *   This report calculates the average number of days listings spent on the market before being rented or taken off the market.
    *   It uses the `showmojo_detailed_listing_data` table.
    *   The average is calculated for each month based on the `last_on_market` date of the listings.
    *   The line chart shows the trend of this average over time.
    *   The table provides the monthly average values.
    """)
    # --- End description ---

    logger.info("Report execution finished.")
