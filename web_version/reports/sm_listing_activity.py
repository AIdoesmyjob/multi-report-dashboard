# reports/listing_activity.py

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
def load_listing_activity_data(_engine, start_date, end_date):
    """
    Loads total listing view and showing creation events grouped by month.

    Args:
        _engine: SQLAlchemy database engine instance.
        start_date: The start date of the analysis period (event_timestamp).
        end_date: The end date of the analysis period (event_timestamp).

    Returns:
        pandas.DataFrame: DataFrame with columns like
            ['month_start_dt', 'Total Listing Views', 'Total Showings Created'],
            or an empty DataFrame on error/no data.
    """
    logger.info(f"Loading total listing activity data from {start_date} to {end_date}")
    end_date_adjusted = end_date + timedelta(days=1)

    # Query counts total views and showings per month
    query = text("""
        SELECT
            DATE_TRUNC('month', event_timestamp)::date AS month_start_dt,
            SUM(CASE WHEN entry_type = 'LISTING_VIEW' THEN 1 ELSE 0 END) AS "Total Listing Views",
            SUM(CASE WHEN entry_type = 'SHOWING_CREATE' THEN 1 ELSE 0 END) AS "Total Showings Created"
        FROM
            showmojo_listing_showing_metrics
        WHERE
            event_timestamp >= :start_date
            AND event_timestamp < :end_date_adj -- Use adjusted end date
            AND event_timestamp IS NOT NULL
            AND entry_type IN ('LISTING_VIEW', 'SHOWING_CREATE')
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
        logger.info(f"Loaded {len(db_results)} total monthly listing activity records from DB.")

        # --- Create complete monthly range ---
        start_date_dt = pd.to_datetime(start_date)
        end_date_dt = pd.to_datetime(end_date)
        start_month = start_date_dt.replace(day=1)
        end_month = end_date_dt.replace(day=1)

        if start_month > end_month:
            logger.warning("Start date is after end date, resulting in empty month range.")
            all_months = pd.DatetimeIndex([])
            all_listings = []
        else:
            all_months = pd.date_range(start=start_month, end=end_month, freq='MS')

        df_months = pd.DataFrame({'month_start_dt': all_months})

        # Merge results with full month range
        if not df_months.empty:
             merged_df = pd.merge(df_months, db_results, on='month_start_dt', how='left')
             # Fill NaN counts with 0 after merge
             merged_df = merged_df.fillna(0)
             # Ensure integer types
             for col in ["Total Listing Views", "Total Showings Created"]:
                 merged_df[col] = merged_df[col].astype(int)
        else:
             merged_df = db_results # Use only db results if month range is empty
             # Ensure types even if only db_results exist
             if not merged_df.empty:
                 merged_df = merged_df.fillna(0)
                 for col in ["Total Listing Views", "Total Showings Created"]:
                     merged_df[col] = merged_df[col].astype(int)

        logger.info(f"Processed total listing activity data, final shape: {merged_df.shape}")
        return merged_df.sort_values(by='month_start_dt')

    except Exception as e:
        error_msg = f"Error querying or processing total listing activity data: {e}"
        logger.exception(error_msg)
        st.error(error_msg)
        return pd.DataFrame(columns=['month_start_dt', 'Total Listing Views', 'Total Showings Created'])

# --- Main Report Function ---
def main(start_date, end_date):
    """
    Main function for the Total Listing Activity report.
    """
    st.info("Analyzes total listing views and showings created across all listings over time.")
    logger.info(f"Running Total Listing Activity report for {start_date} to {end_date}")

    engine = get_engine()
    if engine is None:
        logger.error("Failed to get DB engine. Aborting report.")
        return

    activity_data = load_listing_activity_data(engine, start_date, end_date)

    if activity_data.empty:
        st.warning("No listing activity data found for the selected date range, or an error occurred.")
        logger.warning("No listing activity data loaded or processed.")
        return

    # --- Plotting ---
    st.subheader("Total Monthly Listing Views vs. Showings Created")
    logger.info("Generating total listing activity plot...")
    try:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=activity_data['month_start_dt'], y=activity_data['Total Listing Views'],
            mode='lines+markers', name='Total Listing Views', line=dict(color='blue'),
            hovertemplate='<b>%{x|%b %Y}</b><br>Total Views: %{y}<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            x=activity_data['month_start_dt'], y=activity_data['Total Showings Created'],
            mode='lines+markers', name='Total Showings Created', line=dict(color='green'),
            hovertemplate='<b>%{x|%b %Y}</b><br>Total Showings Created: %{y}<extra></extra>'
        ))

        fig.update_layout(
            title="Total Monthly Listing Activity (Views vs. Showings Created)",
            xaxis_title="Month",
            yaxis_title="Count",
            xaxis_tickformat="%b %Y",
            hovermode="x unified"
        )

        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Count",
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
    st.subheader("Total Listing Activity Data Table")
    try:
        display_df = activity_data.copy()
        display_df['Month'] = display_df['month_start_dt'].dt.strftime('%B %Y')
        # Rename columns if needed (already done in query)
        display_df = display_df[['Month', 'Total Listing Views', 'Total Showings Created']]

        # Sort by Month (desc)
        try:
             display_df_sorted = display_df.sort_values(
                 by='Month',
                 ascending=False,
                 key=lambda s: pd.to_datetime(s, format='%B %Y', errors='coerce'),
                 na_position='last'
             )
        except Exception as sort_e:
             logger.warning(f"Could not sort total listing activity data table: {sort_e}. Displaying unsorted.")
             display_df_sorted = display_df

        # Filter out rows where both counts are zero for cleaner table display
        st.dataframe(display_df_sorted[(display_df_sorted['Total Listing Views'] > 0) | (display_df_sorted['Total Showings Created'] > 0)],
                     use_container_width=True, hide_index=True)
    except Exception as e:
        error_msg = f"Failed to display total listing activity data table: {e}"
        logger.exception(error_msg)
        st.error(error_msg)

    # --- Add plain English description ---
    st.markdown("""
    ---
    **How this report works:**
    *   This report tracks the **total** number of listing views and showings created across **all** listings using data from the `showmojo_listing_showing_metrics` table.
    *   It counts the total number of `LISTING_VIEW` and `SHOWING_CREATE` events per month, based on the `event_timestamp`.
    *   The line chart shows the overall monthly trend for 'Total Listing Views' and 'Total Showings Created'.
    *   The table provides the detailed monthly total counts for views and showings created. Rows with zero activity are hidden in the table.
    """)
    # --- End description ---

    logger.info("Report execution finished.")
