# reports/showing_performance.py

import streamlit as st
import pandas as pd
from sqlalchemy import text, exc as sqlalchemy_exc
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px # Added for easier bar charts
import os
import logging
import numpy as np

# Import shared functions from utils
from web_version.utils import get_engine

# Configure logging for this report module
logger = logging.getLogger(__name__)

# --- Data Loading Function ---
@st.cache_data(ttl=600)
def load_showing_performance_data(_engine, start_date, end_date):
    """
    Calculates showing performance metrics (prospects, scheduled, no-shows, cancellations)
    grouped by the month the prospect was created. Also aggregates showing method counts.

    Args:
        _engine: SQLAlchemy database engine instance.
        start_date: The start date of the analysis period (for prospect created_at).
        end_date: The end date of the analysis period (for prospect created_at).

    Returns:
        tuple: (pandas.DataFrame for monthly trends, pandas.DataFrame for method counts)
               or (empty DataFrame, empty DataFrame) on error/no data.
    """
    logger.info(f"Loading showing performance data from {start_date} to {end_date}")
    # Adjust end_date to be exclusive for the query
    end_date_adjusted = end_date + timedelta(days=1)

    # Query for monthly trends
    query_monthly = text("""
        WITH MonthlyMetrics AS (
            SELECT
                DATE_TRUNC('month', created_at)::date AS month_start_dt,
                COUNT(prospect_id) AS total_prospects,
                SUM(CASE WHEN showing_was_scheduled = TRUE THEN 1 ELSE 0 END) AS scheduled_showings,
                SUM(CASE WHEN no_show = TRUE THEN 1 ELSE 0 END) AS no_shows,
                COUNT(DISTINCT CASE WHEN canceled_by IS NOT NULL THEN prospect_id ELSE NULL END) AS cancellations
                -- Note: Cancellation count might need refinement depending on how 'canceled_by' is used.
                -- This counts prospects who had at least one cancellation event associated.
            FROM
                showmojo_detailed_prospect_data
            WHERE
                created_at >= :start_date
                AND created_at < :end_date_adj -- Use adjusted end date
                AND created_at IS NOT NULL
            GROUP BY
                month_start_dt
        )
        SELECT * FROM MonthlyMetrics ORDER BY month_start_dt ASC;
    """)

    # Query for showing method counts (overall for the period)
    query_methods = text("""
        SELECT
            showing_method,
            COUNT(prospect_id) AS count
        FROM
            showmojo_detailed_prospect_data
        WHERE
            created_at >= :start_date
            AND created_at < :end_date_adj
            AND created_at IS NOT NULL
            AND showing_was_scheduled = TRUE -- Only count methods for scheduled showings
            AND showing_method IS NOT NULL
        GROUP BY
            showing_method
        ORDER BY
            count DESC;
    """)

    try:
        with _engine.connect() as conn:
            params = {
                "start_date": start_date,
                "end_date_adj": end_date_adjusted
            }
            # Load monthly data
            db_results_monthly = pd.read_sql_query(query_monthly, conn, params=params, parse_dates=['month_start_dt'])
            logger.info(f"Loaded {len(db_results_monthly)} monthly showing performance records from DB.")

            # Load method data
            db_results_methods = pd.read_sql_query(query_methods, conn, params=params)
            logger.info(f"Loaded {len(db_results_methods)} showing method counts from DB.")

        # --- Process Monthly Data ---
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
             merged_df_monthly = pd.merge(df_months, db_results_monthly, on='month_start_dt', how='left')
             # Fill NaN counts with 0 after merge
             merged_df_monthly[['total_prospects', 'scheduled_showings', 'no_shows', 'cancellations']] = merged_df_monthly[['total_prospects', 'scheduled_showings', 'no_shows', 'cancellations']].fillna(0).astype(int)
        else:
             merged_df_monthly = db_results_monthly # Use only db results if month range is empty

        # Ensure month_start_dt is datetime before formatting
        merged_df_monthly['month_start_dt'] = pd.to_datetime(merged_df_monthly['month_start_dt'])
        merged_df_monthly['month_start'] = merged_df_monthly['month_start_dt'].dt.strftime('%Y-%m-%d')

        # Select and order final columns
        final_df_monthly = merged_df_monthly[['month_start', 'month_start_dt', 'total_prospects', 'scheduled_showings', 'no_shows', 'cancellations']]
        logger.info(f"Processed monthly showing performance data, shape: {final_df_monthly.shape}")

        # --- Process Method Data ---
        # No extra processing needed for methods, just return the query result
        final_df_methods = db_results_methods
        logger.info(f"Processed showing method counts, shape: {final_df_methods.shape}")

        return final_df_monthly, final_df_methods

    except Exception as e:
        error_msg = f"Error querying or processing showing performance data: {e}"
        logger.exception(error_msg)
        st.error(error_msg)
        return pd.DataFrame(columns=['month_start', 'month_start_dt', 'total_prospects', 'scheduled_showings', 'no_shows', 'cancellations']), pd.DataFrame(columns=['showing_method', 'count'])

# --- Main Report Function ---
def main(start_date, end_date):
    """
    Main function for the Showing Performance report.
    """
    st.info("Analyzes prospect showing funnel metrics based on prospect creation date.")
    logger.info(f"Running Showing Performance report for {start_date} to {end_date}")

    engine = get_engine()
    if engine is None:
        logger.error("Failed to get DB engine. Aborting report.")
        return

    monthly_df, methods_df = load_showing_performance_data(engine, start_date, end_date)

    # Check if DataFrame is completely empty
    if monthly_df.empty:
        st.warning("No monthly showing performance data found for the selected date range, or an error occurred.")
        logger.warning("No monthly showing performance data loaded or processed.")
        # Don't return yet, method data might exist
    else:
        # --- Plotting Monthly Trends ---
        st.subheader("Monthly Showing Funnel Trends")
        logger.info("Generating monthly plot...")
        try:
            fig_monthly = go.Figure()

            fig_monthly.add_trace(go.Scatter(
                x=monthly_df['month_start_dt'], y=monthly_df['total_prospects'],
                mode='lines+markers', name='Total Prospects', line=dict(color='blue'),
                hovertemplate='<b>%{x|%b %Y}</b><br>Total Prospects: %{y}<extra></extra>'
            ))
            fig_monthly.add_trace(go.Scatter(
                x=monthly_df['month_start_dt'], y=monthly_df['scheduled_showings'],
                mode='lines+markers', name='Scheduled Showings', line=dict(color='green'),
                hovertemplate='<b>%{x|%b %Y}</b><br>Scheduled: %{y}<extra></extra>'
            ))
            fig_monthly.add_trace(go.Scatter(
                x=monthly_df['month_start_dt'], y=monthly_df['no_shows'],
                mode='lines+markers', name='No Shows', line=dict(color='red'),
                hovertemplate='<b>%{x|%b %Y}</b><br>No Shows: %{y}<extra></extra>'
            ))
            fig_monthly.add_trace(go.Scatter(
                x=monthly_df['month_start_dt'], y=monthly_df['cancellations'],
                mode='lines+markers', name='Cancellations', line=dict(color='orange'),
                hovertemplate='<b>%{x|%b %Y}</b><br>Cancellations: %{y}<extra></extra>'
            ))

            fig_monthly.update_layout(
                title="Prospect Showing Funnel Over Time",
                xaxis_title="Prospect Creation Month",
                yaxis_title="Count",
                xaxis_tickformat="%b %Y",
                hovermode="x unified"
            )
            fig_monthly.update_yaxes(rangemode='tozero')
            st.plotly_chart(fig_monthly, use_container_width=True)

        except Exception as e:
            error_msg = f"Failed to generate monthly plot: {e}"
            logger.exception(error_msg)
            st.error(error_msg)

        # --- Data Table Monthly ---
        st.subheader("Monthly Data Table")
        try:
            display_df_monthly = monthly_df[['month_start_dt', 'total_prospects', 'scheduled_showings', 'no_shows', 'cancellations']].copy()
            display_df_monthly['Month'] = display_df_monthly['month_start_dt'].dt.strftime('%B %Y')
            display_df_monthly = display_df_monthly.rename(columns={
                'total_prospects': 'Total Prospects',
                'scheduled_showings': 'Scheduled Showings',
                'no_shows': 'No Shows',
                'cancellations': 'Cancellations'
            })
            display_df_monthly = display_df_monthly[['Month', 'Total Prospects', 'Scheduled Showings', 'No Shows', 'Cancellations']]

            # Sort by Month
            try:
                 display_df_monthly_sorted = display_df_monthly.sort_values(
                     by='Month',
                     ascending=False,
                     key=lambda s: pd.to_datetime(s, format='%B %Y', errors='coerce'),
                     na_position='last'
                 )
            except Exception as sort_e:
                 logger.warning(f"Could not sort monthly dataframe by month string: {sort_e}. Displaying unsorted.")
                 display_df_monthly_sorted = display_df_monthly

            st.dataframe(display_df_monthly_sorted, use_container_width=True, hide_index=True)
        except Exception as e:
            error_msg = f"Failed to display monthly data table: {e}"
            logger.exception(error_msg)
            st.error(error_msg)

    # --- Plotting Showing Methods ---
    if methods_df.empty:
        st.warning("No showing method data found for the selected date range.")
        logger.warning("No showing method data loaded or processed.")
    else:
        st.subheader("Showing Methods Used (Overall)")
        logger.info("Generating showing methods plot...")
        try:
            fig_methods = px.bar(methods_df, x='showing_method', y='count',
                                 title=f"Showing Methods Used ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})",
                                 labels={'showing_method': 'Showing Method', 'count': 'Number of Showings'},
                                 text_auto=True) # Show counts on bars
            fig_methods.update_layout(xaxis_title="Showing Method", yaxis_title="Number of Showings")
            st.plotly_chart(fig_methods, use_container_width=True)

            # --- Data Table Methods ---
            st.subheader("Showing Methods Data Table")
            st.dataframe(methods_df.rename(columns={'showing_method': 'Showing Method', 'count': 'Count'}),
                         use_container_width=True, hide_index=True)

        except Exception as e:
            error_msg = f"Failed to generate/display showing methods plot/table: {e}"
            logger.exception(error_msg)
            st.error(error_msg)

    # --- Add plain English description ---
    st.markdown("""
    ---
    **How this report works:**
    *   This report analyzes the prospect showing funnel using data from the `showmojo_detailed_prospect_data` table.
    *   It counts the total number of prospects created each month within the selected date range.
    *   It then tracks how many of those prospects had a showing scheduled (`showing_was_scheduled = TRUE`), how many were marked as no-shows (`no_show = TRUE`), and how many had a cancellation recorded (`canceled_by` is not NULL).
    *   The line chart displays these counts over time.
    *   The bar chart shows the total count for each `showing_method` used during the selected period.
    *   Tables provide the underlying monthly and showing method data.
    """)
    # --- End description ---

    logger.info("Report execution finished.")
