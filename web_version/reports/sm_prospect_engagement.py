# reports/prospect_engagement.py

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
def load_prospect_engagement_data(_engine, start_date, end_date):
    """
    Loads high-level prospect engagement metrics data grouped by month.

    Args:
        _engine: SQLAlchemy database engine instance.
        start_date: The start date of the analysis period (report_end_date).
        end_date: The end date of the analysis period (report_end_date).

    Returns:
        pandas.DataFrame: DataFrame with engagement metrics columns,
            or an empty DataFrame on error/no data.
    """
    logger.info(f"Loading prospect engagement data from {start_date} to {end_date}")
    end_date_adjusted = end_date + timedelta(days=1)

    # Query aggregates daily data into monthly sums
    query = text("""
        SELECT
            DATE_TRUNC('month', report_end_date)::date AS month_start_dt,
            SUM(total_feedback_surveys_completed) AS "Feedback Surveys Completed",
            SUM(total_prospects_notified_new_listings) AS "Notified (New Listings)",
            SUM(total_prospects_notified_rent_reductions) AS "Notified (Rent Reductions)"
            -- Add other relevant engagement metrics if needed
        FROM
            showmojo_high_level_metrics
        WHERE
            report_end_date >= :start_date
            AND report_end_date < :end_date_adj -- Use adjusted end date
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
        logger.info(f"Loaded {len(db_results)} monthly prospect engagement records from DB.")

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
             for col in merged_df.columns:
                 if col != 'month_start_dt':
                     merged_df[col] = merged_df[col].astype(int)
        else:
             merged_df = db_results
             if not merged_df.empty:
                 merged_df = merged_df.fillna(0)
                 for col in merged_df.columns:
                     if col != 'month_start_dt':
                         merged_df[col] = merged_df[col].astype(int)

        logger.info(f"Processed prospect engagement data, final shape: {merged_df.shape}")
        return merged_df

    except Exception as e:
        error_msg = f"Error querying or processing prospect engagement data: {e}"
        logger.exception(error_msg)
        st.error(error_msg)
        return pd.DataFrame(columns=['month_start_dt', 'Feedback Surveys Completed', 'Notified (New Listings)', 'Notified (Rent Reductions)'])

# --- Main Report Function ---
def main(start_date, end_date):
    """
    Main function for the Prospect Engagement report.
    """
    st.info("Displays metrics related to prospect engagement like notifications and feedback.")
    logger.info(f"Running Prospect Engagement report for {start_date} to {end_date}")

    engine = get_engine()
    if engine is None:
        logger.error("Failed to get DB engine. Aborting report.")
        return

    engagement_data = load_prospect_engagement_data(engine, start_date, end_date)

    if engagement_data.empty:
        st.warning("No prospect engagement data found for the selected date range, or an error occurred.")
        logger.warning("No prospect engagement data loaded or processed.")
        return

    # --- Plotting ---
    st.subheader("Monthly Prospect Engagement Metrics")
    logger.info("Generating prospect engagement plot...")
    try:
        # Melt dataframe for easier plotting
        metrics_to_plot = [col for col in engagement_data.columns if col != 'month_start_dt']
        melted_df = engagement_data.melt(id_vars=['month_start_dt'],
                                         value_vars=metrics_to_plot,
                                         var_name='Metric', value_name='Count')

        fig = px.line(melted_df, x='month_start_dt', y='Count', color='Metric',
                      title="Monthly Prospect Engagement Metrics",
                      labels={'month_start_dt': 'Month', 'Count': 'Count', 'Metric': 'Engagement Metric'},
                      markers=True)

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
    st.subheader("Monthly Engagement Metrics Data Table")
    try:
        display_df = engagement_data.copy()
        display_df['Month'] = display_df['month_start_dt'].dt.strftime('%B %Y')
        # Rename columns if needed (already done in query)
        display_cols = ['Month'] + metrics_to_plot
        display_df = display_df[display_cols]

        # Sort by Month (desc)
        try:
             display_df_sorted = display_df.sort_values(
                 by='Month',
                 ascending=False,
                 key=lambda s: pd.to_datetime(s, format='%B %Y', errors='coerce'),
                 na_position='last'
             )
        except Exception as sort_e:
             logger.warning(f"Could not sort engagement data table: {sort_e}. Displaying unsorted.")
             display_df_sorted = display_df

        st.dataframe(display_df_sorted, use_container_width=True, hide_index=True)
    except Exception as e:
        error_msg = f"Failed to display engagement data table: {e}"
        logger.exception(error_msg)
        st.error(error_msg)

    # --- Add plain English description ---
    st.markdown("""
    ---
    **How this report works:**
    *   This report shows metrics related to prospect engagement using data from the `showmojo_high_level_metrics` table.
    *   It sums the daily counts for each metric within each month for the selected date range.
    *   **Feedback Surveys Completed:** Total number of feedback surveys completed by prospects.
    *   **Notified (New Listings):** Total number of prospects notified about new listings matching their criteria.
    *   **Notified (Rent Reductions):** Total number of prospects notified about rent reductions on listings they might be interested in.
    *   The line chart displays the trends of these engagement metrics over time.
    *   The table provides the detailed monthly counts for each metric.
    """)
    # --- End description ---

    logger.info("Report execution finished.")
