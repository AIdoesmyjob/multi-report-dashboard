# reports/inquiry_channel_analysis.py

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
def load_inquiry_data(_engine, start_date, end_date):
    """
    Loads high-level inquiry metrics data grouped by month.

    Args:
        _engine: SQLAlchemy database engine instance.
        start_date: The start date of the analysis period (report_end_date).
        end_date: The end date of the analysis period (report_end_date).

    Returns:
        pandas.DataFrame: DataFrame with columns like
            ['month_start_dt', 'Phone Calls', 'Text Inquiries', 'Email Inquiries', 'Web Inquiries'],
            or an empty DataFrame on error/no data.
    """
    logger.info(f"Loading inquiry channel data from {start_date} to {end_date}")
    # Adjust end_date to be inclusive for the query if needed, or use < end_date + 1 day
    # Assuming report_end_date represents the day the metrics are for.
    end_date_adjusted = end_date + timedelta(days=1)

    # Query aggregates daily data into monthly sums
    query = text("""
        SELECT
            DATE_TRUNC('month', report_end_date)::date AS month_start_dt,
            SUM(total_incoming_phone_calls) AS "Phone Calls",
            SUM(total_incoming_text_inquiries) AS "Text Inquiries",
            SUM(total_incoming_email_inquiries) AS "Email Inquiries",
            SUM(total_inquiries_from_the_web) AS "Web Inquiries"
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
        logger.info(f"Loaded {len(db_results)} monthly inquiry channel records from DB.")

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
             # Fill NaN counts with 0 after merge
             merged_df = merged_df.fillna(0)
             # Ensure integer types where appropriate
             for col in ["Phone Calls", "Text Inquiries", "Email Inquiries", "Web Inquiries"]:
                 merged_df[col] = merged_df[col].astype(int)
        else:
             merged_df = db_results # Use only db results if month range is empty
             # Ensure types even if only db_results exist
             if not merged_df.empty:
                 merged_df = merged_df.fillna(0)
                 for col in ["Phone Calls", "Text Inquiries", "Email Inquiries", "Web Inquiries"]:
                     merged_df[col] = merged_df[col].astype(int)


        logger.info(f"Processed inquiry channel data, final shape: {merged_df.shape}")
        return merged_df

    except Exception as e:
        error_msg = f"Error querying or processing inquiry channel data: {e}"
        logger.exception(error_msg)
        st.error(error_msg)
        return pd.DataFrame(columns=['month_start_dt', 'Phone Calls', 'Text Inquiries', 'Email Inquiries', 'Web Inquiries'])

# --- Main Report Function ---
def main(start_date, end_date):
    """
    Main function for the Inquiry Channel Analysis report.
    """
    st.info("Analyzes the volume of incoming inquiries by channel (Phone, Text, Email, Web) over time.")
    logger.info(f"Running Inquiry Channel Analysis report for {start_date} to {end_date}")

    engine = get_engine()
    if engine is None:
        logger.error("Failed to get DB engine. Aborting report.")
        return

    inquiry_data = load_inquiry_data(engine, start_date, end_date)

    if inquiry_data.empty:
        st.warning("No inquiry data found for the selected date range, or an error occurred.")
        logger.warning("No inquiry data loaded or processed.")
        return

    # --- Plotting ---
    st.subheader("Monthly Inquiry Volume by Channel")
    logger.info("Generating inquiry channel plot...")
    try:
        # Melt dataframe for easier plotting with Plotly Express
        melted_df = inquiry_data.melt(id_vars=['month_start_dt'],
                                      value_vars=['Phone Calls', 'Text Inquiries', 'Email Inquiries', 'Web Inquiries'],
                                      var_name='Channel', value_name='Inquiry Count')

        fig = px.line(melted_df, x='month_start_dt', y='Inquiry Count', color='Channel',
                      title="Monthly Inquiry Volume by Channel",
                      labels={'month_start_dt': 'Month', 'Inquiry Count': 'Number of Inquiries', 'Channel': 'Inquiry Channel'},
                      markers=True)

        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Number of Inquiries",
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
    st.subheader("Monthly Inquiry Data Table")
    try:
        display_df = inquiry_data.copy()
        display_df['Month'] = display_df['month_start_dt'].dt.strftime('%B %Y')
        # Rename columns for display if needed, already done in query
        display_df = display_df[['Month', 'Phone Calls', 'Text Inquiries', 'Email Inquiries', 'Web Inquiries']]

        # Sort by Month (desc)
        try:
             display_df_sorted = display_df.sort_values(
                 by='Month',
                 ascending=False,
                 key=lambda s: pd.to_datetime(s, format='%B %Y', errors='coerce'),
                 na_position='last'
             )
        except Exception as sort_e:
             logger.warning(f"Could not sort inquiry data table: {sort_e}. Displaying unsorted.")
             display_df_sorted = display_df

        st.dataframe(display_df_sorted, use_container_width=True, hide_index=True)
    except Exception as e:
        error_msg = f"Failed to display inquiry data table: {e}"
        logger.exception(error_msg)
        st.error(error_msg)

    # --- Add plain English description ---
    st.markdown("""
    ---
    **How this report works:**
    *   This report shows the total number of incoming inquiries received each month, broken down by channel (Phone, Text, Email, Web).
    *   It uses the `showmojo_high_level_metrics` table, summing up the daily counts for each channel within each month.
    *   The line chart displays the trends for each channel over the selected date range.
    *   The table provides the detailed monthly counts for each inquiry channel.
    """)
    # --- End description ---

    logger.info("Report execution finished.")
