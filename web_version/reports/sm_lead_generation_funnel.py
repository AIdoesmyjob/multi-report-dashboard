# reports/lead_generation_funnel.py

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
def load_lead_funnel_data(_engine, start_date, end_date):
    """
    Loads high-level metrics data for the lead generation funnel, grouped by month.

    Args:
        _engine: SQLAlchemy database engine instance.
        start_date: The start date of the analysis period (report_end_date).
        end_date: The end date of the analysis period (report_end_date).

    Returns:
        pandas.DataFrame: DataFrame with columns like
            ['month_start_dt', 'Total Inquiries', 'New Leads', 'New Showings'],
            or an empty DataFrame on error/no data.
    """
    logger.info(f"Loading lead generation funnel data from {start_date} to {end_date}")
    end_date_adjusted = end_date + timedelta(days=1)

    # Query aggregates daily data into monthly sums
    query = text("""
        SELECT
            DATE_TRUNC('month', report_end_date)::date AS month_start_dt,
            -- Approximate total inquiries by summing channel inquiries
            SUM(COALESCE(total_incoming_phone_calls, 0) +
                COALESCE(total_incoming_text_inquiries, 0) +
                COALESCE(total_incoming_email_inquiries, 0) +
                COALESCE(total_inquiries_from_the_web, 0)) AS "Total Inquiries",
            SUM(total_new_leads) AS "New Leads",
            SUM(total_new_showings) AS "New Showings"
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
        logger.info(f"Loaded {len(db_results)} monthly lead funnel records from DB.")

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
             for col in ["Total Inquiries", "New Leads", "New Showings"]:
                 merged_df[col] = merged_df[col].astype(int)
        else:
             merged_df = db_results
             if not merged_df.empty:
                 merged_df = merged_df.fillna(0)
                 for col in ["Total Inquiries", "New Leads", "New Showings"]:
                     merged_df[col] = merged_df[col].astype(int)

        # Calculate Conversion Rates (optional, can add complexity)
        # merged_df['Inquiry_to_Lead_Rate'] = (merged_df['New Leads'] / merged_df['Total Inquiries'].replace(0, pd.NA)) * 100
        # merged_df['Lead_to_Showing_Rate'] = (merged_df['New Showings'] / merged_df['New Leads'].replace(0, pd.NA)) * 100

        logger.info(f"Processed lead funnel data, final shape: {merged_df.shape}")
        return merged_df

    except Exception as e:
        error_msg = f"Error querying or processing lead funnel data: {e}"
        logger.exception(error_msg)
        st.error(error_msg)
        return pd.DataFrame(columns=['month_start_dt', 'Total Inquiries', 'New Leads', 'New Showings'])

# --- Main Report Function ---
def main(start_date, end_date):
    """
    Main function for the Lead Generation Funnel report.
    """
    st.info("Tracks the progression from total inquiries to new leads and new showings over time.")
    logger.info(f"Running Lead Generation Funnel report for {start_date} to {end_date}")

    engine = get_engine()
    if engine is None:
        logger.error("Failed to get DB engine. Aborting report.")
        return

    funnel_data = load_lead_funnel_data(engine, start_date, end_date)

    if funnel_data.empty:
        st.warning("No lead funnel data found for the selected date range, or an error occurred.")
        logger.warning("No lead funnel data loaded or processed.")
        return

    # --- Plotting ---
    st.subheader("Monthly Lead Generation Funnel")
    logger.info("Generating lead funnel plot...")
    try:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=funnel_data['month_start_dt'], y=funnel_data['Total Inquiries'],
            mode='lines+markers', name='Total Inquiries', line=dict(color='blue'),
            hovertemplate='<b>%{x|%b %Y}</b><br>Total Inquiries: %{y}<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            x=funnel_data['month_start_dt'], y=funnel_data['New Leads'],
            mode='lines+markers', name='New Leads', line=dict(color='orange'),
            hovertemplate='<b>%{x|%b %Y}</b><br>New Leads: %{y}<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            x=funnel_data['month_start_dt'], y=funnel_data['New Showings'],
            mode='lines+markers', name='New Showings', line=dict(color='green'),
            hovertemplate='<b>%{x|%b %Y}</b><br>New Showings: %{y}<extra></extra>'
        ))

        fig.update_layout(
            title="Monthly Lead Generation Funnel (Inquiries -> Leads -> Showings)",
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
    st.subheader("Monthly Funnel Data Table")
    try:
        display_df = funnel_data.copy()
        display_df['Month'] = display_df['month_start_dt'].dt.strftime('%B %Y')
        # Rename columns if needed (already done in query)
        display_df = display_df[['Month', 'Total Inquiries', 'New Leads', 'New Showings']]

        # Sort by Month (desc)
        try:
             display_df_sorted = display_df.sort_values(
                 by='Month',
                 ascending=False,
                 key=lambda s: pd.to_datetime(s, format='%B %Y', errors='coerce'),
                 na_position='last'
             )
        except Exception as sort_e:
             logger.warning(f"Could not sort funnel data table: {sort_e}. Displaying unsorted.")
             display_df_sorted = display_df

        st.dataframe(display_df_sorted, use_container_width=True, hide_index=True)
    except Exception as e:
        error_msg = f"Failed to display funnel data table: {e}"
        logger.exception(error_msg)
        st.error(error_msg)

    # --- Add plain English description ---
    st.markdown("""
    ---
    **How this report works:**
    *   This report visualizes the lead generation funnel using data from the `showmojo_high_level_metrics` table.
    *   It sums the daily counts for each month within the selected date range.
    *   'Total Inquiries' is approximated by summing the counts from phone, text, email, and web inquiries.
    *   'New Leads' comes directly from the `total_new_leads` column.
    *   'New Showings' comes directly from the `total_new_showings` column.
    *   The line chart shows the trend of these three key stages (Inquiries, Leads, Showings) over time.
    *   The table provides the detailed monthly counts for each stage.
    """)
    # --- End description ---

    logger.info("Report execution finished.")
