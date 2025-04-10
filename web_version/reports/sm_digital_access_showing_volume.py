# reports/peak_access_dates.py

import streamlit as st
import pandas as pd
from sqlalchemy import text, exc as sqlalchemy_exc
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import os
import logging
import re # For regex extraction

# Import shared functions from utils
from web_version.utils import get_engine

# Configure logging for this report module
logger = logging.getLogger(__name__)

# --- Date Parsing Function (Copied from access_by_user_type.py) ---
def parse_access_date(date_str):
    """
    Attempts to parse a date from the varied 'access_date_raw' string.
    Looks for a 'D Mon YYYY' pattern.
    """
    if not isinstance(date_str, str):
        return pd.NaT
    # Regex to find 'D Mon YYYY' or 'DD Mon YYYY' pattern
    match = re.search(r'(\d{1,2}\s+\w{3}\s+\d{4})', date_str)
    if match:
        try:
            # Attempt to parse the extracted date string
            return pd.to_datetime(match.group(1), format='%d %b %Y')
        except ValueError:
            return pd.NaT
    return pd.NaT

# --- Data Loading Function ---
@st.cache_data(ttl=600)
def load_peak_access_data(_engine, start_date, end_date):
    """
    Loads digital access data, parses dates, and prepares data for daily/day-of-week analysis.

    Args:
        _engine: SQLAlchemy database engine instance.
        start_date: The start date of the analysis period (for parsed access date).
        end_date: The end date of the analysis period (for parsed access date).

    Returns:
        pandas.DataFrame: DataFrame with columns ['parsed_date', 'day_of_week_num', 'day_of_week_name'],
                          or an empty DataFrame on error/no data.
    """
    logger.info(f"Loading peak access dates data from {start_date} to {end_date}")

    # Query data, including pulled_by_type for filtering
    query = text("""
        SELECT
            access_date_raw,
            pulled_by_type
        FROM
            showmojo_digital_access;
    """)

    try:
        with _engine.connect() as conn:
            db_results = pd.read_sql_query(query, conn)
        logger.info(f"Loaded {len(db_results)} total digital access records from DB.")

        if db_results.empty:
            logger.warning("No digital access records found in the database.")
            return pd.DataFrame(columns=['parsed_date', 'day_of_week_num', 'day_of_week_name'])

        # --- Filter out Team Member accesses ---
        # Ensure pulled_by_type is string and handle potential NaN before filtering
        db_results['pulled_by_type'] = db_results['pulled_by_type'].fillna('Unknown').astype(str)
        non_team_member_df = db_results[db_results['pulled_by_type'].str.lower() != 'team member'].copy()
        logger.info(f"{len(non_team_member_df)} records remain after filtering out 'Team member'.")

        if non_team_member_df.empty:
            logger.warning("No non-team member access records found.")
            return pd.DataFrame(columns=['parsed_date', 'day_of_week_num', 'day_of_week_name'])

        # --- Parse Dates ---
        non_team_member_df['parsed_date'] = non_team_member_df['access_date_raw'].apply(parse_access_date)
        parsed_df = non_team_member_df.dropna(subset=['parsed_date']).copy()
        logger.info(f"Successfully parsed dates for {len(parsed_df)} non-team member records.")

        if parsed_df.empty:
            logger.warning("No non-team member records with parseable dates found.")
            return pd.DataFrame(columns=['parsed_date', 'day_of_week_num', 'day_of_week_name'])

        # --- Filter by Date Range ---
        start_date_dt = pd.to_datetime(start_date)
        end_date_dt = pd.to_datetime(end_date)
        filtered_df = parsed_df[
            (parsed_df['parsed_date'] >= start_date_dt) &
            (parsed_df['parsed_date'] <= end_date_dt)
        ].copy()
        logger.info(f"{len(filtered_df)} records remain after date range filtering.")

        if filtered_df.empty:
            logger.warning("No records found within the selected date range after parsing.")
            return pd.DataFrame(columns=['parsed_date', 'day_of_week_num', 'day_of_week_name'])

        # --- Add Day of Week ---
        filtered_df['day_of_week_num'] = filtered_df['parsed_date'].dt.dayofweek # Monday=0, Sunday=6
        filtered_df['day_of_week_name'] = filtered_df['parsed_date'].dt.strftime('%A') # Full name like 'Monday'

        final_df = filtered_df[['parsed_date', 'day_of_week_num', 'day_of_week_name']]
        logger.info(f"Processed peak access data, final shape: {final_df.shape}")
        return final_df

    except Exception as e:
        error_msg = f"Error querying or processing peak access data: {e}"
        logger.exception(error_msg)
        st.error(error_msg)
        return pd.DataFrame(columns=['parsed_date', 'day_of_week_num', 'day_of_week_name'])

# --- Main Report Function ---
def main(start_date, end_date):
    """
    Main function for the Peak Access Dates/Days report.
    """
    st.info("Analyzes digital access frequency by date and day of the week.")
    logger.info(f"Running Peak Access Dates report for {start_date} to {end_date}")

    engine = get_engine()
    if engine is None:
        logger.error("Failed to get DB engine. Aborting report.")
        return

    access_data = load_peak_access_data(engine, start_date, end_date)

    if access_data.empty:
        st.warning("No access data with parseable dates found for the selected range, or an error occurred.")
        logger.warning("No peak access data loaded or processed.")
        return

    # --- Analysis 1: Access Counts per Day ---
    st.subheader("Daily Access Counts")
    logger.info("Generating daily access count plot...")
    try:
        daily_counts = access_data.groupby('parsed_date').size().reset_index(name='access_count')

        fig_daily = px.line(daily_counts, x='parsed_date', y='access_count',
                            title="Digital Access Counts Per Day",
                            labels={'parsed_date': 'Date', 'access_count': 'Number of Accesses'},
                            markers=True)
        fig_daily.update_layout(xaxis_title="Date", yaxis_title="Number of Accesses")
        fig_daily.update_yaxes(rangemode='tozero')
        st.plotly_chart(fig_daily, use_container_width=True)

        # Optional: Data Table for Daily Counts
        # st.dataframe(daily_counts.rename(columns={'parsed_date':'Date', 'access_count':'Access Count'}).sort_values('Date', ascending=False), use_container_width=True, hide_index=True)

    except Exception as e:
        error_msg = f"Failed to generate daily access plot: {e}"
        logger.exception(error_msg)
        st.error(error_msg)


    # --- Analysis 2: Access Counts by Day of Week ---
    st.subheader("Total Access Counts by Day of Week")
    logger.info("Generating day of week access count plot...")
    try:
        dow_counts = access_data.groupby(['day_of_week_num', 'day_of_week_name']).size().reset_index(name='total_access_count')
        dow_counts = dow_counts.sort_values('day_of_week_num') # Sort Mon-Sun

        fig_dow = px.bar(dow_counts, x='day_of_week_name', y='total_access_count',
                         title=f"Total Digital Accesses by Day of Week ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})",
                         labels={'day_of_week_name': 'Day of Week', 'total_access_count': 'Total Accesses'},
                         text_auto=True) # Show counts on bars
        fig_dow.update_layout(xaxis_title="Day of Week", yaxis_title="Total Number of Accesses")
        fig_dow.update_xaxes(categoryorder='array', categoryarray=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']) # Ensure correct order
        st.plotly_chart(fig_dow, use_container_width=True)

        # Data Table for Day of Week Counts
        st.dataframe(dow_counts[['day_of_week_name', 'total_access_count']].rename(columns={'day_of_week_name':'Day', 'total_access_count':'Total Accesses'}),
                     use_container_width=True, hide_index=True)

    except Exception as e:
        error_msg = f"Failed to generate day of week plot/table: {e}"
        logger.exception(error_msg)
        st.error(error_msg)

    # --- Add plain English description ---
    st.markdown("""
    ---
    **How this report works:**
    *   This report analyzes the frequency of digital lockbox/code access based on date and day of the week, using data from the `showmojo_digital_access` table.
    *   It **excludes** accesses made by 'Team member' (based on the `pulled_by_type` column) to focus on external access like showings.
    *   It attempts to extract a valid date from the `access_date_raw` column (looking for a 'D Mon YYYY' format). Records without a parseable date in this format are excluded.
    *   The first line chart shows the total number of non-team member accesses per day over the selected period.
    *   The second bar chart shows the total number of non-team member accesses aggregated by the day of the week (e.g., total accesses on Mondays vs Tuesdays).
    *   The table provides the total counts for each day of the week.
    """)
    # --- End description ---

    logger.info("Report execution finished.")
