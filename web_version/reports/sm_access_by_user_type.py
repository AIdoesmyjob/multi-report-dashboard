# reports/access_by_user_type.py

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

# --- Date Parsing Function ---
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
            # Handle cases where the extracted string is not a valid date
            return pd.NaT
    return pd.NaT

# --- Data Loading Function ---
@st.cache_data(ttl=600)
def load_access_by_type_data(_engine, start_date, end_date):
    """
    Loads digital access data, parses dates, and aggregates counts by user type and month.

    Args:
        _engine: SQLAlchemy database engine instance.
        start_date: The start date of the analysis period (for parsed access date).
        end_date: The end date of the analysis period (for parsed access date).

    Returns:
        pandas.DataFrame: DataFrame with columns like
            ['month_start_dt', 'pulled_by_type', 'access_count'],
            or an empty DataFrame on error/no data.
    """
    logger.info(f"Loading access by user type data from {start_date} to {end_date}")

    # Query all data within a broader range initially, then filter by parsed date
    # This is less efficient but necessary because we can't filter on the raw text date easily in SQL
    # Consider adding report_start_date/report_end_date filters if applicable and helpful
    query = text("""
        SELECT
            access_date_raw,
            pulled_by_type
        FROM
            showmojo_digital_access;
        -- WHERE clause might be added later if performance is an issue
        -- and report_start/end_date can effectively pre-filter
    """)

    try:
        with _engine.connect() as conn:
            # Load all data first
            db_results = pd.read_sql_query(query, conn)
        logger.info(f"Loaded {len(db_results)} total digital access records from DB.")

        if db_results.empty:
            logger.warning("No digital access records found in the database.")
            return pd.DataFrame(columns=['month_start_dt', 'pulled_by_type', 'access_count'])

        # --- Parse Dates ---
        db_results['parsed_date'] = db_results['access_date_raw'].apply(parse_access_date)

        # Filter out rows where date parsing failed
        parsed_df = db_results.dropna(subset=['parsed_date']).copy()
        logger.info(f"Successfully parsed dates for {len(parsed_df)} records.")

        if parsed_df.empty:
            logger.warning("No records with parseable dates found.")
            return pd.DataFrame(columns=['month_start_dt', 'pulled_by_type', 'access_count'])

        # --- Filter by Date Range ---
        # Convert start_date and end_date to datetime objects for comparison
        start_date_dt = pd.to_datetime(start_date)
        end_date_dt = pd.to_datetime(end_date)

        filtered_df = parsed_df[
            (parsed_df['parsed_date'] >= start_date_dt) &
            (parsed_df['parsed_date'] <= end_date_dt)
        ].copy()
        logger.info(f"{len(filtered_df)} records remain after date range filtering.")

        if filtered_df.empty:
            logger.warning("No records found within the selected date range after parsing.")
            return pd.DataFrame(columns=['month_start_dt', 'pulled_by_type', 'access_count'])

        # --- Aggregate Data ---
        # Ensure 'pulled_by_type' is treated as string, handle potential None/NaN
        filtered_df['pulled_by_type'] = filtered_df['pulled_by_type'].fillna('Unknown').astype(str)

        # Group by month and user type
        filtered_df['month_start_dt'] = filtered_df['parsed_date'].dt.to_period('M').dt.start_time
        aggregated_df = filtered_df.groupby(['month_start_dt', 'pulled_by_type']).size().reset_index(name='access_count')

        logger.info(f"Aggregated data into {len(aggregated_df)} rows.")

        # --- Create complete monthly range for each type ---
        all_months = pd.date_range(start=start_date_dt.replace(day=1), end=end_date_dt.replace(day=1), freq='MS')
        all_types = aggregated_df['pulled_by_type'].unique()

        if len(all_months) > 0 and len(all_types) > 0:
            idx = pd.MultiIndex.from_product([all_months, all_types], names=['month_start_dt', 'pulled_by_type'])
            df_full_grid = pd.DataFrame(index=idx).reset_index()

            # Merge aggregated data with the full grid
            final_df = pd.merge(df_full_grid, aggregated_df, on=['month_start_dt', 'pulled_by_type'], how='left')
            final_df['access_count'] = final_df['access_count'].fillna(0).astype(int)
            logger.info(f"Processed access by user type data, final shape: {final_df.shape}")
            return final_df.sort_values(by=['month_start_dt', 'pulled_by_type'])
        else:
            logger.warning("Could not create full grid, returning potentially sparse aggregated data.")
            return aggregated_df.sort_values(by=['month_start_dt', 'pulled_by_type'])


    except Exception as e:
        error_msg = f"Error querying or processing access by user type data: {e}"
        logger.exception(error_msg)
        st.error(error_msg)
        return pd.DataFrame(columns=['month_start_dt', 'pulled_by_type', 'access_count'])

# --- Main Report Function ---
def main(start_date, end_date):
    """
    Main function for the Access by User Type report.
    """
    st.info("Analyzes digital lockbox/code access counts by user type over time.")
    logger.info(f"Running Access by User Type report for {start_date} to {end_date}")

    engine = get_engine()
    if engine is None:
        logger.error("Failed to get DB engine. Aborting report.")
        return

    access_data = load_access_by_type_data(engine, start_date, end_date)

    if access_data.empty:
        st.warning("No access data found for the selected date range, or an error occurred.")
        logger.warning("No access data loaded or processed.")
        return

    # --- Plotting ---
    st.subheader("Monthly Access Counts by User Type")
    logger.info("Generating access by user type plot...")
    try:
        fig = px.line(access_data, x='month_start_dt', y='access_count', color='pulled_by_type',
                      title="Digital Access Counts by User Type Over Time",
                      labels={'month_start_dt': 'Month', 'access_count': 'Number of Accesses', 'pulled_by_type': 'User Type'},
                      markers=True)

        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Number of Accesses",
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
    st.subheader("Access Data Table")
    try:
        display_df = access_data.copy()
        display_df['Month'] = display_df['month_start_dt'].dt.strftime('%B %Y')
        display_df = display_df.rename(columns={
            'pulled_by_type': 'User Type',
            'access_count': 'Access Count'
        })
        display_df = display_df[['Month', 'User Type', 'Access Count']]

        # Sort by Month (desc), then User Type (asc)
        try:
             display_df_sorted = display_df.sort_values(
                 by=['Month', 'User Type'],
                 ascending=[False, True],
                 key=lambda s: pd.to_datetime(s, format='%B %Y', errors='coerce') if s.name == 'Month' else s,
                 na_position='last'
             )
        except Exception as sort_e:
             logger.warning(f"Could not sort access data table: {sort_e}. Displaying unsorted.")
             display_df_sorted = display_df

        st.dataframe(display_df_sorted[display_df_sorted['Access Count'] > 0], use_container_width=True, hide_index=True) # Hide rows with 0 count
    except Exception as e:
        error_msg = f"Failed to display access data table: {e}"
        logger.exception(error_msg)
        st.error(error_msg)

    # --- Add plain English description ---
    st.markdown("""
    ---
    **How this report works:**
    *   This report analyzes digital lockbox/code access counts by user type using data from the `showmojo_digital_access` table.
    *   It attempts to extract a valid date from the `access_date_raw` column (looking for a 'D Mon YYYY' format). Records without a parseable date in this format are excluded.
    *   It counts the number of accesses for each `pulled_by_type` (e.g., 'Showing', 'Team member') within the selected date range. User types not specified are grouped under 'Unknown'.
    *   The data is grouped by the month of the parsed access date.
    *   The line chart shows the trend of access counts for each user type over time.
    *   The table provides the detailed monthly counts per user type.
    """)
    # --- End description ---

    logger.info("Report execution finished.")
