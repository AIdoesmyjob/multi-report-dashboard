# reports/team_member_showing_activity.py

import streamlit as st
import pandas as pd
from sqlalchemy import text, exc as sqlalchemy_exc
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import os
import logging
import numpy as np

# Import shared functions from utils
from web_version.utils import get_engine

# Configure logging for this report module
logger = logging.getLogger(__name__)

# --- Data Loading Function ---
@st.cache_data(ttl=600)
def load_team_activity_data(_engine, start_date, end_date):
    """
    Calculates showing activity metrics (prospects, scheduled, no-shows)
    grouped by team member and the month the prospect was created.

    Args:
        _engine: SQLAlchemy database engine instance.
        start_date: The start date of the analysis period (for prospect created_at).
        end_date: The end date of the analysis period (for prospect created_at).

    Returns:
        pandas.DataFrame: DataFrame with columns like
            ['month_start_dt', 'team_member', 'total_prospects', 'scheduled_showings', 'no_shows'],
            or an empty DataFrame on error/no data.
    """
    logger.info(f"Loading team member showing activity data from {start_date} to {end_date}")
    # Adjust end_date to be exclusive for the query
    end_date_adjusted = end_date + timedelta(days=1)

    # Query for monthly trends per team member
    query = text("""
        SELECT
            DATE_TRUNC('month', created_at)::date AS month_start_dt,
            COALESCE(team_member, 'Unassigned') AS team_member, -- Handle NULL team members
            COUNT(prospect_id) AS total_prospects,
            SUM(CASE WHEN showing_was_scheduled = TRUE THEN 1 ELSE 0 END) AS scheduled_showings,
            SUM(CASE WHEN no_show = TRUE THEN 1 ELSE 0 END) AS no_shows
        FROM
            showmojo_detailed_prospect_data
        WHERE
            created_at >= :start_date
            AND created_at < :end_date_adj -- Use adjusted end date
            AND created_at IS NOT NULL
        GROUP BY
            month_start_dt,
            team_member
        ORDER BY
            month_start_dt ASC,
            team_member ASC;
    """)

    try:
        with _engine.connect() as conn:
            params = {
                "start_date": start_date,
                "end_date_adj": end_date_adjusted
            }
            db_results = pd.read_sql_query(query, conn, params=params, parse_dates=['month_start_dt'])
        logger.info(f"Loaded {len(db_results)} team member monthly activity records from DB.")

        # --- Create complete monthly range for each team member ---
        start_date_dt = pd.to_datetime(start_date)
        end_date_dt = pd.to_datetime(end_date)
        start_month = start_date_dt.replace(day=1)
        end_month = end_date_dt.replace(day=1)

        if start_month > end_month:
            logger.warning("Start date is after end date, resulting in empty month range.")
            all_months = pd.DatetimeIndex([])
            all_team_members = []
        else:
            all_months = pd.date_range(start=start_month, end=end_month, freq='MS')
            # Get unique team members from the results
            all_team_members = db_results['team_member'].unique() if not db_results.empty else []


        # Create a full grid of months and team members
        if len(all_months) > 0 and len(all_team_members) > 0:
            idx = pd.MultiIndex.from_product([all_months, all_team_members], names=['month_start_dt', 'team_member'])
            df_full_grid = pd.DataFrame(index=idx).reset_index()
        else:
             # If no months or no team members, start with the original results or an empty frame
             df_full_grid = db_results if not db_results.empty else pd.DataFrame(columns=['month_start_dt', 'team_member'])


        # Merge results with the full grid
        if not df_full_grid.empty:
            # Ensure month_start_dt is datetime in both frames before merge
            df_full_grid['month_start_dt'] = pd.to_datetime(df_full_grid['month_start_dt'])
            if not db_results.empty:
                 db_results['month_start_dt'] = pd.to_datetime(db_results['month_start_dt'])
                 merged_df = pd.merge(df_full_grid, db_results, on=['month_start_dt', 'team_member'], how='left')
            else:
                 merged_df = df_full_grid # If no db_results, grid is the starting point
                 # Add the count columns if they don't exist from the grid creation
                 for col in ['total_prospects', 'scheduled_showings', 'no_shows']:
                     if col not in merged_df.columns:
                         merged_df[col] = 0

            # Fill NaN counts with 0 after merge
            merged_df[['total_prospects', 'scheduled_showings', 'no_shows']] = merged_df[['total_prospects', 'scheduled_showings', 'no_shows']].fillna(0).astype(int)
        else:
             merged_df = db_results # Use only db results if grid creation failed

        # Ensure month_start_dt is datetime before formatting
        merged_df['month_start_dt'] = pd.to_datetime(merged_df['month_start_dt'])
        merged_df['month_start'] = merged_df['month_start_dt'].dt.strftime('%Y-%m-%d')


        # Select and order final columns
        final_df = merged_df[['month_start', 'month_start_dt', 'team_member', 'total_prospects', 'scheduled_showings', 'no_shows']]
        logger.info(f"Processed team member activity data, shape: {final_df.shape}")
        return final_df

    except Exception as e:
        error_msg = f"Error querying or processing team member activity data: {e}"
        logger.exception(error_msg)
        st.error(error_msg)
        return pd.DataFrame(columns=['month_start', 'month_start_dt', 'team_member', 'total_prospects', 'scheduled_showings', 'no_shows'])

# --- Main Report Function ---
def main(start_date, end_date):
    """
    Main function for the Team Member Showing Activity report.
    """
    st.info("Analyzes prospect showing activity metrics per team member over time.")
    logger.info(f"Running Team Member Showing Activity report for {start_date} to {end_date}")

    engine = get_engine()
    if engine is None:
        logger.error("Failed to get DB engine. Aborting report.")
        return

    activity_df = load_team_activity_data(engine, start_date, end_date)

    # Check if DataFrame is completely empty
    if activity_df.empty:
        st.warning("No team member activity data found for the selected date range, or an error occurred.")
        logger.warning("No team member activity data loaded or processed.")
        return

    # --- Team Member Selection ---
    st.subheader("Filter by Team Member")
    all_team_members = sorted(activity_df['team_member'].unique())
    selected_team_members = st.multiselect(
        "Select Team Members:",
        options=all_team_members,
        default=all_team_members # Default to all selected
    )

    if not selected_team_members:
        st.warning("Please select at least one team member.")
        return

    filtered_df = activity_df[activity_df['team_member'].isin(selected_team_members)]

    # --- Plotting Monthly Trends per Team Member ---
    st.subheader("Monthly Activity Trends by Team Member")
    logger.info("Generating team activity plot...")
    try:
        # Use Plotly Express for easier grouped bar charts or line charts by color
        # Option 1: Grouped Bar Chart (good for comparing members within a month)
        # fig = px.bar(filtered_df, x='month_start_dt', y='scheduled_showings', color='team_member',
        #              barmode='group', title="Scheduled Showings by Team Member",
        #              labels={'month_start_dt': 'Month', 'scheduled_showings': 'Scheduled Showings', 'team_member': 'Team Member'},
        #              hover_data=['total_prospects', 'no_shows'])

        # Option 2: Line Chart (good for seeing individual trends over time)
        fig = px.line(filtered_df, x='month_start_dt', y='scheduled_showings', color='team_member',
                      title="Scheduled Showings by Team Member Over Time",
                      labels={'month_start_dt': 'Month', 'scheduled_showings': 'Scheduled Showings', 'team_member': 'Team Member'},
                      markers=True, # Add markers to lines
                      hover_data={'month_start_dt': '|%B %Y', # Format hover date
                                  'team_member': True,
                                  'scheduled_showings': True,
                                  'total_prospects': True,
                                  'no_shows': True})

        fig.update_layout(
            xaxis_title="Prospect Creation Month",
            yaxis_title="Count",
            xaxis_tickformat="%b %Y",
            hovermode="x unified" # Or "closest" if preferred for line charts
        )
        fig.update_yaxes(rangemode='tozero')
        st.plotly_chart(fig, use_container_width=True)

        # Add plots for other metrics if desired (e.g., total prospects, no-shows)
        # fig_prospects = px.line(...)
        # st.plotly_chart(fig_prospects, use_container_width=True)
        # fig_noshows = px.line(...)
        # st.plotly_chart(fig_noshows, use_container_width=True)


    except Exception as e:
        error_msg = f"Failed to generate team activity plot: {e}"
        logger.exception(error_msg)
        st.error(error_msg)

    # --- Data Table ---
    st.subheader("Team Activity Data Table")
    try:
        display_df = filtered_df[['month_start_dt', 'team_member', 'total_prospects', 'scheduled_showings', 'no_shows']].copy()
        display_df['Month'] = display_df['month_start_dt'].dt.strftime('%B %Y')
        display_df = display_df.rename(columns={
            'team_member': 'Team Member',
            'total_prospects': 'Total Prospects',
            'scheduled_showings': 'Scheduled Showings',
            'no_shows': 'No Shows'
        })
        display_df = display_df[['Month', 'Team Member', 'Total Prospects', 'Scheduled Showings', 'No Shows']]

        # Sort by Month, then Team Member
        try:
             display_df_sorted = display_df.sort_values(
                 by=['Month', 'Team Member'],
                 ascending=[False, True], # Sort newest month first, then alphabetically by team member
                 key=lambda s: pd.to_datetime(s, format='%B %Y', errors='coerce') if s.name == 'Month' else s,
                 na_position='last'
             )
        except Exception as sort_e:
             logger.warning(f"Could not sort team activity dataframe: {sort_e}. Displaying unsorted.")
             display_df_sorted = display_df

        st.dataframe(display_df_sorted, use_container_width=True, hide_index=True)
    except Exception as e:
        error_msg = f"Failed to display team activity data table: {e}"
        logger.exception(error_msg)
        st.error(error_msg)

    # --- Add plain English description ---
    st.markdown("""
    ---
    **How this report works:**
    *   This report analyzes showing activity metrics per team member using data from the `showmojo_detailed_prospect_data` table.
    *   It counts the total prospects, scheduled showings (`showing_was_scheduled = TRUE`), and no-shows (`no_show = TRUE`) associated with each `team_member`. Prospects with no assigned team member are grouped under 'Unassigned'.
    *   The data is grouped by the month the prospect was created (`created_at`).
    *   The line chart displays the trend of scheduled showings for each selected team member over time. You can select/deselect team members using the filter.
    *   The table provides the detailed monthly counts for total prospects, scheduled showings, and no-shows for the selected team members.
    """)
    # --- End description ---

    logger.info("Report execution finished.")
