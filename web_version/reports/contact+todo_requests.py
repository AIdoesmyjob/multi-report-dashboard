import streamlit as st
import pandas as pd
import logging # <-- Added
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
from sqlalchemy import text # <-- Removed create_engine
import plotly.express as px

# Import shared functions from utils
from web_version.utils import get_engine # <-- Import shared engine function

# Configure logging for this report module
logger = logging.getLogger(__name__)

# -----------------------
# 2. Load & Group Contact Requests
# -----------------------
@st.cache_data(ttl=600)
def load_contact_requests_data(start_date, end_date):
    """
    Query contact_requests for rows with created_datetime between start_date and end_date.
    Group by month and return a DataFrame with columns:
      [month_str, request_count, request_type]
    where request_type = "Contact Requests"
    """
    logger.info(f"Loading contact requests data from {start_date} to {end_date}")
    engine = get_engine() # <-- Use shared engine
    if engine is None:
        logger.error("Failed to get DB engine for contact requests.")
        return pd.DataFrame()

    query = text("""
        SELECT
            id,
            created_datetime
        FROM contact_requests
        WHERE created_datetime >= :start_date
          AND created_datetime <= :end_date
    """)
    try:
        with engine.connect() as conn:
            df = pd.read_sql_query(
                query, conn,
                params={"start_date": start_date, "end_date": end_date}
            )
        logger.info(f"Loaded {len(df)} contact requests.")
    except Exception as e:
        logger.exception("Failed to load contact requests.")
        st.error(f"DB Error loading contact requests: {e}")
        return pd.DataFrame()

    if df.empty:
        logger.info("No contact requests found in the date range.")
        return pd.DataFrame(columns=["month_str", "request_count", "request_type"])

    # Convert to datetime if not already
    df["created_datetime"] = pd.to_datetime(df["created_datetime"], errors="coerce")
    df = df.dropna(subset=["created_datetime"]) # Drop rows where conversion failed

    if df.empty:
        logger.info("No valid contact requests after date conversion.")
        return pd.DataFrame(columns=["month_str", "request_count", "request_type"])

    # Create month_str (YYYY-MM)
    df["month_str"] = df["created_datetime"].dt.to_period("M").astype(str)

    # Group by month_str
    grouped = df.groupby("month_str").size().reset_index(name="request_count")
    grouped["request_type"] = "Contact Requests"
    return grouped

# -----------------------
# 3. Load & Group To-Do Requests
# -----------------------
@st.cache_data(ttl=600)
def load_to_do_requests_data(start_date, end_date):
    """
    Query to_do_requests for rows with created_datetime between start_date and end_date.
    Group by month and return a DataFrame with columns:
      [month_str, request_count, request_type]
    where request_type = "To-Do Requests"
    """
    logger.info(f"Loading to-do requests data from {start_date} to {end_date}")
    engine = get_engine() # <-- Use shared engine
    if engine is None:
        logger.error("Failed to get DB engine for to-do requests.")
        return pd.DataFrame()

    query = text("""
        SELECT
            id,
            created_datetime
        FROM to_do_requests
        WHERE created_datetime >= :start_date
          AND created_datetime <= :end_date
    """)
    try:
        with engine.connect() as conn:
            df = pd.read_sql_query(
                query, conn,
                params={"start_date": start_date, "end_date": end_date}
            )
        logger.info(f"Loaded {len(df)} to-do requests.")
    except Exception as e:
        logger.exception("Failed to load to-do requests.")
        st.error(f"DB Error loading to-do requests: {e}")
        return pd.DataFrame()

    if df.empty:
        logger.info("No to-do requests found in the date range.")
        return pd.DataFrame(columns=["month_str", "request_count", "request_type"])

    # Convert to datetime if not already
    df["created_datetime"] = pd.to_datetime(df["created_datetime"], errors="coerce")
    df = df.dropna(subset=["created_datetime"]) # Drop rows where conversion failed

    if df.empty:
        logger.info("No valid to-do requests after date conversion.")
        return pd.DataFrame(columns=["month_str", "request_count", "request_type"])

    # Create month_str (YYYY-MM)
    df["month_str"] = df["created_datetime"].dt.to_period("M").astype(str)

    # Group by month_str
    grouped = df.groupby("month_str").size().reset_index(name="request_count")
    grouped["request_type"] = "To-Do Requests"
    return grouped

# -----------------------
# 4. Combine & Plot
# -----------------------
def main(start_date, end_date):
    st.title("Contact Requests & To-Do Requests by Month")
    st.write(f"Date Range: {start_date} to {end_date}")
    logger.info(f"Running Contact/To-Do Requests report for {start_date} to {end_date}")

    # Load each request type
    contact_df = load_contact_requests_data(start_date, end_date)
    todo_df = load_to_do_requests_data(start_date, end_date)

    # Combine them
    combined_df = pd.concat([contact_df, todo_df], ignore_index=True)
    if combined_df.empty:
        st.warning("No requests found in the given date range.")
        logger.warning("Combined dataframe is empty.")
        return

    # Convert month_str to an actual datetime for plotting (1st day of each month)
    try:
        combined_df["month_dt"] = pd.to_datetime(combined_df["month_str"] + "-01", format="%Y-%m-%d")
        combined_df.sort_values("month_dt", inplace=True)
    except Exception as e:
        logger.exception("Failed to convert month_str to datetime or sort.")
        st.error("Error processing dates for plotting.")
        # Optionally display unsorted data or stop
        st.dataframe(combined_df)
        return


    # Plot with Plotly Express line chart, color by request_type
    logger.info("Generating plot...")
    try:
        fig = px.line(
            combined_df,
            x="month_dt",
            y="request_count",
            color="request_type",
            markers=True,
            title="Monthly Contact & To-Do Requests"
        )
        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Count of Requests",
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            )
        )
        fig.update_yaxes(rangemode='tozero') # Ensure y-axis starts at 0

        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        logger.exception("Failed to generate plot.")
        st.error(f"Error generating plot: {e}")

    # Display Data Table
    st.subheader("Data Table")
    try:
        # Select and rename columns for display
        display_df = combined_df[['month_str', 'request_type', 'request_count']].copy()
        display_df.rename(columns={
            'month_str': 'Month',
            'request_type': 'Request Type',
            'request_count': 'Count'
        }, inplace=True)
        st.dataframe(display_df.sort_values(by=['Month', 'Request Type']))
    except Exception as e:
        logger.exception("Failed to display data table.")
        st.error(f"Error displaying data table: {e}")

    logger.info("Report execution finished.")

# -----------------------
# 5. Run as a script - REMOVED
# -----------------------
# if __name__ == "__main__":
#     main(date(2020, 3, 15), date(2025, 3, 15))
