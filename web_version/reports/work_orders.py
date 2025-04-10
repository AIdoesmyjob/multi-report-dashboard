import streamlit as st
import pandas as pd
import logging # <-- Added
from sqlalchemy import text # <-- Removed create_engine
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
# import plotly.express as px # Switch to graph_objects for fill
import plotly.graph_objects as go # Import graph_objects

# Import shared functions from utils
from web_version.utils import get_engine # <-- Import shared engine function

# Configure logging for this report module
logger = logging.getLogger(__name__)


# Cache the work orders DataFrame.
@st.cache_data(ttl=600)
def load_work_orders_data(start_date, end_date):
    """
    Loads work orders joined with tasks based on task creation date.
    Groups results by month.
    """
    logger.info(f"Loading work order data from {start_date} to {end_date}")
    engine = get_engine() # <-- Use shared engine
    if engine is None:
        logger.error("Failed to get DB engine for work orders.")
        return pd.DataFrame()

    query = text("""
        SELECT
            w.id AS work_order_id,
            w.task_id,
            t.created_datetime AS task_created_datetime
        FROM work_orders w
        JOIN tasks t ON w.task_id = t.id
        WHERE t.created_datetime >= :start_date
          AND t.created_datetime <= :end_date
        ORDER BY w.id
    """)
    try:
        with engine.connect() as conn:
            df = pd.read_sql_query(query, conn, params={"start_date": start_date, "end_date": end_date})
        logger.info(f"Loaded {len(df)} work order records.")
    except Exception as e:
        logger.exception("Failed to load work order data.")
        st.error(f"DB Error loading work orders: {e}")
        return pd.DataFrame()

    if df.empty:
        logger.info("No work orders found in the date range.")
        return pd.DataFrame(columns=["year_month", "work_order_count", "year_month_dt"])

    # Convert the task creation datetime to a proper datetime type.
    df["task_created_datetime"] = pd.to_datetime(df["task_created_datetime"], errors="coerce")
    df = df.dropna(subset=["task_created_datetime"]) # Drop rows where conversion failed
    if df.empty:
        logger.warning("No work orders with valid task creation dates found.")
        return pd.DataFrame(columns=["year_month", "work_order_count", "year_month_dt"])


    # Create a 'year_month' column (e.g., "2023-05") based on task_created_datetime.
    df["year_month"] = df["task_created_datetime"].dt.to_period("M").astype(str)

    # Group by month and count the number of work orders.
    monthly_counts = df.groupby("year_month").size().reset_index(name="work_order_count")

    # Convert 'year_month' to a datetime (using the first day of the month) for plotting.
    monthly_counts["year_month_dt"] = pd.to_datetime(monthly_counts["year_month"] + "-01", format="%Y-%m-%d")
    monthly_counts.sort_values("year_month_dt", inplace=True)

    logger.info(f"Finished processing work order data, shape: {monthly_counts.shape}")
    return monthly_counts

def main(start_date, end_date):
    # st.header("Monthly Work Order Counts (by Task Creation Date)") # Removed - Redundant
    # st.write(f"Date Range: {start_date} to {end_date}") # Removed - Redundant
    logger.info(f"Running Work Orders report for {start_date} to {end_date}")

    monthly_counts = load_work_orders_data(start_date, end_date)
    if monthly_counts.empty:
        st.warning("No work orders found in the given date range.")
        logger.warning("No work order data loaded or processed.")
        return

    # Create a line chart with markers using Plotly graph_objects.
    logger.info("Generating plot...")
    try:
        fig = go.Figure() # Use go.Figure

        fig.add_trace(go.Scatter( # Use go.Scatter
            x=monthly_counts["year_month_dt"],
            y=monthly_counts["work_order_count"],
            mode='lines+markers', # Specify mode
            name='Work Orders',
            line=dict(color='royalblue'), # Use same blue
            fill='tozeroy', # Add fill
            fillcolor='rgba(65, 105, 225, 0.2)', # Semi-transparent blue
            hovertemplate = '<b>Month</b>: %{x|%Y-%m}<br><b>Work Orders</b>: %{y}<extra></extra>' # Custom hover
        ))

        fig.update_layout(
            title="Monthly Work Order Counts (by Task Creation Date)", # Keep title
            xaxis_title="Month",
            yaxis_title="Number of Work Orders",
            # template="plotly_white", # Removed template
            hovermode="x unified" # Add hovermode
        )
        fig.update_yaxes(rangemode='tozero')

        st.plotly_chart(fig, use_container_width=True)

        # --- Add plain English description ---
        st.markdown("""
        **How this graph is created:**
        *   This graph shows the total number of Work Orders created each month.
        *   The count is based on the creation date of the underlying *Task* associated with the Work Order.
        """)
        # --- End description ---

    except Exception as e:
        logger.exception("Failed to generate plot.")
        st.error(f"Error generating plot: {e}")


    st.subheader("Data Table")
    try:
        display_df = monthly_counts[['year_month_dt', 'work_order_count']].copy()
        display_df['Month'] = display_df['year_month_dt'].dt.strftime('%Y-%m')
        display_df = display_df[['Month', 'work_order_count']]
        display_df.rename(columns={'work_order_count': 'Work Order Count'}, inplace=True)
        st.dataframe(display_df.sort_values(by='Month'), hide_index=True)
    except Exception as e:
        logger.exception("Failed to display data table.")
        st.error(f"Error displaying data table: {e}")

    logger.info("Report execution finished.")

# --- Removed __main__ block ---
# if __name__ == "__main__":
#     from datetime import date
#     main(date(2020, 3, 10), date(2025, 3, 10))
