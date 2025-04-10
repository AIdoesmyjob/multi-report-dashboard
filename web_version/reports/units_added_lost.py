import streamlit as st
import pandas as pd
import json
import logging # <-- Added
from sqlalchemy import text # <-- Removed create_engine
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import shared functions from utils
from web_version.utils import get_engine # <-- Import shared engine function

# Configure logging for this report module
logger = logging.getLogger(__name__)

@st.cache_data(ttl=600)
def load_units_data(start_date, end_date):
    """
    Retrieves rent transactions and lease details, merges them to obtain unit_id,
    and computes for each unit:
      - first_rent_date (when the unit started renting)
      - last_rent_date (most recent rent transaction)

    A unit is flagged as "lost" if today is more than 75 days past its last rent transaction.

    Then, for each month in the analysis period, we count:
      - Units Added: units with a first_rent_date in that month.
      - Units Lost: units with a last_rent_date in that month and that are flagged as lost.
      - Total Units: distinct count of unit_ids with at least one rent transaction in that month.

    Returns a DataFrame with columns:
      [month_start, units_added, units_lost, total_units, month_start_dt]
    """
    logger.info(f"Loading units added/lost data from {start_date} to {end_date}")
    engine = get_engine() # <-- Use shared engine
    if engine is None:
        logger.error("Failed to get DB engine for units data.")
        return pd.DataFrame()

    # --- Step 1: Retrieve Rent Transactions ---
    query_transactions = """
        SELECT
            lease_id,
            transaction_date,
            total_amount,
            transaction_type
        FROM lease_transactions
        WHERE transaction_type ILIKE 'Payment'
           OR transaction_type ILIKE 'Applied Prepayment'
           OR transaction_type ILIKE 'Charge'
    """
    try:
        with engine.connect() as conn:
            rent_txn = pd.read_sql_query(text(query_transactions), conn)
        logger.info(f"Loaded {rent_txn.shape[0]} rent transactions.")
    except Exception as e:
        logger.exception("Failed to load rent transactions.")
        st.error(f"DB Error loading transactions: {e}")
        return pd.DataFrame()

    rent_txn['transaction_date'] = pd.to_datetime(rent_txn['transaction_date'], errors='coerce').dt.date
    rent_txn = rent_txn.dropna(subset=['transaction_date']) # Drop invalid dates
    # st.write(f"Total rent transactions: {rent_txn.shape[0]}") # Replaced with logging

    # --- Step 2: Retrieve Lease Data ---
    query_leases = """
        SELECT
            id AS lease_id,
            unit_id,
            lease_status,
            lease_from_date,
            lease_to_date
        FROM leases
    """
    try:
        with engine.connect() as conn:
            leases_df = pd.read_sql_query(text(query_leases), conn)
        logger.info(f"Loaded {leases_df.shape[0]} leases.")
    except Exception as e:
        logger.exception("Failed to load leases.")
        st.error(f"DB Error loading leases: {e}")
        return pd.DataFrame()

    leases_df['lease_from_date'] = pd.to_datetime(leases_df['lease_from_date'], errors='coerce').dt.date
    leases_df['lease_to_date'] = pd.to_datetime(leases_df['lease_to_date'], errors='coerce').dt.date
    # st.write(f"Total leases: {leases_df.shape[0]}") # Replaced with logging

    # --- Step 3: Merge Rent Transactions with Lease Data ---
    merged_txn = rent_txn.merge(leases_df[['lease_id', 'unit_id']], on="lease_id", how="left")
    merged_txn = merged_txn[merged_txn['unit_id'].notnull()] # Keep only transactions linked to a unit
    logger.info(f"Transactions with valid unit_id: {merged_txn.shape[0]}")
    # st.write(f"Transactions with valid unit_id: {merged_txn.shape[0]}") # Replaced with logging

    if merged_txn.empty:
        logger.warning("No transactions linked to units found.")
        return pd.DataFrame()

    # Compute total unique units per month (based on transactions)
    merged_txn['txn_month'] = pd.to_datetime(merged_txn['transaction_date']).dt.to_period('M').dt.to_timestamp()
    total_units = merged_txn.groupby('txn_month')['unit_id'].nunique().reset_index()
    total_units.rename(columns={'txn_month': 'month_start_dt', 'unit_id': 'total_units'}, inplace=True) # Rename for merge

    # --- Step 4: Group by unit_id to get first and last rent transaction dates ---
    grouped_units = merged_txn.groupby("unit_id")["transaction_date"].agg(
        first_rent_date="min",
        last_rent_date="max"
    ).reset_index()
    logger.info(f"Found {grouped_units.shape[0]} unique units with rent transactions.")
    # st.write(f"Found {grouped_units.shape[0]} unique units with rent transactions.") # Replaced with logging

    # --- Step 5: Flag units as lost if last_rent_date is more than 75 days old ---
    today = datetime.today().date()
    grouped_units['lost'] = grouped_units['last_rent_date'].apply(lambda d: (today - d).days > 75 if pd.notna(d) else False)

    # Create month keys (timestamps) for first and last rent dates
    grouped_units['first_month'] = pd.to_datetime(grouped_units['first_rent_date']).dt.to_period('M').dt.to_timestamp()
    grouped_units['last_month'] = pd.to_datetime(grouped_units['last_rent_date']).dt.to_period('M').dt.to_timestamp()

    # Aggregate units added by month (using first_rent_date)
    added_counts = grouped_units.groupby('first_month').size().reset_index(name='units_added')
    # Aggregate units lost by month (using last_rent_date for units flagged as lost)
    lost_counts = grouped_units[grouped_units['lost']].groupby('last_month').size().reset_index(name='units_lost')

    # --- Step 6: Create a complete monthly range for the analysis period ---
    try:
        start_month = pd.to_datetime(start_date).to_period('M').to_timestamp()
        end_month = pd.to_datetime(end_date).to_period('M').to_timestamp()
        months = pd.date_range(start=start_month, end=end_month, freq='MS')
        df = pd.DataFrame({'month_start_dt': months})
    except Exception as e:
        logger.exception("Error creating date range.")
        st.error(f"Error creating date range: {e}")
        return pd.DataFrame()

    # Merge added and lost counts.
    df = df.merge(added_counts, left_on='month_start_dt', right_on='first_month', how='left')
    df = df.merge(lost_counts, left_on='month_start_dt', right_on='last_month', how='left')
    # Merge total_units
    df = df.merge(total_units, on='month_start_dt', how='left') # Use month_start_dt after rename

    # Fill missing values and cast to int.
    df['units_added'] = df['units_added'].fillna(0).astype(int)
    df['units_lost'] = df['units_lost'].fillna(0).astype(int)
    df['total_units'] = df['total_units'].fillna(0).astype(int)

    # Create a formatted month_start string.
    df['month_start'] = df['month_start_dt'].dt.strftime("%Y-%m-%d")
    df = df[['month_start', 'units_added', 'units_lost', 'total_units', 'month_start_dt']]

    logger.info(f"Finished processing units data, shape: {df.shape}")
    return df

def main(start_date, end_date):
    # st.header("Units Added, Lost, and Total Units per Month") # Removed - Redundant
    st.caption("Based on first/last rent transaction dates per unit. 'Lost' if no rent txn in last 75 days.") # Keep caption
    # st.write(f"Date Range: {start_date} to {end_date}") # Removed - Redundant
    logger.info(f"Running Units Added/Lost report for {start_date} to {end_date}")

    counts_df = load_units_data(start_date, end_date)
    if counts_df.empty:
        st.warning("No data available for the given date range.")
        logger.warning("No units data loaded or processed.")
        return

    # Create a figure with a secondary y-axis.
    logger.info("Generating plot...")
    try:
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Trace for units added (primary y-axis)
        fig.add_trace(go.Scatter(
            x=counts_df['month_start_dt'],
            y=counts_df['units_added'],
            mode='lines+markers',
            name='Units Added',
            line=dict(color='green')
        ), secondary_y=False)

        # Trace for units lost (primary y-axis)
        fig.add_trace(go.Scatter(
            x=counts_df['month_start_dt'],
            y=counts_df['units_lost'],
            mode='lines+markers',
            name='Units Lost',
            line=dict(color='red')
        ), secondary_y=False)

        # Trace for total units (secondary y-axis)
        fig.add_trace(go.Scatter(
            x=counts_df['month_start_dt'],
            y=counts_df['total_units'],
            mode='lines+markers',
            name='Total Units (with rent txn)', # Clarified label
            line=dict(color='blue')
        ), secondary_y=True)

        fig.update_layout(
            title="Units Added, Lost, and Total Units per Month", # Removed date range from title
            xaxis_title="Month",
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.2,
                xanchor="center",
                x=0.5
            )
        )

        fig.update_yaxes(title_text="Units Added / Lost", secondary_y=False, rangemode='tozero')
        fig.update_yaxes(title_text="Total Units", secondary_y=True, rangemode='tozero')

        st.plotly_chart(fig, use_container_width=True)

        # --- Add plain English description ---
        st.markdown("""
        **How this graph is created:**
        *   **Units Added (Green Line, Left Axis):** Shows the number of units that had their *first* recorded rent transaction (Charge, Payment, or Applied Prepayment) in that specific month.
        *   **Units Lost (Red Line, Left Axis):** Shows the number of units whose *last* recorded rent transaction occurred in that specific month, AND where today's date is more than 75 days after that last transaction date. This indicates units potentially lost from the portfolio.
        *   **Total Units (Blue Line, Right Axis):** Shows the total number of distinct units that had *any* rent transaction recorded during that specific month.
        """)
        # --- End description ---

    except Exception as e:
        logger.exception("Failed to generate plot.")
        st.error(f"Error generating plot: {e}")


    st.subheader("Data Table")
    try:
        display_df = counts_df[['month_start_dt', 'units_added', 'units_lost', 'total_units']].copy()
        display_df['Month'] = display_df['month_start_dt'].dt.strftime('%Y-%m')
        display_df = display_df[['Month', 'units_added', 'units_lost', 'total_units']]
        display_df.rename(columns={
            'units_added': 'Units Added',
            'units_lost': 'Units Lost',
            'total_units': 'Total Units (w/ Rent Txn)'
        }, inplace=True)
        st.dataframe(display_df.sort_values(by='Month'), hide_index=True)
    except Exception as e:
        logger.exception("Failed to display data table.")
        st.error(f"Error displaying data table: {e}")

    logger.info("Report execution finished.")

# --- Removed __main__ block ---
# if __name__ == "__main__":
#     from datetime import date
#     main(date(2020, 3, 10), date(2025, 3, 10))
