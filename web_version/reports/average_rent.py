import streamlit as st
import pandas as pd
import json
import logging # <-- Added
from sqlalchemy import text # <-- Removed create_engine
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import plotly.graph_objects as go

# Import shared functions from utils
from web_version.utils import get_engine # <-- Import shared engine function

# Configure logging for this report module
logger = logging.getLogger(__name__)


@st.cache_data(ttl=600) # Added caching decorator
def load_data_avg_rent(start_date, end_date):
    """
    Query rent-related transactions and lease details, then compute monthly:
      - overall_avg_rent
      - new_avg_rent (and its 3-month rolling average)
    Returns a DataFrame with columns including:
      [month_start_dt, overall_avg_rent, new_avg_rent, new_avg_rent_3m]
    """
    logger.info(f"Loading average rent data from {start_date} to {end_date}")
    engine = get_engine() # <-- Use shared engine function
    if engine is None:
        logger.error("Failed to get DB engine. Aborting report.")
        return pd.DataFrame()

    # Query rent-related transactions.
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
    # Removed commented-out date filters that were causing parameter errors
    try:
        with engine.connect() as conn:
            rent_txn = pd.read_sql_query(text(query_transactions), conn) # No params needed now
        logger.info(f"Loaded {len(rent_txn)} rent-related transactions.")
        rent_txn['transaction_date'] = pd.to_datetime(rent_txn['transaction_date'], errors='coerce')
        rent_txn['total_amount'] = pd.to_numeric(rent_txn['total_amount'], errors='coerce')
        rent_txn = rent_txn.dropna(subset=['transaction_date', 'total_amount']) # Drop rows where conversion failed
    except Exception as e:
        logger.exception("Failed to load rent transactions.")
        st.error(f"DB Error loading transactions: {e}")
        return pd.DataFrame()


    # Get first/last rent dates per lease.
    if not rent_txn.empty:
        grouped = rent_txn.groupby("lease_id")["transaction_date"].agg(
            first_rent_date="min",
            last_rent_date="max"
        ).reset_index()
    else:
        logger.warning("No valid rent transactions found.")
        grouped = pd.DataFrame(columns=["lease_id", "first_rent_date", "last_rent_date"])

    # Retrieve lease details.
    query_leases = """
        SELECT
            id AS lease_id,
            property_id,
            unit_id,
            lease_status,
            lease_from_date,
            lease_to_date
        FROM leases
        -- Removed problematic commented-out WHERE clause entirely
    """
    try:
        with engine.connect() as conn:
            leases_df = pd.read_sql_query(text(query_leases), conn) # No params needed now
        logger.info(f"Loaded {len(leases_df)} lease records.")
        leases_df['lease_from_date'] = pd.to_datetime(leases_df['lease_from_date'], errors='coerce')
        leases_df['lease_to_date'] = pd.to_datetime(leases_df['lease_to_date'], errors='coerce')
    except Exception as e:
        logger.exception("Failed to load leases.")
        st.error(f"DB Error loading leases: {e}")
        return pd.DataFrame()

    # Merge transactions with lease details.
    merged = leases_df.merge(grouped, on="lease_id", how="left")

    # Create a list of months based on the selected date range.
    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.min.time())
    start_period = start_dt.replace(day=1)
    # Ensure end_period calculation is correct
    end_period = (end_dt.replace(day=1) + relativedelta(months=1))

    months = []
    current = start_period
    while current < end_period:
        months.append(current)
        current += relativedelta(months=1)

    if not months:
         logger.warning("No months generated for the selected date range.")
         return pd.DataFrame()

    results = []
    logger.info(f"Analyzing {len(months)} months...")
    for month_start in months:
        month_start_date = month_start.date()
        month_end_date = (month_start + relativedelta(months=1) - timedelta(days=1)).date()
        logger.debug(f"Processing month: {month_start_date.strftime('%Y-%m')}")

        # Convert dates for comparison
        month_start_dt_pd = pd.to_datetime(month_start_date)
        month_end_dt_pd = pd.to_datetime(month_end_date)

        # Overall active leases: those with first_rent_date ≤ month_end and last_rent_date ≥ month_start.
        active_condition = (
            merged['first_rent_date'].notnull() &
            (merged['first_rent_date'] <= month_end_dt_pd) &
            merged['last_rent_date'].notnull() &
            (merged['last_rent_date'] >= month_start_dt_pd)
        )
        active_leases = merged[active_condition]
        active_count = active_leases.shape[0]

        # Sum rent transactions for "Charge" in the month for active leases.
        month_txn = rent_txn[
            (rent_txn['transaction_date'] >= month_start_dt_pd) &
            (rent_txn['transaction_date'] <= month_end_dt_pd) &
            (rent_txn['transaction_type'].str.contains('Charge', case=False, na=False))
        ]
        month_txn_active = month_txn[month_txn['lease_id'].isin(active_leases['lease_id'])]
        overall_rent_total = month_txn_active['total_amount'].sum()
        overall_avg_rent = overall_rent_total / active_count if active_count > 0 else 0.0

        # New leases: those with first_rent_date in the 3 months prior to month_start.
        new_lease_start_cutoff = month_start_dt_pd - relativedelta(months=3)
        new_condition = (
            merged['first_rent_date'].notnull() &
            (merged['first_rent_date'] >= new_lease_start_cutoff) &
            (merged['first_rent_date'] < month_start_dt_pd)
        )
        new_leases = merged[new_condition]
        new_count = new_leases.shape[0]

        # Filter transactions for these new leases within the current month
        new_txn = month_txn[month_txn['lease_id'].isin(new_leases['lease_id'])]
        new_rent_total = new_txn['total_amount'].sum()
        new_avg_rent = new_rent_total / new_count if new_count > 0 else 0.0

        results.append({
            'month_start_dt': month_start,
            'overall_avg_rent': overall_avg_rent,
            'new_avg_rent': new_avg_rent
        })
        logger.debug(f"Month {month_start_date.strftime('%Y-%m')}: Overall Avg={overall_avg_rent:.2f}, New Avg={new_avg_rent:.2f}")

    if not results:
         logger.warning("No results generated after processing months.")
         return pd.DataFrame()

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('month_start_dt')
    # Calculate rolling average safely
    if 'new_avg_rent' in results_df.columns:
        results_df['new_avg_rent_3m'] = results_df['new_avg_rent'].rolling(window=3, min_periods=1).mean()
    else:
        results_df['new_avg_rent_3m'] = 0.0 # Or np.nan

    logger.info(f"Finished calculating average rents, shape: {results_df.shape}")
    return results_df

def main(start_date, end_date):
    # st.header("Average Rent per Active Lease & New Leases") # Removed - Redundant
    # st.write(f"Date Range: {start_date} to {end_date}") # Removed - Redundant
    logger.info(f"Running Average Rent report for {start_date} to {end_date}")

    results_df = load_data_avg_rent(start_date, end_date)
    if results_df.empty:
        st.warning("No relevant rent data found in the given date range.")
        logger.warning("No average rent data loaded or processed.")
        return

    # Create a month label column for the x-axis.
    results_df['month_label'] = results_df['month_start_dt'].dt.strftime('%b %Y')

    logger.info("Generating plot...")
    try:
        fig = go.Figure()
        # Line for Overall Average Rent.
        fig.add_trace(go.Scatter( # Changed from Bar to Scatter
            x=results_df['month_label'],
            y=results_df['overall_avg_rent'],
            mode='lines+markers', # Added mode
            name='Overall Avg Rent',
            line=dict(color='#EB89B5'), # Use line dict for color
            fill='tozeroy', # Add fill
            fillcolor='rgba(235, 137, 181, 0.2)' # Semi-transparent pink
        ))
        # Line for New Lease 3-Month Rolling Average.
        fig.add_trace(go.Scatter( # Changed from Bar to Scatter
            x=results_df['month_label'],
            y=results_df['new_avg_rent_3m'],
            mode='lines+markers', # Added mode
            name='New Lease 3-Month Rolling Avg',
            line=dict(color='#330C73'), # Use line dict for color
            fill='tozeroy', # Add fill
            fillcolor='rgba(51, 12, 115, 0.2)' # Semi-transparent purple
        ))

        fig.update_layout(
            title_text="Monthly Average Rent Comparison",
            xaxis_title_text="Month",
            yaxis_title_text="Average Rent ($)", # Added units
            # barmode='group', # Removed barmode
            # template="plotly_white", # Removed template
            hovermode="x unified", # Added hovermode
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.2,
                xanchor="center",
                x=0.5
            )
        )
        fig.update_yaxes(tickprefix="$", rangemode='tozero') # Add dollar prefix

        st.plotly_chart(fig, use_container_width=True)

        # --- Add plain English description ---
        st.markdown("""
        **How this graph is created:**
        *   The **pink line** shows the average monthly rent charged across all active leases during that month.
        *   The **purple line** shows a 3-month rolling average of the rent charged for *new* leases (leases that started within the 3 months prior to the displayed month). This helps smooth out fluctuations and show the trend in rent for new tenants.
        """)
        # --- End description ---

    except Exception as e:
        error_msg = f"Failed to generate plot: {e}"
        logger.exception(error_msg)
        st.error(error_msg)

    st.subheader("Rent Data Details")
    try:
        # Format for display
        display_df = results_df[['month_label', 'overall_avg_rent', 'new_avg_rent', 'new_avg_rent_3m']].copy()
        display_df['overall_avg_rent'] = display_df['overall_avg_rent'].map('${:,.2f}'.format)
        display_df['new_avg_rent'] = display_df['new_avg_rent'].map('${:,.2f}'.format)
        display_df['new_avg_rent_3m'] = display_df['new_avg_rent_3m'].map('${:,.2f}'.format)
        display_df.rename(columns={
            'month_label': 'Month',
            'overall_avg_rent': 'Overall Avg Rent',
            'new_avg_rent': 'New Lease Avg Rent (Month)',
            'new_avg_rent_3m': 'New Lease Avg Rent (3M Rolling)'
        }, inplace=True)
        st.dataframe(display_df, hide_index=True)
    except Exception as e:
        error_msg = f"Failed to display data table: {e}"
        logger.exception(error_msg)
        st.error(error_msg)

    logger.info("Report execution finished.")

# Removed __main__ block
