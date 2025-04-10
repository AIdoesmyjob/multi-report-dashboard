import streamlit as st
import pandas as pd
import logging
import numpy as np # Import numpy
from sqlalchemy import text
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import plotly.graph_objects as go

# Import shared functions from utils
from web_version.utils import get_engine

# Configure logging for this report module
logger = logging.getLogger(__name__)

@st.cache_data(ttl=600)
def load_data_delinquency(start_date, end_date):
    """
    Query general ledger transactions for rent charges and payments.
    Returns a DataFrame with monthly counts for delinquency calculation.
    """
    logger.info(f"Loading delinquency-related data from {start_date} to {end_date}")
    engine = get_engine()
    if engine is None:
        logger.error("Failed to get DB engine. Aborting report.")
        return pd.DataFrame()

    # Query 'Charge' and 'Payment' transaction types from general ledger
    # based on user-provided examples.
    query_gl = """
        SELECT
            transaction_date,
            transaction_type,
            total_amount,
            unit_id,
            unit_number
        FROM general_ledger_transactions
        WHERE (transaction_type ILIKE 'Charge' OR transaction_type ILIKE 'Payment') -- More specific types
          AND transaction_date BETWEEN :start_date AND :end_date
    """
    try:
        with engine.connect() as conn:
            gl_txn = pd.read_sql_query(
                text(query_gl),
                conn,
                params={'start_date': start_date, 'end_date': end_date}
            )
        logger.info(f"Loaded {len(gl_txn)} relevant general ledger transactions.")

        if gl_txn.empty:
            logger.warning("No relevant GL transactions found for the date range.")
            return pd.DataFrame()

        gl_txn['transaction_date'] = pd.to_datetime(gl_txn['transaction_date'], errors='coerce')
        gl_txn['total_amount'] = pd.to_numeric(gl_txn['total_amount'], errors='coerce')
        gl_txn = gl_txn.dropna(subset=['transaction_date', 'total_amount', 'unit_id']) # Drop rows where conversion failed or unit_id is missing

        # Create month column
        gl_txn['month'] = gl_txn['transaction_date'].dt.to_period('M').dt.start_time

        # Create flags based on exact transaction types (case-insensitive)
        gl_txn['is_charge'] = gl_txn['transaction_type'].str.lower() == 'charge'
        gl_txn['is_payment'] = gl_txn['transaction_type'].str.lower() == 'payment'

        # Keep the amount for charges - REMOVED as it's not needed for rate calc
        # gl_txn['charge_amount'] = gl_txn.apply(lambda row: row['total_amount'] if row['is_charge'] else 0, axis=1)

        # Group by month and unit, aggregating flags and charge amount
        monthly_unit_summary = gl_txn.groupby(['month', 'unit_id', 'unit_number']).agg(
            was_rent_charged=('is_charge', 'any'), # True if any charge transaction exists
            was_any_rent_paid=('is_payment', 'any'), # True if any payment transaction exists
            # We don't actually need total_rent_charged for the rate calculation
            # total_rent_charged=('charge_amount', 'sum')
        ).reset_index()

        # Now, aggregate by month to get the counts needed for the vacancy rate
        monthly_summary = monthly_unit_summary.groupby('month').agg(
            # Count units that had any rent charge (Managed Units)
            managed_units=('was_rent_charged', lambda x: x[x == True].count()),
            # Count units that were charged AND paid (Paid Units)
            paid_units=('was_any_rent_paid', lambda x: x[monthly_unit_summary.loc[x.index, 'was_rent_charged'] & (x == True)].count())
        ).reset_index()

        logger.info(f"Processed monthly delinquency counts across {monthly_summary['month'].nunique()} months.")
        return monthly_summary

    except Exception as e:
        logger.exception(f"Failed to load or process general ledger data: {e}")
        st.error(f"DB Error loading GL data: {e}")
        return pd.DataFrame()

def main(start_date, end_date):
    # st.header("Monthly Delinquency Rate (Based on GL Transactions)") # Removed - Redundant
    st.caption("Delinquency Rate = (Units Charged Rent - Units Paid Rent) / Units Charged Rent") # Keep formula caption
    # st.write(f"Date Range: {start_date} to {end_date}") # Removed - Redundant
    logger.info(f"Running Delinquency Rate report for {start_date} to {end_date}")

    monthly_counts_df = load_data_delinquency(start_date, end_date)

    if monthly_counts_df.empty:
        st.warning("No relevant General Ledger transaction data found for delinquency analysis in the given date range.")
        logger.warning("No delinquency data loaded or processed.")
        return

    # Calculate Delinquency Rate
    monthly_counts_df['delinquency_rate'] = monthly_counts_df.apply(
        lambda row: (row['managed_units'] - row['paid_units']) / row['managed_units'] if row['managed_units'] > 0 else 0,
        axis=1
    )

    # --- Display Results ---
    # st.subheader("Monthly Delinquency Rate Trend") # Removed - Redundant with plot title

    # --- Plotting ---
    try:
        # Ensure data is sorted by month for trendline calculation
        monthly_counts_df = monthly_counts_df.sort_values('month')

        fig = go.Figure()
        # Original data trace
        fig.add_trace(go.Scatter(
            x=monthly_counts_df['month'],
            y=monthly_counts_df['delinquency_rate'],
            mode='lines+markers',
            name='Delinquency Rate',
            line=dict(color='royalblue'), # Match color from first report
            fill='tozeroy', # Add fill
            fillcolor='rgba(65, 105, 225, 0.2)', # Semi-transparent royal blue
            hovertemplate = '<b>Month</b>: %{x|%Y-%m}<br><b>Delinquency Rate</b>: %{y:.2%}<extra></extra>' # Custom hover text
        ))
        fig.update_layout(
            title="Monthly Delinquency Rate Trend", # Re-add title as header was removed
            xaxis_title="Month",
            yaxis_title="Delinquency Rate",
            yaxis_tickformat=".0%", # Format y-axis as percentage
            # template="plotly_white", # Removed template
            hovermode="x unified" # Show hover info for the x-value
        )

        # --- Calculate and add Trendline ---
        # Convert dates to numerical format (e.g., days since the first date) for polyfit
        x_numeric = (monthly_counts_df['month'] - monthly_counts_df['month'].min()).dt.days
        y_data = monthly_counts_df['delinquency_rate']

        # Drop NaN values before fitting, if any
        valid_indices = ~np.isnan(x_numeric) & ~np.isnan(y_data)
        if valid_indices.sum() > 1: # Need at least 2 points to fit a line
            coeffs = np.polyfit(x_numeric[valid_indices], y_data[valid_indices], 1) # 1 for linear fit
            trendline_y = np.polyval(coeffs, x_numeric)

            fig.add_trace(go.Scatter(
                x=monthly_counts_df['month'],
                y=trendline_y,
                mode='lines',
                name='Trendline',
                line=dict(color='red', dash='dash'),
                hoverinfo='skip' # Don't show hover for trendline itself
            ))
        else:
            logger.warning("Not enough valid data points to calculate trendline.")
        # --- End Trendline ---

        st.plotly_chart(fig, use_container_width=True)

        # --- Add plain English description ---
        st.markdown("""
        **How this graph is created:**
        *   The **blue line** shows the percentage of units that were charged rent in a given month but did not have *any* payment transaction recorded within that same calendar month. It's calculated based on General Ledger transactions ('Charge' and 'Payment' types).
        *   A higher percentage indicates more units were delinquent on rent payments for that month.
        *   The **red dashed line** shows the overall linear trend (best-fit line) for this rate over the selected period.
        """)
        # --- End description ---

        # Optional: Display the counts used for calculation in an expander
        with st.expander("Show Monthly Counts"):
             display_counts = monthly_counts_df[['month', 'managed_units', 'paid_units', 'delinquency_rate']].copy()
             display_counts['month'] = display_counts['month'].dt.strftime('%Y-%m')
             display_counts['delinquency_rate'] = display_counts['delinquency_rate'].map('{:.2%}'.format)
             display_counts.rename(columns={
                 'month': 'Month',
                 'managed_units': 'Units Charged Rent',
                 'paid_units': 'Units Paid Rent',
                 'delinquency_rate': 'Delinquency Rate'
             }, inplace=True)
        st.dataframe(display_counts, hide_index=True, use_container_width=True)

    except Exception as e:
        logger.exception("Failed to generate delinquency rate plot.")
        st.error(f"Error generating plot: {e}")

    # st.caption(""" # Removed - Moved explanation to markdown above
    # **Note:** Delinquency Rate is calculated here as the percentage of units that were charged rent
    # in a given month but did not have any payment transaction recorded within that same calendar month.
    # It reflects payment status based strictly on transactions within the month.
    # """)

    logger.info("Delinquency Rate report execution finished.")

# Note: This report needs to be added to the main app.py to be selectable. (Already done implicitly)
