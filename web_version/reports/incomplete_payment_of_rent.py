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
def load_data_incomplete_payment(start_date, end_date):
    """
    Query general ledger transactions for rent charges and payments.
    Calculates total charged and total paid per unit per month.
    Returns a DataFrame with monthly counts for incomplete payment calculation.
    """
    logger.info(f"Loading incomplete payment data from {start_date} to {end_date}")
    engine = get_engine()
    if engine is None:
        logger.error("Failed to get DB engine. Aborting report.")
        return pd.DataFrame()

    # Query 'Charge' and 'Payment' transaction types from general ledger
    query_gl = """
        SELECT
            transaction_date,
            transaction_type,
            total_amount,
            unit_id,
            unit_number
        FROM general_ledger_transactions
        WHERE (transaction_type ILIKE 'Charge' OR transaction_type ILIKE 'Payment')
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
        gl_txn = gl_txn.dropna(subset=['transaction_date', 'total_amount', 'unit_id'])

        # Create month column
        gl_txn['month'] = gl_txn['transaction_date'].dt.to_period('M').dt.start_time

        # Separate charges and payments
        charges = gl_txn[gl_txn['transaction_type'].str.lower() == 'charge'].copy()
        payments = gl_txn[gl_txn['transaction_type'].str.lower() == 'payment'].copy()

        # Sum charges per unit per month
        monthly_unit_charges = charges.groupby(['month', 'unit_id', 'unit_number'])['total_amount'].sum().reset_index()
        monthly_unit_charges.rename(columns={'total_amount': 'total_charged'}, inplace=True)

        # Sum payments per unit per month
        monthly_unit_payments = payments.groupby(['month', 'unit_id'])['total_amount'].sum().reset_index()
        monthly_unit_payments.rename(columns={'total_amount': 'total_paid'}, inplace=True)

        # Merge charges and payments
        monthly_unit_summary = pd.merge(
            monthly_unit_charges,
            monthly_unit_payments,
            on=['month', 'unit_id'],
            how='left' # Keep all units that were charged
        )
        # Fill NaN payments with 0
        monthly_unit_summary['total_paid'] = monthly_unit_summary['total_paid'].fillna(0)

        # Identify units with incomplete payment (paid < 90% of charged)
        # Ensure total_charged is positive to avoid division by zero or negative issues
        monthly_unit_summary['incomplete_payment'] = (
            (monthly_unit_summary['total_charged'] > 0) &
            (monthly_unit_summary['total_paid'] < 0.9 * monthly_unit_summary['total_charged'])
        )

        # Aggregate by month
        monthly_summary = monthly_unit_summary.groupby('month').agg(
            # Count units that had any rent charge
            units_charged=('unit_id', 'nunique'), # Count distinct units charged
            # Count units with incomplete payment
            incomplete_payment_units=('incomplete_payment', 'sum')
        ).reset_index()

        logger.info(f"Processed monthly incomplete payment counts across {monthly_summary['month'].nunique()} months.")
        return monthly_summary

    except Exception as e:
        logger.exception(f"Failed to load or process general ledger data: {e}")
        st.error(f"DB Error loading GL data: {e}")
        return pd.DataFrame()

def main(start_date, end_date):
    # st.header("Incomplete Payment of Rent Rate") # Removed - Redundant
    st.caption("Rate = Units Paying < 90% of Rent Charged / Total Units Charged Rent") # Keep formula
    logger.info(f"Running Incomplete Payment of Rent Rate report for {start_date} to {end_date}")
    # Trivial change to attempt triggering a refresh

    monthly_counts_df = load_data_incomplete_payment(start_date, end_date)

    if monthly_counts_df.empty:
        st.warning("No relevant General Ledger transaction data found for incomplete payment analysis in the given date range.")
        logger.warning("No incomplete payment data loaded or processed.")
        return

    # Calculate Incomplete Payment Rate
    monthly_counts_df['incomplete_payment_rate'] = monthly_counts_df.apply(
        lambda row: row['incomplete_payment_units'] / row['units_charged'] if row['units_charged'] > 0 else 0,
        axis=1
    )

    # --- Display Results ---
    # st.subheader("Monthly Incomplete Payment Rate Trend") # Removed - Redundant with plot title

    # --- Plotting ---
    try:
        # Ensure data is sorted by month for trendline calculation
        monthly_counts_df = monthly_counts_df.sort_values('month')

        fig = go.Figure()
        # Original data trace
        fig.add_trace(go.Scatter(
            x=monthly_counts_df['month'],
            y=monthly_counts_df['incomplete_payment_rate'],
            mode='lines+markers',
            name='Incomplete Payment Rate',
            line=dict(color='royalblue'), # Standard blue color
            fill='tozeroy', # Add fill
            fillcolor='rgba(65, 105, 225, 0.2)', # Semi-transparent blue
            hovertemplate = '<b>Month</b>: %{x|%Y-%m}<br><b>Rate</b>: %{y:.2%}<extra></extra>' # Custom hover text
        ))
        fig.update_layout(
            title="Monthly Incomplete Payment Rate Trend",
            xaxis_title="Month",
            yaxis_title="Incomplete Payment Rate",
            yaxis_tickformat=".0%", # Format y-axis as percentage
            hovermode="x unified" # Show hover info for the x-value
        )

        # --- Calculate and add Trendline ---
        # Convert dates to numerical format (e.g., days since the first date) for polyfit
        x_numeric = (monthly_counts_df['month'] - monthly_counts_df['month'].min()).dt.days
        y_data = monthly_counts_df['incomplete_payment_rate']

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
        *   The **blue line** shows the percentage of units that paid less than 90% of the total rent charged to them within a given calendar month. It's calculated based on summing 'Charge' and 'Payment' transactions from the General Ledger for each unit within the month.
        *   A higher percentage indicates more units made significantly incomplete rent payments for that month.
        *   The **red dashed line** shows the overall linear trend (best-fit line) for this rate over the selected period.
        """)
        # --- End description ---

        # Optional: Display the counts used for calculation in an expander
        with st.expander("Show Monthly Counts"):
             display_counts = monthly_counts_df[['month', 'units_charged', 'incomplete_payment_units', 'incomplete_payment_rate']].copy()
             display_counts['month'] = display_counts['month'].dt.strftime('%Y-%m')
             display_counts['incomplete_payment_rate'] = display_counts['incomplete_payment_rate'].map('{:.2%}'.format)
             display_counts.rename(columns={
                 'month': 'Month',
                 'units_charged': 'Units Charged Rent',
                 'incomplete_payment_units': 'Units Paying < 90%',
                 'incomplete_payment_rate': 'Incomplete Payment Rate'
             }, inplace=True)
             st.dataframe(display_counts, hide_index=True, use_container_width=True)

    except Exception as e:
        logger.exception("Failed to generate incomplete payment rate plot.")
        st.error(f"Error generating plot: {e}")

    logger.info("Incomplete Payment Rate report execution finished.")
