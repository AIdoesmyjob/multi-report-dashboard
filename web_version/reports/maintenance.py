import streamlit as st
import pandas as pd
import json
import logging # <-- Added
from sqlalchemy import text # <-- Removed create_engine
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
# import plotly.express as px # Switch to graph_objects for fill
import plotly.graph_objects as go # Import graph_objects

# Import shared functions from utils
from web_version.utils import get_engine # <-- Import shared engine function

# Configure logging for this report module
logger = logging.getLogger(__name__)

# GL accounts for maintenance-related expenses.
MAINTENANCE_ACCOUNTS = {
    "Maintenance Expense",
    "Advertising",
    "Appliance Repair",
    "Appliance replacement",
    "Arbitration Filing Fee",
    "Carpet Cleaning",
    "Carpet Installation",
    "Cleaning",
    "Drain Cleaning",
    "Electrical Repairs",
    "Electricity Bill",
    "Elevator Expenses",
    "Elevator Maintenance",
    "Fence Rental",
    "Fire & Safety",
    "Flooring",
    "Garbage Removal",
    "Gas Bill",
    "Glass Repair",
    # "Inspections",  # Excluded here
    "Locksmith",
    "Painting",
    "Pest Control",
    "Plumbing & Heating",
    "Registered Mail",
    "Repair & Maintenance",
    "Restoration",
    "Revenue Canada",
    "Roof Cleaning",
    "Roof Repair",
    "Salting",
    "Security",
    "Security Deposit Returns",
    "Site Security",
    "Snow Removal",
    "Strata Fees",
    "Taxes",
    "Tree Service",
    "Utility Bill",
    "Vent Cleaning",
    "Yard Maintenance"
}

# BILL_TYPES indicates the transaction types that represent the actual expense.
BILL_TYPES = {"Bill", "Charge"}

# --- Removed report-specific engine function ---
# @st.cache_resource
# def get_engine():
#     return create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")

# Cache the resulting DataFrame.
@st.cache_data(ttl=600)
def load_maintenance_data(start_date, end_date):
    """
    Queries general_ledger_transactions for the given date range, filters to include only
    rows with transaction_type in BILL_TYPES, then for each row parses the journal field.
    If any line's GLAccount name is in MAINTENANCE_ACCOUNTS, its amount is recorded.
    The amounts are then grouped by month.

    Returns a DataFrame with columns:
       [year_month, total_maintenance, year_month_dt]
    """
    logger.info(f"Loading maintenance data from {start_date} to {end_date}")
    engine = get_engine() # <-- Use shared engine
    if engine is None:
        logger.error("Failed to get DB engine for maintenance.")
        return pd.DataFrame()

    query = text("""
        SELECT
            id,
            transaction_date,
            transaction_type,
            total_amount,
            journal
        FROM general_ledger_transactions
        WHERE transaction_date >= :start_date
          AND transaction_date <= :end_date
        ORDER BY id
    """)
    try:
        with engine.connect() as conn:
            df = pd.read_sql_query(query, conn, params={"start_date": start_date, "end_date": end_date})
        logger.info(f"Retrieved {df.shape[0]} GL transactions from the database.")
    except Exception as e:
         logger.exception("Failed to load GL transactions.")
         st.error(f"DB Error loading GL transactions: {e}")
         return pd.DataFrame()

    # Convert transaction_date to datetime.
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    # st.write(f"Retrieved {df.shape[0]} transactions from the database.") # Replaced with logging

    maintenance_entries = []
    json_parse_errors = 0
    for idx, row in df.iterrows():
        ttype = row["transaction_type"]
        if ttype not in BILL_TYPES:
            continue

        journal_data = row["journal"]
        if not journal_data:
            continue

        # Parse journal field
        parsed = None
        if isinstance(journal_data, dict):
            parsed = journal_data
        elif isinstance(journal_data, str):
            try:
                parsed = json.loads(journal_data)
            except Exception as e:
                if json_parse_errors < 5:
                     logger.warning(f"Row {idx}: JSON parsing error - {e}. Data: {journal_data[:100]}...")
                json_parse_errors += 1
                continue
        else:
            logger.warning(f"Row {idx}: Unexpected journal data type: {type(journal_data)}")
            continue

        if parsed is None: continue

        lines = parsed.get("Lines", [])
        if not isinstance(lines, list):
             logger.warning(f"Row {idx}: 'Lines' field is not a list: {type(lines)}")
             continue

        for line in lines:
            if not isinstance(line, dict):
                 logger.warning(f"Row {idx}: Journal line is not a dictionary: {type(line)}")
                 continue

            gl_account = line.get("GLAccount")
            if not isinstance(gl_account, dict):
                 continue

            account_name = gl_account.get("Name", "")
            if account_name in MAINTENANCE_ACCOUNTS:
                amount = line.get("Amount", 0.0)
                try:
                    numeric_amount = pd.to_numeric(amount, errors='coerce')
                    if pd.isna(numeric_amount):
                         logger.warning(f"Row {idx}, Txn ID {row['id']}: Non-numeric amount '{amount}' for account '{account_name}'. Skipping line.")
                         continue
                except Exception as parse_e:
                     logger.warning(f"Row {idx}, Txn ID {row['id']}: Error converting amount '{amount}' to numeric: {parse_e}. Skipping line.")
                     continue

                maintenance_entries.append({
                    "transaction_date": row["transaction_date"],
                    "transaction_type": ttype,
                    "account_name": account_name,
                    "amount": numeric_amount,
                    "transaction_id": row["id"]
                })
                # Don't break here, a single transaction might have multiple maintenance lines

    if json_parse_errors > 0:
         logger.warning(f"Total JSON parsing errors encountered: {json_parse_errors}")
         st.toast(f"Note: Encountered {json_parse_errors} errors parsing journal data. See logs for details.", icon="⚠️")

    if not maintenance_entries:
        logger.warning("No maintenance entries found after processing GL transactions.")
        # st.write("No maintenance entries found in the given date range.") # Replaced with logging
        return pd.DataFrame()

    maintenance_df = pd.DataFrame(maintenance_entries)
    logger.info(f"Found {maintenance_df.shape[0]} maintenance lines (Bill/Charge only).")
    # st.write(f"Found {maintenance_df.shape[0]} maintenance lines (Bill/Charge only).") # Replaced with logging

    # Create a 'year_month' column (e.g., '2023-05').
    maintenance_df["year_month"] = maintenance_df["transaction_date"].dt.to_period("M").astype(str)
    monthly_sums = maintenance_df.groupby("year_month")["amount"].sum().reset_index()
    monthly_sums.rename(columns={"amount": "total_maintenance"}, inplace=True)

    # Create a datetime column for plotting.
    monthly_sums["year_month_dt"] = pd.to_datetime(monthly_sums["year_month"] + "-01", format="%Y-%m-%d")
    monthly_sums.sort_values("year_month_dt", inplace=True)

    logger.info(f"Finished processing maintenance data, shape: {monthly_sums.shape}")
    return monthly_sums

def main(start_date, end_date):
    # st.header("Monthly Maintenance Costs (Bills/Charges Only)") # Removed - Redundant
    # st.write(f"Date Range: {start_date} to {end_date}") # Removed - Redundant
    logger.info(f"Running Maintenance report for {start_date} to {end_date}")

    monthly_sums = load_maintenance_data(start_date, end_date)
    if monthly_sums.empty:
        st.warning("No maintenance entries found for the given date range.")
        logger.warning("No maintenance data loaded or processed.")
        return

    # Create a line graph using Plotly graph_objects.
    logger.info("Generating plot...")
    try:
        fig = go.Figure() # Use go.Figure

        fig.add_trace(go.Scatter( # Use go.Scatter
            x=monthly_sums["year_month_dt"],
            y=monthly_sums["total_maintenance"],
            mode='lines+markers', # Specify mode
            name='Maintenance Costs',
            line=dict(color='royalblue'), # Use same blue
            fill='tozeroy', # Add fill
            fillcolor='rgba(65, 105, 225, 0.2)', # Semi-transparent blue
            hovertemplate = '<b>Month</b>: %{x|%Y-%m}<br><b>Cost</b>: $%{y:,.2f}<extra></extra>' # Custom hover
        ))

        fig.update_layout(
            title="Monthly Maintenance Costs", # Keep title
            xaxis_title="Month",
            yaxis_title="Total Maintenance Costs ($)", # Added units
            # template="plotly_white", # Removed template
            hovermode="x unified" # Add hovermode
        )
        fig.update_yaxes(tickprefix="$", rangemode='tozero') # Add dollar prefix

        st.plotly_chart(fig, use_container_width=True)

        # --- Add plain English description ---
        st.markdown("""
        **How this graph is created:**
        *   This graph shows the total monthly costs recorded under various maintenance-related General Ledger accounts (e.g., 'Repairs & Maintenance', 'Plumbing & Heating', 'Electrical Repairs', etc.).
        *   It only includes costs originating from transactions marked as 'Bill' or 'Charge' to avoid double-counting payments.
        *   The costs are extracted by looking into the 'journal' details of each relevant transaction.
        """)
        # --- End description ---

    except Exception as e:
        logger.exception("Failed to generate plot.")
        st.error(f"Error generating plot: {e}")

    st.subheader("Maintenance Data Table")
    try:
        # Format for display
        display_df = monthly_sums[['year_month_dt', 'total_maintenance']].copy()
        display_df['Month'] = display_df['year_month_dt'].dt.strftime('%Y-%m')
        display_df['Total Costs'] = display_df['total_maintenance'].map('${:,.2f}'.format)
        display_df = display_df[['Month', 'Total Costs']]
        st.dataframe(display_df.sort_values(by='Month'), hide_index=True)
    except Exception as e:
        logger.exception("Failed to display data table.")
        st.error(f"Error displaying data table: {e}")

    logger.info("Report execution finished.")

# --- Removed __main__ block ---
# if __name__ == "__main__":
#     from datetime import date
#     main(date(2020, 3, 10), date(2025, 3, 10))
