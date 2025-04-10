import streamlit as st
import pandas as pd
import json
import logging
from sqlalchemy import text
from datetime import datetime
from dateutil.relativedelta import relativedelta
import plotly.graph_objects as go

# Import shared functions from utils
from web_version.utils import get_engine

# Configure logging for this report module
logger = logging.getLogger(__name__)

# GL account names considered part of management fees.
# --- Restored full list - User needs to confirm which are INCOME accounts ---
FEE_ACCOUNT_NAMES = {
    "Management Fee - 020",
    "Management Fee - 030",
    "Management Fee - 120",
    "Management Fee - 130",
    "Management Fee - Sutton 2022",
    "Management Fees - Abbotsford",
    "Management Fees - Chilliwack",
    "Inspection - Income",
    "Lease Up Fee",
}
# logger.info(f"Using FEE_ACCOUNT_NAMES: {FEE_ACCOUNT_NAMES}") # Keep commented unless debugging

# Cache the data result; the returned DataFrame is pickleable.
@st.cache_data(ttl=600)
def load_data(): # Removed start_date, end_date parameters
    """
    Queries general_ledger_transactions for the CURRENT MONTH, filters for lines
    matching FEE_ACCOUNT_NAMES, extracts property_id, and groups the amounts by property.

    Returns a DataFrame with columns:
       [year_month, group_name, total_fees, year_month_dt]
    """
    # --- Calculate start and end of PREVIOUS month ---
    today = datetime.today().date()
    start_of_current_month = today.replace(day=1)
    end_of_previous_month = start_of_current_month - relativedelta(days=1)
    start_of_previous_month = end_of_previous_month.replace(day=1)
    logger.info(f"Loading management fee data by property for the previous month: {start_of_previous_month} to {end_of_previous_month}")
    engine = get_engine()
    if engine is None:
        logger.error("Failed to get DB engine for management fees by group.")
        return pd.DataFrame()

    # --- FIXED QUERY: Join GL Transactions -> Units -> Properties -> Groups ---
    query = text("""
        SELECT
            glt.id,
            glt.transaction_date,
            glt.journal,
            glt.unit_id, -- Keep unit_id for reference if needed
            COALESCE(pg.name, 'No Group Assigned') AS group_name -- Get group name directly
        FROM general_ledger_transactions glt
        LEFT JOIN rental_units ru ON glt.unit_id = ru.id
        LEFT JOIN rental_properties rp ON ru.property_id = rp.id
        LEFT JOIN property_group_memberships pgm ON rp.id = pgm.property_id
        LEFT JOIN property_groups pg ON pgm.property_group_id = pg.id
        WHERE glt.transaction_date >= :start_date
          AND glt.transaction_date <= :end_date
          -- AND glt.unit_id IS NOT NULL -- Removed filter to include transactions without unit_id
        ORDER BY glt.id;
    """)
    logger.warning("Running query without unit_id filter to include all relevant fee transactions.") # Add warning log
    try:
        with engine.connect() as conn:
            # Use calculated dates for PREVIOUS month in params
            df = pd.read_sql_query(query, conn, params={"start_date": start_of_previous_month, "end_date": end_of_previous_month})
        logger.info(f"Retrieved {df.shape[0]} GL transactions with unit_id and joined group_name from the database for the previous month.")
    except Exception as e:
        # No longer need to check for property_id error specifically
        logger.exception("Failed to load GL transactions with joins.")
        st.error(f"DB Error loading GL transactions: {e}")
        # Return empty df with expected columns for downstream processing
        return pd.DataFrame(columns=['year_month', 'group_name', 'total_fees', 'year_month_dt'])

    # Convert transaction_date to datetime.
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    # Ensure group_name is filled for any potential missed joins (though COALESCE should handle it)
    df['group_name'] = df['group_name'].fillna('Unknown Group')

    fee_entries = []
    json_parse_errors = 0
    for idx, row in df.iterrows():
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
            if account_name in FEE_ACCOUNT_NAMES:
                amount = line.get("Amount", 0.0)
                try:
                    numeric_amount = pd.to_numeric(amount, errors='coerce')
                    if pd.isna(numeric_amount):
                         logger.warning(f"Row {idx}, Txn ID {row['id']}: Non-numeric amount '{amount}' for account '{account_name}'. Skipping line.")
                         continue
                except Exception as parse_e:
                     logger.warning(f"Row {idx}, Txn ID {row['id']}: Error converting amount '{amount}' to numeric: {parse_e}. Skipping line.")
                     continue

                # --- FIXED: Use group_name from the joined query ---
                fee_entries.append({
                    "transaction_date": row["transaction_date"],
                    "account_name": account_name,
                    "amount": numeric_amount,
                    "transaction_id": row["id"],
                    "group_name": row["group_name"] # Use group_name directly
                })
                # Don't break, multiple fee lines could exist in one transaction

    if json_parse_errors > 0:
         logger.warning(f"Total JSON parsing errors encountered: {json_parse_errors}")
         st.toast(f"Note: Encountered {json_parse_errors} errors parsing journal data. See logs for details.", icon="⚠️")

    if not fee_entries:
        logger.warning("No fee entries found after processing GL transactions.")
        # Return empty df with expected columns
        return pd.DataFrame(columns=['year_month', 'group_name', 'total_fees', 'year_month_dt'])

    fee_df = pd.DataFrame(fee_entries)
    logger.info(f"Found {len(fee_df)} fee entries matching specified accounts with group names.")

    # Create a 'year_month' column (e.g., '2023-05').
    fee_df["year_month"] = fee_df["transaction_date"].dt.to_period("M").astype(str)

    # --- FIXED: Group by year_month AND group_name ---
    monthly_sums_group = fee_df.groupby(["year_month", "group_name"])["amount"].sum().reset_index()
    monthly_sums_group.rename(columns={"amount": "total_fees"}, inplace=True)

    # Create a datetime column for plotting.
    monthly_sums_group["year_month_dt"] = pd.to_datetime(monthly_sums_group["year_month"] + "-01", format="%Y-%m-%d")
    monthly_sums_group.sort_values(["year_month_dt", "group_name"], inplace=True) # Sort by date then group

    logger.info(f"Finished processing management fee data by group, shape: {monthly_sums_group.shape}")
    return monthly_sums_group


# --- REMOVED load_property_groups function ---


def main(start_date, end_date): # Keep parameters for compatibility with app.py call signature
    # --- Calculate previous month for display ---
    today = datetime.today().date()
    start_of_current_month = today.replace(day=1)
    end_of_previous_month = start_of_current_month - relativedelta(days=1)
    previous_month_str = end_of_previous_month.strftime("%B %Y") # Format previous month
    logger.info(f"Running Management Fees by Property Group report for PREVIOUS MONTH ({previous_month_str})")

    # --- FIXED: Load data directly grouped by month/group ---
    monthly_sums_group = load_data() # Now returns data already grouped for previous month

    # --- REMOVED call to load_property_groups ---
    # --- REMOVED merge operation ---

    if monthly_sums_group.empty:
        st.warning(f"No relevant fee entries found for the previous month ({previous_month_str}).")
        logger.warning(f"No management fee data by group loaded or processed for the previous month ({previous_month_str}).")
        return

    # --- Filter for specific management groups ---
    target_groups = {
        "Elena Morris (Management Group)",
        "Jake Reimer (Management Group)",
        "Matthew Stevenson ( Management Group )" # Note the potential extra space at the end
    }
    logger.info(f"Filtering report data for target groups: {target_groups}")
    filtered_sums_group = monthly_sums_group[monthly_sums_group['group_name'].isin(target_groups)].copy()

    if filtered_sums_group.empty:
        st.warning(f"No relevant fee entries found for the specified management groups in the previous month ({previous_month_str}).")
        logger.warning(f"No data found after filtering for target groups: {target_groups}")
        # Optionally, display the unfiltered table if no target groups found? Or just stop? Stopping for now.
        return

    # Data is now filtered and aggregated in filtered_sums_group

    # --- Plotting - Uses filtered_sums_group ---
    logger.info("Generating plot for filtered data...")
    try:
        fig = go.Figure()
        # Use filtered data for plotting
        groups = filtered_sums_group['group_name'].unique()

        # Create a trace for each group
        for group in sorted(groups):
            # Use filtered data here
            group_data = filtered_sums_group[filtered_sums_group['group_name'] == group]
            fig.add_trace(go.Bar(
                x=group_data['year_month_dt'],
                y=group_data['total_fees'],
                name=group,
                hovertemplate = f'<b>Group</b>: {group}<br><b>Month</b>: %{{x|%Y-%m}}<br><b>Fees</b>: $%{{y:,.2f}}<extra></extra>'
            ))

        fig.update_layout(
            barmode='stack', # Stack bars
            title=f"Previous Month ({previous_month_str}) Management Fees by Property Group", # Updated title
            xaxis_title="Property Group", # X-axis is group name for single month view
            yaxis_title="Total Fees ($)",
            legend_title="Property Group",
            hovermode="x unified"
        )
        fig.update_yaxes(tickprefix="$", rangemode='tozero')

        st.plotly_chart(fig, use_container_width=True)

        # --- Add plain English description ---
        st.markdown(f"""
        **How this graph is created:**
        *   This chart displays the total management-related fees collected **for the previous calendar month ({previous_month_str})**, broken down by the Property Group each property belongs to.
        *   We identify relevant fees by looking at General Ledger transactions within that month that are posted to specific income accounts like 'Management Fee - *', 'Inspection - Income', and 'Lease Up Fee'.
        *   Each fee transaction is linked to a specific property via its associated Unit (`unit_id`). We then determine the Property Group for that property.
        *   Finally, we sum up all the identified fees for each Property Group to get the totals shown in the bars above.
        *   Fees from transactions that aren't linked to a property, or properties not assigned to a group, are categorized as 'Unknown' or 'No Group Assigned'.
        """)
        # --- End description ---

    except Exception as e:
        logger.exception("Failed to generate plot.")
        st.error(f"Error generating plot: {e}")

    # --- MODIFIED: Data Table - Grouped ---
    st.subheader("Data Table (Grouped by Month and Property Group)") # Keep this generic title
    try:
        # --- MODIFIED: Data Table - Simplified for single month ---
        st.subheader(f"Data Table (Previous Month: {previous_month_str})") # Update subheader
        # Use filtered data for table
        display_df = filtered_sums_group[['group_name', 'total_fees']].copy()
        display_df.rename(columns={'group_name': 'Property Group', 'total_fees': 'Total Fees'}, inplace=True)
        display_df['Total Fees'] = display_df['Total Fees'].map('${:,.2f}'.format)
        st.dataframe(display_df.sort_values(by='Property Group'), hide_index=True)

    except Exception as e:
        logger.exception("Failed to display data table.")
        st.error(f"Error displaying data table: {e}")

    logger.info("Report execution finished.")

# Note: No __main__ block needed as this will be called by app.py
