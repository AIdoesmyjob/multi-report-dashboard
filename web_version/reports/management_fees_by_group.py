import streamlit as st
import pandas as pd
import json
import logging
from sqlalchemy import text, exc as sqlalchemy_exc # Import sqlalchemy exceptions
from datetime import datetime
from dateutil.relativedelta import relativedelta
import plotly.graph_objects as go

# Import shared functions from utils
from web_version.utils import get_engine

# Configure logging for this report module
logger = logging.getLogger(__name__)

# GL account names considered part of management fees.
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
    Queries general_ledger_transactions for the PREVIOUS MONTH, filters for lines
    matching FEE_ACCOUNT_NAMES, determines the correct unit_id (from table or journal),
    joins through units -> properties -> groups, and aggregates the amounts.

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

    # --- CORRECTED QUERY V3: Link via unit_id (from table or journal) ---
    query = text("""
        WITH RelevantLines AS (
            -- Expand journal lines and filter for fee accounts
            SELECT
                glt.id AS transaction_id,
                glt.transaction_date,
                line,
                line -> 'GLAccount' ->> 'Name' AS account_name,
                -- Determine the unit_id to use: glt.unit_id first, then fallback to journal's AccountingEntity->Unit->Id
                COALESCE(
                    glt.unit_id,
                    (line -> 'AccountingEntity' -> 'Unit' ->> 'Id')::integer
                ) AS effective_unit_id
            FROM general_ledger_transactions glt,
                 jsonb_array_elements(glt.journal -> 'Lines') AS line
            WHERE glt.transaction_date >= :start_date
              AND glt.transaction_date <= :end_date
              AND glt.journal IS NOT NULL
              AND jsonb_typeof(glt.journal -> 'Lines') = 'array' -- Ensure Lines is an array
              AND line -> 'GLAccount' ->> 'Name' IN :fee_names
        )
        -- Join using the effective_unit_id to get property and then group
        SELECT
            rl.transaction_id,
            rl.transaction_date,
            rl.account_name,
            rl.line ->> 'Amount' AS amount_str, -- Select amount as string for now
            COALESCE(pg.name, 'No Group Assigned') AS group_name
        FROM RelevantLines rl
        LEFT JOIN rental_units ru ON rl.effective_unit_id = ru.id -- Join using effective_unit_id
        LEFT JOIN rental_properties rp ON ru.property_id = rp.id -- Join units to properties
        LEFT JOIN property_group_memberships pgm ON rp.id = pgm.property_id -- Join properties to memberships
        LEFT JOIN property_groups pg ON pgm.property_group_id = pg.id -- Join memberships to groups
        -- WHERE rl.effective_unit_id IS NOT NULL -- Optionally filter out if unit couldn't be determined at all
        ORDER BY rl.transaction_id;
    """)
    logger.info("Running query joining via effective unit_id (table or journal).")
    try:
        with engine.connect() as conn:
            # Use calculated dates for PREVIOUS month in params
            # Pass FEE_ACCOUNT_NAMES as a tuple for the IN clause
            df = pd.read_sql_query(query, conn, params={
                "start_date": start_of_previous_month,
                "end_date": end_of_previous_month,
                "fee_names": tuple(FEE_ACCOUNT_NAMES)
            })
        logger.info(f"Retrieved {df.shape[0]} relevant GL lines joined with group_name from the database for the previous month.")
    except sqlalchemy_exc.DataError as e:
        # Catch potential casting errors like invalid integer format for unit/property ID
        logger.exception("DataError during SQL query execution. Check JSON structure/IDs.")
        st.error(f"DB Data Error: Failed processing transaction links. Check data consistency. Details: {e}")
        return pd.DataFrame(columns=['year_month', 'group_name', 'total_fees', 'year_month_dt'])
    except Exception as e:
        logger.exception("Failed to load GL transactions with joins.")
        st.error(f"DB Error loading GL transactions: {e}")
        # Return empty df with expected columns for downstream processing
        return pd.DataFrame(columns=['year_month', 'group_name', 'total_fees', 'year_month_dt'])

    # Convert transaction_date to datetime.
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    # Ensure group_name is filled for any potential missed joins (though COALESCE should handle it)
    df['group_name'] = df['group_name'].fillna('Unknown Group') # Should be covered by SQL COALESCE, but belt-and-suspenders

    # Process the extracted lines
    fee_entries = []
    for idx, row in df.iterrows():
        amount_str = row["amount_str"]
        account_name = row["account_name"]
        try:
            numeric_amount = pd.to_numeric(amount_str, errors='coerce')
            if pd.isna(numeric_amount):
                 logger.warning(f"Txn ID {row['transaction_id']}: Non-numeric amount '{amount_str}' for account '{account_name}'. Skipping line.")
                 continue
        except Exception as parse_e:
             logger.warning(f"Txn ID {row['transaction_id']}: Error converting amount '{amount_str}' to numeric: {parse_e}. Skipping line.")
             continue

        fee_entries.append({
            "transaction_date": row["transaction_date"],
            "account_name": account_name, # Keep for potential debugging
            "amount": numeric_amount,
            "transaction_id": row["transaction_id"], # Keep for potential debugging
            "group_name": row["group_name"]
        })

    if not fee_entries:
        logger.warning("No fee entries found after processing GL transactions.")
        # Return empty df with expected columns
        return pd.DataFrame(columns=['year_month', 'group_name', 'total_fees', 'year_month_dt'])

    fee_df = pd.DataFrame(fee_entries)
    logger.info(f"Processed {len(fee_df)} fee entries matching specified accounts with group names.")

    # Create a 'year_month' column (e.g., '2023-05').
    fee_df["year_month"] = fee_df["transaction_date"].dt.to_period("M").astype(str)

    # --- Group by year_month AND group_name ---
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
        # --- Create a grouped bar chart (Groups on X-axis) ---
        fig = go.Figure(data=[
            go.Bar(
                x=filtered_sums_group['group_name'], # Groups on X-axis
                y=filtered_sums_group['total_fees'], # Total fees on Y-axis
                # name='Total Fees', # Not needed when only one trace per group
                hovertemplate = '<b>Group</b>: %{x}<br><b>Fees</b>: $%{y:,.2f}<extra></extra>' # Updated hover
            )
        ])

        fig.update_layout(
            # barmode='group', # Not needed for single trace per x-category
            title=f"Previous Month ({previous_month_str}) Management Fees by Property Group", # Keep title
            xaxis_title="Property Group", # X-axis is group name
            yaxis_title="Total Fees ($)",
            legend_title="Property Group",
            hovermode="x unified" # Might be less useful for single-month bar chart
        )
        # Update x-axis to show group names if needed, though legend covers it.
        # fig.update_xaxes(type='category') # Treat x-axis as categorical
        fig.update_yaxes(tickprefix="$", rangemode='tozero')

        st.plotly_chart(fig, use_container_width=True)

        # --- Add plain English description ---
        st.markdown(f"""
        **How this graph is created:**
        *   This chart displays the total management-related fees collected **for the previous calendar month ({previous_month_str})**, broken down by the Property Group each property belongs to.
        *   We identify relevant fees by looking at General Ledger transactions within that month that are posted to specific accounts (e.g., 'Management Fee - *', 'Inspection - Income', 'Lease Up Fee').
        *   Each fee transaction is linked to a specific property, either directly via `unit_id` or through details within the transaction's journal entry. We then determine the Property Group for that property.
        *   Finally, we sum up all the identified fees for each Property Group to get the totals shown in the bars above. This report is filtered to show only the specified management groups.
        *   Fees from transactions that aren't linked to a property, or properties not assigned to a group, are categorized as 'Unknown' or 'No Group Assigned' (and are excluded from this filtered view).
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
