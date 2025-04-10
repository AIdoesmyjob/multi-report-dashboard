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
FEE_ACCOUNT_NAMES = {
    "Management Fee - 020",
    "Management Fee - 030",
    "Management Fee - 120",
    "Management Fee - 130",
    "Management Fee - Sutton 2022",
    "Management Fees - Abbotsford",
    "Management Fees - Chilliwack",
    # Newly added:
    "Inspection - Income",
    "Lease Up Fee",
}

# Cache the data result; the returned DataFrame is pickleable.
@st.cache_data(ttl=600)
def load_data(): # Removed start_date, end_date parameters
    """
    Queries general_ledger_transactions for the CURRENT MONTH, filters for lines
    matching FEE_ACCOUNT_NAMES, extracts property_id, and groups the amounts by property.

    Returns a DataFrame with columns:
       [year_month, property_id, total_fees, year_month_dt]
    """
    # --- Calculate start and end of CURRENT month ---
    today = datetime.today().date()
    start_of_month = today.replace(day=1)
    # Add one month and subtract one day to get the last day of the current month
    end_of_month = (start_of_month + relativedelta(months=1)) - relativedelta(days=1)
    logger.info(f"Loading management fee data by property for the current month: {start_of_month} to {end_of_month}")
    engine = get_engine()
    if engine is None:
        logger.error("Failed to get DB engine for management fees by group.")
        return pd.DataFrame()

    # --- MODIFIED QUERY: Added property_id ---
    # Assuming 'property_id' exists directly on the transactions table.
    # If not, a JOIN would be needed here.
    # --- FIXED: Removed extra line before SELECT ---
    query = text("""
        SELECT
            id,
            transaction_date,
            transaction_type,
            total_amount,
            journal,
            property_id -- Added property_id
        FROM general_ledger_transactions
        WHERE transaction_date >= :start_date -- Use calculated start date
          AND transaction_date <= :end_date -- Use calculated end date
          AND property_id IS NOT NULL -- Exclude transactions without a property
        ORDER BY id
    """)
    try:
        with engine.connect() as conn:
            # Use calculated dates in params
            df = pd.read_sql_query(query, conn, params={"start_date": start_of_month, "end_date": end_of_month})
        logger.info(f"Retrieved {df.shape[0]} GL transactions with property_id from the database for the current month.")
    except Exception as e:
         # Check if the error is due to the missing 'property_id' column
         if "column \"property_id\" does not exist" in str(e).lower():
             logger.error("The 'property_id' column was not found in 'general_ledger_transactions'. Cannot group by property.")
             st.error("Database Error: The required 'property_id' column is missing from the transactions table. This report cannot be generated.")
             # Return an empty DataFrame with expected columns for graceful failure downstream
             return pd.DataFrame(columns=['year_month', 'property_id', 'total_fees', 'year_month_dt'])
         else:
            logger.exception("Failed to load GL transactions.")
            st.error(f"DB Error loading GL transactions: {e}")
            return pd.DataFrame(columns=['year_month', 'property_id', 'total_fees', 'year_month_dt'])


    # Convert transaction_date to datetime.
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")

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

                # --- MODIFIED: Added property_id ---
                fee_entries.append({
                    "transaction_date": row["transaction_date"],
                    "account_name": account_name,
                    "amount": numeric_amount,
                    "transaction_id": row["id"],
                    "property_id": row["property_id"] # Added property_id
                })
                # Don't break, multiple fee lines could exist in one transaction

    if json_parse_errors > 0:
         logger.warning(f"Total JSON parsing errors encountered: {json_parse_errors}")
         st.toast(f"Note: Encountered {json_parse_errors} errors parsing journal data. See logs for details.", icon="⚠️")

    if not fee_entries:
        logger.warning("No fee entries found after processing GL transactions.")
        return pd.DataFrame(columns=['year_month', 'property_id', 'total_fees', 'year_month_dt']) # Return empty with expected columns

    fee_df = pd.DataFrame(fee_entries)
    logger.info(f"Found {len(fee_df)} fee entries matching specified accounts with property IDs.")

    # Create a 'year_month' column (e.g., '2023-05').
    fee_df["year_month"] = fee_df["transaction_date"].dt.to_period("M").astype(str)

    # --- MODIFIED: Group by year_month AND property_id ---
    monthly_sums = fee_df.groupby(["year_month", "property_id"])["amount"].sum().reset_index()
    monthly_sums.rename(columns={"amount": "total_fees"}, inplace=True)

    # Create a datetime column for plotting.
    monthly_sums["year_month_dt"] = pd.to_datetime(monthly_sums["year_month"] + "-01", format="%Y-%m-%d")
    monthly_sums.sort_values(["year_month_dt", "property_id"], inplace=True) # Sort by date then property

    logger.info(f"Finished processing management fee data by property, shape: {monthly_sums.shape}")
    return monthly_sums


@st.cache_data(ttl=600)
def load_property_groups():
    """
    Loads property and property group data to map property_id to group name.

    Returns a DataFrame with columns: [property_id, property_name, group_name]
    """
    logger.info("Loading property group mapping data.")
    engine = get_engine()
    if engine is None:
        logger.error("Failed to get DB engine for property groups.")
        return pd.DataFrame()

    # --- FIXED QUERY: Use correct table names and join logic ---
    # Uses rental_properties, property_groups, and property_group_memberships
    query = text("""
        SELECT DISTINCT -- Use DISTINCT in case a property is in multiple groups (though fees are per property)
            rp.id AS property_id,
            rp.name AS property_name,
            -- pg.id AS group_id, -- Not strictly needed for this report's purpose
            COALESCE(pg.name, 'No Group Assigned') AS group_name -- Handle properties not in any group
        FROM
            rental_properties rp
        LEFT JOIN
            property_group_memberships pgm ON rp.id = pgm.property_id
        LEFT JOIN
            property_groups pg ON pgm.property_group_id = pg.id
        -- WHERE rp.is_active = true -- Optional: Filter for only active properties if needed
        ORDER BY
            rp.id;
    """)
    try:
        with engine.connect() as conn:
            prop_df = pd.read_sql_query(query, conn)
        logger.info(f"Retrieved {prop_df.shape[0]} properties with group info using correct tables.")
        # The COALESCE in the query handles the 'No Group Assigned' case directly.
        # Ensure the columns exist before returning
        if 'property_id' not in prop_df.columns or 'property_name' not in prop_df.columns or 'group_name' not in prop_df.columns:
             logger.error("Query for property groups did not return expected columns.")
             st.error("DB Error: Failed to retrieve property group mapping correctly.")
             return pd.DataFrame(columns=['property_id', 'property_name', 'group_name'])

        return prop_df[['property_id', 'property_name', 'group_name']]
    except Exception as e:
        logger.exception("Failed to load property/group data using correct tables.")
        st.error(f"DB Error loading property/group data: {e}")
        # Return empty with expected columns for graceful failure
        return pd.DataFrame(columns=['property_id', 'property_name', 'group_name'])


def main(start_date, end_date): # Keep parameters for compatibility with app.py call signature
    # --- Calculate current month for display ---
    today = datetime.today().date()
    current_month_str = today.strftime("%B %Y")
    logger.info(f"Running Management Fees by Property Group report for CURRENT MONTH ({current_month_str})")

    # Load fee data (now ignores start/end date parameters) and property group mapping
    monthly_sums_prop = load_data()
    prop_groups = load_property_groups()

    if monthly_sums_prop.empty:
        st.warning(f"No relevant fee entries with property IDs found for the current month ({current_month_str}).")
        logger.warning("No management fee data by property loaded or processed for the current month.")
        return
    if prop_groups.empty:
        st.warning("Could not load property group information. Cannot group fees.")
        logger.warning("Property group mapping failed to load.")
        # Optionally display the data just by property ID if groups fail
        # st.subheader("Data Table (by Property ID - Grouping Failed)")
        # st.dataframe(monthly_sums_prop)
        return

    # Merge fee data with property group info
    logger.info("Merging fee data with property group info...")
    merged_df = pd.merge(monthly_sums_prop, prop_groups, on="property_id", how="left")
    # Handle cases where a property_id from fees might not be in the properties table (unlikely but possible)
    merged_df['group_name'] = merged_df['group_name'].fillna('Unknown Property/Group')
    merged_df['property_name'] = merged_df['property_name'].fillna('Unknown Property')


    # --- MODIFIED: Aggregate by year_month AND group_name ---
    logger.info("Aggregating fees by month and property group...")
    monthly_sums_group = merged_df.groupby(["year_month_dt", "year_month", "group_name"])["total_fees"].sum().reset_index()
    monthly_sums_group.sort_values(["year_month_dt", "group_name"], inplace=True)

    # --- MODIFIED: Plotting - Stacked Bar Chart ---
    logger.info("Generating plot...")
    try:
        fig = go.Figure()
        groups = monthly_sums_group['group_name'].unique()

        # Create a trace for each group
        for group in sorted(groups):
            group_data = monthly_sums_group[monthly_sums_group['group_name'] == group]
            fig.add_trace(go.Bar(
                x=group_data['year_month_dt'],
                y=group_data['total_fees'],
                name=group,
                hovertemplate = f'<b>Group</b>: {group}<br><b>Month</b>: %{{x|%Y-%m}}<br><b>Fees</b>: $%{{y:,.2f}}<extra></extra>'
            ))

        fig.update_layout(
            barmode='stack', # Stack bars
            title=f"Current Month ({current_month_str}) Management Fees by Property Group", # Updated title
            xaxis_title="Property Group", # Changed X-axis title as we only have one month
            yaxis_title="Total Fees ($)",
            legend_title="Property Group",
            hovermode="x unified"
        )
        fig.update_yaxes(tickprefix="$", rangemode='tozero')

        st.plotly_chart(fig, use_container_width=True)

        # --- Add plain English description ---
        st.markdown(f"""
        **How this graph is created:**
        *   This chart displays the total management-related fees collected **for the current calendar month ({current_month_str})**, broken down by the Property Group each property belongs to.
        *   We identify relevant fees by looking at General Ledger transactions within this month that are posted to specific income accounts like 'Management Fee - *', 'Inspection - Income', and 'Lease Up Fee'.
        *   Each fee transaction is linked to a specific property using the `property_id` recorded with the transaction.
        *   We then look up which Property Group that property belongs to.
        *   Finally, we sum up all the identified fees for each Property Group to get the totals shown in the bars above.
        *   Fees from transactions that aren't linked to a property, or properties not assigned to a group, are categorized as 'Unknown' or 'No Group Assigned'.
        """)
        # --- End description ---

    except Exception as e:
        logger.exception("Failed to generate plot.")
        st.error(f"Error generating plot: {e}")

    # --- MODIFIED: Data Table - Grouped ---
    st.subheader("Data Table (Grouped by Month and Property Group)")
    try:
        # --- MODIFIED: Data Table - Simplified for single month ---
        st.subheader(f"Data Table (Current Month: {current_month_str})")
        # Display simple table grouped by group name for the single month
        display_df = monthly_sums_group[['group_name', 'total_fees']].copy()
        display_df.rename(columns={'group_name': 'Property Group', 'total_fees': 'Total Fees'}, inplace=True)
        display_df['Total Fees'] = display_df['Total Fees'].map('${:,.2f}'.format)
        st.dataframe(display_df.sort_values(by='Property Group'), hide_index=True)

    except Exception as e:
        logger.exception("Failed to display data table.")
        st.error(f"Error displaying data table: {e}")

    logger.info("Report execution finished.")

# Note: No __main__ block needed as this will be called by app.py
