import streamlit as st
import pandas as pd
import logging
import json
from sqlalchemy import text
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
import plotly.graph_objects as go
from collections import deque, defaultdict
import copy

import decimal

# Import shared functions from utils
from web_version.utils import get_engine

# Force basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    force=True,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Constants ---
CHARGE_TYPES = {'Charge', 'ReversePayment', 'NSF'}
CREDIT_TYPES = {'Payment', 'Credit', 'ApplyDeposit', 'Refund'}
INACTIVITY_DAYS = 100 # Updated from 75
LOOKBACK_YEARS = 10
LOOKBACK_EXTRA_DAYS = 90

# --- Core Open Item Calculation (FIFO - Corrected Bug, Day-Based Aging, Sign Fix) ---
def calculate_lease_open_items_fifo(lease_transactions_df, snapshot_date):
    """
    Processes transactions for a single lease up to snapshot_date
    using Open Item FIFO and returns the state of open charges.
    Uses day-based aging. Handles negative credit amounts.
    Input DataFrame must contain: id, transaction_date, transaction_type, total_amount, lease_id
    Returns: tuple(aged_balances_dict, total_open_balance_decimal, unapplied_credit_decimal, processing_log_list)
    """
    lease_transactions_df = lease_transactions_df.sort_values(by=['transaction_date', 'id'], ascending=[True, True])
    open_charges = deque()
    processed_txns_log = []
    unapplied_credit = decimal.Decimal(0)
    current_lease_id = lease_transactions_df['lease_id'].iloc[0] if not lease_transactions_df.empty else None

    for _, row in lease_transactions_df.iterrows():
        tx_date = row['transaction_date']
        tx_type = row['transaction_type']
        try:
            original_tx_amount = decimal.Decimal(str(row['total_amount']))
        except (ValueError, TypeError, decimal.InvalidOperation):
            logger.warning(f"Lease {current_lease_id}: Invalid amount '{row['total_amount']}' for Tx ID {row['id']}. Skipping.")
            continue
        tx_id = row['id']
        log_entry = {'tx_id': tx_id, 'date': tx_date, 'type': tx_type, 'amount': original_tx_amount, 'notes': ''}

        if tx_type in CHARGE_TYPES:
            charge_amount = abs(original_tx_amount)
            log_entry['notes'] = f"Processing charge {charge_amount:.2f}. "
            applied_log = []
            if unapplied_credit > 0:
                apply_from_unapplied = min(unapplied_credit, charge_amount)
                applied_log.append(f"Applied {apply_from_unapplied:.2f} from prior unapplied credit.")
                unapplied_credit -= apply_from_unapplied
                charge_amount -= apply_from_unapplied
                log_entry['notes'] += f"Prior unapplied credit reduced to {unapplied_credit:.2f}. "
            if charge_amount > decimal.Decimal('0.005'):
                 open_charges.append({'id': tx_id, 'date': tx_date, 'original': abs(original_tx_amount), 'remaining': charge_amount})
                 applied_log.append(f"Added charge with remaining {charge_amount:.2f} to open items.")
            else:
                 applied_log.append("Charge fully covered by prior unapplied credit.")
            log_entry['notes'] += "; ".join(filter(None, applied_log))
            processed_txns_log.append(log_entry)

        elif tx_type in CREDIT_TYPES:
            credit_amount = abs(original_tx_amount) # Use absolute value
            log_entry['notes'] = f"Applying credit/payment {credit_amount:.2f} (Original DB value: {original_tx_amount:.2f}). "
            applied_log = []
            while credit_amount > 0 and open_charges:
                oldest_charge = open_charges[0]
                apply_amount = min(credit_amount, oldest_charge['remaining'])
                rem_before_apply = oldest_charge['remaining']
                applied_log.append(f"Applied {apply_amount:.2f} to Charge ID {oldest_charge['id']} (Date: {oldest_charge['date']}, Rem Before: {rem_before_apply:.2f})")
                oldest_charge['remaining'] -= apply_amount
                credit_amount -= apply_amount
                if oldest_charge['remaining'] <= decimal.Decimal('0.005'):
                    open_charges.popleft()
            log_entry['notes'] += "; ".join(filter(None, applied_log))
            if credit_amount > 0.005:
                 unapplied_credit += credit_amount
                 log_entry['notes'] += f"; Added {credit_amount:.2f} to unapplied credit balance (New total: {unapplied_credit:.2f})."
            processed_txns_log.append(log_entry)
        else:
            logger.debug(f"Lease {current_lease_id}: Unknown transaction type '{tx_type}' for Tx ID {tx_id}. Ignored in aging.")
            log_entry['notes'] = f"Unknown type '{tx_type}', ignored in aging."
            processed_txns_log.append(log_entry)

    aged_balances = {'0-30': decimal.Decimal(0), '31-60': decimal.Decimal(0),
                     '61-90': decimal.Decimal(0), '90+': decimal.Decimal(0)}
    current_open_balance = decimal.Decimal(0)
    for charge in open_charges:
        if not isinstance(charge['date'], date):
             logger.warning(f"Invalid date type for charge {charge.get('id', 'N/A')}: {charge['date']}. Skipping aging.")
             continue
        age = (snapshot_date - charge['date']).days
        remaining = charge['remaining']
        current_open_balance += remaining
        if 0 <= age <= 30: aged_balances['0-30'] += remaining
        elif 31 <= age <= 60: aged_balances['31-60'] += remaining
        elif 61 <= age <= 90: aged_balances['61-90'] += remaining
        elif age > 90: aged_balances['90+'] += remaining

    return aged_balances, current_open_balance, unapplied_credit, processed_txns_log


# --- Main Data Loading and Processing (FIFO Method - Corrected, New Filter Logic) ---
# @st.cache_data(ttl=600) # Temporarily disable caching during debugging
def load_and_process_aged_delinquency(start_date, end_date, debug_unit_id=None):
    """
    Loads Lease Transactions and calculates aged balances for each month
    using FIFO, day-based aging, and the new filtering logic:
    Include if balance > 0 AND (lease has future activity OR no newer lease exists).
    """
    logger.info("--- load_and_process_aged_delinquency called (FIFO - Corrected, New Filter) ---")
    logger.info(f"Starting aged delinquency calculation from {start_date} to {end_date}")
    engine = get_engine()
    if engine is None: logger.error("DB engine failed."); st.error("Database connection failed."); return pd.DataFrame()

    # --- Step 1: Load Lease Transactions and Leases ---
    query_start_date = start_date - relativedelta(years=LOOKBACK_YEARS, days=LOOKBACK_EXTRA_DAYS)
    query_end_date_buffer = end_date + relativedelta(days=INACTIVITY_DAYS + 5)
    query_end_date = end_date
    sql_params = {'query_end_buffer': query_end_date_buffer, 'query_start': query_start_date, 'query_end': query_end_date}

    logger.info(f"Querying Lease Transactions data from {query_start_date} to {query_end_date_buffer}")
    query_lease_txns = text(f"""
        SELECT
            lt.id, lt.transaction_date, lt.transaction_type, lt.total_amount, lt.lease_id, l.unit_id
        FROM lease_transactions lt
        JOIN leases l ON lt.lease_id = l.id
        WHERE
            lt.transaction_date <= :query_end_buffer
            AND lt.transaction_date >= :query_start
        ORDER BY l.unit_id, lt.lease_id, lt.transaction_date, lt.id
    """)
    query_leases = text(f"""
        SELECT id as lease_id, unit_id, lease_from_date, lease_to_date
        FROM leases
        WHERE
            (lease_to_date IS NULL OR lease_to_date >= :query_start)
            AND lease_from_date <= :query_end_buffer
    """)
    try:
        with engine.connect() as conn:
            lease_txn_full = pd.read_sql_query(query_lease_txns, conn, params=sql_params)
            leases_full = pd.read_sql_query(query_leases, conn, params=sql_params)
        logger.info(f"Loaded {len(lease_txn_full)} Lease Transactions and {len(leases_full)} relevant leases.")
    except Exception as e:
        logger.exception(f"Failed to load Lease Transactions or Lease data: {e}")
        st.error(f"DB Error loading Lease Transactions/Lease data: {e}")
        return pd.DataFrame() if not debug_unit_id else {}

    if lease_txn_full.empty:
        logger.warning("No relevant Lease Transactions found.")
        return pd.DataFrame() if not debug_unit_id else {'transactions': pd.DataFrame(), 'balance': decimal.Decimal(0), 'aged_buckets': {}, 'processing_log': []}

    # --- Step 2: Clean and prepare the data ---
    lease_txn_full['transaction_date'] = pd.to_datetime(lease_txn_full['transaction_date'], errors='coerce').dt.date
    lease_txn_full = lease_txn_full.dropna(subset=['transaction_date', 'total_amount', 'unit_id', 'lease_id'])
    lease_txn_full['unit_id'] = lease_txn_full['unit_id'].astype(int)
    lease_txn_full['lease_id'] = lease_txn_full['lease_id'].astype(int)

    leases_full['lease_from_date'] = pd.to_datetime(leases_full['lease_from_date'], errors='coerce').dt.date
    leases_full['lease_to_date'] = pd.to_datetime(leases_full['lease_to_date'], errors='coerce').dt.date
    leases_full = leases_full.dropna(subset=['unit_id', 'lease_id', 'lease_from_date'])
    leases_full['unit_id'] = leases_full['unit_id'].astype(int)
    leases_full['lease_id'] = leases_full['lease_id'].astype(int)

    # Pre-group Lease Transactions by lease_id for efficiency
    grouped_lease_txns_by_lease = lease_txn_full.groupby('lease_id')
    lease_txn_data_cache = {lease_id: group for lease_id, group in grouped_lease_txns_by_lease}

    # --- Debug Mode Execution ---
    if debug_unit_id:
        logger.info(f"--- RUNNING IN DEBUG MODE FOR UNIT ID: {debug_unit_id} (FIFO - Corrected, New Filter) ---")
        debug_lease_ids = leases_full[leases_full['unit_id'] == debug_unit_id]['lease_id'].unique()
        if len(debug_lease_ids) == 0:
             st.error(f"No leases found for debug unit {debug_unit_id}.")
             return {}

        # Find the first active lease for the debug unit at the end date
        active_leases = leases_full[
            (leases_full['unit_id'] == debug_unit_id) &
            (leases_full['lease_from_date'] <= end_date) &
            (leases_full['lease_to_date'].isna() | (leases_full['lease_to_date'] >= end_date))
        ]
        if active_leases.empty:
             st.error(f"No active lease found covering end date {end_date} for debug unit {debug_unit_id}.")
             return {}

        lease_info = active_leases.iloc[0] # Process first active lease for debug
        lease_id = lease_info['lease_id']
        lease_from = lease_info['lease_from_date']
        lease_effective_to = min(lease_info['lease_to_date'] if pd.notna(lease_info['lease_to_date']) else end_date, end_date)

        lease_txns_full_history = lease_txn_data_cache.get(lease_id)
        if lease_txns_full_history is None:
             st.warning(f"No transactions found for lease {lease_id} (Unit {debug_unit_id}).")
             return {}

        lease_txns_for_fifo = lease_txns_full_history[
            lease_txns_full_history['transaction_date'] <= lease_effective_to
        ].copy()

        if lease_txns_for_fifo.empty:
             st.warning(f"No transactions found within the lease period for debug unit {debug_unit_id}, lease {lease_id}.")
             aged_balances = {'0-30': decimal.Decimal(0), '31-60': decimal.Decimal(0), '61-90': decimal.Decimal(0), '90+': decimal.Decimal(0)}
             final_balance = decimal.Decimal(0)
             processing_log = []
        else:
             aged_balances, final_balance, _, processing_log = calculate_lease_open_items_fifo(lease_txns_for_fifo, end_date)

        debug_data = {
            'unit_id': debug_unit_id,
            'lease_id': lease_id,
            'snapshot_date': end_date,
            'transactions': lease_txns_for_fifo,
            'balance': final_balance,
            'aged_buckets': aged_balances,
            'processing_log': processing_log
        }
        return debug_data

    # --- Normal Report Execution ---
    all_lease_ids_to_process = lease_txn_full['lease_id'].unique()
    logger.info(f"Processing {len(all_lease_ids_to_process)} leases using FIFO method with new filter.")

    # --- Step 3: Generate monthly snapshots ---
    month_list = []
    current_month_start = date(start_date.year, start_date.month, 1)
    report_end_month_start = date(end_date.year, end_date.month, 1)
    while current_month_start <= report_end_month_start:
        last_day_of_month = (current_month_start + relativedelta(months=1)) - relativedelta(days=1)
        calculated_snapshot_date = last_day_of_month - relativedelta(days=3)
        actual_snapshot_date = min(calculated_snapshot_date, end_date)
        if not month_list or month_list[-1] != actual_snapshot_date:
            month_list.append(actual_snapshot_date)
        if actual_snapshot_date == end_date or current_month_start >= report_end_month_start :
             break
        current_month_start += relativedelta(months=1)

    logger.info(f"Generating {len(month_list)} monthly snapshots by recalculating FIFO with new filter.")

    all_results_by_month = defaultdict(lambda: {'0-30': decimal.Decimal(0), '31-60': decimal.Decimal(0),
                                                '61-90': decimal.Decimal(0), '90+': decimal.Decimal(0),
                                                'unapplied_credits': decimal.Decimal(0), 'active_leases_count': 0})

    # --- Step 4: Process each month independently ---
    for snapshot_date in month_list:
        logger.debug(f"Processing snapshot for date: {snapshot_date}")
        monthly_totals = all_results_by_month[snapshot_date]

        for lease_id in all_lease_ids_to_process:
            lease_info_rows = leases_full[leases_full['lease_id'] == lease_id]
            if lease_info_rows.empty: continue
            lease_info = lease_info_rows.iloc[0]
            unit_id = lease_info['unit_id']
            lease_from = lease_info['lease_from_date']
            lease_to = lease_info['lease_to_date']

            if not (lease_from <= snapshot_date and (pd.isna(lease_to) or lease_to >= snapshot_date)):
                continue

            lease_effective_to = min(lease_to if pd.notna(lease_to) else snapshot_date, snapshot_date)
            lease_txns_full_history = lease_txn_data_cache.get(lease_id)
            if lease_txns_full_history is None: continue

            lease_txns_for_fifo = lease_txns_full_history[
                lease_txns_full_history['transaction_date'] <= lease_effective_to
            ].copy()

            if lease_txns_for_fifo.empty: continue

            lease_aged_balances, lease_total_open, lease_unapplied_credit, _ = calculate_lease_open_items_fifo(lease_txns_for_fifo, snapshot_date)

            monthly_totals['unapplied_credits'] += lease_unapplied_credit

            # --- Apply New Filters ---
            if lease_total_open > 0:
                newer_leases_exist = leases_full[
                    (leases_full['unit_id'] == unit_id) &
                    (leases_full['lease_id'] != lease_id) &
                    (leases_full['lease_from_date'] > snapshot_date)
                ].empty == False

                if newer_leases_exist:
                    continue

                future_payment_check_start = snapshot_date + timedelta(days=1)
                future_payment_check_end = snapshot_date + timedelta(days=INACTIVITY_DAYS)
                future_payments_for_lease = lease_txns_full_history[
                        (lease_txns_full_history['transaction_type'].isin(CREDIT_TYPES)) &
                        (lease_txns_full_history['transaction_date'] >= future_payment_check_start) &
                        (lease_txns_full_history['transaction_date'] <= future_payment_check_end)
                ]
                is_active_forward_looking = not future_payments_for_lease.empty

                if is_active_forward_looking:
                    monthly_totals['active_leases_count'] += 1
                    for bucket in lease_aged_balances:
                        monthly_totals[bucket] += lease_aged_balances[bucket]

    # Convert results to DataFrame
    final_results_list = []
    for month_dt, data in all_results_by_month.items():
        data['month_end_dt'] = month_dt
        final_results_list.append(data)

    if not final_results_list: logger.warning("No aged results generated."); return pd.DataFrame()

    results_df = pd.DataFrame(final_results_list)
    results_df['month_end_dt'] = pd.to_datetime(results_df['month_end_dt'])
    results_df = results_df.sort_values('month_end_dt')
    results_df['total_delinquency'] = results_df[['0-30', '31-60', '61-90', '90+']].sum(axis=1)
    for col in ['0-30', '31-60', '61-90', '90+', 'total_delinquency', 'unapplied_credits', 'active_leases_count']:
         if col not in results_df.columns:
             results_df[col] = 0
         if col in ['0-30', '31-60', '61-90', '90+', 'total_delinquency', 'unapplied_credits']:
              results_df[col] = results_df[col].fillna(decimal.Decimal(0)).astype(object)
         elif col == 'active_leases_count':
              results_df[col] = results_df[col].fillna(0).astype(int)

    logger.info(f"Finished calculation (FIFO - Corrected, New Filter), shape: {results_df.shape}")
    return results_df


# --- Streamlit Main Function (UI, Plotting, Debug Handling) ---
def main(start_date, end_date):
    st.caption("Tracks aged outstanding balances (FIFO). Excludes leases with $0 balance, leases replaced by newer ones, or leases inactive > 75 days.") # Updated caption
    logger.info(f"Running Aged Delinquency report for {start_date} to {end_date} (FIFO - Corrected, New Filter)")

    # --- Add Debug Input ---
    debug_unit_id_input = st.sidebar.text_input("Debug Unit ID (Optional)", "")
    debug_unit_id = None
    if debug_unit_id_input:
        try:
            debug_unit_id = int(debug_unit_id_input)
            logger.info(f"--- RUNNING IN DEBUG MODE FOR UNIT ID: {debug_unit_id} (FIFO - Corrected, New Filter) ---")
            debug_snapshot_date = end_date
            st.sidebar.info(f"Debug snapshot date: {debug_snapshot_date}")

            with st.spinner(f"Loading transactions and calculating aged open items for Unit {debug_unit_id} at {debug_snapshot_date}..."):
                debug_data = load_and_process_aged_delinquency(start_date, debug_snapshot_date, debug_unit_id=debug_unit_id)

            if not debug_data:
                 st.error(f"Could not load data for debug unit {debug_unit_id}.")
                 return

            st.subheader(f"Debug Info for Unit {debug_data.get('unit_id')} / Lease {debug_data.get('lease_id')} at {debug_data.get('snapshot_date')} (FIFO - Corrected, New Filter)")

            st.write(f"Input Transactions (Lease {debug_data.get('lease_id')}, up to {debug_snapshot_date}):")
            debug_txns_df = debug_data.get('transactions', pd.DataFrame())
            if not debug_txns_df.empty:
                 debug_txns_df_display = debug_txns_df[['transaction_date', 'transaction_type', 'total_amount', 'id']].copy()
                 debug_txns_df_display.rename(columns={'transaction_date': 'Date', 'transaction_type': 'Type', 'total_amount': 'Amount', 'id': 'Tx ID'}, inplace=True)
                 debug_txns_df_display['Amount'] = debug_txns_df_display['Amount'].apply(lambda x: f"{x:.2f}")
                 st.dataframe(debug_txns_df_display.sort_values(by='Date'))
            else:
                 st.write(f"No relevant transactions found for this lease up to {debug_snapshot_date}.")

            st.write("Payment Application Log (FIFO):")
            processing_log_df = pd.DataFrame(debug_data.get('processing_log', []))
            if not processing_log_df.empty:
                log_display_df = processing_log_df[['tx_id', 'date', 'type', 'amount', 'notes']].copy()
                log_display_df['amount'] = log_display_df['amount'].apply(lambda x: f"{x:.2f}")
                st.dataframe(log_display_df)
            else:
                st.write("No payment application steps logged.")

            st.metric("Total Open Balance (Sum of Unpaid Charges)", f"${debug_data.get('balance', decimal.Decimal(0)):,.2f}")
            st.write("Calculated Aged Buckets (FIFO - Day-Based):")
            aged_buckets_dict = debug_data.get('aged_buckets', {})
            for b in ['0-30', '31-60', '61-90', '90+']: aged_buckets_dict.setdefault(b, decimal.Decimal(0))
            aged_df = pd.DataFrame([aged_buckets_dict])
            for col in aged_df.columns: aged_df[col] = aged_df[col].apply(lambda x: f"${x:,.2f}")
            st.dataframe(aged_df[['0-30', '31-60', '61-90', '90+']])

            st.info("Debug mode active. Clear the Debug Unit ID input and rerun to see the full report.")
            return
        except ValueError:
            st.sidebar.error("Invalid Unit ID. Please enter a number.")
            return
        except Exception as e:
            st.error(f"Error during debug calculation: {e}")
            logger.exception("Error during debug calculation.")
            return

    # --- Normal Report Execution ---
    with st.spinner("Calculating aged delinquency (FIFO - Corrected, New Filter)..."):
        results_df = load_and_process_aged_delinquency(start_date, end_date)

    if results_df.empty:
        st.warning("No aged delinquency data found for the selected period.")
        return

    # --- Plotting ---
    st.subheader("Monthly Aged Delinquency Trend (FIFO - Corrected, New Filter)")
    logger.info("Generating plot...")
    try:
        fig = go.Figure()
        aging_buckets = ['0-30', '31-60', '61-90', '90+']
        colors = ['#1f77b4', '#ff7f0e', '#d62728', '#8c564b']

        plot_df = results_df.copy()
        for bucket in aging_buckets + ['total_delinquency', 'unapplied_credits']:
             if bucket in plot_df.columns:
                 try: plot_df[bucket] = plot_df[bucket].astype(float)
                 except Exception as conv_err:
                      logger.warning(f"Could not convert column {bucket} to float for plotting: {conv_err}")
                      plot_df[bucket] = pd.to_numeric(plot_df[bucket], errors='coerce').fillna(0)

        for i, bucket in enumerate(aging_buckets):
            if bucket in plot_df.columns:
                fig.add_trace(go.Scatter(
                    x=plot_df['month_end_dt'], y=plot_df[bucket], mode='lines',
                    name=f'{bucket} Days', stackgroup='one',
                    line=dict(width=0.5, color=colors[i % len(colors)]),
                    fillcolor=colors[i % len(colors)],
                    hovertemplate = f'<b>Month End</b>: %{{x|%Y-%m-%d}}<br><b>{bucket} Days</b>: $%{{y:,.2f}}<extra></extra>'
                ))
            else: logger.warning(f"Aging bucket column '{bucket}' not found for plotting.")

        fig.update_layout(
            title="Aged Delinquency by Month (FIFO - Corrected, New Filter)",
            xaxis_title="Month End",
            yaxis_title="Outstanding Balance ($)", hovermode="x unified",
            legend_title_text='Aging Bucket'
        )
        fig.update_yaxes(tickprefix="$", rangemode='tozero')

        if 'total_delinquency' in plot_df.columns:
            fig.add_trace(go.Scatter(
                x=plot_df['month_end_dt'], y=plot_df['total_delinquency'], mode='markers',
                name='Total', marker=dict(size=8, color='rgba(0,0,0,0.5)'),
                hovertemplate='<b>Month End</b>: %{x|%Y-%m-%d}<br><b>Total</b>: $%{y:,.2f}<extra></extra>'
            ))

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **How this graph is created:**
        *   This graph shows the total outstanding balance from **unpaid charges** at the end of each month (or selected snapshot date), broken down by the age of those specific charges.
        *   The calculation uses the **"Open Item (FIFO)"** method:
            1. All charges (Rent, Fees, etc.) are tracked individually per lease.
            2. Payments and credits are applied chronologically to the **oldest outstanding charge** first (First-In, First-Out) for that specific lease. This application is recalculated for each month's snapshot based *only* on transactions up to that snapshot date.
            3. At the end of each snapshot period, the remaining balance of any partially or fully unpaid charges is calculated.
            4. The age of each unpaid charge amount is determined by the actual number of days between its original transaction date and the snapshot date.
        *   The sum of the aged buckets for any given snapshot equals the total balance of all unpaid charges across all included leases for that date.
        *   Leases are included only if:
            a) Had an **active lease** covering the snapshot date.
            b) Had a **positive open balance** after FIFO calculation as of the snapshot date.
            c) EITHER received a **payment/credit *for that specific lease* within 100 days *after*** the snapshot date, OR no newer lease started for the same unit after the snapshot date.
        *   The stacked areas show the contribution of each aging bucket to the total delinquency based on open items across all qualifying leases.
        """) # Updated description

    except Exception as e:
        logger.exception("Failed to generate plot.")
        st.error(f"Error generating plot: {e}")

    # --- Data Table ---
    st.subheader("Aged Delinquency Data Table (FIFO - Corrected, New Filter)")
    try:
        display_df = results_df.copy()
        if 'month_end_dt' in display_df.columns:
             display_df['Month End'] = display_df['month_end_dt'].dt.strftime('%Y-%m-%d')
             # Format Decimal columns as currency strings
             columns_to_format = aging_buckets + ['total_delinquency', 'unapplied_credits']
             for bucket in columns_to_format:
                 if bucket in display_df.columns:
                     display_df[bucket] = display_df[bucket].apply(lambda x: f"${x:,.2f}" if isinstance(x, decimal.Decimal) else f"${float(x):,.2f}")
                 else:
                     display_df[bucket] = '$0.00'

             if 'total_delinquency' in display_df.columns: display_df['Total Delinquency'] = display_df['total_delinquency']
             if 'unapplied_credits' in display_df.columns: display_df['Unapplied Credits'] = display_df['unapplied_credits']

             # Define desired column order
             final_display_cols_ordered = ['Month End'] + aging_buckets + ['Total Delinquency', 'Unapplied Credits']
             final_display_cols = [col for col in final_display_cols_ordered if col in display_df.columns]

             display_df = display_df[final_display_cols]
             st.dataframe(display_df.sort_values(by='Month End', ascending=False), hide_index=True)
             csv = display_df.to_csv(index=False)
             st.download_button(label="Download Data as CSV", data=csv, file_name="aged_delinquency_fifo_new_filter_report.csv", mime="text/csv") # Updated filename
        else: st.warning("Could not display data table.")
    except Exception as e:
        logger.exception("Failed to display data table."); st.error(f"Error displaying table: {e}")

    logger.info("Aged Delinquency report execution finished (FIFO - Corrected, New Filter).")
