import streamlit as st
import pandas as pd
import logging # <-- Added
from datetime import date, datetime, time
from dateutil.relativedelta import relativedelta
from sqlalchemy import text # <-- Removed create_engine
import plotly.express as px

# Import shared functions and constants from utils
from web_version.utils import ( # <-- Absolute import
    get_engine,
    load_user_map,
    parse_created_by_user, # Assuming this might be needed if logic changes
    utc_to_pacific,      # Assuming this might be needed
    OPEN_STATUSES,
    DEFERRED_STATUSES,
    CLOSED_STATUSES
)

# Configure logging for this report module
logger = logging.getLogger(__name__)

@st.cache_data(ttl=600)
def load_task_data(snapshot_end_date):
    """
    Load task data by combining:
      - The tasks table (initial records) filtered by created_datetime <= snapshot_end_date,
      - The task_history table (all events so we capture status changes).
    Both queries include the assigned_to_user_id column.
    """
    logger.info(f"Loading combined task and history data up to {snapshot_end_date}...")
    engine = get_engine()
    if engine is None: return pd.DataFrame()

    tasks_query = text("""
        SELECT
            id AS task_id,
            created_datetime AS event_time,
            task_status AS event_status,
            assigned_to_user_id
        FROM tasks
        WHERE created_datetime <= :snapshot_end_date
    """)
    history_query = text("""
        SELECT
            task_id,
            created_datetime AS event_time,
            task_status AS event_status,
            assigned_to_user_id
        FROM task_history
        WHERE created_datetime <= :snapshot_end_date -- Filter history too for efficiency
    """)

    try:
        with engine.connect() as conn:
            tasks_df = pd.read_sql_query(tasks_query, conn, params={"snapshot_end_date": snapshot_end_date})
            history_df = pd.read_sql_query(history_query, conn, params={"snapshot_end_date": snapshot_end_date})
        logger.info(f"Loaded {len(tasks_df)} initial tasks and {len(history_df)} history events.")
    except Exception as e:
        logger.exception("Failed to load task/history data.")
        st.error(f"DB Error loading task/history data: {e}")
        return pd.DataFrame()

    combined_df = pd.concat([tasks_df, history_df], ignore_index=True)
    combined_df.sort_values(by=["task_id", "event_time"], inplace=True)
    combined_df["event_time"] = pd.to_datetime(combined_df["event_time"], errors="coerce")
    combined_df = combined_df.dropna(subset=["event_time"]) # Drop invalid dates
    logger.debug(f"Combined task/history shape after concat and dropna: {combined_df.shape}")
    return combined_df

def build_status_timeline(df):
    """
    Build a timeline dictionary for each task:
      {task_id: [(event_time, event_status, assigned_to_user_id), ...]}
    Each timeline is sorted in chronological order.
    """
    logger.debug("Building status timelines...")
    timeline_dict = {}
    for task_id, group in df.groupby("task_id"):
        # Ensure group is sorted by event_time
        group = group.sort_values(by="event_time")
        timeline = list(group[['event_time', 'event_status', 'assigned_to_user_id']].itertuples(index=False, name=None))
        # No need to sort again if group was sorted
        # timeline.sort(key=lambda x: x[0])
        timeline_dict[task_id] = timeline
    logger.debug(f"Built timelines for {len(timeline_dict)} tasks.")
    return timeline_dict

def compute_effective_closed_date_up_to(timeline, month_end):
    """
    Simulate state changes up to month_end and return the effective closed date.
    - If a CLOSED_STATUSES event is encountered, record that time.
    - If later (<= month_end) an OPEN_STATUSES or DEFERRED_STATUSES event occurs, reset to None.
    """
    effective_closed = None
    for event in timeline:
        if len(event) < 2: continue
        event_time, event_status = event[0], event[1]
        if not isinstance(event_time, datetime) or pd.isna(event_time): continue

        if event_time.date() > month_end:
            break
        status = str(event_status).strip() if event_status is not None else ""
        if status in CLOSED_STATUSES:
            effective_closed = event_time
        elif (status in OPEN_STATUSES or status in DEFERRED_STATUSES) and effective_closed is not None:
             if event_time > effective_closed:
                  effective_closed = None
    return effective_closed

def get_last_status_and_user_up_to(timeline, month_end):
    """
    Return the last event_status and assigned_to_user_id on or before month_end.
    """
    last_status = None
    last_user = None
    for event in timeline:
        if len(event) < 3: continue
        event_time, event_status, user_id = event
        if not isinstance(event_time, datetime) or pd.isna(event_time): continue

        if event_time.date() <= month_end:
            last_status = event_status
            last_user = user_id
        else:
            break
    return last_status, last_user

def generate_month_list(start_date, end_date):
    """
    Generate a list of date objects representing the first day of each month
    from start_date up to end_date.
    """
    months = []
    current = date(start_date.year, start_date.month, 1)
    end_month = date(end_date.year, end_date.month, 1)
    while current <= end_month:
        months.append(current)
        current += relativedelta(months=1)
    return months

# --- NEW: Interval-Based Function ---
def get_open_assigned_user_in_month(timeline, month_start, month_end):
    """
    For a given task timeline (a list of tuples: (event_time, event_status, assigned_to_user_id)),
    determine if the task was in an open state at any point during the month.
    Returns the assigned_to_user_id for that interval if found, else None.
    """
    month_start_dt = datetime.combine(month_start, time.min)
    month_end_dt = datetime.combine(month_end, time.max)

    for i, event_data in enumerate(timeline):
        if len(event_data) < 3: continue
        event_time, event_status, assigned_user = event_data
        if not isinstance(event_time, datetime) or pd.isna(event_time): continue

        status_str = str(event_status).strip() if event_status is not None else ""
        if status_str not in OPEN_STATUSES:
            continue

        interval_start = event_time
        interval_end = datetime.max

        if i < len(timeline) - 1:
            next_event_data = timeline[i + 1]
            if len(next_event_data) > 0 and isinstance(next_event_data[0], datetime):
                 interval_end = next_event_data[0]

        # Basic timezone handling (assumes consistency or converts to naive)
        tz = interval_start.tzinfo
        if tz:
             month_start_dt = month_start_dt.replace(tzinfo=tz)
             month_end_dt = month_end_dt.replace(tzinfo=tz)
             if interval_end != datetime.max and interval_end.tzinfo is None:
                  interval_end = interval_end.replace(tzinfo=tz)
        elif interval_end != datetime.max and interval_end.tzinfo is not None:
             logger.warning(f"Timezone mismatch in interval at {interval_start}. Converting to naive.")
             interval_start = interval_start.replace(tzinfo=None)
             interval_end = interval_end.replace(tzinfo=None)
             month_start_dt = month_start_dt.replace(tzinfo=None)
             month_end_dt = month_end_dt.replace(tzinfo=None)


        latest_start = max(interval_start, month_start_dt)
        earliest_end = min(interval_end, month_end_dt)

        if earliest_end > latest_start:
            overlap = (earliest_end - latest_start).total_seconds()
            if overlap > 0:
                return assigned_user # Return user ID associated with the open interval
    return None
# --- End of new function ---

@st.cache_data(ttl=600)
def compute_monthly_open_tasks(start_date, end_date):
    """
    Compute monthly open tasks per user (excluding deferred tasks) using interval-based logic.
    Returns a DataFrame with columns:
      [month, user_id, open_tasks, month_str]
    """
    logger.info("Computing monthly open tasks...")
    # Load data up to the end date for accurate timeline
    combined_df = load_task_data(end_date)
    if combined_df.empty: return pd.DataFrame()

    timeline_dict = build_status_timeline(combined_df)
    if not timeline_dict: return pd.DataFrame()

    task_creation = {}
    for task_id, events in timeline_dict.items():
         if events and isinstance(events[0][0], datetime):
             task_creation[task_id] = events[0][0].date()

    month_list = generate_month_list(start_date, end_date)
    results = []
    logger.info(f"Analyzing {len(month_list)} months for open tasks...")

    for m in month_list:
        month_end = m + relativedelta(day=31)
        logger.debug(f"Processing month {m.strftime('%Y-%m')} for open tasks...")
        user_counts = {}
        for task_id, events in timeline_dict.items():
            creation_date = task_creation.get(task_id)
            if creation_date is None or creation_date > month_end:
                continue
            assigned_user = get_open_assigned_user_in_month(events, m, month_end)
            if assigned_user is not None:
                # Ensure user_id is treated as integer/hashable
                try:
                    user_id_key = int(assigned_user) if pd.notna(assigned_user) else -1 # Use -1 for unassigned/NaN
                except (ValueError, TypeError):
                    logger.warning(f"Could not convert assigned_user '{assigned_user}' to int for task {task_id}. Assigning to -1.")
                    user_id_key = -1
                user_counts[user_id_key] = user_counts.get(user_id_key, 0) + 1

        for user_id, count in user_counts.items():
            results.append({
                "month": m,
                "user_id": user_id,
                "open_tasks": count
            })
        logger.debug(f"Month {m.strftime('%Y-%m')}: Found open tasks for {len(user_counts)} users.")

    if not results: return pd.DataFrame()
    results_df = pd.DataFrame(results)
    results_df['month'] = pd.to_datetime(results_df['month'], errors='coerce')
    results_df = results_df.dropna(subset=['month'])
    if results_df.empty: return pd.DataFrame()
    results_df["month_str"] = results_df["month"].dt.strftime("%Y-%m")
    logger.info(f"Finished computing monthly open tasks, shape: {results_df.shape}")
    return results_df

@st.cache_data(ttl=600)
def load_work_orders_data(start_date, end_date):
    """
    Load work orders data by joining work_orders and tasks.
    Groups by month and assigned_to_user_id.
    """
    logger.info(f"Loading work order data from {start_date} to {end_date}...")
    engine = get_engine()
    if engine is None: return pd.DataFrame()
    query = text("""
        SELECT
            w.id AS work_order_id,
            w.task_id,
            t.created_datetime AS task_created_datetime,
            t.assigned_to_user_id
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

    df["task_created_datetime"] = pd.to_datetime(df["task_created_datetime"], errors="coerce")
    df = df.dropna(subset=["task_created_datetime", "assigned_to_user_id"]) # Need user_id for grouping
    if df.empty: return pd.DataFrame()

    df["year_month"] = df["task_created_datetime"].dt.to_period("M").astype(str)
    # Ensure user_id is int/hashable before grouping
    df["assigned_to_user_id"] = pd.to_numeric(df["assigned_to_user_id"], errors='coerce').fillna(-1).astype(int)

    monthly_counts = df.groupby(["year_month", "assigned_to_user_id"]).size().reset_index(name="work_order_count")
    monthly_counts["year_month_dt"] = pd.to_datetime(monthly_counts["year_month"] + "-01", format="%Y-%m-%d")
    monthly_counts.sort_values("year_month_dt", inplace=True)
    logger.info(f"Finished processing work order data, shape: {monthly_counts.shape}")
    return monthly_counts

@st.cache_data(ttl=600)
def compute_monthly_closed_tasks(start_date, end_date):
    """
    Compute monthly closed tasks per user (tasks that were completed/closed in that month).
    Returns a DataFrame with columns:
      [month, user_id, closed_tasks, month_str]
    """
    logger.info("Computing monthly closed tasks...")
    # Load data up to the end date for accurate timeline
    combined_df = load_task_data(end_date)
    if combined_df.empty: return pd.DataFrame()

    timeline_dict = build_status_timeline(combined_df)
    if not timeline_dict: return pd.DataFrame()

    task_creation = {}
    for task_id, events in timeline_dict.items():
         if events and isinstance(events[0][0], datetime):
             task_creation[task_id] = events[0][0].date()

    month_list = generate_month_list(start_date, end_date)
    monthly_user_counts = []
    logger.info(f"Analyzing {len(month_list)} months for closed tasks...")

    for m in month_list:
        month_end = m + relativedelta(day=31)
        logger.debug(f"Processing month {m.strftime('%Y-%m')} for closed tasks...")
        user_counts = {}
        for task_id, events in timeline_dict.items():
            creation_date = task_creation.get(task_id)
            if creation_date is None or creation_date > month_end:
                continue
            effective_closed = compute_effective_closed_date_up_to(events, month_end)
            if effective_closed is not None and m <= effective_closed.date() <= month_end:
                _, last_user = get_last_status_and_user_up_to(events, effective_closed.date()) # Use closure date
                if last_user is not None:
                    try:
                        user_id_key = int(last_user) if pd.notna(last_user) else -1
                    except (ValueError, TypeError):
                        user_id_key = -1
                    user_counts[user_id_key] = user_counts.get(user_id_key, 0) + 1

        for user_id, count in user_counts.items():
            monthly_user_counts.append({
                "month": m,
                "user_id": user_id,
                "closed_tasks": count
            })
        logger.debug(f"Month {m.strftime('%Y-%m')}: Found closed tasks for {len(user_counts)} users.")

    if not monthly_user_counts: return pd.DataFrame()
    results_df = pd.DataFrame(monthly_user_counts)
    results_df['month'] = pd.to_datetime(results_df['month'], errors='coerce')
    results_df = results_df.dropna(subset=['month'])
    if results_df.empty: return pd.DataFrame()
    results_df["month_str"] = results_df["month"].dt.strftime("%Y-%m")
    logger.info(f"Finished computing monthly closed tasks, shape: {results_df.shape}")
    return results_df

@st.cache_data(ttl=600)
def compute_monthly_total_tasks(start_date, end_date):
    """
    Compute monthly counts per user by combining:
      - Open tasks (from compute_monthly_open_tasks)
      - Work orders (from load_work_orders_data)
      - Closed tasks (from compute_monthly_closed_tasks)
    Returns a DataFrame with columns:
      [month_str, user_id, user_name, open_tasks, work_order_count, closed_tasks, total]
    """
    logger.info("Computing combined monthly totals (Open+WO+Closed)...")
    open_df = compute_monthly_open_tasks(start_date, end_date)
    wo_df = load_work_orders_data(start_date, end_date)
    closed_df = compute_monthly_closed_tasks(start_date, end_date)

    # Prepare WO df for merge
    if not wo_df.empty:
        # Rename columns before merge
        wo_df_renamed = wo_df.rename(columns={"year_month": "month_str", "assigned_to_user_id": "user_id"})
        # Select only necessary columns for merge
        wo_to_merge = wo_df_renamed[["month_str", "user_id", "work_order_count"]]
    else:
        wo_to_merge = pd.DataFrame(columns=["month_str", "user_id", "work_order_count"])

    # Prepare Open df for merge
    if not open_df.empty:
        open_to_merge = open_df[["month_str", "user_id", "open_tasks"]]
    else:
        open_to_merge = pd.DataFrame(columns=["month_str", "user_id", "open_tasks"])

    # Prepare Closed df for merge
    if not closed_df.empty:
        closed_to_merge = closed_df[["month_str", "user_id", "closed_tasks"]]
    else:
        closed_to_merge = pd.DataFrame(columns=["month_str", "user_id", "closed_tasks"])


    # Perform merges
    merge_cols = ["month_str", "user_id"]
    combined = pd.merge(open_to_merge, wo_to_merge, on=merge_cols, how="outer")
    combined = pd.merge(combined, closed_to_merge, on=merge_cols, how="outer")

    # Fill NaNs and calculate total
    activity_cols = ["open_tasks", "work_order_count", "closed_tasks"]
    for col in activity_cols:
        if col not in combined.columns: combined[col] = 0
        combined[col] = combined[col].fillna(0).astype(int)

    combined["total"] = combined[activity_cols].sum(axis=1)

    # Map user names
    user_map = load_user_map()
    combined["user_name"] = combined["user_id"].map(user_map).fillna("Unknown/Unassigned") # Handle -1 or other missing IDs

    # Ensure month_str exists before sorting
    if "month_str" not in combined.columns:
         combined["month_str"] = "" # Add empty string if missing

    logger.info(f"Finished computing combined totals, shape: {combined.shape}")
    return combined.sort_values(["month_str", "user_name"])

def main(start_date, end_date):
    st.header("User Monthly Task Summary (Open + WO + Closed)") # Clarified title
    st.write(f"Analysis period: **{start_date}** to **{end_date}**")
    logger.info(f"Running Tasks+WO by User report for {start_date} to {end_date}")

    results_df = compute_monthly_total_tasks(start_date, end_date)
    if results_df.empty:
        st.warning("No data found for the given period.")
        logger.warning("No combined task/WO data computed.")
        return

    all_user_names = sorted([name for name in results_df["user_name"].unique() if pd.notna(name) and name != "Unknown/Unassigned"])

    if not all_user_names:
         st.warning("No users found in the results.")
         logger.warning("No valid user names found in results.")
         filtered_df = results_df # Show unassigned if needed
         selected_users = []
    else:
        # Define default user IDs and map them to names.
        default_user_ids = [6601768, 2810100, 6590396, 2054428, 4925505, 6149464, 4325638]
        user_map = load_user_map() # Load map from utils
        default_user_names = [user_map.get(uid) for uid in default_user_ids if user_map.get(uid) is not None]
        valid_defaults = [name for name in default_user_names if name in all_user_names]

        selected_users = st.multiselect(
            "Select Users to Display",
            options=all_user_names,
            default=valid_defaults
        )
        if not selected_users:
             st.info("Select one or more users to display activity.")
             filtered_df = pd.DataFrame()
        else:
             filtered_df = results_df[results_df["user_name"].isin(selected_users)]

    if not filtered_df.empty:
        logger.info("Generating plot...")
        try:
            fig = px.line(
                filtered_df,
                x="month_str",
                y="total",
                color="user_name",
                markers=True,
                title="Monthly Total (Open Tasks + Work Orders + Closed Tasks) by User"
            )
            fig.update_layout(
                xaxis_title="Month",
                yaxis_title="Total Count",
                template="plotly_white",
                height=600,
                margin=dict(l=50, r=50, t=50, b=150), # Adjusted margin
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.2,
                    x=0.5,
                    xanchor="center"
                )
            )
            fig.update_yaxes(rangemode='tozero')

            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
             logger.exception("Failed to generate plot.")
             st.error(f"Error generating plot: {e}")

        st.subheader("Data Summary")
        try:
            display_cols = ["month_str", "user_name", "open_tasks", "work_order_count", "closed_tasks", "total"]
            display_df = filtered_df[[col for col in display_cols if col in filtered_df.columns]].copy()
            st.dataframe(display_df.sort_values(["month_str", "user_name"]), hide_index=True)
        except Exception as e:
             logger.exception("Failed to display data table.")
             st.error(f"Error displaying data table: {e}")

    elif selected_users:
         st.info("No data found for the selected users in this period.")

    logger.info("Report execution finished.")

# --- Removed __main__ block ---
# if __name__ == "__main__":
#     main(date(2020, 3, 15), date(2025, 3, 15))
