import streamlit as st
import pandas as pd
import json
import logging # <-- Added
from datetime import date, datetime, time, timedelta
from dateutil.relativedelta import relativedelta
from sqlalchemy import text # <-- Removed create_engine
import plotly.express as px
# from dateutil import tz # <-- Removed

# Import shared functions and constants from utils
from web_version.utils import ( # <-- Absolute import
    get_engine,
    load_user_map,
    parse_created_by_user,
    utc_to_pacific,
    OPEN_STATUSES,
    DEFERRED_STATUSES,
    CLOSED_STATUSES,
    EXCLUDE_TIME_PACIFIC # If needed by this report specifically
)

# Configure logging for this report module
logger = logging.getLogger(__name__)

# --- Load Task Data (using task_history only) ---
@st.cache_data(ttl=600)
def load_task_data(snapshot_end_date):
    """
    Load task data from the task_history table only (so that we have created_by_user).
    """
    logger.info(f"Loading task history data up to {snapshot_end_date}...")
    engine = get_engine()
    if engine is None:
        logger.error("Failed to get DB engine for task data.")
        return pd.DataFrame()

    query = text("""
        SELECT
            task_id,
            created_datetime AS event_time,
            task_status AS event_status,
            created_by_user
        FROM task_history
        WHERE created_datetime <= :snapshot_end_date
    """)
    try:
        with engine.connect() as conn:
            df = pd.read_sql_query(query, conn, params={"snapshot_end_date": snapshot_end_date})
        logger.info(f"Loaded {len(df)} task history events.")
    except Exception as e:
        logger.exception("Failed to load task history data.")
        st.error(f"DB Error loading task history: {e}")
        return pd.DataFrame()

    df.sort_values(by=["task_id", "event_time"], inplace=True)
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    df = df.dropna(subset=["event_time"]) # Drop rows where conversion failed
    return df

# --- Build Status Timeline (using new responsibility logic) ---
def build_status_timeline(df):
    """
    Build a timeline dictionary for each task:
      {task_id: [(event_time, event_status, responsible), ...]}
    where responsible is obtained by parsing created_by_user.
    Each timeline is sorted chronologically.
    """
    logger.debug("Building status timelines...")
    timeline_dict = {}
    for task_id, group in df.groupby("task_id"):
        group = group.sort_values(by="event_time")
        timeline = []
        for row in group.itertuples(index=False):
            # Ensure event_time is valid before parsing user
            if pd.isna(row.event_time):
                 logger.warning(f"Skipping row with NaT event_time for task {task_id}")
                 continue

            responsible = parse_created_by_user(row.created_by_user)
            # We only care about events created by staff for responsibility
            if responsible[0] is not None:
                timeline.append((row.event_time, row.event_status, responsible))
            # else:
                # logger.debug(f"Skipping non-staff event for task {task_id} at {row.event_time}")

        # Only add tasks that have at least one staff event in their timeline
        if timeline:
             timeline_dict[task_id] = timeline
        # else:
             # logger.debug(f"Task {task_id} has no staff events, excluding from timeline dict.")

    logger.debug(f"Built timelines for {len(timeline_dict)} tasks with staff events.")
    return timeline_dict

def compute_effective_closed_date_up_to(timeline, month_end):
    """
    For a given task timeline, simulate state changes up to month_end.
    Returns the effective closed date if the task transitions to a CLOSED status
    and remains closed (i.e. no subsequent OPEN or DEFERRED event).
    """
    effective_closed = None
    last_status = None
    for event in timeline:
        if len(event) < 2: continue # Basic validation
        event_time, event_status = event[0], event[1]

        # Ensure event_time is valid datetime before comparison
        if not isinstance(event_time, datetime) or pd.isna(event_time):
             logger.warning(f"Invalid event_time encountered in timeline: {event_time}")
             continue

        if event_time.date() > month_end:
            break # Stop processing events after the target month end

        status = str(event_status).strip() if event_status is not None else ""
        last_status = status # Track the most recent status within the timeframe

        # If it enters a closed state, record the time
        if status in CLOSED_STATUSES:
            effective_closed = event_time
        # If it enters an open/deferred state *after* being closed, reset effective_closed
        elif (status in OPEN_STATUSES or status in DEFERRED_STATUSES) and effective_closed is not None:
             # Check if this reopening event happened after the potential closure
             if event_time > effective_closed:
                  effective_closed = None # Task was reopened

    # Final check: If the loop finished and the *very last* status wasn't closed, it's not effectively closed
    # This handles cases where the last event before month_end was a reopen
    # However, the logic above already resets effective_closed if reopened after closure.
    # This check might be redundant unless the last event *is* the reopening event.
    # Let's rely on the loop logic.

    return effective_closed


def get_last_status_and_user_up_to(timeline, month_end):
    """
    Return the last event's responsible info (from created_by_user) on or before month_end.
    """
    last_status = None
    last_responsible = None
    for event in timeline:
        if len(event) < 3: continue # Basic validation
        event_time, event_status, responsible = event

        if not isinstance(event_time, datetime) or pd.isna(event_time):
             logger.warning(f"Invalid event_time encountered in timeline: {event_time}")
             continue

        if event_time.date() <= month_end:
            last_status = event_status
            last_responsible = responsible # This is the user tuple (id, name, type)
        else:
            break # Stop processing events after the target month end
    return last_status, last_responsible

def generate_month_list(start_date, end_date):
    months = []
    current = date(start_date.year, start_date.month, 1)
    end_month = date(end_date.year, end_date.month, 1)
    while current <= end_month:
        months.append(current)
        current += relativedelta(months=1)
    return months

@st.cache_data(ttl=600)
def compute_monthly_closed_tasks(start_date, end_date):
    """
    Compute monthly counts of tasks that were closed (i.e. transitioned to a CLOSED status)
    within the snapshot month, attributing the task to the responsible user (from created_by_user).
    """
    logger.info("Computing monthly closed tasks...")
    user_map = load_user_map()
    # Load data up to the end date of the analysis period
    combined_df = load_task_data(end_date)
    if combined_df.empty:
        logger.warning("No task history data loaded.")
        return pd.DataFrame()

    timeline_dict = build_status_timeline(combined_df)
    if not timeline_dict:
         logger.warning("No valid task timelines built.")
         return pd.DataFrame()

    # Calculate task creation dates safely
    task_creation = {}
    for task_id, events in timeline_dict.items():
         if events and isinstance(events[0][0], datetime):
             task_creation[task_id] = events[0][0].date()
         else:
             logger.warning(f"Task {task_id} has invalid/empty timeline for creation date.")

    month_list = generate_month_list(start_date, end_date)
    monthly_user_counts = []
    logger.info(f"Analyzing {len(month_list)} months for closed tasks...")

    for m in month_list:
        month_end = m + relativedelta(day=31)
        logger.debug(f"Processing month: {m.strftime('%Y-%m')}")
        user_counts = {}
        for task_id, events in timeline_dict.items():
            # Ensure task was created before the end of the month being processed
            creation_date = task_creation.get(task_id)
            if creation_date is None or creation_date > month_end:
                continue

            # Find the effective closed date considering only events up to month_end
            effective_closed = compute_effective_closed_date_up_to(events, month_end)

            # Check if the task was effectively closed *within* the current month 'm'
            if effective_closed is not None and m <= effective_closed.date() <= month_end:
                # Get the user responsible for the *last* event up to the closure date
                # (or month_end, whichever is earlier, though closure date makes more sense)
                _, last_responsible = get_last_status_and_user_up_to(events, effective_closed.date())

                if last_responsible and last_responsible[0] is not None: # Check if user ID is valid
                    uid = last_responsible[0]
                    user_counts[uid] = user_counts.get(uid, 0) + 1
                # else:
                    # logger.debug(f"Task {task_id} closed in {m.strftime('%Y-%m')} but last user unknown or not staff.")

        # Record counts for the month
        for uid, count in user_counts.items():
            monthly_user_counts.append({
                "month": m,
                "user_id": uid,
                "closed_tasks": count
            })
        logger.debug(f"Month {m.strftime('%Y-%m')}: Found closed tasks for {len(user_counts)} users.")

    if not monthly_user_counts:
         logger.warning("No closed tasks found across all months.")
         return pd.DataFrame()

    results_df = pd.DataFrame(monthly_user_counts)
    # Ensure 'month' column is datetime before using .dt accessor
    results_df['month'] = pd.to_datetime(results_df['month'], errors='coerce')
    results_df = results_df.dropna(subset=['month']) # Drop rows where month conversion failed
    if results_df.empty:
         logger.warning("DataFrame became empty after ensuring 'month' column is datetime.")
         return pd.DataFrame()

    results_df["month_str"] = results_df["month"].dt.strftime("%Y-%m")
    results_df["user_name"] = results_df["user_id"].map(user_map).fillna("Unassigned")

    logger.info(f"Finished computing monthly closed tasks, shape: {results_df.shape}")
    return results_df

def main(start_date, end_date):
    st.header("Monthly Closed Tasks by User") # Simplified title
    st.write(f"Displaying tasks closed between **{start_date}** and **{end_date}**")
    st.caption("Counts tasks that transitioned to 'Closed' or 'Completed' status within each month, attributed to the user of the last event before or at closure.")
    logger.info(f"Running Closed Tasks by User report for {start_date} to {end_date}")

    results_df = compute_monthly_closed_tasks(start_date, end_date)
    if results_df.empty:
        st.warning("No closed tasks found for the given period.")
        logger.warning("No closed task data computed.")
        return

    all_user_names = sorted([name for name in results_df["user_name"].unique() if pd.notna(name) and name != "Unassigned"])

    if not all_user_names:
         st.warning("No users found associated with closed tasks.")
         logger.warning("No valid user names found in results.")
         filtered_df = results_df # Show unassigned if needed
         selected_users = []
    else:
        # Define default user IDs (duplicates removed) and map them to names.
        default_user_ids = [6601768, 2810100, 6590396, 2054428, 4925505, 6149464, 4325638]
        user_map = load_user_map() # Load map from utils
        default_user_names = [user_map.get(uid) for uid in default_user_ids if user_map.get(uid) is not None]
        valid_defaults = [name for name in default_user_names if name in all_user_names]

        selected_users = st.multiselect("Select Users to Display", options=all_user_names, default=valid_defaults)
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
                y="closed_tasks",
                color="user_name",
                markers=True,
                title="Monthly Closed Tasks by User" # Simplified title
            )
            fig.update_layout(
                xaxis_title="Month",
                yaxis_title="Number of Closed Tasks",
                template="plotly_white",
                height=600,
                margin=dict(l=50, r=50, t=50, b=150), # Adjusted margin
                legend=dict(
                    orientation="h",
                    yanchor="top", # Anchor legend below plot
                    y=-0.2,      # Position below plot
                    x=0.5,
                    xanchor="center"
                )
            )
            fig.update_yaxes(rangemode='tozero')

            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
             logger.exception("Failed to generate plot.")
             st.error(f"Error generating plot: {e}")

        st.subheader("Closed Tasks Data")
        try:
            display_cols = ["month_str", "user_name", "closed_tasks"]
            display_df = filtered_df[[col for col in display_cols if col in filtered_df.columns]].copy()
            st.dataframe(display_df.sort_values(["month_str", "user_name"]), hide_index=True)
        except Exception as e:
             logger.exception("Failed to display data table.")
             st.error(f"Error displaying data table: {e}")

    elif selected_users:
         st.info("No closed tasks found for the selected users in this period.")

    logger.info("Report execution finished.")

# --- Removed __main__ block ---
# if __name__ == "__main__":
#     main(date(2020, 3, 15), date(2025, 3, 15))
