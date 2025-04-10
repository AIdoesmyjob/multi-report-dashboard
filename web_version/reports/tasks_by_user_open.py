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
    DEFERRED_STATUSES, # Not used in this specific report's logic, but keep for consistency?
    CLOSED_STATUSES,   # Not used in this specific report's logic, but keep for consistency?
    EXCLUDE_TIME_PACIFIC # If needed by this report specifically
)

# Configure logging for this report module
logger = logging.getLogger(__name__)

# --- Build Status Timeline using created_by_user ---
def build_status_timeline(df):
    """
    Build a timeline dictionary for each task:
      {task_id: [(event_time, event_status, creator_user_info), ...]}
    where creator_user_info is the result of parse_created_by_user.
    Each timeline is sorted chronologically.
    """
    logger.debug("Building status timelines...")
    timeline_dict = {}
    for task_id, group in df.groupby("task_id"):
        group = group.sort_values(by="event_time")
        timeline = []
        for row in group.itertuples(index=False):
            if pd.isna(row.event_time):
                 logger.warning(f"Skipping row with NaT event_time for task {task_id}")
                 continue
            creator_info = parse_created_by_user(row.created_by_user)
            # Only add events with valid staff info.
            if creator_info[0] is not None:
                timeline.append((row.event_time, row.event_status, creator_info))
        if timeline:
            timeline_dict[task_id] = timeline
    logger.debug(f"Built timelines for {len(timeline_dict)} tasks with staff events.")
    return timeline_dict

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

def get_open_creator_in_month(timeline, month_start, month_end):
    """
    For a given task timeline (tuples of (event_time, event_status, creator_info)),
    determine if the task has an open interval (with a status in OPEN_STATUSES) overlapping the month.
    Returns the creator_info tuple of the event that initiated the open interval if found, else None.
    """
    month_start_dt = datetime.combine(month_start, time.min)
    month_end_dt = datetime.combine(month_end, time.max)

    for i, event_data in enumerate(timeline):
        # Basic validation
        if len(event_data) < 3: continue
        event_time, event_status, creator_info = event_data
        if not isinstance(event_time, datetime) or pd.isna(event_time): continue

        # Check if the status is an OPEN status
        status_str = str(event_status).strip() if event_status is not None else ""
        if status_str not in OPEN_STATUSES:
            continue

        # Determine the end of this status interval
        interval_start = event_time
        interval_end = datetime.max # Assume open indefinitely if it's the last event

        if i < len(timeline) - 1:
            next_event_data = timeline[i + 1]
            if len(next_event_data) > 0 and isinstance(next_event_data[0], datetime):
                 interval_end = next_event_data[0]
            else:
                 # If next event is invalid, assume status continues indefinitely for overlap check
                 pass # Keep interval_end as datetime.max

        # Ensure timezone consistency for comparison (basic handling)
        tz = interval_start.tzinfo
        if tz:
             month_start_dt = month_start_dt.replace(tzinfo=tz)
             month_end_dt = month_end_dt.replace(tzinfo=tz)
             if interval_end != datetime.max and interval_end.tzinfo is None:
                  interval_end = interval_end.replace(tzinfo=tz)
        elif interval_end != datetime.max and interval_end.tzinfo is not None:
             # If interval_end has timezone but interval_start doesn't, this is ambiguous
             # For simplicity, assume they should match or log a warning
             logger.warning(f"Timezone mismatch in interval for task event at {interval_start}")
             # Attempt to proceed assuming UTC if interval_end has tz, otherwise skip interval
             if interval_end.tzinfo:
                  try:
                       interval_start = interval_start.replace(tzinfo=interval_end.tzinfo)
                       month_start_dt = month_start_dt.replace(tzinfo=interval_end.tzinfo)
                       month_end_dt = month_end_dt.replace(tzinfo=interval_end.tzinfo)
                  except ValueError: # Handle potential issues with replace
                       continue
             else:
                  continue


        # Calculate overlap
        latest_start = max(interval_start, month_start_dt)
        earliest_end = min(interval_end, month_end_dt)

        # Check for valid overlap duration
        if earliest_end > latest_start:
            overlap = (earliest_end - latest_start).total_seconds()
            if overlap > 0:
                # Task was open during this month, return the creator of this event
                return creator_info
    return None # No open interval found overlapping the month

@st.cache_data(ttl=600)
def compute_monthly_open_tasks(start_date, end_date):
    """
    Compute monthly counts of tasks that were open (at any point during the month)
    per creator (from created_by_user).

    Returns a DataFrame with columns:
      [month, creator_user_id, open_tasks, month_str, user_name]
    """
    logger.info("Computing monthly open tasks...")
    user_map = load_user_map() # Load map from utils
    # Load all task_history events up to the final month end for accurate state tracking.
    final_end_date = (date(end_date.year, end_date.month, 1) + relativedelta(day=31))
    engine = get_engine() # Use shared engine
    if engine is None: return pd.DataFrame()

    query = text("""
        SELECT
            task_id,
            created_datetime AS event_time,
            task_status AS event_status,
            created_by_user
        FROM task_history
        WHERE created_datetime <= :final_end_date -- Load history up to the end date
    """)
    try:
        with engine.connect() as conn:
            df = pd.read_sql_query(query, conn, params={"final_end_date": final_end_date})
        logger.info(f"Loaded {len(df)} task history events for open task analysis.")
    except Exception as e:
        logger.exception("Failed to load task history for open tasks.")
        st.error(f"DB Error loading task history: {e}")
        return pd.DataFrame()

    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    df = df.dropna(subset=["event_time"])
    if df.empty: return pd.DataFrame()

    # Build timeline dictionary using our parsing method.
    timeline_dict = build_status_timeline(df)
    if not timeline_dict:
         logger.warning("No valid task timelines built.")
         return pd.DataFrame()

    # Record each task's creation date (from the first event in its timeline).
    task_creation = {}
    for task_id, events in timeline_dict.items():
         if events and isinstance(events[0][0], datetime):
             task_creation[task_id] = events[0][0].date()
         else:
             logger.warning(f"Task {task_id} has invalid/empty timeline for creation date.")


    month_list = generate_month_list(start_date, end_date)
    results = []
    logger.info(f"Analyzing {len(month_list)} months for open tasks...")

    for m in month_list:
        month_end = m + relativedelta(day=31)
        logger.debug(f"Processing month: {m.strftime('%Y-%m')}")
        monthly_counts = {} # Count open tasks per user for this month
        for task_id, timeline in timeline_dict.items():
            # Ensure task was created before the end of the month being processed
            creation_date = task_creation.get(task_id)
            if creation_date is None or creation_date > month_end:
                continue

            # Check if the task had an open interval overlapping this month
            creator_info = get_open_creator_in_month(timeline, m, month_end)
            if creator_info is not None:
                uid, name, _ = creator_info # Unpack the tuple
                if uid is not None: # Ensure we have a valid user ID
                    monthly_counts[uid] = monthly_counts.get(uid, 0) + 1

        # Record counts for the month
        for uid, count in monthly_counts.items():
            results.append({
                "month": m,
                "creator_user_id": uid,
                "open_tasks": count
            })
        logger.debug(f"Month {m.strftime('%Y-%m')}: Found open tasks for {len(monthly_counts)} users.")

    if not results:
         logger.warning("No open tasks found across all months.")
         return pd.DataFrame()

    results_df = pd.DataFrame(results)
    # Ensure 'month' column is datetime before using .dt accessor
    results_df['month'] = pd.to_datetime(results_df['month'], errors='coerce')
    results_df = results_df.dropna(subset=['month'])
    if results_df.empty:
         logger.warning("DataFrame became empty after ensuring 'month' column is datetime.")
         return pd.DataFrame()

    results_df["month_str"] = results_df["month"].dt.strftime("%Y-%m")
    results_df["user_name"] = results_df["creator_user_id"].map(user_map).fillna("Unassigned")

    logger.info(f"Finished computing monthly open tasks, shape: {results_df.shape}")
    return results_df


def main(start_date, end_date):
    # st.header("Monthly Open Tasks by User") # Removed - Redundant
    # st.write(f"Displaying tasks open at any point between **{start_date}** and **{end_date}**") # Removed - Redundant
    st.caption("Counts tasks that had an 'Open' or 'In Progress' status during each month, attributed based on the user who created the event initiating the open status.") # Keep caption
    logger.info(f"Running Open Tasks by User report for {start_date} to {end_date}")

    results_df = compute_monthly_open_tasks(start_date, end_date)
    if results_df.empty:
        st.warning("No open tasks found for the given period.")
        logger.warning("No open task data computed.")
        return

    all_user_names = sorted([name for name in results_df["user_name"].unique() if pd.notna(name) and name != "Unassigned"])

    if not all_user_names:
         st.warning("No users found associated with open tasks.")
         logger.warning("No valid user names found in results.")
         filtered_df = results_df # Show unassigned if needed
         selected_users = []
    else:
        # Define default user IDs and map them to names.
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
                y="open_tasks",
                color="user_name",
                markers=True,
                title="Monthly Open Tasks by User"
            )
            fig.update_layout(
                xaxis_title="Month",
                yaxis_title="Number of Open Tasks",
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

            # --- Add plain English description ---
            st.markdown("""
            **How this graph is created:**
            *   This graph shows the number of tasks that were in an 'Open' or 'In Progress' state at any point during each month, broken down by the user who created the event that put the task into that open state.
            *   It helps visualize the open task load associated with different users over time.
            *   A task is counted for a month if its open period (from the time it was set to 'Open'/'In Progress' until its status changed again) overlaps with that month.
            """)
            # --- End description ---

        except Exception as e:
             logger.exception("Failed to generate plot.")
             st.error(f"Error generating plot: {e}")

        st.subheader("Open Tasks Data")
        try:
            display_cols = ["month_str", "user_name", "open_tasks"]
            display_df = filtered_df[[col for col in display_cols if col in filtered_df.columns]].copy()
            st.dataframe(display_df.sort_values(["month_str", "user_name"]), hide_index=True)
        except Exception as e:
             logger.exception("Failed to display data table.")
             st.error(f"Error displaying data table: {e}")

    elif selected_users:
         st.info("No open tasks found for the selected users in this period.")

    logger.info("Report execution finished.")

# --- Removed __main__ block ---
# if __name__ == "__main__":
#     main(date(2020, 3, 15), date(2025, 3, 15))
