import streamlit as st
import pandas as pd
import json
import logging # <-- Added
from datetime import datetime, date, time, timedelta
from dateutil.relativedelta import relativedelta
from sqlalchemy import text # <-- Removed create_engine
import plotly.express as px
# from dateutil import tz # <-- No longer needed directly if using utils.utc_to_pacific

# Import shared functions and constants from utils
from web_version.utils import ( # <-- Changed back to absolute import
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

# --- Load task data from task_history only ---
@st.cache_data(ttl=600)
def load_task_data_for_reporting(final_end_date):
    """
    Loads all task_history events up to final_end_date.
    """
    logger.info(f"Loading task history data up to {final_end_date}...")
    engine = get_engine()
    if engine is None:
        logger.error("Cannot load task data, DB engine not available.")
        return pd.DataFrame() # Return empty DataFrame if engine failed

    # Consider adding a lower bound based on start_date if feasible
    # start_bound = start_date - relativedelta(years=1) # Example lower bound
    query = text("""
        SELECT
            task_id,
            created_datetime AS event_time,
            task_status AS event_status,
            created_by_user
        FROM task_history
        WHERE created_datetime <= :final_end_date
    """)
    try:
        with engine.connect() as conn:
            df = pd.read_sql_query(query, conn, params={"final_end_date": final_end_date})
        df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
        logger.info(f"Loaded {len(df)} task history events.")
        return df
    except Exception as e:
        error_msg = f"Failed to load task history data: {e}"
        logger.exception(error_msg)
        st.error(error_msg)
        return pd.DataFrame() # Return empty DataFrame on error

# --- Build a timeline dictionary for each task ---
def build_status_timeline(df):
    """
    Build a timeline dictionary for each task:
      {task_id: [(event_time, event_status, created_by_user), ...]}
    Sorted chronologically.
    """
    logger.debug("Building status timelines for tasks...")
    timeline_dict = {}
    for task_id, group in df.groupby("task_id"):
        # Ensure event_time is not NaT before sorting
        group = group.dropna(subset=['event_time'])
        if group.empty:
            continue
        timeline = list(group[['event_time', 'event_status', 'created_by_user']].itertuples(index=False, name=None))
        try:
            # Ensure all event times are comparable datetimes before sorting
            valid_timeline = True
            for event in timeline:
                if not isinstance(event[0], datetime):
                    logger.warning(f"Non-datetime event_time found for task {task_id}: {event[0]}. Skipping task.")
                    valid_timeline = False
                    break
            if not valid_timeline:
                continue

            timeline.sort(key=lambda x: x[0])
            timeline_dict[task_id] = timeline
        except TypeError as e:
             logger.warning(f"Error sorting timeline for task {task_id}: {e}. Skipping task.")
             continue # Skip tasks with unorderable event times
    logger.debug(f"Built timelines for {len(timeline_dict)} tasks.")
    return timeline_dict

# --- Determine if task was open in a month ---
def was_task_open_in_month(timeline, month_start, month_end):
    """
    For a given task timeline, determine if the task was in an "open" state at any point during the month.
    Returns "open" if an interval overlapping the month has a status in OPEN_STATUSES,
    "deferred" if none is open but an overlapping interval has DEFERRED status,
    or None if no overlapping open/deferred interval is found.
    """
    month_start_dt = datetime.combine(month_start, time.min)
    month_end_dt = datetime.combine(month_end, time.max)
    category = None
    for i, event_data in enumerate(timeline):
        # Ensure event_data has the expected structure
        if len(event_data) < 2 or not isinstance(event_data[0], datetime):
             logger.warning(f"Skipping invalid event data in timeline: {event_data}")
             continue

        event_time, event_status, *_ = event_data # Unpack safely

        interval_start = event_time
        interval_end = datetime.max # Assume open indefinitely if it's the last event

        if i < len(timeline) - 1:
            next_event_data = timeline[i + 1]
            if len(next_event_data) > 0 and isinstance(next_event_data[0], datetime):
                 interval_end = next_event_data[0]
            else:
                 logger.warning(f"Skipping invalid next event data: {next_event_data}")
                 continue # Skip if next event is invalid

        # Ensure interval_end is timezone-aware if interval_start is
        if interval_start.tzinfo is not None and interval_end.tzinfo is None:
             interval_end = interval_end.replace(tzinfo=interval_start.tzinfo) # Basic assumption
        elif interval_start.tzinfo is None and interval_end.tzinfo is not None:
             interval_start = interval_start.replace(tzinfo=interval_end.tzinfo) # Basic assumption

        # Ensure month boundaries are timezone-aware if intervals are
        if interval_start.tzinfo is not None:
             month_start_dt = month_start_dt.replace(tzinfo=interval_start.tzinfo)
             month_end_dt = month_end_dt.replace(tzinfo=interval_start.tzinfo)


        # Calculate overlap
        latest_start = max(interval_start, month_start_dt)
        earliest_end = min(interval_end, month_end_dt)

        # Check for valid overlap duration
        if earliest_end > latest_start:
            overlap = (earliest_end - latest_start).total_seconds()
            if overlap > 0:
                if event_status in OPEN_STATUSES:
                    return "open"
                elif event_status in DEFERRED_STATUSES:
                    category = "deferred" # Continue checking in case an 'open' status appears later

    return category


def generate_month_list(start_date, end_date):
    """
    Generate a list of the first day of each month between start_date and end_date.
    """
    months = []
    current = date(start_date.year, start_date.month, 1)
    end_month = date(end_date.year, end_date.month, 1)
    while current <= end_month:
        months.append(current)
        current += relativedelta(months=1)
    return months

# --- Main Function ---
def main(start_date, end_date):
    """Generates the monthly open/deferred tasks report."""
    st.header("Monthly Tasks Open/Deferred Report")
    st.write(f"Reporting period: **{start_date}** to **{end_date}**")
    logger.info(f"Running Tasks Total report for {start_date} to {end_date}")

    # Set final_end_date to the last day of the reporting period's month.
    # This ensures we capture all events needed to determine status at month end.
    final_end_date = (date(end_date.year, end_date.month, 1) + relativedelta(day=31))

    combined_df = load_task_data_for_reporting(final_end_date)
    if combined_df.empty:
        st.warning("No task history data found for the period.")
        logger.warning("No task history data found, exiting report.")
        return

    timeline_dict = build_status_timeline(combined_df)
    if not timeline_dict:
         st.warning("Could not build task timelines from the data.")
         logger.warning("Task timeline dictionary is empty.")
         return

    # Calculate task creation dates safely
    task_creation = {}
    for task_id, events in timeline_dict.items():
         if events and isinstance(events[0][0], datetime): # Ensure events list is not empty and first event time is valid
             task_creation[task_id] = events[0][0].date()
         else:
             logger.warning(f"Task {task_id} has an invalid or empty timeline in task_creation calculation.")


    month_list = generate_month_list(start_date, end_date)
    results = []
    logger.info(f"Analyzing {len(month_list)} months...")

    for m in month_list:
        logger.debug(f"Processing month: {m.strftime('%Y-%m')}")
        month_end = m + relativedelta(day=31)
        open_count = 0
        deferred_count = 0
        # user_counts = {} # Removed - user breakdown logic was flawed for this report

        for task_id, timeline in timeline_dict.items():
            # Ensure task existed before the end of the month being processed
            creation_date = task_creation.get(task_id)
            if creation_date is None or creation_date > month_end:
                # logger.debug(f"Skipping task {task_id} created after month {m.strftime('%Y-%m')}")
                continue

            cat = was_task_open_in_month(timeline, m, month_end)
            if cat is not None:
                # Count based on category
                if cat == "open":
                    open_count += 1
                elif cat == "deferred":
                    deferred_count += 1

        # Append results for the month (no user breakdown here for open/deferred counts)
        results.append({
            "month": m,
            "open_tasks": open_count,
            "deferred_tasks": deferred_count
        })
        logger.debug(f"Month {m.strftime('%Y-%m')}: Open={open_count}, Deferred={deferred_count}")


    if not results:
        st.warning("No monthly results generated.")
        logger.warning("Results list is empty after processing months.")
        return

    results_df = pd.DataFrame(results)
    # Ensure 'month' column is datetime before formatting
    results_df['month'] = pd.to_datetime(results_df['month'])
    results_df["month_str"] = results_df["month"].dt.strftime("%Y-%m") # Use .dt accessor

    # --- Plotting ---
    logger.info("Generating plot...")
    try:
        fig = px.line(
            results_df,
            x="month_str",
            y=["open_tasks", "deferred_tasks"],
            markers=True,
            title="Monthly Tasks Open / Deferred (Active During Month)"
        )
        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Number of Tasks",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        error_msg = f"Failed to generate plot: {e}"
        logger.exception(error_msg)
        st.error(error_msg)

    # --- Display Data Table ---
    st.subheader("Monthly Data Summary")
    display_df = results_df[['month_str', 'open_tasks', 'deferred_tasks']].sort_values(by="month_str")
    st.dataframe(display_df)
    logger.info("Report execution finished.")

if __name__ == "__main__":
    # Example usage for running the script directly
    # Ensure .env is in the parent directory relative to this script if running directly
    script_dir = os.path.dirname(__file__)
    dotenv_path = os.path.join(script_dir, '..', '.env')
    if os.path.exists(dotenv_path):
        from dotenv import load_dotenv
        load_dotenv(dotenv_path)
        print(f"Loaded .env from {dotenv_path} for direct script run")
    else:
        print("Warning: .env file not found for direct script run. DB connection might fail.")

    # Setup basic logging for direct run
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    log_handler = logging.StreamHandler()
    log_handler.setFormatter(log_formatter)
    root_logger = logging.getLogger()
    root_logger.addHandler(log_handler)
    root_logger.setLevel(logging.INFO) # Or DEBUG for more detail

    # Define example date range
    example_start = date(2024, 1, 1)
    example_end = date.today()
    print(f"Running report directly for {example_start} to {example_end}")
    main(example_start, example_end)
