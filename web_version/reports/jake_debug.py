import streamlit as st
import pandas as pd
import logging # <-- Added
from sqlalchemy import text # <-- Removed create_engine
from dateutil import tz
from datetime import datetime

# Import shared functions from utils
from web_version.utils import get_engine # <-- Import shared engine function

# Configure logging for this report module
logger = logging.getLogger(__name__)

# --- Timezone Objects ---
PACIFIC_TZ = tz.gettz('America/Los_Angeles')
UTC_TZ = tz.UTC

# --- Main Function (Specific to User 2054428) ---
def main(start_date, end_date):
    st.title("Activity Log for User 2054428 (Jake Debug)") # Clarified title
    st.write(f"Date Range: {start_date} to {end_date}")
    logger.info(f"Running Jake Debug report for {start_date} to {end_date}")

    user_id = 2054428
    engine = get_engine() # <-- Use shared engine
    if engine is None:
        logger.error("Failed to get DB engine. Aborting report.")
        return

    all_events = []  # List to store all events

    # --- Time to exclude (Pacific Time) ---
    # Updated exclusion time to match events: 2025-03-08 22:54:24 PST converts to 2025-03-09 06:54:24+00:00 UTC
    exclude_time_pacific = datetime(2025, 3, 8, 22, 54, 24, tzinfo=PACIFIC_TZ)
    exclude_time_utc = exclude_time_pacific.astimezone(UTC_TZ)  # Convert to UTC for comparison
    logger.debug(f"Exclusion time (UTC): {exclude_time_utc}")

    # Define datetime range once
    try:
        start_datetime = datetime.combine(start_date, datetime.min.time())
        end_datetime = datetime.combine(end_date, datetime.max.time())
        start_datetime_utc = pd.Timestamp(start_datetime).tz_localize(PACIFIC_TZ).tz_convert(UTC_TZ) # Assume input dates are Pacific? Or make explicit?
        end_datetime_utc = pd.Timestamp(end_datetime).tz_localize(PACIFIC_TZ).tz_convert(UTC_TZ)
        logger.debug(f"Querying UTC range: {start_datetime_utc} to {end_datetime_utc}")
    except Exception as e:
        logger.exception("Error creating date range.")
        st.error(f"Error creating date range: {e}")
        return


    # --- 1. Tasks Started ---
    logger.info("Querying tasks started...")
    try:
        with engine.connect() as conn:
            # Using created_at from tasks table directly
            query = text("""
                SELECT
                    id AS task_id,
                    created_at, -- Keep as timestamp with timezone if available
                    'Task Started' AS reason,
                    '' as details
                FROM tasks
                WHERE assigned_to_user_id = :user_id
                  AND created_at >= :start_dt_utc -- Filter in DB
                  AND created_at <= :end_dt_utc   -- Filter in DB
                  AND created_at <> :exclude_utc -- Filter in DB
            """)
            tasks_started = pd.read_sql_query(query, conn, params={
                "user_id": user_id,
                "start_dt_utc": start_datetime_utc,
                "end_dt_utc": end_datetime_utc,
                "exclude_utc": exclude_time_utc
            }, parse_dates=['created_at']) # Let pandas handle parsing

            # Ensure timezone is UTC after loading
            if not tasks_started.empty and tasks_started['created_at'].dt.tz is None:
                 logger.warning("Localizing tasks.created_at to UTC as timezone info was missing.")
                 tasks_started['created_at'] = tasks_started['created_at'].dt.tz_localize(UTC_TZ)
            elif not tasks_started.empty:
                 tasks_started['created_at'] = tasks_started['created_at'].dt.tz_convert(UTC_TZ)


            tasks_started.rename(columns={"created_at": "timestamp"}, inplace=True)
            all_events.extend(tasks_started.to_dict('records'))
            logger.info(f"Found {len(tasks_started)} 'Task Started' events.")
    except Exception as e:
        logger.exception("Error querying tasks started.")
        st.error(f"Error querying tasks started: {e}")

    # --- 2. Work Orders Started ---
    logger.info("Querying work orders started...")
    try:
        with engine.connect() as conn:
            # Using created_at from tasks table directly
            query = text("""
                SELECT
                    t.id AS task_id,
                    t.created_at, -- Keep as timestamp with timezone if available
                    'Work Order Started' AS reason,
                    '' as details
                FROM work_orders w
                JOIN tasks t ON w.task_id = t.id
                WHERE t.assigned_to_user_id = :user_id
                  AND t.created_at >= :start_dt_utc -- Filter in DB
                  AND t.created_at <= :end_dt_utc   -- Filter in DB
                  AND t.created_at <> :exclude_utc -- Filter in DB
            """)
            work_orders_started = pd.read_sql_query(query, conn, params={
                "user_id": user_id,
                "start_dt_utc": start_datetime_utc,
                "end_dt_utc": end_datetime_utc,
                "exclude_utc": exclude_time_utc
            }, parse_dates=['created_at'])

            # Ensure timezone is UTC after loading
            if not work_orders_started.empty and work_orders_started['created_at'].dt.tz is None:
                 logger.warning("Localizing work_orders.tasks.created_at to UTC as timezone info was missing.")
                 work_orders_started['created_at'] = work_orders_started['created_at'].dt.tz_localize(UTC_TZ)
            elif not work_orders_started.empty:
                 work_orders_started['created_at'] = work_orders_started['created_at'].dt.tz_convert(UTC_TZ)

            work_orders_started.rename(columns={"created_at": "timestamp"}, inplace=True)
            all_events.extend(work_orders_started.to_dict('records'))
            logger.info(f"Found {len(work_orders_started)} 'Work Order Started' events.")
    except Exception as e:
        logger.exception("Error querying work orders started.")
        st.error(f"Error querying work orders started: {e}")

    # --- 3. Notes/Uploads ---
    logger.info("Querying notes/uploads...")
    try:
        with engine.connect() as conn:
            # Using created_datetime from task_history
            query = text("""
                SELECT
                    th.task_id AS task_id,
                    th.created_datetime, -- Keep as timestamp with timezone
                    th.message,
                    CASE
                        WHEN thf.task_history_id IS NOT NULL THEN 'Note with Upload'
                        WHEN th.message IS NOT NULL AND th.message <> '' THEN 'Note'
                        ELSE 'Unknown Action' -- Should not happen based on WHERE clause
                    END as reason,
                    th.message as details
                FROM task_history th
                LEFT JOIN task_history_files thf ON th.id = thf.task_history_id
                WHERE (th.created_by_user ->> 'Id')::int = :user_id
                  AND (th.message IS NOT NULL AND th.message <> '' OR thf.task_history_id IS NOT NULL)
                  AND th.created_datetime >= :start_dt_utc -- Filter in DB
                  AND th.created_datetime <= :end_dt_utc   -- Filter in DB
                  -- No exclusion needed here as it's from task_history
            """)
            notes = pd.read_sql_query(query, conn, params={
                "user_id": user_id,
                "start_dt_utc": start_datetime_utc,
                "end_dt_utc": end_datetime_utc
            }, parse_dates=['created_datetime'])

            # Ensure timezone is UTC after loading
            if not notes.empty and notes['created_datetime'].dt.tz is None:
                 logger.warning("Localizing task_history.created_datetime for notes to UTC.")
                 notes['created_datetime'] = notes['created_datetime'].dt.tz_localize(UTC_TZ)
            elif not notes.empty:
                 notes['created_datetime'] = notes['created_datetime'].dt.tz_convert(UTC_TZ)

            notes.rename(columns={"created_datetime": "timestamp"}, inplace=True)
            all_events.extend(notes[["task_id", "timestamp", "reason", "details"]].to_dict('records'))
            logger.info(f"Found {len(notes)} 'Note/Upload' events.")
    except Exception as e:
        logger.exception("Error querying notes.")
        st.error(f"Error querying notes: {e}")

    # --- 4. Tasks Closed/Completed ---
    logger.info("Querying tasks closed/completed...")
    try:
        with engine.connect() as conn:
            # Using created_datetime from task_history
            query = text("""
                SELECT
                    task_id,
                    created_datetime, -- Keep as timestamp with timezone
                    'Task ' || task_status AS reason,
                    '' as details
                FROM task_history
                WHERE (created_by_user ->> 'Id')::int = :user_id
                  AND task_status IN ('Closed', 'Completed')
                  AND created_datetime >= :start_dt_utc -- Filter in DB
                  AND created_datetime <= :end_dt_utc   -- Filter in DB
                  -- No exclusion needed here
            """)
            tasks_closed = pd.read_sql_query(query, conn, params={
                "user_id": user_id,
                "start_dt_utc": start_datetime_utc,
                "end_dt_utc": end_datetime_utc
                }, parse_dates=['created_datetime'])

            # Ensure timezone is UTC after loading
            if not tasks_closed.empty and tasks_closed['created_datetime'].dt.tz is None:
                 logger.warning("Localizing task_history.created_datetime for closed tasks to UTC.")
                 tasks_closed['created_datetime'] = tasks_closed['created_datetime'].dt.tz_localize(UTC_TZ)
            elif not tasks_closed.empty:
                 tasks_closed['created_datetime'] = tasks_closed['created_datetime'].dt.tz_convert(UTC_TZ)

            tasks_closed.rename(columns={"created_datetime": "timestamp"}, inplace=True)
            all_events.extend(tasks_closed.to_dict('records'))
            logger.info(f"Found {len(tasks_closed)} 'Task Closed/Completed' events.")
    except Exception as e:
        logger.exception("Error querying tasks closed.")
        st.error(f"Error querying tasks closed: {e}")

    # --- Combine and Display ---
    if not all_events:
        st.warning("No activity found for this user in the given date range.")
        logger.warning("No events found for user 2054428.")
        return

    logger.info(f"Total events collected: {len(all_events)}")
    df_all = pd.DataFrame(all_events)

    # Ensure timestamp column exists and is datetime before sorting/converting
    if 'timestamp' not in df_all.columns or df_all['timestamp'].isnull().all():
         logger.warning("Timestamp column missing or all null after collecting events.")
         st.warning("Could not process timestamps for display.")
         # Display without time sorting/conversion
         st.dataframe(df_all)
         return

    # Convert to datetime if not already, coercing errors
    df_all['timestamp'] = pd.to_datetime(df_all['timestamp'], errors='coerce', utc=True)
    df_all = df_all.dropna(subset=['timestamp']) # Remove rows where conversion failed

    if df_all.empty:
         st.warning("No valid timestamps found after processing.")
         return

    # Sort by UTC timestamp
    df_all = df_all.sort_values(by="timestamp")

    # Convert UTC to Pacific for display
    try:
        df_all["pacific_time"] = df_all["timestamp"].dt.tz_convert(PACIFIC_TZ).dt.strftime('%Y-%m-%d %H:%M:%S %Z%z')
    except Exception as tz_e:
         logger.exception("Error converting timestamp to Pacific for display.")
         st.error(f"Error formatting time for display: {tz_e}")
         df_all["pacific_time"] = df_all["timestamp"].astype(str) + " (UTC conversion failed)" # Fallback display


    # Select and order columns for display
    display_cols = ["reason", "task_id", "pacific_time", "details"]
    df_display = df_all[[col for col in display_cols if col in df_all.columns]]

    st.dataframe(df_display, hide_index=True)
    logger.info("Report execution finished.")

# --- Removed __main__ block ---
# if __name__ == "__main__":
#     start_date = st.date_input("Start Date", datetime(2024, 1, 1))
#     end_date = st.date_input("End Date", datetime.now())
#     main(start_date, end_date)
