import streamlit as st
import pandas as pd
import json
import logging # <-- Added
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
from sqlalchemy import text # <-- Removed create_engine
import plotly.express as px
# from dateutil import tz # <-- No longer needed directly

# Import shared functions and constants from utils
from web_version.utils import ( # <-- Absolute import
    get_engine,
    load_user_map,
    parse_created_by_user,
    utc_to_pacific,
    EXCLUDE_TIME_PACIFIC # If needed by this report specifically
)

# Configure logging for this report module
logger = logging.getLogger(__name__)

# --- Daily Tasks Started (from task_history) ---
@st.cache_data(ttl=600)
def compute_daily_tasks_started(start_date, end_date):
    logger.info("Computing daily tasks started...")
    engine = get_engine()
    if engine is None: return pd.DataFrame(columns=["activity_date", "creator_user_id", "creator_user_name", "tasks_started"])
    query = text("""
        SELECT DISTINCT ON (task_id)
            task_id,
            created_datetime,
            created_by_user
        FROM task_history
        WHERE created_datetime >= :start_date
          AND created_datetime <= :end_date
        ORDER BY task_id, created_datetime ASC
    """)
    try:
        with engine.connect() as conn:
            df = pd.read_sql_query(query, conn, params={"start_date": start_date, "end_date": end_date})
        logger.info(f"Loaded {len(df)} initial task events.")
    except Exception as e:
        logger.exception("Failed to query daily tasks started.")
        st.error(f"DB Error loading tasks started: {e}")
        return pd.DataFrame(columns=["activity_date", "creator_user_id", "creator_user_name", "tasks_started"])

    df["created_datetime"] = pd.to_datetime(df["created_datetime"], errors="coerce")
    df["created_date"] = df["created_datetime"].dt.date
    # Apply parsing safely
    parsed_users = df["created_by_user"].apply(parse_created_by_user)
    df[["creator_user_id", "creator_user_name", "creator_user_type"]] = pd.DataFrame(parsed_users.tolist(), index=df.index)

    df = df[df["creator_user_id"].notnull()] # Filter out non-staff/unparseable
    if df.empty:
        logger.warning("No staff-created tasks found after parsing.")
        return pd.DataFrame(columns=["activity_date", "creator_user_id", "creator_user_name", "tasks_started"])

    daily_counts = df.groupby(["created_date", "creator_user_id", "creator_user_name"]).size()\
                     .reset_index(name="tasks_started")
    daily_counts.rename(columns={"created_date": "activity_date"}, inplace=True) # Rename here
    return daily_counts

# --- Daily Work Orders Started (from work_orders joined to task_history) ---
@st.cache_data(ttl=600)
def compute_daily_work_orders_started(start_date, end_date):
    logger.info("Computing daily work orders started...")
    engine = get_engine()
    if engine is None: return pd.DataFrame(columns=["activity_date", "creator_user_id", "creator_user_name", "work_orders_started"])
    query = text("""
        SELECT DISTINCT ON (w.id)
            w.id AS work_order_id,
            th.created_datetime,
            th.created_by_user
        FROM work_orders w
        JOIN task_history th ON w.task_id = th.task_id
        WHERE th.created_datetime >= :start_date
          AND th.created_datetime <= :end_date
        ORDER BY w.id, th.created_datetime ASC
    """)
    try:
        with engine.connect() as conn:
            df = pd.read_sql_query(query, conn, params={"start_date": start_date, "end_date": end_date})
        logger.info(f"Loaded {len(df)} initial work order events.")
    except Exception as e:
        logger.exception("Failed to query daily work orders started.")
        st.error(f"DB Error loading work orders started: {e}")
        return pd.DataFrame(columns=["activity_date", "creator_user_id", "creator_user_name", "work_orders_started"])

    df["created_datetime"] = pd.to_datetime(df["created_datetime"], errors="coerce")
    df["created_date"] = df["created_datetime"].dt.date
    # Apply parsing safely
    parsed_users = df["created_by_user"].apply(parse_created_by_user)
    df[["creator_user_id", "creator_user_name", "creator_user_type"]] = pd.DataFrame(parsed_users.tolist(), index=df.index)

    df = df[df["creator_user_id"].notnull()] # Filter out non-staff/unparseable
    if df.empty:
        logger.warning("No staff-created work orders found after parsing.")
        return pd.DataFrame(columns=["activity_date", "creator_user_id", "creator_user_name", "work_orders_started"])

    daily_counts = df.groupby(["created_date", "creator_user_id", "creator_user_name"]).size()\
                     .reset_index(name="work_orders_started")
    daily_counts.rename(columns={"created_date": "activity_date"}, inplace=True) # Rename here
    return daily_counts

# --- Daily Notes/Uploads (from task_history) ---
@st.cache_data(ttl=600)
def compute_daily_notes(start_date, end_date):
    logger.info("Computing daily notes/uploads...")
    engine = get_engine()
    if engine is None: return pd.DataFrame(columns=["activity_date", "creator_user_id", "creator_user_name", "notes_count"])
    query = text("""
        SELECT
            id AS history_id,
            created_datetime AS event_time,
            created_by_user,
            message
        FROM task_history
        WHERE created_datetime >= :start_date
          AND created_datetime <= :end_date
    """)
    try:
        with engine.connect() as conn:
            df_history = pd.read_sql_query(query, conn, params={"start_date": start_date, "end_date": end_date})
        logger.info(f"Loaded {len(df_history)} history events for notes check.")
    except Exception as e:
        logger.exception("Failed to query task history for notes.")
        st.error(f"DB Error loading history for notes: {e}")
        return pd.DataFrame(columns=["activity_date", "creator_user_id", "creator_user_name", "notes_count"])

    df_history["event_time"] = pd.to_datetime(df_history["event_time"], errors="coerce")

    # Query file associations
    files_query = text("""
        SELECT DISTINCT task_history_id
        FROM task_history_files
    """)
    try:
        with engine.connect() as conn:
            df_files = pd.read_sql_query(files_query, conn)
        logger.info(f"Loaded {len(df_files)} file association records.")
        file_events = set(df_files["task_history_id"].dropna().unique())
    except Exception as e:
        logger.exception("Failed to query task history files.")
        st.error(f"DB Error loading file associations: {e}")
        # Continue without file info, logging a warning
        logger.warning("Proceeding with note calculation without file association data.")
        file_events = set() # Empty set

    # Apply parsing safely
    parsed_users = df_history["created_by_user"].apply(parse_created_by_user)
    df_history[["creator_user_id", "creator_user_name", "creator_user_type"]] = pd.DataFrame(parsed_users.tolist(), index=df_history.index)

    df_history = df_history[df_history["creator_user_id"].notnull()] # Filter out non-staff/unparseable

    def is_note_or_upload(row):
        has_message = bool(row["message"] and str(row["message"]).strip() != "")
        has_file = row["history_id"] in file_events
        return has_message or has_file

    df_history["is_note_or_upload"] = df_history.apply(is_note_or_upload, axis=1)
    df_notes = df_history[df_history["is_note_or_upload"]].copy()

    if df_notes.empty:
        logger.warning("No staff notes or uploads found after filtering.")
        return pd.DataFrame(columns=["activity_date", "creator_user_id", "creator_user_name", "notes_count"])

    df_notes["event_date"] = df_notes["event_time"].dt.date
    daily_counts = df_notes.groupby(["event_date", "creator_user_id", "creator_user_name"]).size()\
                            .reset_index(name="notes_count")
    daily_counts.rename(columns={"event_date": "activity_date"}, inplace=True) # Rename here
    return daily_counts

# --- Daily Tasks Closed/Completed (from task_history) ---
@st.cache_data(ttl=600)
def compute_daily_tasks_closed(start_date, end_date):
    logger.info("Computing daily tasks closed...")
    engine = get_engine()
    if engine is None: return pd.DataFrame(columns=["activity_date", "creator_user_id", "creator_user_name", "tasks_closed"])
    query = text("""
        SELECT
            task_id,
            created_datetime,
            task_status,
            created_by_user
        FROM task_history
        WHERE created_datetime >= :start_date
          AND created_datetime <= :end_date
          AND task_status IN ('Closed', 'Completed')
        ORDER BY task_id, created_datetime ASC -- Order needed if using DISTINCT ON, but not strictly necessary for groupby
    """)
    try:
        with engine.connect() as conn:
            df = pd.read_sql_query(query, conn, params={"start_date": start_date, "end_date": end_date})
        logger.info(f"Loaded {len(df)} closed/completed task events.")
    except Exception as e:
        logger.exception("Failed to query daily tasks closed.")
        st.error(f"DB Error loading closed tasks: {e}")
        return pd.DataFrame(columns=["activity_date", "creator_user_id", "creator_user_name", "tasks_closed"])

    df["created_datetime"] = pd.to_datetime(df["created_datetime"], errors="coerce")
    df["created_date"] = df["created_datetime"].dt.date
    # Apply parsing safely
    parsed_users = df["created_by_user"].apply(parse_created_by_user)
    df[["creator_user_id", "creator_user_name", "creator_user_type"]] = pd.DataFrame(parsed_users.tolist(), index=df.index)

    df = df[df["creator_user_id"].notnull()] # Filter out non-staff/unparseable
    if df.empty:
        logger.warning("No staff-closed tasks found after parsing.")
        return pd.DataFrame(columns=["activity_date", "creator_user_id", "creator_user_name", "tasks_closed"])

    daily_counts = df.groupby(["created_date", "creator_user_id", "creator_user_name"]).size()\
                     .reset_index(name="tasks_closed")
    daily_counts.rename(columns={"created_date": "activity_date"}, inplace=True) # Rename here
    return daily_counts

# --- Combine All Metrics ---
@st.cache_data(ttl=600)
def compute_daily_activity(start_date, end_date):
    logger.info("Computing combined daily activity...")
    tasks_started = compute_daily_tasks_started(start_date, end_date)
    work_orders_started = compute_daily_work_orders_started(start_date, end_date)
    notes = compute_daily_notes(start_date, end_date)
    tasks_closed = compute_daily_tasks_closed(start_date, end_date)

    # Define expected columns for the final DataFrame
    final_cols = ["activity_date", "creator_user_id", "user_name",
                  "tasks_started", "work_orders_started", "notes_count", "tasks_closed", "total_activity"]
    empty_df = pd.DataFrame(columns=final_cols)


    # Check if all input dataframes are empty
    if tasks_started.empty and work_orders_started.empty and notes.empty and tasks_closed.empty:
         logger.warning("All input dataframes for daily activity are empty.")
         return empty_df

    # Standardize date column names (already done in compute functions)

    merge_cols = ["activity_date", "creator_user_id", "creator_user_name"]

    # Perform merges sequentially, starting with a potentially empty frame if the first is empty
    dfs_to_merge = [tasks_started, work_orders_started, notes, tasks_closed]
    # Initialize df with the first non-empty dataframe or an empty one with merge columns
    df = pd.DataFrame(columns=merge_cols)
    first_df_found = False
    for i, current_df in enumerate(dfs_to_merge):
         if not current_df.empty:
             # Ensure merge columns exist
             for col in merge_cols:
                 if col not in current_df.columns:
                     logger.warning(f"Merge column '{col}' missing in dataframe {i}. Adding with NaNs.")
                     current_df[col] = pd.NA # Use pd.NA for consistency
             if not first_df_found:
                 df = current_df.copy() # Use copy to avoid modifying original cached df
                 first_df_found = True
             else:
                 # Ensure merge columns are of compatible types before merge
                 for col in merge_cols:
                      if col in df.columns and col in current_df.columns:
                           if df[col].dtype != current_df[col].dtype:
                                try:
                                     # Attempt to convert to object type for safe merging
                                     df[col] = df[col].astype(object)
                                     current_df[col] = current_df[col].astype(object)
                                except Exception as e:
                                     logger.warning(f"Could not align dtype for merge column '{col}': {e}")
                 df = pd.merge(df, current_df, how="outer", on=merge_cols)


    if not first_df_found: # If all input dfs were empty
         logger.warning("DataFrame is empty after attempting merges.")
         return empty_df


    # Fill NaNs and calculate total
    activity_cols = ["tasks_started", "work_orders_started", "notes_count", "tasks_closed"]
    for col in activity_cols:
        if col not in df.columns: # Handle cases where a df was empty and merge didn't add the col
             df[col] = 0
        df[col] = df[col].fillna(0).astype(int)

    df["total_activity"] = df[activity_cols].sum(axis=1)

    # Ensure user_name column exists, handling potential all-NaN cases after merge
    if "creator_user_name" in df.columns:
         # Use fillna('') before assigning to handle potential NaNs from outer merge
         df["user_name"] = df["creator_user_name"].fillna("Unknown")
    else:
         df["user_name"] = "Unknown" # Or handle as appropriate

    # Ensure activity_date column exists and convert
    if "activity_date" in df.columns:
         df["activity_date"] = pd.to_datetime(df["activity_date"]).dt.date
    else:
         df["activity_date"] = pd.NaT # Or handle as appropriate


    # Ensure all final columns exist, adding missing ones with default values if necessary
    for col in final_cols:
        if col not in df.columns:
             logger.warning(f"Final column '{col}' was missing after merge. Adding default.")
             if col in activity_cols + ["total_activity"]:
                 df[col] = 0
             elif col == "user_name":
                 df[col] = "Unknown"
             elif col == "activity_date":
                 df[col] = pd.NaT
             else: # creator_user_id
                 df[col] = pd.NA # Use pd.NA for missing IDs


    logger.info(f"Computed daily activity, resulting shape: {df.shape}")
    # Ensure consistent column order
    return df[final_cols]

# --- Main Function ---
def main(start_date, end_date):
    # st.header("Daily User Activity Report") # Removed - Redundant
    # st.write(f"Activity from **{start_date}** to **{end_date}**") # Removed - Redundant
    logger.info(f"Running Daily User Activity report for {start_date} to {end_date}")

    df_activity = compute_daily_activity(start_date, end_date)
    if df_activity.empty:
        st.warning("No activity data found for the selected period.")
        logger.warning("No activity data found, exiting report.")
        return

    # --- User Selection ---
    logger.debug("Populating user selection...")
    # Filter out potential None or NaN user names before sorting unique
    user_options = sorted([name for name in df_activity["user_name"].unique() if pd.notna(name) and name != "Unknown"])

    if not user_options:
         st.warning("No users with activity found in this period.")
         logger.warning("No valid user names found in activity data.")
         # Display unfiltered data? Or maybe just stop?
         filtered_df = df_activity # Show all data including 'Unknown' if no specific users found
         selected_users = [] # Indicate no users were selected
    else:
        # Set default user IDs and map them to names.
        default_user_ids = [6601768, 2810100, 6590396, 2054428, 4925505, 6149464, 4325638]
        user_map = load_user_map() # Load map from utils
        default_user_names = [user_map.get(uid) for uid in default_user_ids if user_map.get(uid) is not None]
        # Only include defaults that actually appear in the data for this period
        valid_defaults = [name for name in default_user_names if name in user_options]

        selected_users = st.multiselect("Select Users to Display", options=user_options, default=valid_defaults)
        if not selected_users:
             st.info("Select one or more users to display activity.")
             filtered_df = pd.DataFrame() # Show empty if no users selected
        else:
             filtered_df = df_activity[df_activity["user_name"].isin(selected_users)]

    # --- Plotting ---
    if not filtered_df.empty:
        logger.info(f"Generating plot for selected users.")
        try:
            # Ensure activity_date is suitable for plotting
            if pd.api.types.is_datetime64_any_dtype(filtered_df['activity_date']) or pd.api.types.is_object_dtype(filtered_df['activity_date']):
                 # Convert object dates if necessary, handling potential errors
                 try:
                      plot_df = filtered_df.copy()
                      plot_df['activity_date'] = pd.to_datetime(plot_df['activity_date'], errors='coerce')
                      plot_df = plot_df.dropna(subset=['activity_date']) # Drop rows where date conversion failed
                 except Exception as date_err:
                      logger.error(f"Error converting activity_date for plotting: {date_err}")
                      st.error("Could not process dates for plotting.")
                      plot_df = pd.DataFrame() # Avoid plotting with bad dates
            else:
                 plot_df = filtered_df

            if not plot_df.empty:
                fig = px.line(plot_df, x="activity_date", y="total_activity", color="user_name",
                              markers=True, title="Daily Total Activity by User")
                fig.update_layout(
                     xaxis_title="Date",
                     yaxis_title="Total Activity",
                     template="plotly_white",
                     height=600,
                     legend=dict(
                        orientation="h",
                        yanchor="top",
                        y=-0.3, # Adjust position if needed
                        xanchor="center",
                        x=0.5,
                        font=dict(size=10)
                     ),
                 margin=dict(b=200) # Adjust bottom margin if needed
                )
                st.plotly_chart(fig, use_container_width=True)

                # --- Add plain English description ---
                st.markdown("""
                **How this graph is created:**
                *   This graph shows the total daily activity for each selected user.
                *   "Total Activity" is calculated by summing the number of:
                    *   Tasks created by the user that day.
                    *   Work Orders created by the user that day.
                    *   Notes added or files uploaded to tasks by the user that day.
                    *   Tasks marked as 'Closed' or 'Completed' by the user that day.
                *   This provides a granular view of user engagement with the task and work order systems.
                """)
                # --- End description ---

            else:
                 st.info("No valid data to plot after date processing.")

        except Exception as e:
            error_msg = f"Failed to generate plot: {e}"
            logger.exception(error_msg)
            st.error(error_msg)

        # --- Display Data Table ---
        st.subheader("Detailed Daily Activity Data")
        # Ensure columns exist before sorting/displaying
        display_cols = ["activity_date", "user_name", "tasks_started", "work_orders_started", "notes_count", "tasks_closed", "total_activity"]
        display_df = filtered_df[[col for col in display_cols if col in filtered_df.columns]].copy() # Use copy

        # Convert activity_date to datetime for proper sorting if it's not already
        if "activity_date" in display_df.columns and not pd.api.types.is_datetime64_any_dtype(display_df['activity_date']):
             display_df['activity_date'] = pd.to_datetime(display_df['activity_date'], errors='coerce')

        # Sort, handling potential NaT dates
        sort_cols = []
        if "activity_date" in display_df.columns: sort_cols.append("activity_date")
        if "user_name" in display_df.columns: sort_cols.append("user_name")

        if sort_cols:
             st.dataframe(display_df.sort_values(by=sort_cols, na_position='first'))
        else:
             st.dataframe(display_df)


    elif selected_users: # Only show if users were selected but resulted in no data
         st.info("No activity data found for the selected users in this period.")


    logger.info("Report execution finished.")


if __name__ == "__main__":
    # Example usage for running the script directly
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
    # Avoid adding handler multiple times if script is re-run in some environments
    if not root_logger.hasHandlers() or not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
         root_logger.addHandler(log_handler)
    root_logger.setLevel(logging.INFO) # Or DEBUG for more detail

    # Define example date range
    example_start = date(2024, 1, 1)
    example_end = date.today()
    print(f"Running report directly for {example_start} to {example_end}")
    main(example_start, example_end)
