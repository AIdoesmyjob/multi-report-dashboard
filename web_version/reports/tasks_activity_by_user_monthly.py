import streamlit as st
import pandas as pd
import json
import logging # <-- Added
from datetime import date, datetime
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
    EXCLUDE_TIME_PACIFIC # If needed by this report specifically
)

# Configure logging for this report module
logger = logging.getLogger(__name__)

# --- Monthly Tasks Started (from task_history) ---
@st.cache_data(ttl=600)
def compute_monthly_tasks_started(start_date, end_date):
    logger.info("Computing monthly tasks started...")
    engine = get_engine()
    if engine is None: return pd.DataFrame(columns=["month", "creator_user_id", "creator_user_name", "tasks_started", "month_str"])
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
        logger.exception("Failed to query monthly tasks started.")
        st.error(f"DB Error loading tasks started: {e}")
        return pd.DataFrame(columns=["month", "creator_user_id", "creator_user_name", "tasks_started", "month_str"])

    df["created_datetime"] = pd.to_datetime(df["created_datetime"], errors="coerce")
    df = df.dropna(subset=["created_datetime"])
    if df.empty: return pd.DataFrame(columns=["month", "creator_user_id", "creator_user_name", "tasks_started", "month_str"])

    # Create a monthly column (first day of the month)
    df["month"] = df["created_datetime"].dt.to_period("M").dt.to_timestamp()
    # Parse creator info from created_by_user.
    parsed_users = df["created_by_user"].apply(parse_created_by_user)
    df[["creator_user_id", "creator_user_name", "creator_user_type"]] = pd.DataFrame(parsed_users.tolist(), index=df.index)

    df = df[df["creator_user_id"].notnull()] # Filter out non-staff/unparseable
    if df.empty:
        logger.warning("No staff-created tasks found after parsing.")
        return pd.DataFrame(columns=["month", "creator_user_id", "creator_user_name", "tasks_started", "month_str"])

    monthly_counts = df.groupby(["month", "creator_user_id", "creator_user_name"]).size()\
                         .reset_index(name="tasks_started")
    monthly_counts["month_str"] = monthly_counts["month"].dt.strftime("%Y-%m")
    return monthly_counts

# --- Monthly Notes/Uploads (from task_history) ---
@st.cache_data(ttl=600)
def compute_monthly_notes(start_date, end_date):
    logger.info("Computing monthly notes/uploads...")
    engine = get_engine()
    if engine is None: return pd.DataFrame(columns=["month", "creator_user_id", "creator_user_name", "notes_count", "month_str"])
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
            df_history = pd.read_sql_query(query, conn,
                                           params={"start_date": start_date, "end_date": end_date})
        logger.info(f"Loaded {len(df_history)} history events for notes check.")
    except Exception as e:
        logger.exception("Failed to query task history for notes.")
        st.error(f"DB Error loading history for notes: {e}")
        return pd.DataFrame(columns=["month", "creator_user_id", "creator_user_name", "notes_count", "month_str"])

    df_history["event_time"] = pd.to_datetime(df_history["event_time"], errors="coerce")
    df_history = df_history.dropna(subset=["event_time"])
    if df_history.empty: return pd.DataFrame(columns=["month", "creator_user_id", "creator_user_name", "notes_count", "month_str"])

    # Create monthly column from event_time.
    df_history["month"] = df_history["event_time"].dt.to_period("M").dt.to_timestamp()

    # Query file upload events.
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
        logger.warning("Proceeding with note calculation without file association data.")
        file_events = set()

    # Parse creator info from created_by_user.
    parsed_users = df_history["created_by_user"].apply(parse_created_by_user)
    df_history[["creator_user_id", "creator_user_name", "creator_user_type"]] = pd.DataFrame(parsed_users.tolist(), index=df_history.index)

    df_history = df_history[df_history["creator_user_id"].notnull()]

    def is_note_or_upload(row):
        has_message = bool(row["message"] and str(row["message"]).strip() != "")
        has_file = row["history_id"] in file_events
        return has_message or has_file

    df_history["is_note_or_upload"] = df_history.apply(is_note_or_upload, axis=1)
    df_events = df_history[df_history["is_note_or_upload"]].copy()

    if df_events.empty:
        logger.warning("No staff notes or uploads found after filtering.")
        return pd.DataFrame(columns=["month", "creator_user_id", "creator_user_name", "notes_count", "month_str"])

    monthly_counts = df_events.groupby(["month", "creator_user_id", "creator_user_name"]).size()\
                                .reset_index(name="notes_count")
    monthly_counts["month_str"] = monthly_counts["month"].dt.strftime("%Y-%m")
    return monthly_counts

# --- Monthly Tasks Closed/Completed (from task_history) ---
@st.cache_data(ttl=600)
def compute_monthly_tasks_closed(start_date, end_date):
    logger.info("Computing monthly tasks closed...")
    engine = get_engine()
    if engine is None: return pd.DataFrame(columns=["month", "creator_user_id", "creator_user_name", "tasks_closed", "month_str"])
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
        ORDER BY task_id, created_datetime ASC
    """)
    try:
        with engine.connect() as conn:
            df = pd.read_sql_query(query, conn, params={"start_date": start_date, "end_date": end_date})
        logger.info(f"Loaded {len(df)} closed/completed task events.")
    except Exception as e:
        logger.exception("Failed to query monthly tasks closed.")
        st.error(f"DB Error loading closed tasks: {e}")
        return pd.DataFrame(columns=["month", "creator_user_id", "creator_user_name", "tasks_closed", "month_str"])

    df["created_datetime"] = pd.to_datetime(df["created_datetime"], errors="coerce")
    df = df.dropna(subset=["created_datetime"])
    if df.empty: return pd.DataFrame(columns=["month", "creator_user_id", "creator_user_name", "tasks_closed", "month_str"])

    df["month"] = df["created_datetime"].dt.to_period("M").dt.to_timestamp()
    parsed_users = df["created_by_user"].apply(parse_created_by_user)
    df[["creator_user_id", "creator_user_name", "creator_user_type"]] = pd.DataFrame(parsed_users.tolist(), index=df.index)

    df = df[df["creator_user_id"].notnull()]
    if df.empty:
        logger.warning("No staff-closed tasks found after parsing.")
        return pd.DataFrame(columns=["month", "creator_user_id", "creator_user_name", "tasks_closed", "month_str"])

    monthly_counts = df.groupby(["month", "creator_user_id", "creator_user_name"]).size()\
                         .reset_index(name="tasks_closed")
    monthly_counts["month_str"] = monthly_counts["month"].dt.strftime("%Y-%m")
    return monthly_counts

# --- Combine All Monthly Metrics (without work orders) ---
@st.cache_data(ttl=600)
def compute_monthly_activity(start_date, end_date):
    logger.info("Computing combined monthly activity...")
    ts = compute_monthly_tasks_started(start_date, end_date)
    nt = compute_monthly_notes(start_date, end_date)
    tc = compute_monthly_tasks_closed(start_date, end_date)

    # Define expected columns for the final DataFrame
    final_cols = ["activity_month", "creator_user_id", "user_name",
                  "tasks_started", "notes_count", "tasks_closed", "total_activity", "month_str"]
    empty_df = pd.DataFrame(columns=final_cols)

    if ts.empty and nt.empty and tc.empty:
        logger.warning("All input dataframes for monthly activity are empty.")
        return empty_df

    merge_cols = ["month", "creator_user_id", "creator_user_name"]

    # Perform merges sequentially
    dfs_to_merge = [ts, nt, tc]
    df = pd.DataFrame(columns=merge_cols)
    first_df_found = False
    for i, current_df in enumerate(dfs_to_merge):
         if not current_df.empty:
             # Ensure merge columns exist
             for col in merge_cols:
                 if col not in current_df.columns:
                     logger.warning(f"Merge column '{col}' missing in dataframe {i}. Adding with NaNs.")
                     current_df[col] = pd.NA
             if not first_df_found:
                 df = current_df.copy()
                 first_df_found = True
             else:
                 # Ensure compatible dtypes before merge
                 for col in merge_cols:
                      if col in df.columns and col in current_df.columns:
                           if df[col].dtype != current_df[col].dtype:
                                try:
                                     df[col] = df[col].astype(object)
                                     current_df[col] = current_df[col].astype(object)
                                except Exception as e:
                                     logger.warning(f"Could not align dtype for merge column '{col}': {e}")
                 df = pd.merge(df, current_df, how="outer", on=merge_cols)

    if not first_df_found:
         logger.warning("DataFrame is empty after attempting merges.")
         return empty_df

    # Fill NaNs and calculate total
    activity_cols = ["tasks_started", "notes_count", "tasks_closed"]
    for col in activity_cols:
        if col not in df.columns:
             df[col] = 0
        df[col] = df[col].fillna(0).astype(int)

    df["total_activity"] = df[activity_cols].sum(axis=1)

    if "creator_user_name" in df.columns:
         df["user_name"] = df["creator_user_name"].fillna("Unknown")
    else:
         df["user_name"] = "Unknown"

    if "month" in df.columns:
         df["activity_month"] = df["month"].dt.date # Keep as date object
         df["month_str"] = df["month"].dt.strftime("%Y-%m")
    else:
         df["activity_month"] = pd.NaT
         df["month_str"] = ""


    # Ensure all final columns exist
    for col in final_cols:
        if col not in df.columns:
             logger.warning(f"Final column '{col}' was missing after merge. Adding default.")
             if col in activity_cols + ["total_activity"]: df[col] = 0
             elif col == "user_name": df[col] = "Unknown"
             elif col == "activity_month": df[col] = pd.NaT
             elif col == "month_str": df[col] = ""
             else: df[col] = pd.NA # creator_user_id

    logger.info(f"Computed monthly activity, resulting shape: {df.shape}")
    return df[final_cols]

def main(start_date, end_date):
    # st.title("Monthly User Activity Dashboard") # Removed - Redundant with app.py title
    # st.write(f"Activity from **{start_date}** to **{end_date}**") # Removed - Redundant with app.py caption
    logger.info(f"Running Monthly User Activity report for {start_date} to {end_date}")

    df_activity = compute_monthly_activity(start_date, end_date)
    if df_activity.empty:
        st.warning("No activity data found for the given period.")
        logger.warning("No monthly activity data found.")
        return

    # Get unique user names.
    user_options = sorted([name for name in df_activity["user_name"].unique() if pd.notna(name) and name != "Unknown"])

    if not user_options:
         st.warning("No users with activity found in this period.")
         logger.warning("No valid user names found in activity data.")
         filtered_df = df_activity
         selected_users = []
    else:
        # Define default user IDs and map them to names.
        default_user_ids = [6601768, 2810100, 6590396, 2054428, 4925505, 6149464, 4325638]
        user_map = load_user_map() # Load map from utils
        default_user_names = [user_map.get(uid) for uid in default_user_ids if user_map.get(uid) is not None]
        valid_defaults = [name for name in default_user_names if name in user_options]

        selected_users = st.multiselect("Select Users to Display", options=user_options, default=valid_defaults)
        if not selected_users:
             st.info("Select one or more users to display activity.")
             filtered_df = pd.DataFrame()
        else:
             filtered_df = df_activity[df_activity["user_name"].isin(selected_users)]

    # First Chart: Line Chart for Monthly Total Activity by User
    if not filtered_df.empty:
        logger.info("Generating monthly activity line chart...")
        try:
            fig = px.line(filtered_df, x="month_str", y="total_activity", color="user_name",
                          markers=True, title="Monthly Total Activity by User")
            fig.update_layout(
                 xaxis_title="Month",
                 yaxis_title="Total Activity",
                 template="plotly_white",
                 height=600,
                 legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.3,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=10)
                 ),
                 margin=dict(b=200)
            )
            st.plotly_chart(fig, use_container_width=True)

            # --- Add plain English description ---
            st.markdown("""
            **How this graph is created:**
            *   This graph shows the total monthly activity for each selected user.
            *   "Total Activity" is calculated by summing the number of:
                *   Tasks created by the user that month.
                *   Notes added or files uploaded to tasks by the user that month.
                *   Tasks marked as 'Closed' or 'Completed' by the user that month.
            *   This provides an overview of user engagement with the task system over time.
            """)
            # --- End description ---

        except Exception as e:
             logger.exception("Failed to generate monthly activity line chart.")
             st.error(f"Error generating line chart: {e}")

        # Second Chart: Bar Chart for Average Monthly Activity by User (active months only)
        logger.info("Generating average monthly activity bar chart...")
        try:
            # Compute the average monthly activity per user using only months with recorded activity.
            df_avg = filtered_df.groupby("user_name", as_index=False)["total_activity"].mean()
            df_avg.rename(columns={"total_activity": "average_monthly_activity"}, inplace=True)

            fig_avg = px.bar(
                df_avg,
                x="user_name",
                y="average_monthly_activity",
                title="Average Monthly Activity by User (Active Months Only)",
                labels={"user_name": "User", "average_monthly_activity": "Avg Monthly Activity"}
            )
            fig_avg.update_layout(
                xaxis_title="User",
                yaxis_title="Average Monthly Activity",
                template="plotly_white",
                height=600
            )
            st.plotly_chart(fig_avg, use_container_width=True)
        except Exception as e:
             logger.exception("Failed to generate average monthly activity bar chart.")
             st.error(f"Error generating bar chart: {e}")


        st.subheader("Detailed Monthly Activity Data")
        try:
            display_cols = ["month_str", "user_name", "tasks_started", "notes_count", "tasks_closed", "total_activity"]
            display_df = filtered_df[[col for col in display_cols if col in filtered_df.columns]].copy()
            st.dataframe(display_df.sort_values(["month_str", "user_name"]), hide_index=True)
        except Exception as e:
             logger.exception("Failed to display data table.")
             st.error(f"Error displaying data table: {e}")

    elif selected_users: # Only show if users were selected but resulted in no data
         st.info("No activity data found for the selected users in this period.")

    logger.info("Report execution finished.")

# --- Removed __main__ block ---
# if __name__ == "__main__":
#     main(date(2023, 1, 1), date(2023, 12, 31))
