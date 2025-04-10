import streamlit as st
import pandas as pd
import json
import logging # <-- Added
from datetime import date, datetime, time # <-- Added time
from sqlalchemy import text # <-- Removed create_engine
import plotly.express as px
from dateutil import tz

# Import shared functions and constants from utils
from web_version.utils import ( # <-- Absolute import
    get_engine,
    load_user_map, # Although not used directly in main, compute functions might use it implicitly via other compute functions? Let's keep for now.
    parse_created_by_user,
    utc_to_pacific,
    EXCLUDE_TIME_PACIFIC, # Keep if needed
    PACIFIC_TZ # <-- Added missing import
)

# Configure logging for this report module
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# 1. TASKS STARTED (by creator)
# ------------------------------------------------------------------------------
@st.cache_data(ttl=600)
def compute_hourly_tasks_started(start_date, end_date):
    """
    For each task, retrieves the earliest record ("task started").
    Uses created_by_user to determine the event creator (Staff only).
    Groups by activity_date, hour, creator_user_id, and creator_user_name.
    """
    logger.info("Computing hourly tasks started...")
    engine = get_engine() # <-- Use shared engine
    if engine is None: return pd.DataFrame()

    start_datetime = datetime.combine(start_date, time.min) # Use time.min
    end_datetime = datetime.combine(end_date, time.max)   # Use time.max

    query = text("""
        SELECT DISTINCT ON (task_id)
            task_id,
            created_by_user,
            created_datetime
        FROM task_history
        WHERE created_datetime >= :start_datetime
          AND created_datetime <= :end_datetime
        ORDER BY task_id, created_datetime ASC
    """)
    try:
        with engine.connect() as conn:
            df = pd.read_sql_query(query, conn,
                                   params={"start_datetime": start_datetime, "end_datetime": end_datetime},
                                   parse_dates=["created_datetime"])
        logger.info(f"Loaded {len(df)} initial task events.")
    except Exception as e:
        logger.exception("Error fetching tasks started.")
        st.error(f"Error fetching tasks started: {e}")
        return pd.DataFrame()

    if df.empty: return pd.DataFrame(columns=["activity_date", "hour", "creator_user_id", "creator_user_name", "tasks_started"])

    # Parse created_by_user (keeping only Staff)
    parsed_users = df["created_by_user"].apply(parse_created_by_user)
    df[["creator_user_id", "creator_user_name", "creator_user_type"]] = pd.DataFrame(parsed_users.tolist(), index=df.index)
    df = df.dropna(subset=["creator_user_id"])

    # Convert timestamp to Pacific Time and filter out the exclusion time
    df["created_datetime_pacific"] = df["created_datetime"].apply(utc_to_pacific)
    df = df.dropna(subset=["created_datetime_pacific"])
    df = df[df["created_datetime_pacific"] != EXCLUDE_TIME_PACIFIC] # Compare Timestamps correctly

    if df.empty: return pd.DataFrame(columns=["activity_date", "hour", "creator_user_id", "creator_user_name", "tasks_started"])

    df["activity_date"] = df["created_datetime_pacific"].dt.date
    df["hour"] = df["created_datetime_pacific"].dt.hour

    # Ensure grouping columns are correct type before grouping
    df['creator_user_id'] = df['creator_user_id'].astype(int)

    grouped = df.groupby(["activity_date", "hour", "creator_user_id", "creator_user_name"], dropna=False).size()\
                .reset_index(name="tasks_started")
    logger.info(f"Finished computing hourly tasks started, shape: {grouped.shape}")
    return grouped

# ------------------------------------------------------------------------------
# 2. NOTES / UPLOADS (by creator) - MODIFIED TO AVOID DOUBLE COUNTING CLOSURES
# ------------------------------------------------------------------------------
@st.cache_data(ttl=600)
def compute_hourly_notes(start_date, end_date):
    """
    Retrieves note/upload events from task_history (Staff only).
    Considers records with a non-empty message or an attached file.
    Excludes records that also represent a 'Closed' or 'Completed' status change.
    Groups by activity_date, hour, creator_user_id, and creator_user_name.
    """
    logger.info("Computing hourly notes/uploads...")
    engine = get_engine() # <-- Use shared engine
    if engine is None: return pd.DataFrame()

    start_datetime = datetime.combine(start_date, time.min)
    end_datetime = datetime.combine(end_date, time.max)

    # Fetch task_status as well
    query = text("""
        SELECT
            id AS history_id,
            created_by_user,
            created_datetime,
            message,
            task_status       -- Fetched task_status
        FROM task_history
        WHERE created_datetime >= :start_datetime
          AND created_datetime <= :end_datetime
    """)
    try:
        with engine.connect() as conn:
            df_history = pd.read_sql_query(query, conn,
                                           params={"start_datetime": start_datetime, "end_datetime": end_datetime},
                                           parse_dates=["created_datetime"])
            logger.info(f"Loaded {len(df_history)} history events for notes check.")
            # Fetch distinct history IDs that have files associated
            files_query = text("SELECT DISTINCT task_history_id FROM task_history_files")
            df_files = pd.read_sql_query(files_query, conn)
            logger.info(f"Loaded {len(df_files)} file association records.")
            # Use a set for potentially faster 'isin' lookups
            file_events_ids = set(df_files["task_history_id"].dropna().astype(int).unique())

    except Exception as e:
        logger.exception("Error fetching notes/uploads data.")
        st.error(f"Error fetching notes/uploads: {e}")
        return pd.DataFrame()

    if df_history.empty: return pd.DataFrame(columns=["activity_date", "hour", "creator_user_id", "creator_user_name", "notes_count"])

    # Parse created_by_user (only Staff)
    parsed_users = df_history["created_by_user"].apply(parse_created_by_user)
    df_history[["creator_user_id", "creator_user_name", "creator_user_type"]] = pd.DataFrame(parsed_users.tolist(), index=df_history.index)
    df_history = df_history.dropna(subset=["creator_user_id"])

    # Convert timestamp and filter
    df_history["created_datetime_pacific"] = df_history["created_datetime"].apply(utc_to_pacific)
    df_history = df_history.dropna(subset=["created_datetime_pacific"])
    df_history = df_history[df_history["created_datetime_pacific"] != EXCLUDE_TIME_PACIFIC]

    if df_history.empty: return pd.DataFrame(columns=["activity_date", "hour", "creator_user_id", "creator_user_name", "notes_count"])

    # --- Modification for Double Counting ---
    has_message = df_history["message"].fillna("").astype(str).str.strip() != ""
    has_file = df_history["history_id"].isin(file_events_ids)
    is_potential_note = has_message | has_file
    is_closure_event = df_history["task_status"].fillna("").isin(['Closed', 'Completed'])
    is_true_note = is_potential_note & (~is_closure_event)
    df_notes = df_history[is_true_note].copy()
    # --- End Modification ---

    # Grouping
    if not df_notes.empty:
        df_notes["activity_date"] = df_notes["created_datetime_pacific"].dt.date
        df_notes["hour"] = df_notes["created_datetime_pacific"].dt.hour
        df_notes['creator_user_id'] = df_notes['creator_user_id'].astype(int)

        grouped = df_notes.groupby(["activity_date", "hour", "creator_user_id", "creator_user_name"], dropna=False).size()\
                    .reset_index(name="notes_count")
        logger.info(f"Finished computing hourly notes, shape: {grouped.shape}")
        return grouped
    else:
        logger.warning("No 'true notes' found after filtering.")
        return pd.DataFrame(columns=["activity_date", "hour", "creator_user_id", "creator_user_name", "notes_count"])

# ------------------------------------------------------------------------------
# 3. TASKS CLOSED / COMPLETED (by creator)
# ------------------------------------------------------------------------------
@st.cache_data(ttl=600)
def compute_hourly_tasks_closed(start_date, end_date):
    """
    Retrieves tasks closed or completed events from task_history (Staff only).
    Groups by activity_date, hour, creator_user_id, and creator_user_name.
    """
    logger.info("Computing hourly tasks closed...")
    engine = get_engine() # <-- Use shared engine
    if engine is None: return pd.DataFrame()

    start_datetime = datetime.combine(start_date, time.min)
    end_datetime = datetime.combine(end_date, time.max)

    query = text("""
        SELECT
            task_id,
            created_by_user,
            created_datetime,
            task_status
        FROM task_history
        WHERE created_datetime >= :start_datetime
          AND created_datetime <= :end_datetime
          AND task_status IN ('Closed', 'Completed')
    """)
    try:
        with engine.connect() as conn:
            df_closed = pd.read_sql_query(query, conn,
                                          params={"start_datetime": start_datetime, "end_datetime": end_datetime},
                                          parse_dates=["created_datetime"])
        logger.info(f"Loaded {len(df_closed)} closed/completed events.")
    except Exception as e:
        logger.exception("Error fetching tasks closed.")
        st.error(f"Error fetching tasks closed: {e}")
        return pd.DataFrame()

    if df_closed.empty: return pd.DataFrame(columns=["activity_date", "hour", "creator_user_id", "creator_user_name", "tasks_closed"])

    # Parse created_by_user (only Staff)
    parsed_users = df_closed["created_by_user"].apply(parse_created_by_user)
    df_closed[["creator_user_id", "creator_user_name", "creator_user_type"]] = pd.DataFrame(parsed_users.tolist(), index=df_closed.index)
    df_closed = df_closed.dropna(subset=["creator_user_id"])

    # Convert timestamp and filter
    df_closed["created_datetime_pacific"] = df_closed["created_datetime"].apply(utc_to_pacific)
    df_closed = df_closed.dropna(subset=["created_datetime_pacific"])
    df_closed = df_closed[df_closed["created_datetime_pacific"] != EXCLUDE_TIME_PACIFIC]

    if df_closed.empty: return pd.DataFrame(columns=["activity_date", "hour", "creator_user_id", "creator_user_name", "tasks_closed"])

    df_closed["activity_date"] = df_closed["created_datetime_pacific"].dt.date
    df_closed["hour"] = df_closed["created_datetime_pacific"].dt.hour
    df_closed['creator_user_id'] = df_closed['creator_user_id'].astype(int)

    # Group and count occurrences
    grouped = df_closed.groupby(["activity_date", "hour", "creator_user_id", "creator_user_name"], dropna=False).size()\
                       .reset_index(name="tasks_closed")
    logger.info(f"Finished computing hourly tasks closed, shape: {grouped.shape}")
    return grouped

# ------------------------------------------------------------------------------
# Combine Hourly Metrics
# ------------------------------------------------------------------------------
@st.cache_data(ttl=600)
def compute_hourly_activity(start_date, end_date):
    """
    Merges tasks_started, notes_count, and tasks_closed for Staff events.
    Handles cases where dataframes might be empty.
    """
    logger.info("Computing combined hourly activity...")
    df_ts = compute_hourly_tasks_started(start_date, end_date)
    df_nt = compute_hourly_notes(start_date, end_date)
    df_tc = compute_hourly_tasks_closed(start_date, end_date)

    # Define standard columns for merging and final output
    merge_cols = ["activity_date", "hour", "creator_user_id", "creator_user_name"]
    data_cols = ["tasks_started", "notes_count", "tasks_closed"]
    all_dfs = {"tasks_started": df_ts, "notes_count": df_nt, "tasks_closed": df_tc}

    # Start with an empty DataFrame with the merge columns, or the first non-empty DF
    df_merged = pd.DataFrame(columns=merge_cols)
    first_df = True

    for key, df_current in all_dfs.items():
         # Ensure the DF has the merge columns even if empty
        for col in merge_cols:
            if col not in df_current.columns:
                df_current[col] = pd.NA

        # Ensure the specific data column exists
        if key not in df_current.columns:
            df_current[key] = 0

        # Keep only merge columns + the specific data column
        cols_to_keep = merge_cols + [key]
        # Handle case where df_current might be completely empty
        if df_current.empty:
             df_current = pd.DataFrame(columns=cols_to_keep)
        else:
             df_current = df_current[cols_to_keep]


        if not df_current.empty:
            if first_df:
                df_merged = df_current.copy() # Use copy
                first_df = False
            else:
                # Ensure compatible types before merge
                for col in merge_cols:
                     if col in df_merged.columns and col in df_current.columns:
                          if df_merged[col].dtype != df_current[col].dtype:
                               try:
                                    df_merged[col] = df_merged[col].astype(object)
                                    df_current[col] = df_current[col].astype(object)
                               except Exception as e:
                                    logger.warning(f"Could not align dtype for merge column '{col}': {e}")
                df_merged = pd.merge(df_merged, df_current, how="outer", on=merge_cols)

    # If df_merged is still empty (all inputs were empty), return it
    if df_merged.empty:
        logger.warning("Combined hourly activity dataframe is empty.")
        df_merged = pd.DataFrame(columns=merge_cols + data_cols)
        df_merged["total_activity"] = 0
        return df_merged

    # Fill NaNs introduced by merges with 0 for count columns
    for col in data_cols:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].fillna(0).astype(int)
        else:
            df_merged[col] = 0

    # Calculate total activity
    df_merged["total_activity"] = df_merged[data_cols].sum(axis=1)

    # Ensure correct types for merge columns before returning (especially numeric ID)
    df_merged['hour'] = df_merged['hour'].fillna(-1).astype(int) # Fill potential NaN hours
    df_merged['creator_user_id'] = df_merged['creator_user_id'].fillna(-1).astype(int) # Fill potential NaN IDs
    # Convert activity_date back to date objects if they became objects during merge
    df_merged['activity_date'] = pd.to_datetime(df_merged['activity_date']).dt.date

    logger.info(f"Finished computing combined hourly activity, shape: {df_merged.shape}")
    return df_merged.sort_values(by=merge_cols).reset_index(drop=True)


# ------------------------------------------------------------------------------
# Main Dashboard - MODIFIED FOR WEIGHTED AVERAGE
# ------------------------------------------------------------------------------
def main(start_date, end_date):
    """Renders the Streamlit dashboard elements with weighted averaging."""
    st.title("Hourly Staff Activity Dashboard (Weighted Average)")
    logger.info(f"Running Hourly Activity report for {start_date} to {end_date}")

    if start_date > end_date:
        st.error("Error: Start date must be before or the same as end date.")
        return

    st.write(f"Displaying activity from **{start_date.strftime('%Y-%m-%d')}** to **{end_date.strftime('%Y-%m-%d')}** (Pacific Time)")

    # --- Configuration for Weighted Average ---
    WEEKDAY_WEIGHT = 1.0
    WEEKEND_WEIGHT = 0.25 # Adjust this value as needed (e.g., 0.2, 0.3)
    st.sidebar.info(f"Averaging Weights:\n- Weekday: {WEEKDAY_WEIGHT}\n- Weekend: {WEEKEND_WEIGHT}")
    # --- End Configuration ---

    df_activity_raw = compute_hourly_activity(start_date, end_date) # Renamed to avoid confusion

    if df_activity_raw.empty or df_activity_raw['total_activity'].sum() == 0:
        st.info("No staff activity found for the selected period.")
        logger.warning("No raw activity data found.")
        return

    st.subheader("Detailed Hourly Activity Data (Raw)")
    try:
        st.dataframe(
            df_activity_raw.sort_values(["activity_date", "hour", "creator_user_name"]),
            use_container_width=True,
            hide_index=True,
             column_config={
                "creator_user_id": "User ID", "creator_user_name": "Staff Name",
                "activity_date": "Date", "hour": st.column_config.NumberColumn("Hour (Pacific)", format="%d"),
                "tasks_started": "Tasks Started", "notes_count": "Notes/Uploads",
                "tasks_closed": "Tasks Closed", "total_activity": "Total Actions"
            }
        )
    except Exception as e:
         logger.exception("Error displaying raw data table.")
         st.error(f"Error displaying raw data: {e}")


    # --- Individual Staff Charts with Weighted Average ---
    # Ensure creator_user_id is not NaN before getting unique values
    valid_staff_ids = df_activity_raw["creator_user_id"].dropna().unique()
    staff_mapping = df_activity_raw.drop_duplicates(subset=['creator_user_id'])[['creator_user_id', 'creator_user_name']].set_index('creator_user_id')['creator_user_name'].to_dict()

    if not staff_mapping or len(valid_staff_ids) == 0:
        st.info("No specific staff members found in the activity data.")
        logger.warning("No staff IDs or mapping found.")
        return

    st.subheader("Weighted Average Hourly Activity per Staff Member")
    st.caption(f"Shows a weighted average of actions per hour. Weekdays count fully (weight={WEEKDAY_WEIGHT}), weekends count partially (weight={WEEKEND_WEIGHT}). Includes hours with zero activity on relevant days.")

    # --- Create the complete date/hour grid for the selected range ---
    try:
        # Use the actual min/max dates from the data if available, else use input range
        min_date = df_activity_raw['activity_date'].min() if not df_activity_raw.empty else start_date
        max_date = df_activity_raw['activity_date'].max() if not df_activity_raw.empty else end_date

        all_hours_range = pd.date_range(start=min_date, end=max_date + pd.Timedelta(days=1), freq='h', tz=PACIFIC_TZ)[:-1] # Inclusive of end date hours
    except Exception as e:
         logger.exception("Error creating full hourly range.")
         st.error(f"Error creating hourly range: {e}")
         return


    if not valid_staff_ids.size > 0:
         st.warning("No staff IDs found to process.")
         return

    # Create multi-index for all staff, all hours in the range
    multi_index = pd.MultiIndex.from_product(
        [valid_staff_ids, all_hours_range],
        names=['creator_user_id', 'datetime_pacific']
    )
    df_full_grid = pd.DataFrame(index=multi_index).reset_index()

    # Prepare columns for merging with actual activity
    df_full_grid['activity_date'] = df_full_grid['datetime_pacific'].dt.date
    df_full_grid['hour'] = df_full_grid['datetime_pacific'].dt.hour
    df_full_grid['creator_user_id'] = df_full_grid['creator_user_id'].astype(int) # Ensure type match

    # Merge actual activity onto the full grid
    # Ensure df_activity_raw types are correct before merge
    df_activity_raw['creator_user_id'] = df_activity_raw['creator_user_id'].astype(int)
    df_activity_raw['hour'] = df_activity_raw['hour'].astype(int)
    df_activity_raw['activity_date'] = pd.to_datetime(df_activity_raw['activity_date']).dt.date # Ensure date objects

    df_merged_weighted = pd.merge(
        df_full_grid[['activity_date', 'hour', 'creator_user_id']],
        df_activity_raw[['activity_date', 'hour', 'creator_user_id', 'total_activity']],
        on=['activity_date', 'hour', 'creator_user_id'],
        how='left'
    )

    # Fill NaN activity with 0 (representing hours with no recorded actions)
    df_merged_weighted['total_activity'] = df_merged_weighted['total_activity'].fillna(0).astype(int)

    # Add day of week and weights
    df_merged_weighted['day_of_week'] = pd.to_datetime(df_merged_weighted['activity_date']).dt.dayofweek # Monday=0, Sunday=6
    df_merged_weighted['weight'] = df_merged_weighted['day_of_week'].apply(lambda d: WEEKDAY_WEIGHT if d < 5 else WEEKEND_WEIGHT)

    # --- Calculate weighted average for each staff member ---
    logger.info("Calculating and plotting weighted averages per staff member...")
    for staff_id in sorted(valid_staff_ids):
        staff_name = staff_mapping.get(staff_id, f"Unknown Staff ID {staff_id}")
        st.markdown(f"#### {staff_name}")

        staff_data_weighted = df_merged_weighted[df_merged_weighted['creator_user_id'] == staff_id]

        if staff_data_weighted.empty:
            st.write("No data for this staff member in the selected range.")
            logger.debug(f"No weighted data for staff {staff_name} ({staff_id})")
            continue

        # Define weighted average calculation function
        def weighted_avg(x):
            total_weighted_activity = (x['total_activity'] * x['weight']).sum()
            total_weight = x['weight'].sum()
            return total_weighted_activity / total_weight if total_weight > 0 else 0

        # Apply weighted average calculation per hour
        try:
            hourly_weighted_avg = staff_data_weighted.groupby('hour').apply(weighted_avg, include_groups=False).reset_index(name='weighted_avg_activity')
        except Exception as group_e:
             logger.exception(f"Error calculating weighted average for {staff_name}: {group_e}")
             st.error(f"Could not calculate average for {staff_name}.")
             continue


        # Ensure all hours 0-23 are present for the chart
        all_hours_df = pd.DataFrame({'hour': range(24)})
        hourly_weighted_avg_full = pd.merge(all_hours_df, hourly_weighted_avg, on='hour', how='left').fillna(0)

        # Create and display the bar chart
        try:
            fig = px.bar(
                hourly_weighted_avg_full,
                x="hour",
                y="weighted_avg_activity",
                labels={"hour": "Hour of Day (Pacific Time)", "weighted_avg_activity": "Weighted Avg Actions per Hour"},
                title=f"Weighted Average Hourly Activity Pattern for {staff_name}",
                hover_data={'weighted_avg_activity': ':.2f'}
            )
            fig.update_xaxes(tickmode='linear', dtick=1)
            fig.update_yaxes(rangemode='tozero')
            st.plotly_chart(fig, use_container_width=True)
        except Exception as plot_e:
             logger.exception(f"Error plotting weighted average for {staff_name}: {plot_e}")
             st.error(f"Could not generate plot for {staff_name}.")

    logger.info("Report execution finished.")


# ------------------------------------------------------------------------------
# Streamlit App Execution (remains the same)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Setup basic logging if running standalone
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    log_handler = logging.StreamHandler()
    log_handler.setFormatter(log_formatter)
    root_logger = logging.getLogger()
    if not root_logger.hasHandlers():
         root_logger.addHandler(log_handler)
    root_logger.setLevel(logging.INFO)

    # Default dates
    default_end_date = date.today()
    default_start_date = default_end_date.replace(day=1) - relativedelta(months=1) # Default to previous full month

    st.sidebar.header("Date Range Selection")
    start_date_input = st.sidebar.date_input("Start Date", default_start_date)
    end_date_input = st.sidebar.date_input("End Date", default_end_date)

    main(start_date_input, end_date_input)
