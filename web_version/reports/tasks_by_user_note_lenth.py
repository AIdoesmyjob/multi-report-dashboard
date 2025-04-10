import streamlit as st
import pandas as pd
import json
import logging # <-- Added
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
from sqlalchemy import text # <-- Removed create_engine
import plotly.express as px

# Import shared functions from utils
from web_version.utils import ( # <-- Absolute import
    get_engine,
    load_user_map,
    parse_created_by_user
)

# Configure logging for this report module
logger = logging.getLogger(__name__)

@st.cache_data(ttl=600)
def compute_monthly_note_characters(start_date, end_date):
    """
    For each task_history event between start_date and end_date:
      - Compute the number of characters in the 'message' (if any).
      - Query task_history_files to get file sizes (assuming a "size" column in KB)
        and treat each file's character equivalent as (size / 3).
      - Sum these values to get a total character count for the event.
    Then, using our new logic:
      - Parse created_by_user to extract (creator_user_id, creator_user_name, creator_user_type),
        keeping only Staff events.
      - Group these events by month (derived from event_time) and by the parsed creator.
    Returns a DataFrame with columns:
      [month, creator_user_id, creator_user_name, total_characters, month_str]
    """
    logger.info(f"Computing monthly note characters from {start_date} to {end_date}")
    engine = get_engine() # <-- Use shared engine
    if engine is None:
        logger.error("Failed to get DB engine for note length.")
        return pd.DataFrame()

    # Query task_history events between start_date and end_date, using created_by_user.
    query = text("""
        SELECT
            id AS history_id,
            created_datetime AS event_time,
            created_by_user,
            message
        FROM task_history
        WHERE created_datetime >= :snapshot_start_date
          AND created_datetime <= :snapshot_end_date
    """)
    try:
        with engine.connect() as conn:
            df_history = pd.read_sql_query(query, conn,
                                           params={"snapshot_start_date": start_date, "snapshot_end_date": end_date})
        logger.info(f"Loaded {len(df_history)} task history events.")
    except Exception as e:
        logger.exception("Failed to load task history for note length.")
        st.error(f"DB Error loading task history: {e}")
        return pd.DataFrame()

    # Convert event_time to datetime.
    df_history["event_time"] = pd.to_datetime(df_history["event_time"], errors="coerce")
    df_history = df_history.dropna(subset=["event_time"]) # Drop rows where conversion failed

    # Calculate note character count from message.
    df_history["note_char_count"] = df_history["message"].apply(
        lambda m: len(m.strip()) if pd.notnull(m) and isinstance(m, str) and m.strip() != "" else 0
    )

    # Query file upload events from task_history_files (assuming column "size" in KB).
    files_query = text("""
        SELECT
            task_history_id,
            size AS file_size
        FROM task_history_files
        -- Potentially filter by history_ids from df_history for efficiency if needed
    """)
    try:
        with engine.connect() as conn:
            df_files = pd.read_sql_query(files_query, conn)
        logger.info(f"Loaded {len(df_files)} file records.")
    except Exception as e:
        logger.exception("Failed to load task history files.")
        st.error(f"DB Error loading file info: {e}")
        # Continue without file info, logging a warning
        logger.warning("Proceeding with note length calculation without file size data.")
        df_files = pd.DataFrame(columns=["task_history_id", "file_size"]) # Empty df

    if not df_files.empty:
        # Ensure file_size is numeric before calculation
        df_files["file_size"] = pd.to_numeric(df_files["file_size"], errors='coerce').fillna(0)
        df_files["file_char_count"] = df_files["file_size"] / 3.0 # Assuming size is in KB
        df_files_grouped = df_files.groupby("task_history_id")["file_char_count"].sum().reset_index()
    else:
        df_files_grouped = pd.DataFrame(columns=["task_history_id", "file_char_count"])

    # Merge file character counts with history events.
    df_history = pd.merge(
        df_history, df_files_grouped, how="left", left_on="history_id", right_on="task_history_id"
    )
    df_history["file_char_count"] = df_history["file_char_count"].fillna(0)

    # Compute total character count for each event.
    df_history["total_characters"] = df_history["note_char_count"] + df_history["file_char_count"]

    # Create a "month" column (first day of the month) from event_time.
    df_history["month"] = df_history["event_time"].dt.to_period("M").dt.to_timestamp()

    # --- New Logic: Parse created_by_user to get creator info ---
    parsed_users = df_history["created_by_user"].apply(parse_created_by_user)
    df_history[["creator_user_id", "creator_user_name", "creator_user_type"]] = pd.DataFrame(parsed_users.tolist(), index=df_history.index)

    # Keep only events with valid creator info (i.e. where UserType is "Staff").
    df_history = df_history[df_history["creator_user_id"].notnull()]

    if df_history.empty:
        logger.warning("No staff events found after parsing user info.")
        return pd.DataFrame()

    # Group by month and the parsed creator info, summing total_characters.
    grouped = df_history.groupby(["month", "creator_user_id", "creator_user_name"])["total_characters"]\
                        .sum().reset_index()
    grouped["month_str"] = grouped["month"].dt.strftime("%Y-%m")

    logger.info(f"Finished computing monthly note characters, shape: {grouped.shape}")
    return grouped

def main(start_date, end_date):
    # st.header("User Task Note Characters Report") # Removed - Redundant
    # st.write(f"Analysis period: **{start_date}** to **{end_date}**") # Removed - Redundant
    st.caption("Calculates total characters from notes and approximates characters from file uploads (Size in KB / 3).") # Keep caption
    logger.info(f"Running Note Length report for {start_date} to {end_date}")

    results_df = compute_monthly_note_characters(start_date, end_date)
    if results_df.empty:
        st.warning("No note/upload events found for the given period.")
        logger.warning("No note character data computed.")
        return

    all_user_names = sorted([name for name in results_df["creator_user_name"].unique() if pd.notna(name)])

    if not all_user_names:
         st.warning("No users found associated with notes/uploads.")
         logger.warning("No valid user names found in results.")
         filtered_df = results_df
         selected_users = []
    else:
        # Define default user IDs (duplicates removed) and map them to names.
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
             filtered_df = results_df[results_df["creator_user_name"].isin(selected_users)]

    # Plot the monthly total character counts.
    if not filtered_df.empty:
        logger.info("Generating plot...")
        try:
            fig = px.line(
                filtered_df,
                x="month_str",
                y="total_characters",
                color="creator_user_name",
                markers=True,
                title="Monthly Total Note/Upload Characters by User"
            )
            fig.update_layout(
                xaxis_title="Month",
                yaxis_title="Total Characters (Notes + Files/3)",
                template="plotly_white",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.2,
                    xanchor="center",
                    x=0.5
                )
            )
            fig.update_yaxes(rangemode='tozero')

            st.plotly_chart(fig, use_container_width=True)

            # --- Add plain English description ---
            st.markdown("""
            **How this graph is created:**
            *   This graph shows the total monthly "character count" for each selected user, based on their task notes and file uploads.
            *   The count includes:
                *   The actual number of characters in the text of any notes added to tasks.
                *   An estimated character count for file uploads, calculated as (File Size in KB / 3). This provides a rough measure of the "volume" of uploaded content.
            *   This metric attempts to quantify the amount of information contributed by users via notes and uploads.
            """)
            # --- End description ---

        except Exception as e:
             logger.exception("Failed to generate plot.")
             st.error(f"Error generating plot: {e}")

        st.subheader("Data Table")
        try:
            display_cols = ["month_str", "creator_user_name", "total_characters"]
            display_df = filtered_df[[col for col in display_cols if col in filtered_df.columns]].copy()
            display_df.rename(columns={'creator_user_name': 'User Name', 'total_characters': 'Total Characters'}, inplace=True)
            st.dataframe(display_df.sort_values(["month_str", "User Name"]), hide_index=True)
        except Exception as e:
             logger.exception("Failed to display data table.")
             st.error(f"Error displaying data table: {e}")

    elif selected_users:
         st.info("No data found for the selected users in this period.")

    logger.info("Report execution finished.")

# --- Removed __main__ block ---
# if __name__ == "__main__":
#     main(date(2020, 3, 15), date(2025, 3, 15))
