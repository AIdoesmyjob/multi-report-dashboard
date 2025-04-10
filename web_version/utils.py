import os
import json
import logging
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from dateutil import tz
from datetime import datetime

# Configure logging for the utility module
logger = logging.getLogger(__name__)

# --- Timezone Definitions ---
PACIFIC_TZ = tz.gettz('America/Los_Angeles')
UTC_TZ = tz.UTC

@st.cache_resource
def get_engine():
    """
    Create a SQLAlchemy engine using credentials from environment variables.
    Assumes .env has been loaded by the main app script.
    """
    db_host = os.environ.get("DB_HOST")
    db_name = os.environ.get("DB_NAME")
    db_user = os.environ.get("DB_USER")
    db_password = os.environ.get("DB_PASSWORD")

    if not all([db_host, db_name, db_user, db_password]):
        error_msg = "Database credentials (DB_HOST, DB_NAME, DB_USER, DB_PASSWORD) not found in environment variables."
        logger.error(error_msg)
        st.error(error_msg)
        st.stop()
        # return None # Or raise an exception depending on desired handling

    try:
        engine = create_engine(f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}/{db_name}")
        # Test connection - optional but recommended
        with engine.connect() as connection:
            logger.info("Database engine created and connection tested successfully.")
        return engine
    except Exception as e:
        error_msg = f"Failed to create database engine: {e}"
        logger.exception(error_msg) # Log full traceback
        st.error(error_msg)
        st.stop()
        # return None

@st.cache_data(ttl=600)
def load_user_map():
    """Loads user ID to full name mapping from the database."""
    logger.info("Loading user map...")
    engine = get_engine()
    if engine is None:
        logger.error("Cannot load user map, DB engine not available.")
        return {} # Return empty dict if engine failed

    query = text("SELECT id, first_name, last_name FROM users")
    try:
        with engine.connect() as conn:
            df_users = pd.read_sql_query(query, conn)
        df_users["full_name"] = df_users["first_name"] + " " + df_users["last_name"]
        user_dict = dict(zip(df_users["id"], df_users["full_name"]))
        logger.info(f"User map loaded successfully with {len(user_dict)} users.")
        return user_dict
    except Exception as e:
        error_msg = f"Failed to load user map: {e}"
        logger.exception(error_msg)
        st.error(error_msg)
        return {} # Return empty dict on error

def parse_created_by_user(val):
    """
    Parses a JSON string or dict from task_history.created_by_user.
    Returns (user_id, full_name, user_type) if UserType is "Staff", else (None, None, None).
    """
    if not val or pd.isnull(val):
        return (None, None, None)
    try:
        if isinstance(val, dict):
            user_dict = val
        else:
            # Handle potential non-string inputs before json.loads
            if not isinstance(val, str):
                 val = str(val) # Attempt to convert to string
            user_dict = json.loads(val)

        user_id = user_dict.get("Id")
        first = user_dict.get("FirstName", "")
        last = user_dict.get("LastName", "")
        user_type = user_dict.get("UserType", "").strip()

        # Ensure user_id is treated as an integer if present
        if user_id is not None:
            try:
                user_id = int(user_id)
            except (ValueError, TypeError):
                 logger.warning(f"Could not convert user_id '{user_id}' to int in parse_created_by_user. Value: {val}")
                 return (None, None, None) # Invalid ID format

        if user_type.lower() != "staff":
            # logger.debug(f"Non-staff user type encountered: {user_type}. Value: {val}")
            return (None, None, None)

        full_name = f"{first} {last}".strip()
        return (user_id, full_name, user_type)

    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"Error parsing created_by_user JSON: {e}. Value: {val}")
        return (None, None, None)
    except Exception as e: # Catch unexpected errors
        logger.exception(f"Unexpected error in parse_created_by_user. Value: {val}")
        return (None, None, None)


def utc_to_pacific(ts):
    """Converts a UTC or naive timestamp to Pacific Time."""
    if pd.isnull(ts):
        return None
    pacific_tz = tz.gettz('America/Los_Angeles')
    # Ensure ts is a datetime object
    if not isinstance(ts, datetime):
        try:
            ts = pd.to_datetime(ts)
            if pd.isnull(ts): return None # Handle conversion failure
        except Exception:
             logger.warning(f"Could not convert {ts} to datetime in utc_to_pacific.")
             return None

    try:
        if ts.tzinfo is None:
            # logger.debug(f"Localizing naive timestamp {ts} to UTC.")
            ts = ts.tz_localize('UTC')
        # logger.debug(f"Converting timestamp {ts} to Pacific.")
        return ts.astimezone(pacific_tz)
    except Exception as e:
        logger.exception(f"Error converting timestamp {ts} to Pacific: {e}")
        return None

# --- Status definitions (Consider if these should live here or be passed) ---
OPEN_STATUSES = {"New", "In Progress"}
DEFERRED_STATUSES = {"Deferred"}
CLOSED_STATUSES = {"Closed", "Completed"}

# --- Global Exclusion Time (Consider if this should live here or be passed) ---
# If used by many reports, keep here. If specific, move to report.
EXCLUDE_TIME_PACIFIC = datetime(2025, 3, 8, 22, 54, 24, tzinfo=tz.gettz('America/Los_Angeles'))

# --- Add the daily activity computation functions here if desired ---
# Example:
# @st.cache_data(ttl=600)
# def compute_daily_activity(start_date, end_date):
#     # ... implementation using get_engine(), parse_created_by_user() etc. ...
#     pass
