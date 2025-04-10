import os
import sys
import logging # <-- Added
from dotenv import load_dotenv
import importlib
import streamlit as st
import streamlit_authenticator as stauth # <-- Added
import yaml                             # <-- Added
from yaml.loader import SafeLoader      # <-- Added
from datetime import datetime, date
from dateutil.relativedelta import relativedelta

# --- Logging Configuration ---  # <-- Added Block
log_file_path = os.path.join(os.path.dirname(__file__), 'app.log')
logging.basicConfig(
    level=logging.INFO, # Set to logging.DEBUG for more verbose output
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        # logging.StreamHandler() # Optionally add to see logs in console too
    ]
)
logging.info("Application started.") # <-- Added

# --- Calculate Absolute Path for Reports Directory ---
# .env loading removed; Streamlit secrets should be used directly.
script_dir = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(script_dir, "reports")
logging.info(f"Absolute reports directory path set to: {REPORTS_DIR}") # <-- Added logging

# --- Ensure web_version package is importable ---
# Add the parent directory of web_version (the project root) to sys.path
# This ensures 'from web_version.utils' works within imported report modules.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root) # Insert at beginning to prioritize
    logging.info(f"Added project root {project_root} to sys.path")

# --- Set Page Config FIRST --- # <-- Moved Up
st.set_page_config(layout="wide")

# --- Authentication Setup --- #
# Load configuration directly from Streamlit secrets instead of config.yaml
try:
    # Check if essential keys exist in secrets
    if "credentials" not in st.secrets or "cookie" not in st.secrets:
        st.error("Authentication configuration missing in Streamlit secrets.")
        logging.error("Required 'credentials' or 'cookie' section not found in st.secrets.")
        st.stop()

    # Construct the config dictionary from secrets
    config = {
        'credentials': {
             # Explicitly access and convert the 'usernames' sub-section
            'usernames': st.secrets['credentials']['usernames'].to_dict()
        },
        'cookie': st.secrets['cookie'].to_dict()
    }
    # --- Debugging: Log the constructed config ---
    logging.info(f"DEBUG: Config constructed from secrets: {config}")
    # --- End Debugging ---
    logging.info("Authentication configuration loaded from Streamlit secrets.")

    # Validate essential sub-keys (optional but recommended)
    if not all(k in config['cookie'] for k in ['name', 'key', 'expiry_days']):
         st.error("Cookie configuration incomplete in Streamlit secrets (missing name, key, or expiry_days).")
         logging.error("Cookie configuration incomplete in st.secrets.")
         st.stop()
    if not config['credentials'].get('usernames'):
         st.error("Usernames configuration missing in Streamlit secrets.")
         logging.error("Usernames configuration missing in st.secrets.")
         st.stop()

except Exception as e:
    st.error(f"Error loading authentication configuration from Streamlit secrets: {e}")
    logging.exception("Error loading authentication configuration from st.secrets:")
    st.stop()

# Initialize the authenticator with the config derived from secrets
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
    # config['preauthorized'] # Removed deprecated parameter
)

# Initialize variables
name = None
authentication_status = None
username = None

# --- Main Application Logic AND Login --- #
# Check cookie/session first for existing authentication
# Note: This part might be handled internally by authenticator, but let's try structuring logic around status
# It's possible authenticator needs to read cookie status *before* login is called.
# Let's try getting the status from the cookie first if possible (though login usually does this)
# This is speculative based on the unusual behavior.

# Reverting to a simpler structure based on examples:
# Call login() first, then branch based on the resulting status.
# Using defensive check again as direct unpacking fails consistently
# Explicitly setting location back to 'main'
# Call login() to render the widget. Status is checked via session_state below.
authenticator.login(location='main')

# Check authentication status from session_state (pattern for v0.4.x)
authentication_status = st.session_state.get("authentication_status")
name = st.session_state.get("name")
username = st.session_state.get("username")

# --- Streamlit App UI ---
# Display title only *after* successful login
if authentication_status:
    st.title("Multi-Report Dashboard") # Moved title here
    # --- Sidebar ---
    # Logout button now uses location='sidebar' directly if needed, or default
    authenticator.logout('Logout', location='sidebar')
    st.sidebar.write(f'Welcome *{name}*')
    st.sidebar.divider() # Add a visual separator

    # Sidebar: Date Inputs
    st.sidebar.header("Select Date Range")
    # Define default dates within the authenticated block
    default_end_date = datetime.today().date()
    default_start_date = (datetime.today() - relativedelta(years=5)).date()
    start_date = st.sidebar.date_input("Start Date", default_start_date)
    end_date = st.sidebar.date_input("End Date", default_end_date)

    # Move date validation inside the authenticated block
    if start_date > end_date:
        st.sidebar.error("Error: Start date must be before or the same as end date.")
        logging.error(f"Date validation failed: Start date {start_date} > End date {end_date}") # <-- Added logging
        st.stop()

    # --- Report Discovery and Selection --- (Moved inside auth block)
    st.sidebar.header("Select Report")
    report_option = None # Initialize report_option
    try: # Correct indentation for try block
        if not os.path.isdir(REPORTS_DIR):
            st.sidebar.error(f"Error: Reports directory '{REPORTS_DIR}' not found.")
            logging.error(f"Reports directory '{REPORTS_DIR}' not found.") # <-- Added logging
            st.stop() # Indent stop under the if

        all_files = os.listdir(REPORTS_DIR)
        logging.debug(f"Files found by os.listdir: {all_files}") # <-- Replaced print

        report_files = [f for f in all_files if f.endswith(".py") and f != "__init__.py" and not f.startswith('.')]
        logging.debug(f"Filtered report_files: {report_files}") # <-- Replaced print


        if not report_files:
            st.sidebar.warning(f"No report files (.py) found in the '{REPORTS_DIR}' directory.")
            logging.warning(f"No report files (.py) found in the '{REPORTS_DIR}' directory.") # <-- Added logging
            st.stop()

        report_names = sorted([f[:-3].replace("_", " ").title() for f in report_files])
        logging.debug(f"Generated report_names: {report_names}") # <-- Replaced print

        report_mapping = {f[:-3].replace("_", " ").title(): f[:-3] for f in report_files}
        report_option = st.sidebar.selectbox("Choose a report:", report_names)
        if report_option:
                logging.info(f"Report selected: {report_option}") # <-- Added logging

    except Exception as e: # Correct indentation for except block
        st.sidebar.error(f"Error discovering reports: {e}")
        logging.exception("Error discovering reports:") # <-- Enhanced logging
        st.stop() # Indent stop under the except

    # --- Main Area: Report Execution --- (Moved inside auth block)
    # Correct indentation for this entire block
    if report_option: # Check if a report was successfully selected
        st.subheader(f"Displaying: {report_option}")
        st.caption(f"Data Range: {start_date.strftime('%B %d, %Y')} to {end_date.strftime('%B %d, %Y')}")
        logging.info(f"Attempting to run report: {report_option} for dates {start_date} to {end_date}") # <-- Added logging

        # No engine created here anymore

        try: # Indent try block for report execution
            module_name = report_mapping[report_option]
            # Use relative path for Python import system
            full_module_name = f"web_version.reports.{module_name}" # <-- Reverted import path logic
            logging.info(f"Attempting to import module: {full_module_name}") # <-- Log attempt

            report_module = importlib.import_module(full_module_name)
            logging.info(f"Successfully imported module: {full_module_name}")
            # importlib.reload(report_module) # Optional for development

            if hasattr(report_module, "main") and callable(report_module.main):
                 # --- Call Report's Main Function --- # <-- Reverted CALL (only dates)
                logging.info(f"Calling main function for report: {report_option}") # <-- Added logging
                report_module.main(start_date, end_date) # Assumes main(start_date, end_date)
                logging.info(f"Finished running report: {report_option}") # <-- Added logging
            else:
                st.error(f"Report '{report_option}' (module: {module_name}.py) does not have a callable 'main' function.")
                logging.error(f"Report '{report_option}' (module: {module_name}.py) does not have a callable 'main' function.") # <-- Added logging

        except Exception as e: # Indent except block
            st.error(f"An error occurred while loading/running report '{report_option}': {e}")
            logging.exception(f"An error occurred while loading/running report '{report_option}':") # <-- Enhanced logging
            # import traceback # Not needed when using logging.exception
            # st.error(traceback.format_exc()) # Not needed when using logging.exception

    else: # Ensure this else aligns with 'if report_option:'
        # Correctly indent lines under the else
        st.info("Please select a report from the sidebar.")
        logging.info("No report selected.") # <-- Added logging

elif authentication_status is False:
    st.error('Username/password is incorrect')
    # Use the username from session_state if available for logging
    log_username = st.session_state.get("username", "N/A")
    logging.warning(f"Failed login attempt for username: {log_username}")

elif authentication_status is None:
    # The login() call above should render the form fields in the 'main' location.
    # This block handles the initial state where status is None.
    # Optionally add a placeholder or message if needed, but login() handles the form.
    # st.info("Please log in using the form above.") # Example placeholder
    pass
