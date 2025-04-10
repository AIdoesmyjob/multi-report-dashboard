import os
import logging
import caldav
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit as st
import pandas as pd
from datetime import datetime, date, time # Ensure time is imported

# Assuming ai_calendar_categorizer is in the same directory or web_version
try:
    # Use absolute import now that web_version is a package
    from web_version.ai_calendar_categorizer import AICalendarCategorizer
except ImportError:
    # Fallback might still be needed if run standalone in certain ways
    try:
        from ai_calendar_categorizer import AICalendarCategorizer
    except ImportError as e:
         # Log or raise a more informative error if needed
         raise ImportError("Could not import AICalendarCategorizer. Ensure it's accessible.") from e

# Configure logging for this report module
logger = logging.getLogger(__name__)

def fetch_calendar_events(calendar_url, username, password, start, end):
    logger.info(f"Attempting to fetch calendar events from {calendar_url} between {start} and {end}")
    try:
        # Ensure start/end are datetime objects for caldav
        if isinstance(start, date) and not isinstance(start, datetime):
            start_dt = datetime.combine(start, time.min)
        else:
            start_dt = start
        if isinstance(end, date) and not isinstance(end, datetime):
            end_dt = datetime.combine(end, time.max) # Use time.max for end date to include whole day
        else:
            end_dt = end
        logger.debug(f"Using datetime range for search: {start_dt} to {end_dt}")

        client = caldav.DAVClient(url=calendar_url, username=username, password=password)

        # --- Use the specific calendar URL directly ---
        logger.info(f"Attempting to access calendar directly at URL: {calendar_url}")
        calendar = caldav.Calendar(client=client, url=calendar_url)
        logger.info(f"Searching for events between {start_dt} and {end_dt} (inclusive)...")
        events_data = calendar.date_search(start=start_dt, end=end_dt, expand=True)
        logger.info(f"calendar.date_search returned type: {type(events_data)}")

        events = []
        event_count = 0
        # Iterate safely, logging potential issues
        # Outer try for the iteration process itself
        try:
            for event in events_data:
                event_count += 1
                event_url = getattr(event, 'url', 'N/A')
                logger.debug(f"Processing event {event_count} - URL: {event_url}")
                # Inner try for parsing individual event data
                try:
                    vevent = event.instance.vevent
                    summary = getattr(vevent, 'summary', None)
                    description = getattr(vevent, 'description', None)
                    location = getattr(vevent, 'location', None)

                    events.append({
                        "title": str(summary.value) if summary else '',
                        "description": str(description.value) if description else '',
                        "location": str(location.value) if location else ''
                    })
                except Exception as event_parse_e:
                     logger.warning(f"Could not parse event data: {event_parse_e}. Event URL: {event_url}", exc_info=True)
                     continue # Skip this event, continue loop
        except Exception as iteration_e:
             logger.error(f"Error during iteration over events_data: {iteration_e}", exc_info=True)
             st.warning("An error occurred while processing calendar events. Some events might be missing.")

        logger.info(f"Successfully processed {event_count} events from date_search, yielding {len(events)} valid events.")
        return events

    except caldav.lib.error.NotFoundError as nf_e:
         logger.error(f"Calendar not found at URL {calendar_url}: {nf_e}", exc_info=True)
         st.error(f"Calendar not found at the specified URL. Please check MAILCOW_CALENDAR_URL.")
         return []
    except Exception as e:
        error_msg = f"Failed to fetch calendar events: {e}"
        logger.exception(error_msg) # Log full traceback
        st.error(error_msg)
        return []


def categorize_event(agent, event):
    # Add logging within the categorization if needed, especially for errors
    try:
        category = agent.categorize_event(
            event["title"], event["description"], event["location"]
        )
        # logger.debug(f"Categorized '{event['title']}' as '{category}'")
        return event, category
    except Exception as e:
        logger.error(f"Error during categorization for event '{event.get('title', 'N/A')}': {e}", exc_info=True)
        raise # Re-raise to be caught in the main loop


def main(start_date, end_date):
    st.header("Calendar Categorization Report")
    logger.info(f"Running Calendar Categorization report for {start_date} to {end_date}")

    # Retrieve credentials from environment variables (loaded in app.py)
    calendar_url = os.getenv("MAILCOW_CALENDAR_URL")
    username = os.getenv("MAILCOW_USERNAME")
    password = os.getenv("MAILCOW_PASSWORD")
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434") # Allow overriding Ollama URL via env
    ollama_model = os.getenv("OLLAMA_MODEL", "gemma3:12b") # Allow overriding model via env

    # Log retrieved creds (mask password)
    logger.info(f"Calendar URL: {calendar_url}")
    logger.info(f"Username: {username}")
    logger.info(f"Ollama URL: {ollama_url}")
    logger.info(f"Ollama Model: {ollama_model}")

    if not calendar_url or not username or not password:
        st.error("Please ensure MAILCOW_CALENDAR_URL, MAILCOW_USERNAME, and MAILCOW_PASSWORD are set in the .env file.")
        logger.error("Missing calendar credentials in environment variables.")
        return

    with st.spinner("Fetching calendar events..."):
        events = fetch_calendar_events(calendar_url, username, password, start_date, end_date)

    if not events:
        logger.warning("No events returned by fetch_calendar_events.")
        st.warning("No events found in the selected date range or could not be fetched.")
        return

    try:
        logger.info(f"Initializing AI Categorizer with model {ollama_model} at {ollama_url}")
        agent = AICalendarCategorizer(ollama_url=ollama_url, model=ollama_model)
    except Exception as e:
        error_msg = f"Error initializing AI categorizer: {e}"
        logger.exception(error_msg)
        st.error(error_msg)
        return

    category_counts = Counter()
    processed_count = 0
    total_events = len(events)
    max_workers = int(os.getenv("CALENDAR_MAX_WORKERS", 5))

    st.info(f"Categorizing {total_events} events using {max_workers} workers...")
    progress_text = st.empty()
    progress_bar = st.progress(0)

    logger.info(f"Starting categorization with {max_workers} workers.")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(categorize_event, agent, event): event for event in events}
        for future in as_completed(futures):
            event_data = futures[future]
            try:
                event, category = future.result()
                logger.debug(f"Successfully categorized event '{event.get('title', 'N/A')}' as '{category}'")
                if category != "Personal / Lunch / Non-business":
                    category_counts[category] += 1
            except Exception as e:
                st.warning(f"Could not categorize event: '{event_data.get('title', 'N/A')}'. See logs for details.")
            finally:
                processed_count += 1
                progress_percentage = processed_count / total_events
                try:
                    progress_text.text(f"Categorized {processed_count}/{total_events} events")
                    progress_bar.progress(progress_percentage)
                except Exception as st_update_e:
                     logger.warning(f"Error updating Streamlit progress: {st_update_e}")

    progress_text.text(f"Categorization Complete ✅ ({processed_count}/{total_events})")
    progress_bar.empty()
    logger.info("Categorization process finished.")

    st.subheader("Event Categorization Summary:")

    if not category_counts:
         st.info("No business-related categories found after filtering.")
         logger.info("Category counts are empty after filtering.")
         return

    try:
        df = pd.DataFrame(list(category_counts.items()), columns=["Category", "Count"])
        df = df.sort_values(by="Count", ascending=False)
        df_display = df.set_index("Category")
        st.bar_chart(df_display)
        st.dataframe(df)
    except Exception as e:
        error_msg = f"Failed to display categorization results: {e}"
        logger.exception(error_msg)
        st.error(error_msg)

    logger.info("Report execution finished.")

# Removed __main__ block
