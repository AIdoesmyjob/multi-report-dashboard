import caldav
from datetime import datetime
from collections import Counter
from ai_calendar_categorizer import AICalendarCategorizer
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def fetch_calendar_events(calendar_url, username, password, start, end):
    client = caldav.DAVClient(
        url=calendar_url,
        username=username,
        password=password
    )

    calendar = caldav.Calendar(client=client, url=calendar_url)

    events = []
    for event in calendar.date_search(start, end):
        vevent = event.instance.vevent
        events.append({
            "title": str(vevent.summary.value),
            "description": str(vevent.description.value) if hasattr(vevent, 'description') else '',
            "location": str(vevent.location.value) if hasattr(vevent, 'location') else ''
        })
    return events

def categorize_event(agent, event):
    category = agent.categorize_event(
        event["title"], 
        event["description"], 
        event["location"]
    )
    return event, category

if __name__ == '__main__':
    calendar_url = "https://webmail.skofo.ca/SOGo/dav/lukas@lukasmatheson.com/Calendar/141-67E2F780-1-42B8D300/"
    username = "lukas@lukasmatheson.com"
    password = "1!Qy!qKRRK*d6A0pKY@y"

    start = datetime(2025, 2, 1)
    end = datetime(2025, 3, 1)

    events = fetch_calendar_events(calendar_url, username, password, start, end)
    
    if not events:
        print("No events found in February 2025.")
        exit()

    agent = AICalendarCategorizer(
        ollama_url="http://localhost:11434", 
        model="gemma3:12b"
    )

    category_counts = Counter()

    print("Categorizing events concurrently...")

    # Adjust max_workers depending on your system capabilities
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(categorize_event, agent, event): event for event in events}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Events"):
            event, category = future.result()
            print(f"\n--- Event Details ---")
            print(f"Title: {event['title']}")
            print(f"Category: {category}")

            if category != "Personal / Lunch / Non-business":
                category_counts[category] += 1

    print("\n✅ Categorized Calendar Events for February 2025:")
    for cat, count in category_counts.items():
        print(f"{cat}: {count}")
