import requests

class AICalendarCategorizer:
    def __init__(self, ollama_url="http://localhost:11434", model="gemma3:12b"):
        self.ollama_url = ollama_url
        self.model = model

    def categorize_event(self, title, description='', location=''):
        prompt = prompt = f"""
You're an expert calendar categorization assistant for a property management company.

Categorize each calendar event into exactly ONE of these categories:

1. Owner Meeting
   - Any interaction or communication explicitly involving property owners.

2. Property Showing
   - Events involving showing, touring properties, or explicitly mentioning "showing", "tour", or prospective tenants by name.

3. Move-in Inspection
   - Inspections clearly done when tenants move in.

4. Move-out Inspection
   - Inspections clearly done when tenants move out.

5. Tenant Meeting
   - Meetings or calls with tenants, prospective tenants, applicants, or events referencing tenant portals.

6. Contractor Meeting
   - Meetings explicitly involving contractors, maintenance personnel, or clear maintenance-related tasks.

7. Internal/Admin Tasks
   - Clearly internal or administrative tasks, including reporting, internal key pickups, testing keys, internal meetings, daily/weekly internal updates.

8. Business Lunch / External Business Meeting
   - Lunch or dining meetings explicitly involving external discussions about properties, pricing, or leases.

9. Personal / Lunch / Non-business
   - Clearly personal, non-business, vacation, or explicitly "no appointments" events.

Important clarification for unclear events:
- If only a first name is provided without additional context, default to "Tenant Meeting".
- If "portal" or tenant-specific software is mentioned (like "Showmojo"), default to "Tenant Meeting".
- Internal meetings clearly stating "daily" or "weekly" meetings without external context default to "Internal/Admin Tasks".

Event details:
Title: "{title}"
Description: "{description}"
Location: "{location}"

Respond with exactly one category.
"""

        response = requests.post(f"{self.ollama_url}/api/generate", json={
            "model": self.model,
            "prompt": prompt,
            "stream": False
        })

        if response.status_code == 200:
            return response.json()['response'].strip()
        else:
            raise Exception(f"Ollama categorization failed: {response.text}")
