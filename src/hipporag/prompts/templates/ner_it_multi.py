ner_system = """You are an IT support assistant. Your task is to extract named entities from the given IT support-related paragraph. 
Respond with a JSON list of entities, capturing organization names, product names, dates, email addresses, phone numbers, technical terms, locations, error codes, and other relevant entities.
"""

# Example 1 (User Login)
paragraph_1 = """User Login Issue
On April 20, 2025, John Doe from the Sales department was unable to log in to the company portal at portal.megacorp.com.
He received a '403 Forbidden' error while using Google Chrome on Windows 10.
The issue was escalated to IT support via ticket #15432 on Jira.
"""

output_1 = """{"named_entities": [
    "April 20, 2025", "John Doe", "Sales department", "company portal", "portal.megacorp.com",
    "403 Forbidden", "Google Chrome", "Windows 10", "IT support", "ticket #15432", "Jira"
]}
"""

# Example 2 (Mobile Email Setup)
paragraph_2 = """Mobile Email Setup
To set up company email on your mobile device, ensure the Mobile Device Management (MDM) profile is installed. 
If unsure, contact the IT department for assistance. 
After installing MDM, open the Email app, select 'Add Account', and enter your company email credentials.
"""

output_2 = """{"named_entities": [
    "Mobile Device Management (MDM) profile", "IT department", "Email app", "Add Account", "company email credentials"
]}
"""

# Example 3 (PIN Reset)
paragraph_3 = """Resetting a Forgotten PIN
If you have forgotten your PIN, go to the IT Support page, click on the Self-Service tab, and select 'PIN Reset'. 
Authenticate with your company username and password, then answer your security question to reset your PIN.
If you cannot reset your PIN, contact the IT Helpdesk.
"""

output_3 = """{"named_entities": [
    "IT Support page", "Self-Service tab", "PIN Reset", "company username", "password", "security question", "IT Helpdesk"
]}
"""

prompt_template = [
    {"role": "system", "content": ner_system},
    {"role": "user", "content": paragraph_1},
    {"role": "assistant", "content": output_1},
    {"role": "user", "content": paragraph_2},
    {"role": "assistant", "content": output_2},
    {"role": "user", "content": paragraph_3},
    {"role": "assistant", "content": output_3},
    {"role": "user", "content": "${passage}"}
]