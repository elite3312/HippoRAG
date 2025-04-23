ner_system = """Your task is to extract named entities from the given paragraph. 
Respond with a JSON list of entities.
"""

# Example 1
paragraph_1 = """Acme Corp IT Helpdesk
Acme Corp's IT Helpdesk provides 24/7 technical support for all employees. 
The team uses ServiceNow for ticket management and supports Windows 11, macOS Ventura, and Ubuntu 22.04 devices. 
In January 2024, the helpdesk upgraded its remote assistance tool to TeamViewer. 
All IT support requests can be submitted through support@acmecorp.com or by calling +1-800-555-1234.
"""

output_1 = """{"named_entities": [
    "Acme Corp", "IT Helpdesk", "ServiceNow", "Windows 11", "macOS Ventura", "Ubuntu 22.04",
    "January 2024", "TeamViewer", "support@acmecorp.com", "+1-800-555-1234"
]}
"""

# Example 2
paragraph_2 = """GlobalTech Solutions
GlobalTech Solutions migrated its cloud infrastructure to Microsoft Azure in March 2023. 
The migration included critical databases managed through SQL Server 2022. 
Employees use Okta for identity management and Duo Security for multi-factor authentication. 
For urgent issues, staff can call the IT hotline at 555-7890.
"""

output_2 = """{"named_entities": [
    "GlobalTech Solutions", "Microsoft Azure", "March 2023", "SQL Server 2022", 
    "Okta", "Duo Security", "IT hotline", "555-7890"
]}
"""

# Example 3 (User Login)
paragraph_3 = """User Login Issue
On April 20, 2025, John Doe from the Sales department was unable to log in to the company portal at portal.megacorp.com.
He received a '403 Forbidden' error while using Google Chrome on Windows 10.
The issue was escalated to IT support via ticket #15432 on Jira.
"""

output_3 = """{"named_entities": [
    "April 20, 2025", "John Doe", "Sales department", "company portal", "portal.megacorp.com",
    "403 Forbidden", "Google Chrome", "Windows 10", "IT support", "ticket #15432", "Jira"
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