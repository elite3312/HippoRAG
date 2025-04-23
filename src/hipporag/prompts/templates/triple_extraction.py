# from .ner import one_shot_ner_paragraph, one_shot_ner_output
# from ...utils.llm_utils import convert_format_to_template

# ner_conditioned_re_system = """Your task is to construct an RDF (Resource Description Framework) graph from the given passages and named entity lists. 
# Respond with a JSON list of triples, with each triple representing a relationship in the RDF graph. 

# Pay attention to the following requirements:
# - Each triple should contain at least one, but preferably two, of the named entities in the list for each passage.
# - Clearly resolve pronouns to their specific names to maintain clarity.

# """


# ner_conditioned_re_frame = """Convert the paragraph into a JSON dict, it has a named entity list and a triple list.
# Paragraph:
# ```
# {passage}
# ```

# {named_entity_json}
# """


# ner_conditioned_re_input = ner_conditioned_re_frame.format(passage=one_shot_ner_paragraph, named_entity_json=one_shot_ner_output)


# ner_conditioned_re_output = """{"triples": [
#             ["Radio City", "located in", "India"],
#             ["Radio City", "is", "private FM radio station"],
#             ["Radio City", "started on", "3 July 2001"],
#             ["Radio City", "plays songs in", "Hindi"],
#             ["Radio City", "plays songs in", "English"],
#             ["Radio City", "forayed into", "New Media"],
#             ["Radio City", "launched", "PlanetRadiocity.com"],
#             ["PlanetRadiocity.com", "launched in", "May 2008"],
#             ["PlanetRadiocity.com", "is", "music portal"],
#             ["PlanetRadiocity.com", "offers", "news"],
#             ["PlanetRadiocity.com", "offers", "videos"],
#             ["PlanetRadiocity.com", "offers", "songs"]
#     ]
# }
# """


# prompt_template = [
#     {"role": "system", "content": ner_conditioned_re_system},
#     {"role": "user", "content": ner_conditioned_re_input},
#     {"role": "assistant", "content": ner_conditioned_re_output},
#     {"role": "user", "content": convert_format_to_template(original_string=ner_conditioned_re_frame, placeholder_mapping=None, static_values=None)}
# ]

from ...utils.llm_utils import convert_format_to_template

ner_conditioned_re_system = """Your task is to construct an RDF (Resource Description Framework) graph from the given passages and named entity lists. 
Respond with a JSON list of triples, with each triple representing a relationship in the RDF graph. 

Pay attention to the following requirements:
- Each triple should contain at least one, but preferably two, of the named entities in the list for each passage.
- Clearly resolve pronouns to their specific names to maintain clarity.

"""

ner_conditioned_re_frame = """Convert the paragraph into a JSON dict, it has a named entity list and a triple list.
Paragraph:
{passage}
{named_entity_json}
"""

# ----------- EXAMPLE 1 -----------
it_paragraph_1 = """Acme Corp IT Helpdesk provides 24/7 technical support for all employees.
The team uses ServiceNow for ticket management and supports Windows 11, macOS Ventura, and Ubuntu 22.04 devices.
In January 2024, the helpdesk upgraded its remote assistance tool to TeamViewer."""

it_entities_1 = """{"named_entities": [
    "Acme Corp", "IT Helpdesk", "ServiceNow", "Windows 11", "macOS Ventura", "Ubuntu 22.04",
    "January 2024", "TeamViewer"
]}"""

ner_conditioned_re_input_1 = ner_conditioned_re_frame.format(
    passage=it_paragraph_1,
    named_entity_json=it_entities_1
)

ner_conditioned_re_output_1 = """{"triples": [
    ["Acme Corp", "has", "IT Helpdesk"],
    ["IT Helpdesk", "provides", "technical support"],
    ["IT Helpdesk", "uses", "ServiceNow"],
    ["IT Helpdesk", "supports", "Windows 11"],
    ["IT Helpdesk", "supports", "macOS Ventura"],
    ["IT Helpdesk", "supports", "Ubuntu 22.04"],
    ["IT Helpdesk", "upgraded tool to", "TeamViewer"],
    ["TeamViewer", "was upgraded in", "January 2024"]
]}
"""

# ----------- EXAMPLE 2 -----------
it_paragraph_2 = """GlobalTech Solutions migrated its cloud infrastructure to Microsoft Azure in March 2023.
Critical databases were transferred to SQL Server 2022.
Okta and Duo Security are used for authentication."""

it_entities_2 = """{"named_entities": [
    "GlobalTech Solutions", "Microsoft Azure", "March 2023", "SQL Server 2022",
    "Okta", "Duo Security"
]}"""

ner_conditioned_re_input_2 = ner_conditioned_re_frame.format(
    passage=it_paragraph_2,
    named_entity_json=it_entities_2
)

ner_conditioned_re_output_2 = """{"triples": [
    ["GlobalTech Solutions", "migrated to", "Microsoft Azure"],
    ["GlobalTech Solutions", "migrated in", "March 2023"],
    ["Critical databases", "transferred to", "SQL Server 2022"],
    ["GlobalTech Solutions", "uses", "Okta"],
    ["GlobalTech Solutions", "uses", "Duo Security"]
]}
"""

# ----------- EXAMPLE 3 -----------
it_paragraph_3 = """On April 20, 2025, John Doe from the Sales department was unable to log in to the company portal at portal.megacorp.com.
He received a '403 Forbidden' error while using Google Chrome on Windows 10.
The issue was escalated to IT support via ticket #15432 on Jira."""

it_entities_3 = """{"named_entities": [
    "April 20, 2025", "John Doe", "Sales department", "company portal", "portal.megacorp.com",
    "403 Forbidden", "Google Chrome", "Windows 10", "IT support", "ticket #15432", "Jira"
]}"""

ner_conditioned_re_input_3 = ner_conditioned_re_frame.format(
    passage=it_paragraph_3,
    named_entity_json=it_entities_3
)

ner_conditioned_re_output_3 = """{"triples": [
    ["John Doe", "belongs to", "Sales department"],
    ["John Doe", "could not log in to", "company portal"],
    ["company portal", "URL", "portal.megacorp.com"],
    ["John Doe", "received error", "403 Forbidden"],
    ["John Doe", "used", "Google Chrome"],
    ["John Doe", "used", "Windows 10"],
    ["Issue", "escalated to", "IT support"],
    ["Issue", "tracked by", "ticket #15432"],
    ["ticket #15432", "logged on", "Jira"],
    ["Issue", "occurred on", "April 20, 2025"]
]}
"""

# ----------- EXAMPLE 4 -----------
it_paragraph_4 = """On May 2, 2024, Sarah Lee from the Engineering department experienced issues connecting to the corporate VPN via Cisco AnyConnect on her MacBook Pro running macOS Sonoma.
She reported error code 'Login Failed' and contacted the Network Operations Center (NOC) at 09:35 AM.
The case was tracked as incident #2024-509 in ServiceNow."""

it_entities_4 = """{"named_entities": [
    "May 2, 2024", "Sarah Lee", "Engineering department", "corporate VPN", "Cisco AnyConnect",
    "MacBook Pro", "macOS Sonoma", "Login Failed", "Network Operations Center", "NOC",
    "09:35 AM", "incident #2024-509", "ServiceNow"
]}"""

ner_conditioned_re_input_4 = ner_conditioned_re_frame.format(
    passage=it_paragraph_4,
    named_entity_json=it_entities_4
)

ner_conditioned_re_output_4 = """{"triples": [
    ["Sarah Lee", "belongs to", "Engineering department"],
    ["Sarah Lee", "used", "MacBook Pro"],
    ["MacBook Pro", "runs", "macOS Sonoma"],
    ["Sarah Lee", "attempted connection to", "corporate VPN"],
    ["corporate VPN", "accessed via", "Cisco AnyConnect"],
    ["Sarah Lee", "received error", "Login Failed"],
    ["Sarah Lee", "reported issue to", "Network Operations Center"],
    ["Network Operations Center", "also called", "NOC"],
    ["Issue", "reported at", "09:35 AM"],
    ["Issue", "occurred on", "May 2, 2024"],
    ["Issue", "tracked as", "incident #2024-509"],
    ["incident #2024-509", "logged in", "ServiceNow"]
]}
"""

# ----------- EXAMPLE 5 -----------
it_paragraph_5 = """In June 2023, the Information Security team detected a phishing email targeting Finance department users at DataVantage Inc.
The malicious email contained a fake Microsoft 365 login page.
The team initiated a password reset for affected accounts using the Okta dashboard and notified users via helpdesk@datavantage.com."""

it_entities_5 = """{"named_entities": [
    "June 2023", "Information Security team", "phishing email", "Finance department", "DataVantage Inc",
    "Microsoft 365", "Okta", "helpdesk@datavantage.com"
]}"""

ner_conditioned_re_input_5 = ner_conditioned_re_frame.format(
    passage=it_paragraph_5,
    named_entity_json=it_entities_5
)

ner_conditioned_re_output_5 = """{"triples": [
    ["Information Security team", "detected", "phishing email"],
    ["phishing email", "targeted", "Finance department"],
    ["Finance department", "is part of", "DataVantage Inc"],
    ["phishing email", "contained", "fake Microsoft 365 login page"],
    ["Information Security team", "initiated password reset via", "Okta"],
    ["Information Security team", "notified users via", "helpdesk@datavantage.com"],
    ["Incident", "occurred in", "June 2023"]
]}
"""

# --------- (Append these to your prompt_template) ---------
prompt_template = [
    {"role": "system", "content": ner_conditioned_re_system},
    {"role": "user", "content": ner_conditioned_re_input_1},
    {"role": "assistant", "content": ner_conditioned_re_output_1},
    {"role": "user", "content": ner_conditioned_re_input_2},
    {"role": "assistant", "content": ner_conditioned_re_output_2},
    {"role": "user", "content": ner_conditioned_re_input_3},
    {"role": "assistant", "content": ner_conditioned_re_output_3},
    {"role": "user", "content": ner_conditioned_re_input_4},
    {"role": "assistant", "content": ner_conditioned_re_output_4},
    {"role": "user", "content": ner_conditioned_re_input_5},
    {"role": "assistant", "content": ner_conditioned_re_output_5},
    {"role": "user", "content": convert_format_to_template(
        original_string=ner_conditioned_re_frame,
        placeholder_mapping=None,
        static_values=None
    )}
]