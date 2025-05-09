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

it_paragraph_a = """To set up company email on your mobile device, ensure that the Mobile Device Management (MDM) profile is installed. 
This profile allows your device to connect to the company network and access company email. 
Contact your IT department if you are unsure whether MDM is required."""

it_entities_a = """{"named_entities": [
    "Mobile Device Management (MDM) profile", "company network", "company email", "IT department"
]}"""

ner_conditioned_re_input_a = ner_conditioned_re_frame.format(
    passage=it_paragraph_a,
    named_entity_json=it_entities_a
)

ner_conditioned_re_output_a = """{"triples": [
    ["Mobile Device Management (MDM) profile", "must be installed for", "company email"],
    ["Mobile Device Management (MDM) profile", "enables connection to", "company network"],
    ["User", "should contact", "IT department"]
]}
"""

it_paragraph_b = """If you forgot your PIN, go to the IT Support page, click the Self-Service tab, and select 'PIN Reset.' 
You will be redirected to the PIN Reset Tool login page. 
Authenticate with your credentials and answer your security question to reset your PIN."""

it_entities_b = """{"named_entities": [
    "IT Support page", "Self-Service tab", "PIN Reset", "PIN Reset Tool", "credentials", "security question"
]}"""

ner_conditioned_re_input_b = ner_conditioned_re_frame.format(
    passage=it_paragraph_b,
    named_entity_json=it_entities_b
)

ner_conditioned_re_output_b = """{"triples": [
    ["User", "navigates to", "IT Support page"],
    ["User", "selects", "Self-Service tab"],
    ["User", "clicks", "PIN Reset"],
    ["User", "uses", "PIN Reset Tool"],
    ["User", "authenticates with", "credentials"],
    ["User", "answers", "security question"],
    ["User", "resets", "PIN"]
]}
"""

it_paragraph_c = """To verify your VPN connection, open a web browser and navigate to company.com. 
Ensure you are connected to the VPN and can access company resources."""

it_entities_c = """{"named_entities": [
    "VPN connection", "web browser", "company.com", "company resources"
]}"""

ner_conditioned_re_input_c = ner_conditioned_re_frame.format(
    passage=it_paragraph_c,
    named_entity_json=it_entities_c
)

ner_conditioned_re_output_c = """{"triples": [
    ["User", "opens", "web browser"],
    ["User", "navigates to", "company.com"],
    ["User", "verifies", "VPN connection"],
    ["VPN connection", "enables access to", "company resources"]
]}
"""

it_paragraph_d = """To factory reset your company-issued tablet, go to Settings > System > Advanced > Reset Options > Erase all data (factory reset). 
Ensure all important data is backed up before proceeding."""

it_entities_d = """{"named_entities": [
    "company-issued tablet", "Settings", "System", "Advanced", "Reset Options", "Erase all data (factory reset)", "important data"
]}"""

ner_conditioned_re_input_d = ner_conditioned_re_frame.format(
    passage=it_paragraph_d,
    named_entity_json=it_entities_d
)

ner_conditioned_re_output_d = """{"triples": [
    ["User", "navigates to", "Settings"],
    ["User", "selects", "System"],
    ["User", "chooses", "Advanced"],
    ["User", "opens", "Reset Options"],
    ["User", "executes", "Erase all data (factory reset)"],
    ["User", "backs up", "important data"]
]}
"""

it_paragraph_e = """For a secure wireless network, use WPA2-PSK (AES) or WPA3-PSK (AES-256) as the wireless protocol. 
Avoid outdated protocols like WEP or WPA."""

it_entities_e = """{"named_entities": [
    "WPA2-PSK (AES)", "WPA3-PSK (AES-256)", "wireless protocol", "WEP", "WPA", "wireless network"
]}"""

ner_conditioned_re_input_e = ner_conditioned_re_frame.format(
    passage=it_paragraph_e,
    named_entity_json=it_entities_e
)

ner_conditioned_re_output_e = """{"triples": [
    ["wireless network", "should use", "WPA2-PSK (AES)"],
    ["wireless network", "should use", "WPA3-PSK (AES-256)"],
    ["wireless network", "should avoid", "WEP"],
    ["wireless network", "should avoid", "WPA"]
]}
"""

prompt_template = [
    {"role": "system", "content": ner_conditioned_re_system},
    {"role": "user", "content": ner_conditioned_re_input_a},
    {"role": "assistant", "content": ner_conditioned_re_output_a},
    {"role": "user", "content": ner_conditioned_re_input_b},
    {"role": "assistant", "content": ner_conditioned_re_output_b},
    {"role": "user", "content": ner_conditioned_re_input_c},
    {"role": "assistant", "content": ner_conditioned_re_output_c},
    {"role": "user", "content": ner_conditioned_re_input_d},
    {"role": "assistant", "content": ner_conditioned_re_output_d},
    {"role": "user", "content": ner_conditioned_re_input_e},
    {"role": "assistant", "content": ner_conditioned_re_output_e},
    {"role": "user", "content": convert_format_to_template(
        original_string=ner_conditioned_re_frame,
        placeholder_mapping=None,
        static_values=None
    )}
]