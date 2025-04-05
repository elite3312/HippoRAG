import json

def convert_to_json_string(raw_text):
    """
    Converts raw multi-line text into a JSON-compatible single-line string.
    """
    # Use json.dumps to escape special characters and format the string
    json_compatible_string = json.dumps(raw_text)
    return json_compatible_string

# Example input
raw_text_0 = """To set up your company email on your mobile device, please follow these steps:

**First, ensure that you have a supported operating system (iOS, Android, or Windows) and a company email account.**

1. **Check if a Mobile Device Management (MDM) profile is required**: If your company requires MDM for mobile devices, ensure that the profile is installed on your device. If you're unsure, contact your IT department for assistance.
2. **Set up your email account**:
    * Go to the Settings app on your mobile device.
    * Select "Mail" or "Email" (depending on your device's operating system).
    * Tap "Add Account" or "Create a new account".
    * Select "Exchange" or "Corporate" as the account type.
    * Enter your company email address and password.
    * If prompted, enter the company's email server address (e.g., mail.company.com).
    * Select the desired synchronization options (e.g., sync email, contacts, calendar).
3. **Configure your email settings**:
    * In the email account settings, select the "Advanced" or "Security" option.
    * Ensure that the "Use SSL/TLS" or "Use secure connection" option is enabled.
    * Set the authentication method to "Username and Password" or "Domain\\Username".
    * If prompted, enter your company's email domain (e.g., company.com).
4. **Verify your email account**:
    * Wait for the email account to synchronize with the company email server.
    * Send a test email to yourself or a colleague to verify that email is sending and receiving correctly.

If you encounter any issues, refer to the troubleshooting tips provided. If you're still having trouble, don't hesitate to contact your IT department for assistance."""


raw_text_0=r"""**Setting Up a Mobile Device for Company Email**

**Prerequisites:**

* Mobile device with a supported operating system (iOS, Android, or Windows)
* Company email account credentials
* Mobile device management (MDM) profile installed (if required by company policy)

**Step 1: Ensure Mobile Device Management (MDM) Profile is Installed (if required)**

If your company requires MDM for mobile devices, ensure that the profile is installed on your device. This profile will allow your device to connect to the company network and access company email. If you are unsure whether MDM is required, contact your IT department for assistance.

**Step 2: Set Up Email Account on Mobile Device**

1. Go to the Settings app on your mobile device.
2. Select "Mail" or "Email" (depending on your device's operating system).
3. Tap "Add Account" or "Create a new account".
4. Select "Exchange" or "Corporate" as the account type.
5. Enter your company email address and password.
6. If prompted, enter the company's email server address (e.g., mail.company.com).
7. Select the desired synchronization options (e.g., sync email, contacts, calendar).

**Step 3: Configure Email Settings**

1. In the email account settings, select the "Advanced" or "Security" option.
2. Ensure that the "Use SSL/TLS" or "Use secure connection" option is enabled.
3. Set the authentication method to "Username and Password" or "Domain\Username".
4. If prompted, enter your company's email domain (e.g., company.com).

**Step 4: Verify Email Account**

1. Wait for the email account to synchronize with the company email server.
2. Send a test email to yourself or a colleague to verify that email is sending and receiving correctly.

**Troubleshooting Tips:**

* If you encounter issues setting up your email account, ensure that your device has a stable internet connection and that your email credentials are correct.
* If you are unable to connect to the company email server, contact your IT department for assistance.
* If you experience issues with email synchronization, try restarting your device or checking the email account settings.

**Additional Information:**

* For security reasons, it is recommended to set up a password or PIN lock on your mobile device.
* Company email policies may require additional security measures, such as encryption or two-factor authentication. Contact your IT department for more information.
* If you need further assistance or have questions about company email policies, contact your IT department or refer to the company's email policy documentation."""

with open("./input.txt", "r", encoding="utf-8") as file:
    raw_text= file.read()
    
# Convert the raw text to a JSON-compatible string
json_string = convert_to_json_string(raw_text)

# Print the result
print(json_string)

# Write the string to an output file
with open("./output.txt", "w", encoding="utf-8") as file:
    file.write(json_string)
