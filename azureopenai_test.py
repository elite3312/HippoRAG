import os
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key="<key>",
    api_version="2024-10-21",
    azure_endpoint="https://hkust.azure-api.net"
)

chat_completion = client.chat.completions.create(
    messages=[
        {       
                
            "role": "user",
            "content": "Say this is a test",
        }
    ],
    model="gpt-4o-mini",
)
print(chat_completion.choices[0].message.content)