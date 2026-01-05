LongCat API Platform Quick Start Guide
Welcome to the LongCat API Platform! This document will help you get started quickly and begin using our large model services.

How to Get an API Key?
Register an Account
Visit LongCat API Platform
Fill in the required information to complete account registration
Create Your API Key
After registering and logging in, go to the API Keys page and click “Create API Key” to manually generate a key.
Once created, you can view the following details in the list:
Name: Your custom identifier for the application
Key: The system-generated secret (please keep it safe and do not disclose it)
Supported API Types
LongCat API Platform is compatible with two mainstream API formats. You can choose according to your needs:

OpenAI API Format
Fully compatible with the OpenAI API specification, supporting the following endpoints:

Chat Completion: /v1/chat/completions
Anthropic API Format
Compatible with the Anthropic Claude API specification, supporting the following endpoint:

Message Chat: /v1/messages
Endpoints
OpenAI Format: https://api.longcat.chat/openai
Anthropic Format: https://api.longcat.chat/anthropic
Supported Models
Model Name	API Format	Description
LongCat-Flash-Chat	OpenAI/Anthropic	High-performance general-purpose chat model
LongCat-Flash-Thinking	OpenAI/Anthropic	Deep-thinking model
How to Get Usage Quota?
Daily Free Quota
Each account automatically receives 500,000 Tokens free quota per day
The free quota is refreshed automatically at midnight (Beijing Time) every day
Unused quota from the previous day will be cleared and will not roll over to the next day
Ways to Request More Free Quota
You can visit the Usage to apply for an increase in your free Tokens quota. Upon approval, your free quota will be raised to 5,000,000 Tokens/day.
For additional quota, please contact us by email at longcat-team@meituan.com .
Quota Usage Instructions
Both input and output Tokens are counted towards consumption
Streaming and non-streaming endpoints consume quota equally
Quota Inquiry
You can view your usage in real time at Usage

Note: The platform is currently in public beta and does not support paid quota purchases.

Rate Limiting Rules
Single Request Limit
Output text: Maximum 8K Tokens
Over-limit Handling
When rate limiting is triggered, the API will return HTTP status code 429. Example response:


{
  "error": {
    "code": "rate_limit_exceeded",
    "message": "Request rate limit exceeded, please try again later",
    "type": "rate_limit_error",
    "retry_after": 60
  }
}
It is recommended to implement exponential backoff retry mechanism on the client side.

Quick Integration Examples
OpenAI API Format Example
Python Example

import requests

url = "https://api.longcat.chat/openai/v1/chat/completions"
headers = {
    "Authorization": "Bearer YOUR_APP_KEY",
    "Content-Type": "application/json"
}

data = {
    "model": "LongCat-Flash-Chat",
    "messages": [
        {"role": "user", "content": "Hello, please introduce yourself."}
    ],
    "max_tokens": 1000,
    "temperature": 0.7
}

response = requests.post(url, headers=headers, json=data)
print(response.json())
Using OpenAI SDK

from openai import OpenAI

client = OpenAI(
    api_key="YOUR_APP_KEY",
    base_url="https://api.longcat.chat/openai"
)

response = client.chat.completions.create(
    model="LongCat-Flash-Chat",
    messages=[
        {"role": "user", "content": "Hello!"}
    ],
    max_tokens=1000
)

print(response.choices[0].message.content)
Anthropic API Format Example
Using Anthropic SDK

from anthropic import Anthropic

client = Anthropic(
    api_key="Authorization: Bearer YOUR_APP_KEY",
    base_url="https://api.longcat.chat/anthropic/",
    default_headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer YOUR_APP_KEY",
    }
)


response = client.messages.create(
    model="LongCat-Flash-Chat",
    max_tokens=1000,
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.content[0].text)
cURL Example
OpenAI Format

curl -X POST https://api.longcat.chat/openai/v1/chat/completions \
  -H "Authorization: Bearer YOUR_APP_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "LongCat-Flash-Chat",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 1000
  }'
Anthropic Format

curl -X POST https://api.longcat.chat/anthropic/v1/messages \
  -H "Authorization: Bearer YOUR_APP_KEY" \
  -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "LongCat-Flash-Chat",
    "max_tokens": 1000,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
Important Reminders
Please use your daily free quota reasonably. Unused quota will not be retained for the next day
Please keep your API Key safe to avoid quota theft due to leakage
Now you have learned the basics of using the LongCat API Platform. Go ahead and try it out!