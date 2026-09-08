curl --location --request POST 'https://cognitiveinsprod.azure-api.net/openai/deployments/gpt-4o-mini/chat/completions?api-version=2025-01-01-preview' \
--header 'api-key: FFIE8TZtUTnP5HKU7ARHGyN3Fo18OrYNQJ5dxnS3VK2Cfw5sOHcZJQQJ99CDAC77bzfXJ3w3AAAAACOGk7cs' \
--header 'Content-Type: application/json' \
--data-raw '{
    "temperature": 1,
    "top_p": 1,
    "stream": false,
    "max_tokens": 4096,
    "presence_penalty": 0,
    "frequency_penalty": 0,
    "logit_bias": {},
    "user": "user-1234",
    "messages": [
        {
            "role": "user",
            "content": "What is the capital of india"
        }
    ]
}'
