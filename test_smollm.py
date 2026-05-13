import requests

ENDPOINT = "http://localhost:12434/engines/llama.cpp/v1/chat/completions"
MODEL    = "ai/smollm2:360M-instruct-q4_K_M"

response = requests.post(
    ENDPOINT,
    json={
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "Explain what Docker is in one sentence."}
        ],
    },
    timeout=60,
)

data = response.json()
print("Status:", response.status_code)
print("Reply :", data["choices"][0]["message"]["content"])
print("\nFull response:")
print(data)
