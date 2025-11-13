import os

import requests


def main():
    LLM_API_URI = os.getenv("LLM_API_URI")
    url_api_chat = "api/chat"

    payload = {
        "model": "mistral",
        "messages": [{"role": "user", "content": "Hello there!"}],
    }

    response = requests.post(f"{LLM_API_URI}/{url_api_chat}", json=payload, stream=True)

    if response.status_code != 200:
        print(f"status_code: {response.status_code}")
        return

    print(response.text)


if __name__ == "__main__":
    main()
