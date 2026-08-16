from ollama import Client

OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "qwen3.5:4b"

client = Client(host=OLLAMA_HOST)


def chat(prompt: str) -> str:
    response = client.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        options={
            "num_ctx": 4096,
        },
    )

    return response.message.content