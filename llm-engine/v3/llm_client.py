import os
from typing import Dict, List, Optional

from ollama import Client


OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")
NUM_CONTEXT_TOKENS = int(os.getenv("OLLAMA_NUM_CTX", "4096"))

client = Client(host=OLLAMA_HOST)


def chat(
    prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Send a prompt or an existing conversation to the configured Ollama model."""
    if messages is None:
        if prompt is None:
            raise ValueError("A prompt or messages are required")
        messages = [{"role": "user", "content": prompt}]

    response = client.chat(
        model=MODEL_NAME,
        messages=messages,
        options={"num_ctx": NUM_CONTEXT_TOKENS},
    )

    # Ollama response objects expose attributes in current versions, while older
    # versions return dictionaries. Supporting both keeps this client portable.
    try:
        return response.message.content
    except AttributeError:
        return response["message"]["content"]
