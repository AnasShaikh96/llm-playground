import os
from typing import Any, Dict, List, Optional, Sequence, Union

from ollama import Client


OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")
EMBED_MODEL_NAME = os.getenv("OLLAMA_EMBED_MODEL", MODEL_NAME)
NUM_CONTEXT_TOKENS = int(os.getenv("OLLAMA_NUM_CTX", "4096"))

client = Client(host=OLLAMA_HOST)


def _as_dict(response: Any) -> Dict[str, Any]:
    """Convert Ollama's typed responses into JSON-serializable dictionaries."""
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json", exclude_none=True)
    if hasattr(response, "dict"):
        return response.dict()
    return dict(response)


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


def health() -> Dict[str, Any]:
    """Verify that Ollama is reachable and the configured chat model exists."""
    response = _as_dict(client.show(MODEL_NAME))
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "capabilities": response.get("capabilities", []),
    }


def generate(prompt: str, system: Optional[str] = None) -> Dict[str, Any]:
    """Generate a completion using the configured chat model."""
    response = client.generate(
        model=MODEL_NAME,
        prompt=prompt,
        system=system,
        options={"num_ctx": NUM_CONTEXT_TOKENS},
    )
    result = _as_dict(response)
    return {
        "model": result.get("model", MODEL_NAME),
        "response": result.get("response", ""),
        "done": result.get("done", True),
    }


def embed(
    input_text: Union[str, Sequence[str]],
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate vector embeddings for one or more input strings."""
    selected_model = model or EMBED_MODEL_NAME
    response = _as_dict(client.embed(model=selected_model, input=input_text))
    return {
        "model": response.get("model", selected_model),
        "embeddings": response.get("embeddings", []),
    }


def show(model: Optional[str] = None) -> Dict[str, Any]:
    """Return Ollama metadata for a model."""
    return _as_dict(client.show(model or MODEL_NAME))
