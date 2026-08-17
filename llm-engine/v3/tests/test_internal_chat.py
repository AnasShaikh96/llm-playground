from unittest.mock import patch

from v3.factory import create_app


def _client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_chat_accepts_prompt():
    with patch("v3.api.internal.llm_client.chat", return_value="Hello!") as chat:
        response = _client().post("/internal/chat", json={"prompt": "Hi"})

    assert response.status_code == 200
    assert response.get_json()["message"] == {
        "role": "assistant",
        "content": "Hello!",
    }
    chat.assert_called_once_with(prompt="Hi", messages=None)


def test_chat_accepts_message_history():
    messages = [
        {"role": "user", "content": "My name is Sam."},
        {"role": "assistant", "content": "Hi Sam!"},
        {"role": "user", "content": "What is my name?"},
    ]

    with patch("v3.api.internal.llm_client.chat", return_value="Sam") as chat:
        response = _client().post("/internal/chat", json={"messages": messages})

    assert response.status_code == 200
    chat.assert_called_once_with(prompt=None, messages=messages)


def test_chat_rejects_invalid_request():
    response = _client().post("/internal/chat", json={"prompt": ""})

    assert response.status_code == 400
    assert "error" in response.get_json()


def test_chat_returns_service_unavailable_when_ollama_fails():
    with patch(
        "v3.api.internal.llm_client.chat",
        side_effect=ConnectionError("Ollama is offline"),
    ):
        response = _client().post("/internal/chat", json={"prompt": "Hi"})

    assert response.status_code == 503
    assert response.get_json() == {
        "error": "The language model is currently unavailable"
    }
