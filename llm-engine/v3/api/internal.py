from flask import Blueprint, current_app, jsonify, request

from .. import llm_client


internal = Blueprint("internal", __name__)
internal.url_prefix = "/internal"


ALLOWED_ROLES = {"system", "user", "assistant"}


@internal.route("/")
def index():
    return jsonify("Internal API start")


def _error(message, status_code):
    return jsonify({"error": message}), status_code


def _validate_messages(messages):
    if not isinstance(messages, list) or not messages:
        return "'messages' must be a non-empty array"

    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            return f"messages[{index}] must be an object"

        role = message.get("role")
        content = message.get("content")
        if role not in ALLOWED_ROLES:
            return (
                f"messages[{index}].role must be one of: "
                f"{', '.join(sorted(ALLOWED_ROLES))}"
            )
        if not isinstance(content, str) or not content.strip():
            return f"messages[{index}].content must be a non-empty string"

    return None


@internal.route("/chat", methods=["POST"])
def chat():
    """Generate an assistant response from a prompt or conversation history."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return _error("Request body must be a JSON object", 400)

    prompt = payload.get("prompt")
    messages = payload.get("messages")

    if prompt is not None and messages is not None:
        return _error("Provide either 'prompt' or 'messages', not both", 400)

    if messages is not None:
        validation_error = _validate_messages(messages)
        if validation_error:
            return _error(validation_error, 400)
    elif not isinstance(prompt, str) or not prompt.strip():
        return _error("'prompt' must be a non-empty string", 400)

    try:
        content = llm_client.chat(
            prompt=prompt.strip() if prompt is not None else None,
            messages=messages,
        )
    except Exception:
        # Do not leak connection details or model internals to API consumers.
        current_app.logger.exception("LLM chat request failed")
        return _error("The language model is currently unavailable", 503)

    return jsonify({
        "message": {"role": "assistant", "content": content},
        "model": llm_client.MODEL_NAME,
    })
