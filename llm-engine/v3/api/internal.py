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


@internal.route("/health")
def health():
    """Report whether Ollama and the configured model are available."""
    try:
        return jsonify(llm_client.health())
    except Exception:
        current_app.logger.exception("LLM health check failed")
        return jsonify({
            "status": "unhealthy",
            "model": llm_client.MODEL_NAME,
            "error": "The language model is currently unavailable",
        }), 503


@internal.route("/generate", methods=["POST"])
def generate():
    """Generate a completion from a prompt."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return _error("Request body must be a JSON object", 400)

    prompt = payload.get("prompt")
    system = payload.get("system")
    if not isinstance(prompt, str) or not prompt.strip():
        return _error("'prompt' must be a non-empty string", 400)
    if system is not None and (not isinstance(system, str) or not system.strip()):
        return _error("'system' must be a non-empty string when provided", 400)

    try:
        result = llm_client.generate(
            prompt=prompt.strip(),
            system=system.strip() if system is not None else None,
        )
    except Exception:
        current_app.logger.exception("LLM generation request failed")
        return _error("The language model is currently unavailable", 503)

    return jsonify(result)


@internal.route("/embed", methods=["POST"])
def embed():
    """Generate embeddings for a string or an array of strings."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return _error("Request body must be a JSON object", 400)

    input_text = payload.get("input")
    model = payload.get("model")
    valid_input = (
        isinstance(input_text, str) and bool(input_text.strip())
    ) or (
        isinstance(input_text, list)
        and bool(input_text)
        and all(isinstance(item, str) and item.strip() for item in input_text)
    )
    if not valid_input:
        return _error(
            "'input' must be a non-empty string or an array of non-empty strings",
            400,
        )
    if model is not None and (not isinstance(model, str) or not model.strip()):
        return _error("'model' must be a non-empty string when provided", 400)

    normalized_input = (
        input_text.strip()
        if isinstance(input_text, str)
        else [item.strip() for item in input_text]
    )
    try:
        result = llm_client.embed(
            input_text=normalized_input,
            model=model.strip() if model is not None else None,
        )
    except Exception:
        current_app.logger.exception("LLM embedding request failed")
        return _error("The embedding model is currently unavailable", 503)

    return jsonify(result)


@internal.route("/show")
def show():
    """Return metadata for the requested or configured model."""
    model = request.args.get("model")
    if model is not None and not model.strip():
        return _error("'model' must be a non-empty string when provided", 400)

    try:
        return jsonify(llm_client.show(model=model.strip() if model else None))
    except Exception:
        current_app.logger.exception("LLM model details request failed")
        return _error("The requested model is currently unavailable", 503)


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
