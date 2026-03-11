"""
vLLM OpenAI-Compatible API Client
Lists all available models and runs a chat completion against the first one.
Prints the raw JSON request and response for full visibility.
"""

import json
import os
import sys
import httpx
from openai import OpenAI

# ─── Endpoint config ──────────────────────────────────────────────────────────
BASE_URL    = os.environ.get("VLLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
API_KEY     = os.environ.get("VLLM_API_KEY", "token-abc123")
TEMPERATURE = 0.7
MAX_TOKENS  = 65535
PROMPT      = "Explain the difference between vLLM and Ollama in 2 sentences."
# ──────────────────────────────────────────────────────────────────────────────


def print_json(label: str, data: dict) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")
    print(json.dumps(data, indent=2))


class LoggingTransport(httpx.HTTPTransport):
    """httpx transport that prints raw JSON request and response."""

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        # ── Print request ──────────────────────────────────────────
        try:
            req_body = json.loads(request.content)
        except Exception:
            req_body = request.content.decode(errors="replace")

        print_json("REQUEST  →", {
            "method":  request.method,
            "url":     str(request.url),
            "headers": dict(request.headers),
            "body":    req_body,
        })

        # ── Send and capture response ──────────────────────────────
        response = super().handle_request(request)
        response.read()  # buffer so we can inspect the body

        try:
            res_body = response.json()
        except Exception:
            res_body = response.text

        print_json("RESPONSE ←", {
            "status":  response.status_code,
            "headers": dict(response.headers),
            "body":    res_body,
        })

        return response


client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    http_client=httpx.Client(transport=LoggingTransport()),
)


def list_models() -> list[str]:
    print("=" * 60)
    print("  Available models on vLLM server")
    print("=" * 60)

    models = client.models.list()
    ids = [m.id for m in models.data]

    if not ids:
        print("  (no models found – is vLLM running?)")
    else:
        print()
        for i, model_id in enumerate(ids, 1):
            print(f"  [{i}] {model_id}")
    print()
    return ids


def chat_completion(model_id: str) -> str:
    print("=" * 60)
    print(f"  Running chat completion")
    print(f"  Model  : {model_id}")
    print(f"  Prompt : {PROMPT}")
    print("=" * 60)

    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": PROMPT}],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    reply = response.choices[0].message.content
    print(f"\n{'=' * 60}")
    print("  Assistant reply")
    print(f"{'=' * 60}")
    print(reply)

    usage = response.usage
    if usage:
        print(
            f"\n  Tokens — prompt: {usage.prompt_tokens}, "
            f"completion: {usage.completion_tokens}, "
            f"total: {usage.total_tokens}"
        )
    return reply


def main() -> None:
    print(f"[INFO] Connecting to {BASE_URL}\n")

    try:
        model_ids = list_models()
    except Exception as e:
        print(f"[ERROR] Could not reach vLLM server at {BASE_URL}\n  {e}")
        sys.exit(1)

    if not model_ids:
        sys.exit(0)

    target = sys.argv[1] if len(sys.argv) > 1 else model_ids[0]

    if target not in model_ids:
        print(f"[WARN] '{target}' not found; using '{model_ids[0]}'.")
        target = model_ids[0]

    chat_completion(target)


if __name__ == "__main__":
    main()