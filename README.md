# agy-api

An OpenAI-compatible API server powered by the Google Antigravity (`agy`) CLI.

`agy-api` acts as a drop-in gateway for any OpenAI-compatible client, tool, or UI (e.g. Chatbot web UIs, IDE extensions, or custom agent frameworks) to interact with Antigravity models and custom agent personas.

---

## Features

- **OpenAI Compatibility**: Provides standard `/v1/chat/completions` and `/v1/models` endpoints compatible with official OpenAI client SDKs.
- **Execution Modes**:
  - **Non-Streaming**: Direct batch completions returning OpenAI-standard JSON responses.
  - **Streaming**: Real-time response streaming using Server-Sent Events (SSE).
  - **Interactive (PTY)**: Multi-turn, stateful sessions managed via pseudo-terminals (`pexpect`) for conversational workflows.
- **Dynamic Model Discovery**: Queries `agy models` dynamically with in-memory caching and maps model reasoning variants (e.g., `(High)`, `(Medium)`, `(Low)`).
- **Agent Personas**: Discovers and selects custom agent personas defined in `.agents/agents/` via `GET /v1/agents`.
- **Structured Outputs**: Native support for JSON Schema enforcement (`response_format={"type": "json_schema", ...}`).
- **Session Management**: Session pooling with idle timeout reclamation and concurrency safeguards.
- **Logging & Tracing**: Structured JSON/text logging, configurable log levels, and turn-by-turn transcript extraction in `log/`.
- **Interactive Test CLI**: Built-in terminal CLI (`scripts/test_cli.py`) for testing and debugging endpoints.

---

## Project Structure

```
agy-api/
├── backend/
│   ├── main.py               # FastAPI application factory and ASGI entry point
│   ├── config.py             # Configuration and environment variable settings
│   ├── types.py              # Pydantic schemas (OpenAI-compatible request/response models)
│   ├── session.py            # Session abstractions and state tracking
│   ├── session_manager.py    # Session pool lifecycle and eviction
│   ├── agy_process.py        # Subprocess execution for non-interactive / streaming commands
│   ├── agy_interactive.py    # PTY-driven interactive session handling via pexpect
│   ├── safe_runner.py        # Subprocess execution utility with timeout handling
│   ├── ansi_utils.py         # ANSI escape code sanitization
│   ├── logging_config.py     # Logging setup (supports DEV/DEBUG/INFO, text/json formats)
│   ├── turn_logger.py        # Turn logger and thinking transcript extractor
│   └── routes/
│       ├── chat.py           # POST /v1/chat/completions
│       ├── models.py         # GET /v1/models
│       └── agents.py         # GET /v1/agents
├── scripts/
│   └── test_cli.py           # Interactive terminal CLI to test the API
├── tests/
│   ├── unit/                 # Unit tests (session, runners, types, utils)
│   └── integration/          # Integration tests for API routes
├── .agents/agents/           # Custom agent persona definitions
├── pyproject.toml            # Project packaging and tool configurations
└── requirements.txt          # Python dependencies
```

---

## Prerequisites

- **Python**: 3.12 or later
- **Antigravity CLI**: `agy` command installed and available in your `PATH` (or configured via `AGY_BINARY`).

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ablondel42/agy-api.git
   cd agy-api
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   # Base dependencies:
   pip install -r requirements.txt

   # Or in editable mode with development & test tools:
   pip install -e ".[dev]"
   ```

---

## Configuration

Settings are managed via environment variables (prefixed with `AGY_`) or a `.env` file at the root of the project:

| Variable | Default | Description |
|---|---|---|
| `AGY_HOST` | `127.0.0.1` | API server listen host |
| `AGY_PORT` | `8000` | API server listen port |
| `AGY_BINARY` | `agy` | Path or name of the `agy` CLI binary |
| `AGY_DEFAULT_MODEL` | `Gemini 3.8 Flash` | Default foundational LLM model |
| `AGY_DEFAULT_AGENT` | `default` | Default agent persona |
| `AGY_DEFAULT_REFLECTION` | `high` | Reflection level (`low`, `medium`, `high`) |
| `AGY_DEFAULT_TIMEOUT` | `120` | Execution timeout (seconds) for non-interactive commands |
| `AGY_INTERACTIVE_TIMEOUT`| `180` | PTY interaction timeout (seconds) |
| `AGY_DEFAULT_WORKSPACE` | `.` | Target workspace path (maps to `--add-dir`) |
| `AGY_AGENTS_DIR` | `.agents/agents` | Directory containing agent definition files |
| `AGY_MAX_SESSIONS` | `10` | Maximum active concurrent sessions |
| `AGY_SESSION_IDLE_TIMEOUT` | `600` | Idle session expiration timeout (seconds) |
| `AGY_LOG_LEVEL` | `INFO` | Logging level (`DEV`, `DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `AGY_LOG_FORMAT` | `json` | Log format (`json` or `text`) |

Example `.env` file:
```env
AGY_HOST=127.0.0.1
AGY_PORT=8000
AGY_LOG_LEVEL=INFO
AGY_LOG_FORMAT=text
```

---

## How to Run

### 1. Start the API Server

You can run the server directly with Python or through `uvicorn`:

```bash
# Direct run:
python -m backend.main

# Or with uvicorn (with auto-reload):
uvicorn backend.main:app --reload --port 8000
```

Once running, verify the server is healthy:
```bash
curl http://localhost:8000/
# Output: {"message": "agy-api", "version": "0.1.0"}
```

API documentation is automatically available at:
- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

### 2. Run the Interactive Test CLI

A dedicated interactive CLI is provided in `scripts/test_cli.py` to test and inspect endpoints:

```bash
python scripts/test_cli.py
```

Inside the CLI, you can type commands like:
- `models`: List discovered models
- `agents`: List agent personas
- `chat <message>`: Send a non-streaming message
- `stream <message>`: Stream a message chunk-by-chunk
- `multi`: Start an interactive multi-turn conversation
- `help`: View all available commands

---

### 3. Usage Examples

#### Non-Streaming Completion (cURL)
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Gemini 3.8 Flash",
    "messages": [
      {"role": "user", "content": "Explain async/await in Python in two sentences."}
    ],
    "stream": false
  }'
```

#### Streaming Completion (cURL)
```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Gemini 3.8 Flash",
    "messages": [
      {"role": "user", "content": "Count from 1 to 5 slowly."}
    ],
    "stream": true
  }'
```

#### Using with OpenAI Python Client
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",  # agy-api runs locally
)

response = client.chat.completions.create(
    model="Gemini 3.8 Flash",
    messages=[
        {"role": "system", "content": "You are a concise engineering assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
)

print(response.choices[0].message.content)
```

---

## API Reference Summary

- `GET /`: Health check and version verification.
- `GET /v1/models`: Discovers available models supported by the underlying `agy` CLI.
- `GET /v1/agents`: Lists agent personas found in `.agents/agents/`.
- `POST /v1/chat/completions`: Generates completions with OpenAI compatibility. Supported additional fields:
  - `agent`: Name of the custom agent profile to run.
  - `reflection` or `reasoning_effort`: Thinking effort level (`low`, `medium`, `high`).
  - `conversation_id`: Session or conversation ID for multi-turn persistence.
  - `workspace`: Working directory passed to `agy` via `--add-dir`.
  - `interactive`: Boolean flag to run persistent background PTY sessions.

---

## Development & Testing

Run unit and integration tests using `pytest`:

```bash
# Run all tests:
pytest

# Run unit tests only:
pytest tests/unit

# Run with coverage:
pytest --cov=backend
```

Run code formatting and linting:
```bash
ruff check .
```
