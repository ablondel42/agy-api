# Environment & Tooling Rules

## Python Virtual Environment Isolation
- **No Global Environments**: NEVER invoke global Python interpreters, system-level binaries, or global virtual environments (e.g., paths under `/Library/...`, `/usr/...`, or system-wide `pytest`/`python`).
- **Workspace Virtual Environment Only**: ALWAYS use the virtual environment located within the active workspace (e.g., `.venv/bin/python`, `.venv/bin/<binary>`).
- **Missing Dependencies**: If a required package or tool is not present in the workspace `.venv`, do NOT fall back to global or system interpreters. Work within the local virtual environment or notify the user.
