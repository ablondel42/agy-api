"""Turn logger for recording complete prompts, responses, and evaluation metadata."""
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from config import settings

logger = logging.getLogger(__name__)


def is_test_environment() -> bool:
    """Check if the current process is executing within a test suite."""
    if os.environ.get("AGY_DISABLE_FILE_LOGGING") == "1":
        return True
    if os.environ.get("PYTEST_CURRENT_TEST") is not None:
        return True
    if "pytest" in sys.modules:
        return True
    return False


def get_log_datetime() -> datetime:
    """Return the current datetime in the configured or local system timezone."""
    if settings.log_timezone:
        try:
            return datetime.now(ZoneInfo(settings.log_timezone))
        except Exception as e:
            logger.debug(f"Invalid or unsupported log timezone '{settings.log_timezone}': {e}")
    return datetime.now().astimezone()


def get_transcript_path(conversation_id: str | None) -> Path | None:
    """Resolve the path to the agy internal transcript file for a conversation."""
    if not conversation_id:
        return None

    base_dir = (
        Path(settings.agy_app_data_dir)
        if getattr(settings, "agy_app_data_dir", None)
        else (Path.home() / ".gemini" / "antigravity-cli")
    )
    transcript_full = (
        base_dir
        / "brain"
        / conversation_id
        / ".system_generated"
        / "logs"
        / "transcript_full.jsonl"
    )
    if transcript_full.exists():
        return transcript_full

    fallback = transcript_full.parent / "transcript.jsonl"
    if fallback.exists():
        return fallback

    return transcript_full


def collect_turn_transcript(
    conversation_id: str | None,
    since_line: int = 0,
) -> tuple[list[dict[str, Any]], str, int]:
    """Collect the raw transcript entries for a turn directly as logged by agy.

    Reads every step (prompts, tool calls, arguments, outputs, errors, statuses)
    without dropping or cherry-picking fields.

    Args:
        conversation_id: Conversation UUID.
        since_line: Line index to start reading from.

    Returns:
        Tuple of (parsed_entries_list, raw_jsonl_text, new_line_count).
    """
    path = get_transcript_path(conversation_id)
    if not path or not path.exists():
        return [], "", since_line

    entries: list[dict[str, Any]] = []
    raw_lines: list[str] = []
    total_lines = 0

    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                total_lines += 1
                if i < since_line or not line.strip():
                    continue
                raw_lines.append(line.rstrip("\r\n"))
                try:
                    entries.append(json.loads(line))
                except Exception:
                    pass
    except Exception as e:
        logger.debug(f"Error reading transcript for {conversation_id}: {e}")

    raw_text = "\n".join(raw_lines)
    return entries, raw_text, total_lines


def extract_error_from_transcript(entries: list[dict[str, Any]]) -> str | None:
    """Extract error message from transcript steps if any error occurred."""
    for entry in reversed(entries):
        if entry.get("status") == "ERROR" or entry.get("error"):
            err = entry.get("error") or entry.get("content")
            if err and isinstance(err, str) and err.strip():
                return err.strip()
    return None


def extract_turn_thinking(conversation_id: str | None, since_line: int = 0) -> str:
    """Extract LLM thinking/reflection text from the Antigravity transcript for the turn.

    Args:
        conversation_id: Conversation UUID.
        since_line: Line index to start reading from.

    Returns:
        Concatenated thinking/reflection string.
    """
    entries, _, _ = collect_turn_transcript(conversation_id, since_line=since_line)
    thinking_parts: list[str] = []
    for step in entries:
        t = step.get("thinking")
        if t and isinstance(t, str) and t.strip():
            thinking_parts.append(t.strip())
    return "\n\n".join(thinking_parts)


def log_turn(
    conversation_id: str,
    turn: int,
    agent: str,
    model: str,
    target_model: str,
    reflection: str,
    mode: str,
    prompt: str,
    system_prompt: str | None = None,
    response_text: str = "",
    thinking: str | None = None,
    usage: dict[str, Any] | None = None,
    duration_s: float | None = None,
    workspace: str | None = None,
    force_write: bool = False,
    transcript_raw: str | None = None,
    transcript_entries: list[dict[str, Any]] | None = None,
) -> Path | None:
    """Log a complete turn (prompt + response + metadata + full raw transcript) to the conversation log file.

    Args:
        conversation_id: Unique conversation UUID.
        turn: Current turn index (1-based).
        agent: Agent persona used.
        model: Clean model name.
        target_model: Target model argument passed to agy.
        reflection: Reflection/effort level (low, medium, high).
        mode: Execution mode ('non-streaming', 'streaming', 'interactive').
        prompt: The user prompt text.
        system_prompt: Optional system prompt text.
        response_text: The complete generated response text.
        thinking: Optional extracted thinking/reflection text.
        usage: Optional token usage dict.
        duration_s: Optional duration in seconds.
        workspace: Optional workspace directory path.
        force_write: If True, writes even in test environments.
        transcript_raw: Full raw unformatted JSONL transcript text for the turn.
        transcript_entries: Optional list of parsed step dicts from the transcript.

    Returns:
        Path to written log file, or None if skipped.
    """
    if not force_write and is_test_environment():
        return None

    try:
        ws_path = Path(workspace or ".").expanduser().resolve()
        log_dir = ws_path / settings.agy_log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{conversation_id}.log"

        now = get_log_datetime()
        tz_name = now.strftime("%Z") or now.strftime("%z")
        timestamp = now.strftime(f"%Y-%m-%d %H:%M:%S {tz_name}").strip()

        tokens_line = ""
        if usage:
            in_tok = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
            out_tok = usage.get("completion_tokens") or usage.get("output_tokens") or 0
            think_tok = usage.get("thinking_tokens") or 0
            tot_tok = usage.get("total_tokens") or (in_tok + out_tok)
            tokens_line = f"Tokens: {in_tok} input, {out_tok} output, {think_tok} thinking ({tot_tok} total)\n"

        dur_line = f"Duration: {duration_s:.3f}s\n" if duration_s is not None else ""

        transcript_section = ""
        if transcript_raw and transcript_raw.strip():
            transcript_section = f"{'-' * 80}\n[TRANSCRIPT LOG]\n{transcript_raw.strip()}\n\n"
        elif thinking and thinking.strip():
            transcript_section = f"{'-' * 80}\n[REFLECTION / THINKING]\n{thinking.strip()}\n\n"

        entry = (
            f"{'=' * 80}\n"
            f"[{timestamp}] Conversation ID: {conversation_id} | Turn: {turn}\n"
            f"Agent: {agent} | Model: {model} | Level: {reflection} (Target: {target_model})\n"
            f"Mode: {mode}\n"
            f"{dur_line}"
            f"{tokens_line}"
            f"{'-' * 80}\n"
            f"[SYSTEM INSTRUCTIONS]\n{system_prompt or '(None)'}\n\n"
            f"[USER PROMPT]\n{prompt}\n\n"
            f"{transcript_section}"
            f"{'-' * 80}\n"
            f"[RESPONSE]\n{response_text}\n"
            f"{'=' * 80}\n\n"
        )

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(entry)

        return log_file
    except Exception as e:
        logger.warning(f"Failed to write turn evaluation log for {conversation_id}: {e}")
        return None
