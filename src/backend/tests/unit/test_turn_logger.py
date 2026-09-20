"""Unit tests for backend/turn_logger.py and logging timezone resolution."""
import json
import logging
import tempfile

from logging_config import JSONFormatter
from turn_logger import get_log_datetime, is_test_environment, log_turn


class TestTurnLogger:
    """Tests for turn evaluation logging and timezone resolution."""

    def test_is_test_environment_active(self):
        """Verify is_test_environment detects running under pytest."""
        assert is_test_environment() is True

    def test_get_log_datetime_default(self):
        """Verify get_log_datetime returns datetime with timezone info."""
        dt = get_log_datetime()
        assert dt.tzinfo is not None

    def test_get_log_datetime_custom_timezone(self, monkeypatch):
        """Verify custom AGY_LOG_TIMEZONE is respected."""
        monkeypatch.setattr("turn_logger.settings.log_timezone", "America/New_York")
        dt = get_log_datetime()
        assert dt.tzinfo is not None
        assert "EDT" in dt.tzname() or "EST" in dt.tzname() or "New_York" in str(dt.tzinfo)

    def test_json_formatter_includes_tz_offset(self):
        """Verify JSONFormatter produces timestamp with timezone offset."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="Hello log",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        assert "timestamp" in output
        # ISO timestamp with timezone offset (e.g. +02:00 or -04:00 or Z)
        data = json.loads(output)
        assert ("+" in data["timestamp"]) or ("-" in data["timestamp"][10:])

    def test_log_turn_skipped_in_test_without_force(self):
        """Verify log_turn skips writing when running under pytest without force_write."""
        result = log_turn(
            conversation_id="test-conv-skip",
            turn=1,
            agent="default",
            model="Gemini 3.7 Flash",
            target_model="Gemini 3.7 Flash (High)",
            reflection="high",
            mode="non-streaming",
            prompt="Hello world",
            system_prompt="Be concise",
            response_text="Hi there!",
            force_write=False,
        )
        assert result is None

    def test_log_turn_formatting_and_append(self):
        """Verify log_turn writes structured log file with metadata and appends multi-turn."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cid = "test-conv-eval-123"

            # Turn 1
            log_path = log_turn(
                conversation_id=cid,
                turn=1,
                agent="default",
                model="Gemini 3.7 Flash",
                target_model="Gemini 3.7 Flash (High)",
                reflection="high",
                mode="streaming",
                prompt="What is Python?",
                system_prompt="You are a senior developer.",
                response_text="Python is a high-level programming language.",
                usage={"input_tokens": 100, "output_tokens": 20, "thinking_tokens": 15, "total_tokens": 135},
                duration_s=1.234,
                workspace=tmp_dir,
                force_write=True,
            )

            assert log_path is not None
            assert log_path.exists()

            # Turn 2
            log_turn(
                conversation_id=cid,
                turn=2,
                agent="default",
                model="Gemini 3.7 Flash",
                target_model="Gemini 3.7 Flash (High)",
                reflection="high",
                mode="streaming",
                prompt="Give an example of Python code.",
                system_prompt=None,
                response_text="print('Hello World')",
                usage={"input_tokens": 150, "output_tokens": 10, "total_tokens": 160},
                duration_s=0.850,
                workspace=tmp_dir,
                force_write=True,
            )

            content = log_path.read_text(encoding="utf-8")

            # Assert Turn 1 content
            assert f"Conversation ID: {cid} | Turn: 1" in content
            assert "Agent: default | Model: Gemini 3.7 Flash | Level: high (Target: Gemini 3.7 Flash (High))" in content
            assert "Mode: streaming" in content
            assert "Duration: 1.234s" in content
            assert "Tokens: 100 input, 20 output, 15 thinking (135 total)" in content
            assert "[SYSTEM INSTRUCTIONS]\nYou are a senior developer." in content
            assert "[USER PROMPT]\nWhat is Python?" in content
            assert "[RESPONSE]\nPython is a high-level programming language." in content

            # Assert Turn 2 content
            assert f"Conversation ID: {cid} | Turn: 2" in content
            assert "Duration: 0.850s" in content
            assert "[SYSTEM INSTRUCTIONS]\n(None)" in content
            assert "[USER PROMPT]\nGive an example of Python code." in content
            assert "[RESPONSE]\nprint('Hello World')" in content

    def test_collect_turn_transcript_and_extract_error(self, tmp_path, monkeypatch):
        """Verify collect_turn_transcript and extract_error_from_transcript."""
        from turn_logger import collect_turn_transcript, extract_error_from_transcript

        cid = "test-conv-trans-123"
        log_dir = tmp_path / "brain" / cid / ".system_generated" / "logs"
        log_dir.mkdir(parents=True)
        transcript_file = log_dir / "transcript_full.jsonl"

        step1 = {"step_index": 1, "source": "USER", "type": "USER_INPUT", "content": "what's my name?"}
        step2 = {
            "step_index": 2,
            "source": "MODEL",
            "type": "PLANNER_RESPONSE",
            "status": "ERROR",
            "content": "permission check failed for command 'git config user.name': user denied permission",
        }
        transcript_file.write_text(f"{json.dumps(step1)}\n{json.dumps(step2)}\n", encoding="utf-8")

        monkeypatch.setattr("turn_logger.settings.agy_app_data_dir", str(tmp_path))

        steps, raw, total_lines = collect_turn_transcript(cid, since_line=0)
        assert total_lines == 2
        assert len(steps) == 2
        assert steps[0]["content"] == "what's my name?"
        assert steps[1]["status"] == "ERROR"
        assert "permission check failed" in raw

        error = extract_error_from_transcript(steps)
        assert error is not None
        assert "user denied permission" in error

    def test_collect_turn_transcript_since_line(self, tmp_path, monkeypatch):
        """Verify collect_turn_transcript respects since_line offset."""
        from turn_logger import collect_turn_transcript

        cid = "test-conv-trans-offset"
        log_dir = tmp_path / "brain" / cid / ".system_generated" / "logs"
        log_dir.mkdir(parents=True)
        transcript_file = log_dir / "transcript_full.jsonl"

        step1 = {"step_index": 1, "content": "turn 1"}
        step2 = {"step_index": 2, "content": "turn 2"}
        transcript_file.write_text(f"{json.dumps(step1)}\n{json.dumps(step2)}\n", encoding="utf-8")

        monkeypatch.setattr("turn_logger.settings.agy_app_data_dir", str(tmp_path))

        steps, raw, total_lines = collect_turn_transcript(cid, since_line=1)
        assert total_lines == 2
        assert len(steps) == 1
        assert steps[0]["content"] == "turn 2"

    def test_log_turn_with_transcript_raw(self, tmp_path):
        """Verify log_turn includes [TRANSCRIPT LOG] when transcript_raw is provided."""
        cid = "test-conv-trans-log"
        log_path = log_turn(
            conversation_id=cid,
            turn=1,
            agent="default",
            model="Gemini 3.8 Flash",
            target_model="Gemini 3.8 Flash",
            reflection="high",
            mode="non-streaming",
            prompt="what's my name?",
            response_text="Error: user denied permission",
            transcript_raw='{"step_index": 1, "status": "ERROR"}\n',
            workspace=str(tmp_path),
            force_write=True,
        )
        assert log_path is not None
        content = log_path.read_text(encoding="utf-8")
        assert "[TRANSCRIPT LOG]" in content
        assert '{"step_index": 1, "status": "ERROR"}' in content
