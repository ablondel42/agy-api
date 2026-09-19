"""Tests for agy process command building and execution."""
import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from config import settings
from safe_runner import ExecutionResult

from agy_process import (
    AgyProcess,
    AgyProcessError,
    _build_command,
    run_agy,
    stream_agy,
)


class TestBuildCommand:
    """Tests for the _build_command helper (backwards compatibility)."""

    def test_basic_command(self):
        """Verify basic command structure."""
        cmd = _build_command(
            prompt="Hello",
            agent="default",
            output_format="json",
        )
        assert cmd[0] == "agy"  # binary name from settings
        assert "--print" in cmd
        assert "Hello" in cmd
        assert "--agent" in cmd
        assert "default" in cmd
        assert "--output-format" in cmd
        assert "json" in cmd
        assert "--add-dir" in cmd

    def test_command_with_json_schema(self):
        """Verify --json-schema flag is added when schema is provided."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        cmd = _build_command(
            prompt="Extract",
            agent="default",
            json_schema=schema,
        )
        assert "--json-schema" in cmd
        schema_idx = cmd.index("--json-schema")
        assert json.loads(cmd[schema_idx + 1]) == schema

    def test_command_with_conversation_id(self):
        """Verify --conversation flag is added when conversation_id is provided."""
        cmd = _build_command(
            prompt="Continue",
            agent="default",
            conversation_id="abc-123",
        )
        assert "--conversation" in cmd
        conv_idx = cmd.index("--conversation")
        assert cmd[conv_idx + 1] == "abc-123"

    def test_command_without_optional_flags(self):
        """Verify optional flags are omitted when not provided."""
        cmd = _build_command(
            prompt="Hello",
            agent="default",
        )
        assert "--json-schema" not in cmd
        assert "--conversation" not in cmd

    def test_command_with_stream_json_format(self):
        """Verify stream-json output format."""
        cmd = _build_command(
            prompt="Hello",
            agent="test",
            output_format="stream-json",
        )
        assert "stream-json" in cmd

    def test_no_dangerously_skip_permissions(self):
        """Verify --dangerously-skip-permissions is NEVER included."""
        cmd = _build_command(
            prompt="Hello",
            agent="default",
            extra_flags=["--dangerously-skip-permissions", "--verbose"],
        )
        assert "--dangerously-skip-permissions" not in cmd
        assert "--verbose" in cmd

    def test_command_with_configuration_parameters(self):
        """Verify mode, sandbox, project, and effort are properly passed."""
        cmd = _build_command(
            prompt="Plan task",
            agent="default",
            mode="plan",
            sandbox=True,
            project="test-proj",
            effort="high",
        )
        assert "--mode" in cmd
        assert cmd[cmd.index("--mode") + 1] == "plan"
        assert "--sandbox" in cmd
        assert "--project" in cmd
        assert cmd[cmd.index("--project") + 1] == "test-proj"
        assert "--effort" in cmd
        assert cmd[cmd.index("--effort") + 1] == "high"


class TestAgyProcessClass:
    """Tests for the AgyProcess class attributes and methods."""

    def test_init_defaults(self):
        """Verify default attributes when initialized with prompt."""
        proc = AgyProcess(prompt="Test prompt")
        assert proc.prompt == "Test prompt"
        assert proc.agent == settings.agy_default_agent
        assert proc.workspace == settings.agy_default_workspace
        assert proc.timeout == settings.agy_default_timeout
        assert proc.model == settings.agy_default_model
        assert proc.cwd == Path(settings.agy_default_workspace).expanduser().resolve()
        assert proc.sandbox is False
        assert proc.json_schema is None
        assert proc.conversation_id is None
        assert proc.mode is None
        assert proc.effort is None
        assert proc.project is None
        assert proc.extra_flags == []
        assert proc.env is None

    def test_init_custom_attributes(self, tmp_path):
        """Verify custom attributes are assigned accurately."""
        schema = {"type": "object"}
        flags = ["--flag1", "--flag2"]
        env = {"KEY": "VALUE"}

        proc = AgyProcess(
            prompt="Custom prompt",
            agent="special-agent",
            json_schema=schema,
            conversation_id="conv-456",
            workspace=str(tmp_path),
            timeout=120,
            model="custom-model",
            mode="plan",
            sandbox=True,
            effort="low",
            project="proj-99",
            extra_flags=flags,
            env=env,
        )

        assert proc.prompt == "Custom prompt"
        assert proc.agent == "special-agent"
        assert proc.json_schema == schema
        assert proc.conversation_id == "conv-456"
        assert proc.workspace == str(tmp_path)
        assert proc.timeout == 120
        assert proc.model == "custom-model"
        assert proc.mode == "plan"
        assert proc.sandbox is True
        assert proc.effort == "low"
        assert proc.project == "proj-99"
        assert proc.extra_flags == flags
        assert proc.env == env
        assert proc.cwd == tmp_path.resolve()

    def test_build_command_strips_dangerous_flags(self):
        """Verify build_command strips --dangerously-skip-permissions."""
        proc = AgyProcess(
            prompt="Hello",
            agent="default",
            extra_flags=["--dangerously-skip-permissions", "--keep-this"],
        )
        cmd = proc.build_command(output_format="json")
        assert "--dangerously-skip-permissions" not in cmd
        assert "--keep-this" in cmd

    @pytest.mark.asyncio
    async def test_run_success(self):
        """Verify successful non-streaming execution returns parsed json."""
        proc = AgyProcess(prompt="What is 2+2?", agent="default")
        mock_output = json.dumps({"status": "SUCCESS", "response": "4"})
        mock_result = ExecutionResult(
            stdout=mock_output,
            stderr="",
            returncode=0,
            duration_ms=150,
        )

        with patch("agy_process.safe_run_command", new_callable=AsyncMock, return_value=mock_result) as mock_safe_run:
            result = await proc.run()
            assert result == {"status": "SUCCESS", "response": "4"}
            mock_safe_run.assert_called_once()
            call_kwargs = mock_safe_run.call_args[1]
            assert call_kwargs["cwd"] == proc.cwd
            assert call_kwargs["timeout"] == proc.timeout

    @pytest.mark.asyncio
    async def test_run_nonzero_exit_raises(self):
        """Verify non-zero returncode raises AgyProcessError."""
        proc = AgyProcess(prompt="Fail command", agent="default")
        mock_result = ExecutionResult(
            stdout="",
            stderr="Fatal error occurred",
            returncode=1,
            duration_ms=100,
        )

        with patch("agy_process.safe_run_command", new_callable=AsyncMock, return_value=mock_result):
            with pytest.raises(AgyProcessError) as exc_info:
                await proc.run()
            assert exc_info.value.returncode == 1
            assert "Fatal error occurred" in exc_info.value.stderr

    @pytest.mark.asyncio
    async def test_run_error_status_in_json_raises(self):
        """Verify status: ERROR in json output raises AgyProcessError."""
        proc = AgyProcess(prompt="Bad command", agent="default")
        mock_output = json.dumps({"status": "ERROR", "error": "Internal agent error"})
        mock_result = ExecutionResult(
            stdout=mock_output,
            stderr="",
            returncode=0,
            duration_ms=100,
        )

        with patch("agy_process.safe_run_command", new_callable=AsyncMock, return_value=mock_result):
            with pytest.raises(AgyProcessError) as exc_info:
                await proc.run()
            assert "Internal agent error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_run_empty_stdout_raises(self):
        """Verify empty stdout raises AgyProcessError."""
        proc = AgyProcess(prompt="Empty output", agent="default")
        mock_result = ExecutionResult(
            stdout="   ",
            stderr="",
            returncode=0,
            duration_ms=50,
        )

        with patch("agy_process.safe_run_command", new_callable=AsyncMock, return_value=mock_result):
            with pytest.raises(AgyProcessError) as exc_info:
                await proc.run()
            assert "Empty or invalid stdout output" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_stream_success(self):
        """Verify successful streaming yields parsed event dicts."""
        proc = AgyProcess(prompt="Stream test", agent="default")

        lines = [
            b'{"event": "init", "conversation_id": "c1"}\n',
            b'not a json line\n',
            b'{"event": "result", "text": "hello"}\n',
        ]

        async def line_gen():
            for line in lines:
                yield line

        mock_proc = MagicMock()
        mock_proc.stdout = line_gen()
        mock_proc.stderr = MagicMock()
        mock_proc.wait = AsyncMock(return_value=0)
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_proc):
            events = []
            async for evt in proc.stream():
                events.append(evt)

            assert len(events) == 2
            assert events[0] == {"event": "init", "conversation_id": "c1"}
            assert events[1] == {"event": "result", "text": "hello"}

    @pytest.mark.asyncio
    async def test_stream_nonzero_exit_raises(self):
        """Verify non-zero return code after stream raises AgyProcessError."""
        proc = AgyProcess(prompt="Stream fail", agent="default")

        async def line_gen():
            yield b'{"event": "init"}\n'

        mock_proc = MagicMock()
        mock_proc.stdout = line_gen()
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read = AsyncMock(return_value=b"Process crashed")
        mock_proc.wait = AsyncMock(return_value=1)
        mock_proc.returncode = 1

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_proc):
            with pytest.raises(AgyProcessError) as exc_info:
                async for _ in proc.stream():
                    pass
            assert exc_info.value.returncode == 1
            assert "Process crashed" in exc_info.value.stderr

    @pytest.mark.asyncio
    async def test_run_agy_wrapper_delegation(self):
        """Verify run_agy delegates to AgyProcess.run."""
        with patch.object(AgyProcess, "run", new_callable=AsyncMock, return_value={"ok": True}) as mock_run:
            res = await run_agy(prompt="Hello", agent="default")
            assert res == {"ok": True}
            mock_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_stream_agy_wrapper_delegation(self):
        """Verify stream_agy delegates to AgyProcess.stream."""
        async def mock_stream_gen(self):
            yield {"msg": "streamed"}

        with patch.object(AgyProcess, "stream", mock_stream_gen):
            results = []
            async for item in stream_agy(prompt="Hello", agent="default"):
                results.append(item)
            assert results == [{"msg": "streamed"}]
