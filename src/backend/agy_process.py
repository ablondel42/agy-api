"""Low-level agy CLI subprocess execution for both JSON and stream-json output."""
import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from config import settings
from safe_runner import safe_run_command

logger = logging.getLogger(__name__)


class AgyProcessError(Exception):
    """Raised when agy subprocess fails."""

    def __init__(self, message: str, returncode: int, stderr: str):
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr


class AgyProcess:
    """Encapsulates agy CLI subprocess execution (both batch JSON and NDJSON streaming)."""

    def __init__(
        self,
        prompt: str,
        agent: str | None = None,
        json_schema: dict[str, Any] | None = None,
        conversation_id: str | None = None,
        workspace: str | None = None,
        timeout: int | None = None,
        model: str | None = None,
        mode: str | None = None,
        sandbox: bool = False,
        effort: str | None = None,
        project: str | None = None,
        extra_flags: list[str] | None = None,
        env: dict[str, str] | None = None,
    ):
        self.prompt = prompt
        self.agent = agent or settings.agy_default_agent
        self.json_schema = json_schema
        self.conversation_id = conversation_id
        self.workspace = workspace or settings.agy_default_workspace
        self.timeout = timeout or settings.agy_default_timeout
        self.model = model or settings.agy_default_model
        self.mode = mode
        self.sandbox = sandbox
        self.effort = effort
        self.project = project
        self.extra_flags = list(extra_flags) if extra_flags else []
        self.env = env

    @property
    def cwd(self) -> Path:
        """Resolved workspace directory path."""
        return Path(self.workspace).expanduser().resolve()

    def build_command(self, output_format: str = "json") -> list[str]:
        """Build the agy CLI command list with configuration options.

        Args:
            output_format: Output format ('json' or 'stream-json').

        Returns:
            List of command arguments for subprocess execution.
        """
        cmd = [
            settings.agy_binary,
            "--print", self.prompt,
            "--agent", self.agent,
            "--output-format", output_format,
        ]

        if self.model:
            cmd.extend(["--model", self.model])

        cmd.extend(["--add-dir", str(self.cwd)])

        if self.json_schema:
            cmd.extend(["--json-schema", json.dumps(self.json_schema)])
        if self.conversation_id:
            cmd.extend(["--conversation", self.conversation_id])
        if self.mode:
            cmd.extend(["--mode", self.mode])
        if self.sandbox:
            cmd.append("--sandbox")
        if self.effort:
            cmd.extend(["--effort", self.effort])
        if self.project:
            cmd.extend(["--project", self.project])
        if self.extra_flags:
            safe_flags = [f for f in self.extra_flags if f != "--dangerously-skip-permissions"]
            cmd.extend(safe_flags)

        return cmd

    async def run(self) -> dict[str, Any]:
        """Execute agy with --output-format json (non-streaming).

        Returns:
            Parsed JSON response dict from agy.

        Raises:
            AgyProcessError: If agy exits with non-zero code or output is unparseable.
        """
        cmd = self.build_command(output_format="json")

        logger.info(
            "Executing agy (non-streaming)",
            extra={
                "agent": self.agent,
                "conversation_id": self.conversation_id,
                "model": self.model,
                "mode": self.mode,
            },
        )

        res = await safe_run_command(
            cmd,
            cwd=self.cwd,
            timeout=self.timeout,
            env=self.env,
            override_stdin_devnull=True,
        )

        logger.info(
            "agy completed",
            extra={
                "agent": self.agent,
                "conversation_id": self.conversation_id,
                "duration_ms": res.duration_ms,
            },
        )

        stdout_text = res.stdout
        stderr_text = res.stderr

        data = None
        if stdout_text.strip():
            try:
                data = json.loads(stdout_text)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse agy JSON output: {e} | stdout: {stdout_text[:200]}")

        if res.returncode != 0 or (data and data.get("status") == "ERROR"):
            error_msg = f"agy exited with code {res.returncode}"
            if data and isinstance(data, dict) and data.get("error"):
                error_msg = str(data["error"])
            elif stderr_text.strip():
                error_msg = stderr_text.strip()

            logger.error(
                f"agy execution error: {error_msg}",
                extra={"stderr": stderr_text[:500]},
            )
            raise AgyProcessError(
                error_msg,
                res.returncode or 1,
                stderr_text,
            )

        if data is not None:
            return data

        raise AgyProcessError(
            "Empty or invalid stdout output from agy",
            res.returncode or 0,
            stderr_text,
        )

    async def stream(self) -> AsyncIterator[dict[str, Any]]:
        """Execute agy with --output-format stream-json (streaming).

        Yields parsed NDJSON event dicts line-by-line as agy produces them.

        Yields:
            Parsed JSON event dicts from agy's NDJSON stream.

        Raises:
            AgyProcessError: If agy exits with non-zero code.
        """
        cmd = self.build_command(output_format="stream-json")

        logger.info(
            "Executing agy (streaming)",
            extra={"agent": self.agent, "conversation_id": self.conversation_id, "mode": self.mode},
        )

        exec_env = dict(os.environ)
        if self.env:
            exec_env.update(self.env)
        exec_env["PAGER"] = "cat"
        exec_env["GIT_PAGER"] = "cat"
        exec_env["TERM"] = "dumb"

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,
            cwd=self.cwd,
            env=exec_env,
        )

        assert proc.stdout is not None, "proc.stdout must be PIPE"
        assert proc.stderr is not None, "proc.stderr must be PIPE"

        try:
            async for line in proc.stdout:
                decoded = line.decode("utf-8").strip()
                if not decoded:
                    continue
                try:
                    event = json.loads(decoded)
                    yield event
                except json.JSONDecodeError:
                    logger.warning(f"Skipping non-JSON line from agy: {decoded[:100]}")
                    continue
        finally:
            await proc.wait()
            if proc.returncode != 0:
                stderr = await proc.stderr.read()
                stderr_text = stderr.decode("utf-8", errors="replace")
                logger.error(
                    f"agy stream exited with code {proc.returncode}",
                    extra={"stderr": stderr_text[:500]},
                )
                raise AgyProcessError(
                    f"agy streaming process exited with code {proc.returncode}",
                    proc.returncode or 1,
                    stderr_text,
                )


def _build_command(
    prompt: str,
    agent: str,
    output_format: str = "json",
    json_schema: dict[str, Any] | None = None,
    conversation_id: str | None = None,
    workspace: str | None = None,
    model: str | None = None,
    mode: str | None = None,
    sandbox: bool = False,
    effort: str | None = None,
    project: str | None = None,
    extra_flags: list[str] | None = None,
) -> list[str]:
    """Build the agy CLI command list with configuration options (compatibility wrapper)."""
    return AgyProcess(
        prompt=prompt,
        agent=agent,
        json_schema=json_schema,
        conversation_id=conversation_id,
        workspace=workspace,
        model=model,
        mode=mode,
        sandbox=sandbox,
        effort=effort,
        project=project,
        extra_flags=extra_flags,
    ).build_command(output_format=output_format)


async def run_agy(
    prompt: str,
    agent: str,
    json_schema: dict[str, Any] | None = None,
    conversation_id: str | None = None,
    workspace: str | None = None,
    timeout: int | None = None,
    model: str | None = None,
    mode: str | None = None,
    sandbox: bool = False,
    effort: str | None = None,
    project: str | None = None,
    extra_flags: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Execute agy with --output-format json (compatibility wrapper)."""
    return await AgyProcess(
        prompt=prompt,
        agent=agent,
        json_schema=json_schema,
        conversation_id=conversation_id,
        workspace=workspace,
        timeout=timeout,
        model=model,
        mode=mode,
        sandbox=sandbox,
        effort=effort,
        project=project,
        extra_flags=extra_flags,
        env=env,
    ).run()


async def stream_agy(
    prompt: str,
    agent: str,
    json_schema: dict[str, Any] | None = None,
    conversation_id: str | None = None,
    workspace: str | None = None,
    timeout: int | None = None,
    model: str | None = None,
    mode: str | None = None,
    sandbox: bool = False,
    effort: str | None = None,
    project: str | None = None,
    extra_flags: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Execute agy with --output-format stream-json (compatibility wrapper)."""
    proc = AgyProcess(
        prompt=prompt,
        agent=agent,
        json_schema=json_schema,
        conversation_id=conversation_id,
        workspace=workspace,
        timeout=timeout,
        model=model,
        mode=mode,
        sandbox=sandbox,
        effort=effort,
        project=project,
        extra_flags=extra_flags,
        env=env,
    )
    async for event in proc.stream():
        yield event


__all__ = [
    "AgyProcess",
    "AgyProcessError",
    "run_agy",
    "stream_agy",
    "_build_command",
]
