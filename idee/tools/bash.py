import asyncio
import logging
import os
from typing import Dict, Any, Optional

from .base import BaseTool, ToolError, ToolResult, run_command, maybe_truncate

logger = logging.getLogger(__name__)

class _BashSession:
    """A persistent session of a bash shell for vendor tools."""

    _started: bool
    _process: asyncio.subprocess.Process

    command: str = "/bin/bash"
    _output_delay: float = 0.2  # seconds
    _timeout: float = 120.0  # seconds
    _sentinel: str = "<<exit>>"

    def __init__(self):
        self._started = False
        self._timed_out = False

    async def start(self):
        if self._started:
            return

        self._process = await asyncio.create_subprocess_shell(
            self.command,
            preexec_fn=os.setsid,
            shell=True,
            bufsize=0,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        self._started = True

    def stop(self):
        """Terminate the bash shell."""
        if not self._started:
            raise ToolError("Session has not started.")
        if self._process.returncode is not None:
            return
        self._process.terminate()

    async def run(self, command: str):
        """Execute a command in the bash shell."""
        if not self._started:
            raise ToolError("Session has not started.")
        if self._process.returncode is not None:
            return ToolResult(
                output="tool must be restarted",
                error=f"bash has exited with returncode {self._process.returncode}"
            )
        if self._timed_out:
            raise ToolError(
                f"timed out: bash has not returned in {self._timeout} seconds and must be restarted"
            )

        # we know these are not None because we created the process with PIPEs
        assert self._process.stdin
        assert self._process.stdout
        assert self._process.stderr

        # send command to the process
        self._process.stdin.write(
            command.encode() + f"; echo '{self._sentinel}'\n".encode()
        )
        await self._process.stdin.drain()

        # read output from the process, until the sentinel is found
        try:
            async with asyncio.timeout(self._timeout):
                while True:
                    await asyncio.sleep(self._output_delay)
                    # if we read directly from stdout/stderr, it will wait forever for
                    # EOF. use the StreamReader buffer directly instead.
                    output = self._process.stdout._buffer.decode()  # pyright: ignore[reportAttributeAccessIssue]
                    if self._sentinel in output:
                        # strip the sentinel and break
                        output = output[: output.index(self._sentinel)]
                        break
        except asyncio.TimeoutError:
            self._timed_out = True
            # Capture partial output before timing out
            partial_output = self._process.stdout._buffer.decode()  # pyright: ignore[reportAttributeAccessIssue]
            partial_error = self._process.stderr._buffer.decode()  # pyright: ignore[reportAttributeAccessIssue]
            
            # Clean up trailing newlines
            if partial_output.endswith("\n"):
                partial_output = partial_output[:-1]
            if partial_error.endswith("\n"):
                partial_error = partial_error[:-1]
                
            # Add timeout indication to error
            timeout_msg = f"[COMMAND TIMED OUT after {self._timeout} seconds]"
            if partial_error:
                partial_error += f"\n{timeout_msg}"
            else:
                partial_error = timeout_msg
            
            return ToolResult(
                output=partial_output,
                error=partial_error
            )

        if output.endswith("\n"):
            output = output[:-1]

        error = self._process.stderr._buffer.decode()  # pyright: ignore[reportAttributeAccessIssue]
        if error.endswith("\n"):
            error = error[:-1]

        # clear the buffers so that the next output can be read correctly
        self._process.stdout._buffer.clear()  # pyright: ignore[reportAttributeAccessIssue]
        self._process.stderr._buffer.clear()  # pyright: ignore[reportAttributeAccessIssue]

        return ToolResult(output=output, error=error)

class BashTool(BaseTool):
    """
    A tool for executing bash commands.
    Supports vendor-specific implementations with persistent sessions for better performance.
    """
    name: str = "bash"
    description: str = (
        "Executes a given bash command and returns its standard output, "
        "standard error, and return code. Use this for interacting with "
        "the shell, running scripts, or checking system status. "
        "Be cautious with commands that modify the filesystem or system state."
    )
    input_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The bash command to execute.",
            },
            "restart": {
                "type": "boolean",
                "description": "Whether to restart the bash session (for vendor tools).",
                "default": False
            }
        },
        "required": ["command"],
    }
    
    # Vendor-specific tool specifications with model support
    vendor_specs: Dict[str, Dict[str, Dict[str, Any]]] = {
        "anthropic": {
            # Claude models with bash tool support
            "claude-3-5-*": {
                "type": "bash_20250124",
                "name": "bash"
            },
            "claude-*-4-*": {
                "type": "bash_20250124", 
                "name": "bash"
            },
            # Default fallback for newer Claude models
            "claude-*": {
                "type": "bash_20250124",
                "name": "bash"
            }
        }
    }

    def __init__(self):
        self._session: Optional[_BashSession] = None


    async def run(self, command: str, restart: bool = False) -> ToolResult:
        """
        Executes the bash command.
        Uses persistent session for vendor tools, one-off execution for others.

        Args:
            command: The bash command string.
            restart: Whether to restart the session (for vendor tools).

        Returns:
            A ToolResult containing stdout, stderr, and return_code.
        """
        # Handle restart for persistent sessions
        if restart:
            if self._session:
                self._session.stop()
            self._session = _BashSession()
            await self._session.start()
            return ToolResult(output="tool has been restarted.")

        if not command or not isinstance(command, str):
            raise ToolError("Invalid command provided. Command must be a non-empty string.")

        try:
            # Use persistent session if available (for vendor tools)
            if hasattr(self, '_session') and self._session is not None:
                if not self._session._started:
                    await self._session.start()
                result = await self._session.run(command)
                # Truncate outputs for persistent session (create new ToolResult since it's frozen)
                truncated_output = maybe_truncate(result.output) if result.output else result.output
                truncated_error = maybe_truncate(result.error) if result.error else result.error
                return ToolResult(
                    output=truncated_output,
                    error=truncated_error,
                    base64_image=getattr(result, 'base64_image', None)
                )
            else:
                # Fall back to one-off command execution
                stdout, stderr, return_code = await run_command(command)
                
                # Truncate outputs to prevent context overflow
                stdout_truncated = maybe_truncate(stdout) if stdout else ""
                stderr_truncated = maybe_truncate(stderr) if stderr else ""
                
                result = {
                    "stdout": stdout_truncated,
                    "stderr": stderr_truncated,
                    "return_code": return_code,
                }
                
                # Include stderr in the error field if the command failed
                error_message = None
                if return_code != 0:
                    if stderr_truncated:
                        error_message = f"Command failed with return code {return_code}: {stderr_truncated}"
                    else:
                        error_message = f"Command failed with return code {return_code}"
                    
                return ToolResult(
                    output=str(result) if isinstance(result, dict) else result,
                    error=error_message
                )
        except ToolError:
            # re-raise tool errors as-is
            raise
        except Exception as e:
            logger.exception(f"Failed to execute command '{command}': {e}")
            raise ToolError(f"An unexpected error occurred: {e}")

    def enable_persistent_session(self):
        """Enable persistent session mode for vendor tools."""
        if self._session is None:
            self._session = _BashSession()


