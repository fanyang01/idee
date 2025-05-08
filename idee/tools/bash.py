import logging
import asyncio
import shlex
from typing import Dict, Any, Tuple

from .base import BaseTool, ToolError

logger = logging.getLogger(__name__)

# Define a maximum runtime for commands to prevent hangs
COMMAND_TIMEOUT_SECONDS = 60.0

class BashTool(BaseTool):
    """
    A tool for executing bash commands in a subprocess.
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
            }
        },
        "required": ["command"],
    }

    async def _run_command(self, command: str) -> Tuple[str, str, int]:
        """Runs the command in a subprocess."""
        logger.info(f"Executing bash command: {command}")
        try:
            # Use asyncio.create_subprocess_shell for better async handling
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            # Wait for the command to complete with a timeout
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=COMMAND_TIMEOUT_SECONDS
            )

            return_code = process.returncode or 0 # Ensure returncode is not None

            # Decode stdout and stderr
            stdout_str = stdout.decode('utf-8', errors='replace').strip()
            stderr_str = stderr.decode('utf-8', errors='replace').strip()

            logger.info(f"Command finished with return code: {return_code}")
            if stdout_str:
                logger.debug(f"Stdout: {stdout_str[:200]}...") # Log snippet
            if stderr_str:
                logger.warning(f"Stderr: {stderr_str[:200]}...") # Log snippet

            return stdout_str, stderr_str, return_code

        except asyncio.TimeoutError:
            logger.error(f"Command timed out after {COMMAND_TIMEOUT_SECONDS} seconds: {command}")
            # Attempt to kill the process if it timed out
            if process.returncode is None:
                try:
                    process.kill()
                    await process.wait() # Ensure process is cleaned up
                    logger.info(f"Killed timed-out process for command: {command}")
                except ProcessLookupError:
                    logger.warning("Process already terminated when trying to kill.")
                except Exception as kill_err:
                    logger.error(f"Error trying to kill timed-out process: {kill_err}")
            raise ToolError(f"Command timed out after {COMMAND_TIMEOUT_SECONDS} seconds.")
        except FileNotFoundError:
            logger.error(f"Command not found: {command.split()[0]}")
            raise ToolError(f"Command not found: {command.split()[0]}")
        except Exception as e:
            logger.exception(f"Error executing command '{command}': {e}")
            raise ToolError(f"An unexpected error occurred: {e}")

    async def run(self, command: str) -> Dict[str, Any]:
        """
        Executes the bash command.

        Args:
            command: The bash command string.

        Returns:
            A dictionary containing stdout, stderr, and return_code.
        """
        if not command or not isinstance(command, str):
            raise ToolError("Invalid command provided. Command must be a non-empty string.")

        # Basic safety check: prevent execution of extremely dangerous commands directly?
        # This is tricky. Relying on the LLM's safety features and user oversight is key.
        # Example (very basic):
        # if "rm -rf /" in command:
        #     raise ToolError("Potentially dangerous command detected and blocked.")

        stdout, stderr, return_code = await self._run_command(command)

        return {
            "stdout": stdout,
            "stderr": stderr,
            "return_code": return_code,
        }


