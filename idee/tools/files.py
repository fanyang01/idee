""" https://github.com/anthropics/anthropic-quickstarts/blob/main/agents/tools/file_tools.py """
"""File operation tools for reading and writing files."""

import asyncio
import glob
import os
from pathlib import Path

from .base import BaseTool, ToolError, ToolResult


class FileReadTool(BaseTool):
    """Tool for reading files and listing directories."""

    def __init__(self):
        self.name = "file_read"
        self.description = """
        Read files or list directory contents.

        Operations:
        - read: Read the contents of a file
        - list: List files in a directory
        """
        self.input_schema = {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["read", "list"],
                    "description": "File operation to perform",
                },
                "path": {
                    "type": "string",
                    "description": "File path for read or directory path",
                },
                "max_lines": {
                    "type": "integer",
                    "description": "Maximum lines to read (0 means no limit)",
                },
                "pattern": {
                    "type": "string",
                    "description": "File pattern to match",
                },
            },
            "required": ["operation", "path"],
        }

    async def run(
        self,
        operation: str,
        path: str,
        max_lines: int = 0,
        pattern: str = "*",
        **kwargs
    ) -> ToolResult:
        """Execute a file read operation.

        Args:
            operation: The operation to perform (read or list)
            path: The file or directory path
            max_lines: Maximum lines to read (for read operation, 0 means no limit)
            pattern: File pattern to match (for list operation)

        Returns:
            ToolResult containing the operation result or error
        """
        # Removed the outer try/except ToolError and general Exception blocks.
        # ToolErrors from helper methods (_read_file, _list_files) will propagate.
        # Other unexpected exceptions will also propagate.

        if operation == "read":
            result = await self._read_file(path, max_lines)
            return ToolResult(output=result)
        elif operation == "list":
            result = await self._list_files(path, pattern)
            return ToolResult(output=result)
        else:
            # This indicates an invalid 'operation' value was passed, likely due to
            # the LLM not adhering to the schema or a programming error.
            # Raising a ToolError is appropriate.
            raise ToolError(f"Unsupported operation '{operation}'")

    async def _read_file(self, path: str, max_lines: int = 0) -> str:
        """Read a file from disk.
        
        Args:
            path: Path to the file to read
            max_lines: Maximum number of lines to read (0 means read entire file)
        """
        try:
            file_path = Path(path)

            if not file_path.exists():
                raise ToolError(f"File not found at {path}")
            if not file_path.is_file():
                raise ToolError(f"{path} is not a file")

            def read_sync():
                with open(file_path, encoding="utf-8", errors="replace") as f:
                    if max_lines > 0:
                        lines = []
                        for i, line in enumerate(f):
                            if i >= max_lines:
                                break
                            lines.append(line)
                        return "".join(lines)
                    return f.read()

            return await asyncio.to_thread(read_sync)
        except ToolError as e:
            raise e
        except Exception as e:
            raise ToolError(f"Error reading {path}: {str(e)}")

    async def _list_files(self, directory: str, pattern: str = "*") -> str:
        """List files in a directory."""
        try:
            dir_path = Path(directory)

            if not dir_path.exists():
                raise ToolError(f"Directory not found at {directory}")
            if not dir_path.is_dir():
                raise ToolError(f"{directory} is not a directory")

            def list_sync():
                search_pattern = f"{directory}/{pattern}"
                files = glob.glob(search_pattern)

                if not files:
                    return f"No files found matching {directory}/{pattern}"

                file_list = []
                for file_path in sorted(files):
                    path_obj = Path(file_path)
                    rel_path = str(file_path).replace(str(dir_path) + "/", "")

                    if path_obj.is_dir():
                        file_list.append(f"📁 {rel_path}/")
                    else:
                        file_list.append(f"📄 {rel_path}")

                return "\n".join(file_list)

            return await asyncio.to_thread(list_sync)
        except ToolError as e:
            raise e
        except Exception as e:
            raise ToolError(f"Error listing files in {directory}: {str(e)}")


class FileWriteTool(BaseTool):
    """Tool for writing and editing files."""

    def __init__(self):
        self.name = "file_write"
        self.description = """
        Write or edit files.

        Operations:
        - write: Create or completely replace a file
        - edit: Make targeted changes to parts of a file
        """
        self.input_schema = {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["write", "edit"],
                    "description": "File operation to perform",
                },
                "path": {
                    "type": "string",
                    "description": "File path to write to or edit",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write",
                },
                "old_text": {
                    "type": "string",
                    "description": "Text to replace (for edit operation)",
                },
                "new_text": {
                    "type": "string",
                    "description": "Replacement text (for edit operation)",
                },
            },
            "required": ["operation", "path"],
        }

    async def run(
        self,
        operation: str,
        path: str,
        content: str = "",
        old_text: str = "",
        new_text: str = "",
        **kwargs
    ) -> ToolResult:
        """Execute a file write operation.

        Args:
            operation: The operation to perform (write or edit)
            path: The file path
            content: Content to write (for write operation)
            old_text: Text to replace (for edit operation)
            new_text: Replacement text (for edit operation)

        Returns:
            ToolResult containing the operation result or error
        """
        # Removed the outer try/except ToolError and general Exception blocks.
        # ToolErrors from helper methods (_write_file, _edit_file) will propagate.
        # Other unexpected exceptions will also propagate.

        if operation == "write":
            if not content:
                # Input validation specific to the 'write' operation.
                raise ToolError("Content parameter is required for write operation")
            result = await self._write_file(path, content)
            return ToolResult(output=result)
        elif operation == "edit":
            if not old_text or not new_text:
                # Input validation specific to the 'edit' operation.
                raise ToolError("Both old_text and new_text parameters are required for edit operation")
            result = await self._edit_file(path, old_text, new_text)
            return ToolResult(output=result)
        else:
            # Invalid 'operation' value.
            raise ToolError(f"Unsupported operation '{operation}'")

    async def _write_file(self, path: str, content: str) -> str:
        """Write content to a file."""
        try:
            file_path = Path(path)
            os.makedirs(file_path.parent, exist_ok=True)

            def write_sync():
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                return (
                    f"Successfully wrote {len(content)} "
                    f"characters to {path}"
                )

            return await asyncio.to_thread(write_sync)
        except Exception as e:
            raise ToolError(f"Error writing to {path}: {str(e)}")

    async def _edit_file(self, path: str, old_text: str, new_text: str) -> str:
        """Make targeted changes to a file."""
        try:
            file_path = Path(path)

            if not file_path.exists():
                raise ToolError(f"File not found at {path}")
            if not file_path.is_file():
                raise ToolError(f"{path} is not a file")

            def edit_sync():
                try:
                    with open(
                        file_path, encoding="utf-8", errors="replace"
                    ) as f:
                        content = f.read()

                    if old_text not in content:
                        raise ToolError(
                            f"The specified text was not found in {path}"
                        )

                    # Count occurrences to warn about multiple matches
                    count = content.count(old_text)
                    if count > 1:
                        # Edit with warning about multiple occurrences
                        new_content = content.replace(old_text, new_text)
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(new_content)
                        return (
                            f"Warning: Found {count} occurrences. "
                            f"All were replaced in {path}"
                        )
                    else:
                        # One occurrence, straightforward replacement
                        new_content = content.replace(old_text, new_text)
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(new_content)
                        return f"Successfully edited {path}"
                except UnicodeDecodeError:
                    raise ToolError(f"{path} appears to be a binary file")

            return await asyncio.to_thread(edit_sync)
        except ToolError as e:
            raise e
        except Exception as e:
            raise ToolError(f"Error editing {path}: {str(e)}")