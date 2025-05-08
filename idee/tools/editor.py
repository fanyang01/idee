import logging
import os
import re
from typing import Dict, Any, Optional, List

from .base import BaseTool, ToolError

logger = logging.getLogger(__name__)

# Define allowed paths or restrict access? For now, allow relative/absolute paths.
# Consider adding security checks based on project root or allowed directories.

class TextEditorTool(BaseTool):
    """
    A tool for reading files and performing search-and-replace operations.
    Operates on text files only.
    """
    name: str = "text_editor"
    description: str = (
        "Reads the content of a text file or performs a search-and-replace operation within it. "
        "Use 'read_file' action to view content, and 'search_replace' to modify. "
        "Specify the file path relative to the current working directory or as an absolute path. "
        "Only operates on text files."
    )
    input_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["read_file", "search_replace"],
                "description": "The operation to perform: 'read_file' or 'search_replace'.",
            },
            "file_path": {
                "type": "string",
                "description": "The path to the text file.",
            },
            "search_pattern": {
                "type": "string",
                "description": "The regex pattern to search for (only for 'search_replace').",
            },
            "replace_string": {
                "type": "string",
                "description": "The string to replace matches with (only for 'search_replace').",
            },
            "max_lines": {
                "type": "integer",
                "description": "Maximum number of lines to return when reading (optional, default 100).",
                "default": 100,
            }
        },
        "required": ["action", "file_path"],
    }

    async def _read_file(self, file_path: str, max_lines: int) -> str:
        """Reads content from a file."""
        logger.info(f"Reading file: {file_path} (max_lines: {max_lines})")
        try:
            # Security check: Ensure path is within expected bounds if needed
            # real_path = os.path.realpath(file_path)
            # if not real_path.startswith(ALLOWED_BASE_PATH):
            #     raise ToolError("Access denied: Path is outside allowed directory.")

            if not os.path.exists(file_path):
                raise ToolError(f"File not found: {file_path}")
            if not os.path.isfile(file_path):
                 raise ToolError(f"Path is not a file: {file_path}")

            lines: List[str] = []
            line_count = 0
            # Read file line by line to handle large files and limit lines
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    if line_count >= max_lines:
                        lines.append("... (file truncated due to max_lines limit)")
                        break
                    lines.append(line)
                    line_count += 1

            content = "".join(lines)
            logger.info(f"Successfully read {line_count} lines from {file_path}.")
            return content

        except FileNotFoundError:
            logger.error(f"File not found during read: {file_path}")
            raise ToolError(f"File not found: {file_path}")
        except IsADirectoryError:
             logger.error(f"Attempted to read a directory: {file_path}")
             raise ToolError(f"Path is a directory, not a file: {file_path}")
        except PermissionError:
            logger.error(f"Permission denied reading file: {file_path}")
            raise ToolError(f"Permission denied reading file: {file_path}")
        except UnicodeDecodeError as e:
             logger.error(f"Encoding error reading file {file_path}: {e}. Is it a text file?")
             raise ToolError(f"Could not decode file {file_path} as UTF-8. It might be a binary file.")
        except Exception as e:
            logger.exception(f"Error reading file {file_path}: {e}")
            raise ToolError(f"An unexpected error occurred while reading the file: {e}")

    async def _search_replace(self, file_path: str, search_pattern: str, replace_string: str) -> Dict[str, Any]:
        """Performs search and replace in a file."""
        logger.info(f"Performing search ('{search_pattern}') and replace ('{replace_string}') in file: {file_path}")

        try:
            # Read the file content first
            # Note: This reads the entire file into memory for replacement.
            # Could be problematic for very large files.
            # Consider streaming/chunking for large files if necessary.
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()

            # Perform the replacement using regex
            try:
                new_content, num_replacements = re.subn(search_pattern, replace_string, content)
            except re.error as e:
                 logger.error(f"Invalid regex pattern '{search_pattern}': {e}")
                 raise ToolError(f"Invalid regex pattern provided: {e}")

            if num_replacements > 0:
                # Write the modified content back to the file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                logger.info(f"Successfully performed {num_replacements} replacements in {file_path}.")
                return {"status": "success", "replacements_made": num_replacements}
            else:
                logger.info(f"No matches found for pattern '{search_pattern}' in {file_path}. File not modified.")
                return {"status": "success", "replacements_made": 0, "message": "No matches found."}

        except FileNotFoundError:
            logger.error(f"File not found during search/replace: {file_path}")
            raise ToolError(f"File not found: {file_path}")
        except IsADirectoryError:
             logger.error(f"Attempted to modify a directory: {file_path}")
             raise ToolError(f"Path is a directory, not a file: {file_path}")
        except PermissionError:
            logger.error(f"Permission denied modifying file: {file_path}")
            raise ToolError(f"Permission denied modifying file: {file_path}")
        except UnicodeDecodeError as e:
             logger.error(f"Encoding error reading file for replace {file_path}: {e}.")
             raise ToolError(f"Could not decode file {file_path} as UTF-8 for modification.")
        except Exception as e:
            logger.exception(f"Error during search/replace in {file_path}: {e}")
            raise ToolError(f"An unexpected error occurred during search/replace: {e}")


    async def run(self, action: str, file_path: str, search_pattern: Optional[str] = None, replace_string: Optional[str] = None, max_lines: int = 100) -> Any:
        """
        Executes the specified file action.

        Args:
            action: 'read_file' or 'search_replace'.
            file_path: Path to the file.
            search_pattern: Regex pattern for search_replace.
            replace_string: Replacement string for search_replace.
            max_lines: Max lines to return for read_file.

        Returns:
            File content string for 'read_file', or a status dictionary for 'search_replace'.
        """
        if not file_path:
             raise ToolError("File path must be provided.")

        if action == "read_file":
            return await self._read_file(file_path, max_lines)
        elif action == "search_replace":
            if search_pattern is None or replace_string is None:
                raise ToolError("Both 'search_pattern' and 'replace_string' are required for 'search_replace' action.")
            return await self._search_replace(file_path, search_pattern, replace_string)
        else:
            # This case should ideally be prevented by schema validation if using Pydantic,
            # or by the LLM adhering to the enum definition.
            logger.warning(f"Invalid action '{action}' requested for text_editor tool.")
            raise ToolError(f"Invalid action specified: {action}. Must be 'read_file' or 'search_replace'.")


