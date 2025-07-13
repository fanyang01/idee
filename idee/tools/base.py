import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields, replace
from typing import Any, Dict, Optional, Type, Union, Tuple
import asyncio

from pydantic import BaseModel


logger = logging.getLogger(__name__)

# Define a maximum runtime for commands to prevent hangs
COMMAND_TIMEOUT_SECONDS = 60.0

# Output truncation settings
TRUNCATED_MESSAGE: str = "<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>"
MAX_RESPONSE_LEN: int = 16000


def maybe_truncate(content: str, truncate_after: int | None = MAX_RESPONSE_LEN) -> str:
    """Truncate content and append a notice if content exceeds the specified length."""
    return (
        content
        if not truncate_after or len(content) <= truncate_after
        else content[:truncate_after] + TRUNCATED_MESSAGE
    )

@dataclass
class ToolDefinition:
    """Structured representation of a tool's definition."""
    name: str
    description: str
    input_schema: Dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    output_schema: Optional[Dict[str, Any]] = None
    version: str = "1.0"
    strict: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the tool definition to a dictionary format"""
        result = {
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema,
        }
        if self.strict:
            result["strict"] = self.strict
        return result

class ToolError(Exception):
    """Custom exception for tool execution errors."""
    pass

@dataclass(kw_only=True, frozen=True)
class ToolResult:
    """Represents the result of a tool execution."""

    output: str | None = None
    error: str | None = None
    base64_image: str | None = None
    system: str | None = None

    def __bool__(self):
        return any(getattr(self, field.name) for field in fields(self))

    def __add__(self, other: "ToolResult"):
        def combine_fields(
            field: str | None, other_field: str | None, concatenate: bool = True
        ):
            if field and other_field:
                if concatenate:
                    return field + other_field
                raise ValueError("Cannot combine tool results")
            return field or other_field

        return ToolResult(
            output=combine_fields(self.output, other.output),
            error=combine_fields(self.error, other.error),
            base64_image=combine_fields(self.base64_image, other.base64_image, False),
            system=combine_fields(self.system, other.system),
        )

    def replace(self, **kwargs):
        """Returns a new ToolResult with the given fields replaced."""
        return replace(self, **kwargs)


class BaseTool(ABC):
    """
    Abstract base class for all tools that the agent can use.
    """
    name: str = "base_tool"
    description: str = "This is a placeholder description for the base tool."
    # Define input schema using Pydantic BaseModel or a simple dict structure
    # for validation and generation of tool definition for the LLM.
    # Example using dict:
    input_schema: Optional[Union[Dict[str, Any], Type["BaseModel"]]] = {
        "type": "object",
        "properties": {},
        "required": [],
    }
    strict: bool = False
    
    # Vendor-specific tool specifications
    # Key: vendor name (e.g., "anthropic", "openai", "google")
    # Value: either a single spec dict, or a dict mapping model patterns to specs
    vendor_specs: Optional[Dict[str, Union[Dict[str, Any], Dict[str, Dict[str, Any]]]]] = None

    @abstractmethod
    async def run(self, **kwargs: Any) -> ToolResult:
        """
        Executes the tool's logic asynchronously.

        Args:
            **kwargs: Arguments required by the tool, matching the args_schema.

        Returns:
            The result of the tool execution. This should typically be a
            JSON-serializable type (like str, dict, list, int, float, bool).

        Raises:
            ToolError: If an error occurs during tool execution.
            ValidationError: If using Pydantic and input validation fails.
            Exception: For unexpected errors.
        """
        pass

    def _normalize_schema(self, schema: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Normalize a schema dictionary to ensure valid format.
        
        Handles:
        - Empty properties dict
        - Empty required list
        - Default type if not specified
        """
        if schema is None:
            return {"type": "object", "properties": {}}
        
        # Make a copy to avoid modifying the original
        schema = dict(schema)
        
        # Ensure properties is a dict
        if 'properties' not in schema or not isinstance(schema.get('properties'), dict):
            schema['properties'] = {}
        
        # Remove empty required list if present
        if 'required' in schema and not schema['required']:
            del schema['required']
            
        # Ensure type is specified
        if 'type' not in schema:
            schema['type'] = 'object'
            
        return schema

    def _get_schema_from_pydantic(self, model_class: Type["BaseModel"]) -> Dict[str, Any]:
        """
        Extract JSON Schema from a Pydantic model.
        """        
        return model_class.model_json_schema()

    def has_vendor_spec(self, vendor: str, model: Optional[str] = None) -> bool:
        """Check if this tool has a vendor-specific specification."""
        return self.get_vendor_spec(vendor, model) is not None
    
    def get_vendor_spec(self, vendor: str, model: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get the vendor-specific specification for this tool.
        
        Args:
            vendor: The vendor name (e.g., "anthropic")
            model: The model name (e.g., "claude-3-5-sonnet-20241022")
            
        Returns:
            The vendor specification dict, or None if not found
        """
        if self.vendor_specs is None or vendor not in self.vendor_specs:
            return None
            
        vendor_config = self.vendor_specs[vendor]
        
        # If it's a simple dict with "type" key, it's a single spec
        if "type" in vendor_config:
            return vendor_config
            
        # Otherwise, it's a model-specific mapping
        if model is None:
            # Return the first available spec if no model specified
            for spec in vendor_config.values():
                if isinstance(spec, dict) and "type" in spec:
                    return spec
            return None
            
        # Find the best matching spec for the model
        return self._find_best_model_spec(vendor_config, model)
    
    def _find_best_model_spec(self, model_specs: Dict[str, Dict[str, Any]], model: str) -> Optional[Dict[str, Any]]:
        """Find the best matching spec for a given model."""
        model_lower = model.lower()
        
        # Try exact match first
        if model in model_specs:
            return model_specs[model]
            
        # Try case-insensitive match
        for pattern, spec in model_specs.items():
            if pattern.lower() == model_lower:
                return spec
                
        # Try specific pattern matching first (most specific to least specific)
        # Sort patterns by specificity (longer patterns first)
        sorted_patterns = sorted(model_specs.items(), key=lambda x: (-len(x[0]), x[0]))
        
        for pattern, spec in sorted_patterns:
            if self._model_matches_pattern(model_lower, pattern.lower()):
                return spec
                
        return None
    
    def _model_matches_pattern(self, model: str, pattern: str) -> bool:
        """Check if a model matches a pattern using simple wildcard matching."""
        if "*" in pattern:
            import fnmatch
            return fnmatch.fnmatch(model, pattern)
        
        # Exact pattern match
        return pattern == model

    def get_definition(self) -> ToolDefinition:
        """
        Generates the tool definition as a structured ToolDefinition object.

        Returns:
            A ToolDefinition object representing the tool's metadata.
        """
        # Handle different schema types
        schema: Dict[str, Any]
        
        if self.input_schema is None:
            schema = {"type": "object", "properties": {}}
        elif isinstance(self.input_schema, dict):
            schema = self._normalize_schema(self.input_schema)
        elif isinstance(self.input_schema, type) and issubclass(self.input_schema, BaseModel):
            try:
                schema = self._get_schema_from_pydantic(self.input_schema)
            except Exception as e:
                logger.error(f"Failed to extract schema from Pydantic model: {e}")
                schema = {"type": "object", "properties": {}}
        else:
            logger.warning(f"Unsupported schema type: {type(self.input_schema)}. Using empty schema.")
            schema = {"type": "object", "properties": {}}
            
        return ToolDefinition(
            name=self.name,
            description=self.description,
            input_schema=schema,
            strict=self.strict,
        )

    def __str__(self) -> str:
        return f"Tool(name='{self.name}', description='{self.description}')"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name='{self.name}'>"


async def run_command(command: str, timeout: float = COMMAND_TIMEOUT_SECONDS) -> Tuple[str, str, int]:
    """
    Runs a shell command in a subprocess and returns its output.
    
    This function captures partial output even when a timeout occurs.
    
    Args:
        command: The shell command to execute.
        timeout: Maximum time in seconds to wait for command completion.
        
    Returns:
        A tuple containing (stdout_str, stderr_str, return_code).
        For timeouts, partial outputs will be included with a non-zero return code.
        
    Raises:
        ToolError: If an error occurs during execution.
    """
    logger.info(f"Executing shell command: {command}")
    
    # Store partial outputs
    stdout_parts = []
    stderr_parts = []
    
    try:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Set up async tasks to read from stdout/stderr
        async def read_stream(stream, parts_list):
            while True:
                line = await stream.readline()
                if not line:
                    break
                decoded = line.decode('utf-8', errors='replace')
                parts_list.append(decoded)
                
        # Start tasks to read outputs continuously
        read_stdout = asyncio.create_task(read_stream(process.stdout, stdout_parts))
        read_stderr = asyncio.create_task(read_stream(process.stderr, stderr_parts))
        
        try:
            # Wait for the command to complete with a timeout
            await asyncio.wait_for(process.wait(), timeout=timeout)
            # If we get here, process completed normally - wait for readers to complete
            await read_stdout
            await read_stderr
        except asyncio.TimeoutError:
            # Process timed out - cancel readers and kill process
            read_stdout.cancel()
            read_stderr.cancel()
            try:
                process.kill()
                await process.wait()  # Clean up process
                logger.warning(f"Command timed out after {timeout} seconds: {command}")
            except Exception as kill_err:
                logger.error(f"Error killing timed-out process: {kill_err}")
            
            # Return partial results with timeout indication
            stdout_str = "".join(stdout_parts)
            stderr_str = "".join(stderr_parts)
            stderr_str += f"\n[COMMAND TIMED OUT after {timeout} seconds]"
            return stdout_str, stderr_str, 124  # Use 124 as timeout return code (consistent with 'timeout' command)
            
        # Process completed within timeout
        return_code = process.returncode or 0  # Ensure returncode is not None
        
        # Join all collected output parts
        stdout_str = "".join(stdout_parts)
        stderr_str = "".join(stderr_parts)
        
        logger.info(f"Command finished with return code: {return_code}")
        if stdout_str:
            logger.debug(f"Stdout: {stdout_str[:200]}...") # Log snippet
        if stderr_str:
            logger.warning(f"Stderr: {stderr_str[:200]}...") # Log snippet
            
        return stdout_str, stderr_str, return_code
        
    except FileNotFoundError:
        logger.error(f"Command not found: {command.split()[0]}")
        raise ToolError(f"Command not found: {command.split()[0]}")
    except Exception as e:
        logger.exception(f"Error executing command '{command}': {e}")
        raise ToolError(f"An unexpected error occurred: {e}")
