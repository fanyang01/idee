import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields, replace
from typing import Any, Dict, Optional, Type, Union, Tuple, List, Literal
from pathlib import Path
import asyncio
import base64
import io

from pydantic import BaseModel


logger = logging.getLogger(__name__)

# Define a maximum runtime for commands to prevent hangs
COMMAND_TIMEOUT_SECONDS = 60.0

# Output truncation settings
TRUNCATED_MESSAGE: str = "<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>"
MAX_RESPONSE_LEN: int = 16000

# Content block types for multimodal tool results
ContentBlockType = Literal["text", "image", "audio", "video", "document", "file"]


@dataclass(frozen=True)
class ContentBlock:
    """Base class for content blocks in multimodal tool results."""
    type: ContentBlockType

@dataclass(frozen=True)
class TextBlock(ContentBlock):
    """Text content block."""
    type: Literal["text"] = "text"
    text: str = ""

@dataclass(frozen=True)
class FileBlock(ContentBlock):
    """File content block that can hold local files or base64 data."""
    type: Literal["file"] = "file"
    
    # File source - one of these should be provided
    file_path: Optional[str] = None  # Path to local file
    content: Optional[bytes] = None  # Raw file content  
    base64_data: Optional[str] = None  # Base64 encoded content
    
    # File metadata
    media_type: Optional[str] = None  # MIME type
    filename: Optional[str] = None  # Original filename
    
    def __post_init__(self):
        """Validate that exactly one content source is provided."""
        sources = [self.file_path, self.content, self.base64_data]
        provided_sources = [s for s in sources if s is not None]
        
        if len(provided_sources) != 1:
            raise ValueError("FileBlock must have exactly one of: file_path, content, or base64_data")
    
    @classmethod
    def from_path(cls, file_path: str, media_type: Optional[str] = None, filename: Optional[str] = None) -> "FileBlock":
        """Create a FileBlock from a local file path."""
        return cls(
            file_path=file_path,
            media_type=media_type,
            filename=filename or Path(file_path).name
        )
    
    @classmethod 
    def from_content(cls, content: bytes, media_type: str, filename: Optional[str] = None) -> "FileBlock":
        """Create a FileBlock from raw bytes."""
        return cls(
            content=content,
            media_type=media_type,
            filename=filename
        )
    
    @classmethod
    def from_base64(cls, base64_data: str, media_type: str, filename: Optional[str] = None) -> "FileBlock":
        """Create a FileBlock from base64 data."""
        return cls(
            base64_data=base64_data,
            media_type=media_type,
            filename=filename
        )
    
    def get_base64_data(self) -> str:
        """Get the file content as base64 string."""
        if self.base64_data:
            return self.base64_data
        elif self.file_path:
            from .media_utils import read_file_as_base64
            return read_file_as_base64(self.file_path)
        elif self.content:
            return base64.b64encode(self.content).decode('utf-8')
        else:
            raise ValueError("No content source available for base64 conversion")
    
    def get_bytes_content(self) -> bytes:
        """Get the file content as bytes."""
        if self.content:
            return self.content
        elif self.base64_data:
            return base64.b64decode(self.base64_data)
        elif self.file_path:
            return Path(self.file_path).read_bytes()
        else:
            raise ValueError("No content source available for bytes conversion")

    def get_file_like_object(self) -> Union[Path, io.BytesIO]:
        """Get the file content as a file-like object."""
        if self.file_path:
            return Path(self.file_path)
        elif self.content:
            return io.BytesIO(self.content)
        elif self.base64_data:
            return io.BytesIO(base64.b64decode(self.base64_data))
        else:
            raise ValueError("No content source available for file-like object conversion")

@dataclass(frozen=True)
class MediaBlock(ContentBlock):
    """Base class for media content blocks (image, audio, video, document)."""
    source: Dict[str, Any] = field(default_factory=lambda: {
        "type": "base64",
        "media_type": "",
        "data": ""
    })
    
    @classmethod
    def from_base64(cls, data: str, media_type: str) -> "MediaBlock":
        """Create a MediaBlock from base64 data."""
        return cls(source={
            "type": "base64", 
            "media_type": media_type,
            "data": data
        })
    
    @classmethod
    def from_file_path(cls, file_path: str, media_type: str) -> "MediaBlock":
        """Create a MediaBlock from a file path."""
        return cls(source={
            "type": "file_path",
            "media_type": media_type,
            "data": file_path
        })
    

@dataclass(frozen=True)
class ImageBlock(MediaBlock):
    """Image content block with base64 data, file path, or file reference."""
    type: Literal["image"] = "image"
    
    @classmethod
    def from_base64(cls, data: str, media_type: str = "image/png") -> "ImageBlock":
        """Create an ImageBlock from base64 data."""
        return cls(source={
            "type": "base64",
            "media_type": media_type,
            "data": data
        })
    

@dataclass(frozen=True)
class AudioBlock(MediaBlock):
    """Audio content block with base64 data, file path, or file reference."""
    type: Literal["audio"] = "audio"
    
    @classmethod
    def from_base64(cls, data: str, media_type: str = "audio/wav") -> "AudioBlock":
        """Create an AudioBlock from base64 data."""
        return cls(source={
            "type": "base64",
            "media_type": media_type,
            "data": data
        })
    

@dataclass(frozen=True)
class VideoBlock(MediaBlock):
    """Video content block with base64 data, file path, or file reference."""
    type: Literal["video"] = "video"
    
    @classmethod
    def from_base64(cls, data: str, media_type: str = "video/mp4") -> "VideoBlock":
        """Create a VideoBlock from base64 data."""
        return cls(source={
            "type": "base64",
            "media_type": media_type,
            "data": data
        })
    

@dataclass(frozen=True)
class DocumentBlock(MediaBlock):
    """Document content block with base64 data, file path, or file reference."""
    type: Literal["document"] = "document"
    
    @classmethod
    def from_base64(cls, data: str, media_type: str = "application/pdf") -> "DocumentBlock":
        """Create a DocumentBlock from base64 data."""
        return cls(source={
            "type": "base64", 
            "media_type": media_type,
            "data": data
        })
    

@dataclass(frozen=True)
class ToolOutputPlaceholder(ContentBlock):
    """Placeholder for tool output that will be provided in follow-up message."""
    type: Literal["placeholder"] = "placeholder"
    placeholder_id: str = ""
    doc: str = ""
    media_type: str = ""  # Type of media this placeholder represents


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
    """Represents the result of a tool execution with multimodal content support."""

    content: List[Union[TextBlock, FileBlock, ImageBlock, AudioBlock, VideoBlock, DocumentBlock, ToolOutputPlaceholder]] = field(default_factory=list)
    error: str | None = None
    system: str | None = None

    def __bool__(self):
        return bool(self.content) or bool(self.error) or bool(self.system)

    def __add__(self, other: "ToolResult"):
        """Combine two ToolResult objects."""
        def combine_optional_str(a: str | None, b: str | None) -> str | None:
            if a and b:
                return a + b
            return a or b

        # Combine content blocks
        combined_content = list(self.content) + list(other.content)
        
        return ToolResult(
            content=combined_content,
            error=combine_optional_str(self.error, other.error),
            system=combine_optional_str(self.system, other.system),
        )

    def replace(self, **kwargs):
        """Returns a new ToolResult with the given fields replaced."""
        return replace(self, **kwargs)
    
    def add_text(self, text: str) -> "ToolResult":
        """Add a text block to the content."""
        new_content = list(self.content) + [TextBlock(text=text)]
        return self.replace(content=new_content)
    
    def add_image(self, base64_data: str, media_type: str = "image/png") -> "ToolResult":
        """Add an image block to the content."""
        new_content = list(self.content) + [ImageBlock.from_base64(base64_data, media_type)]
        return self.replace(content=new_content)
    
    def add_audio(self, base64_data: str, media_type: str = "audio/wav") -> "ToolResult":
        """Add an audio block to the content."""
        new_content = list(self.content) + [AudioBlock.from_base64(base64_data, media_type)]
        return self.replace(content=new_content)
    
    def add_video(self, base64_data: str, media_type: str = "video/mp4") -> "ToolResult":
        """Add a video block to the content."""
        new_content = list(self.content) + [VideoBlock.from_base64(base64_data, media_type)]
        return self.replace(content=new_content)
    
    def add_document(self, base64_data: str, media_type: str = "application/pdf") -> "ToolResult":
        """Add a document block to the content."""
        new_content = list(self.content) + [DocumentBlock.from_base64(base64_data, media_type)]
        return self.replace(content=new_content)
    
    def add_placeholder(self, placeholder_id: str, doc: str = "", media_type: str = "") -> "ToolResult":
        """Add a placeholder block for content that will be provided later."""
        new_content = list(self.content) + [ToolOutputPlaceholder(placeholder_id=placeholder_id, doc=doc, media_type=media_type)]
        return self.replace(content=new_content)
    
    def add_file(self, file_path: str, media_type: Optional[str] = None, filename: Optional[str] = None) -> "ToolResult":
        """Add a file block from a local file path. Upload will be handled by the agent."""
        new_content = list(self.content) + [FileBlock.from_path(file_path, media_type, filename)]
        return self.replace(content=new_content)
    
    def add_file_content(self, content: bytes, media_type: str, filename: Optional[str] = None) -> "ToolResult":
        """Add a file block from raw bytes. Upload will be handled by the agent."""
        new_content = list(self.content) + [FileBlock.from_content(content, media_type, filename)]
        return self.replace(content=new_content)
    
    def add_file_base64(self, base64_data: str, media_type: str, filename: Optional[str] = None) -> "ToolResult":
        """Add a file block from base64 data. Upload will be handled by the agent."""
        new_content = list(self.content) + [FileBlock.from_base64(base64_data, media_type, filename)]
        return self.replace(content=new_content)
    
    def get_text_content(self) -> str:
        """Extract all text content as a single string."""
        text_parts = []
        for block in self.content:
            if isinstance(block, TextBlock):
                text_parts.append(block.text)
        return "".join(text_parts)
    
    def get_images(self) -> List[ImageBlock]:
        """Extract all image blocks."""
        return [block for block in self.content if isinstance(block, ImageBlock)]
    
    def get_audio(self) -> List[AudioBlock]:
        """Extract all audio blocks."""
        return [block for block in self.content if isinstance(block, AudioBlock)]
    
    def get_video(self) -> List[VideoBlock]:
        """Extract all video blocks."""
        return [block for block in self.content if isinstance(block, VideoBlock)]
    
    def get_documents(self) -> List[DocumentBlock]:
        """Extract all document blocks."""
        return [block for block in self.content if isinstance(block, DocumentBlock)]
    
    def get_files(self) -> List[FileBlock]:
        """Extract all file blocks."""
        return [block for block in self.content if isinstance(block, FileBlock)]
    
    def get_media_blocks(self) -> List[Union[ImageBlock, AudioBlock, VideoBlock, DocumentBlock, FileBlock]]:
        """Extract all media blocks (images, audio, video, documents, files)."""
        return [block for block in self.content if isinstance(block, (ImageBlock, AudioBlock, VideoBlock, DocumentBlock, FileBlock))]
    
    def has_images(self) -> bool:
        """Check if this result contains any images."""
        return any(isinstance(block, ImageBlock) for block in self.content)
    
    def has_files(self) -> bool:
        """Check if this result contains any file blocks."""
        return any(isinstance(block, FileBlock) for block in self.content)
    
    def has_media(self) -> bool:
        """Check if this result contains any media content."""
        return any(isinstance(block, (ImageBlock, AudioBlock, VideoBlock, DocumentBlock, FileBlock)) for block in self.content)


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
