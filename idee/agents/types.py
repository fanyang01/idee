from dataclasses import dataclass, field
import time
import uuid
from typing import List, Dict, Any, Literal, Optional, Union
from pydantic import BaseModel

from ..tools.base import ToolResult, ImageBlock, AudioBlock, VideoBlock, DocumentBlock, FileBlock, ToolOutputPlaceholder, TextBlock


# --- Unified Message Representation ---
# Used internally for logging, TUI display, and potentially history summarization.
# Specific agents will work with their native API message formats during the loop.

Role = Literal["user", "assistant", "system", "tool"]

@dataclass
class UnifiedToolCall:
    """Represents a tool call requested by the assistant."""
    id: str # Unique ID for the tool call (provided by the LLM API)
    tool_name: str
    tool_input: Dict[str, Any] # Parsed arguments for the tool

@dataclass
class UnifiedToolResult:
    """Represents the result of executing a tool with multimodal support."""
    call_id: str # ID of the tool call this result corresponds to
    tool_name: str
    tool_output: Any # The data returned by the tool's run method (ToolResult object)
    is_error: bool = False # Flag to indicate if the tool execution failed
    
    # Multimodal content handling
    follow_up_content: Optional[List[Dict[str, Any]]] = None # Media attachments for non-Claude models
    placeholder_mappings: Optional[Dict[str, str]] = None # Map placeholder IDs to descriptions
    
    def get_text_content(self) -> str:
        """Extract text content from the tool result."""
        if isinstance(self.tool_output, ToolResult):
            return self.tool_output.get_text_content()
        elif isinstance(self.tool_output, str):
            return self.tool_output
        else:
            return str(self.tool_output)
    
    def has_multimodal_content(self) -> bool:
        """Check if this result contains multimodal content."""        
        if isinstance(self.tool_output, ToolResult):
            return self.tool_output.has_media() or bool(self.follow_up_content)
        return bool(self.follow_up_content)
    
    def get_media(self) -> List[Dict[str, Any]]:
        """Get all media from this tool result."""        
        media = []
        
        # Get media from ToolResult
        if isinstance(self.tool_output, ToolResult):
            for media_block in self.tool_output.get_media_blocks():
                media.append({
                    "type": media_block.type,
                    "source": media_block.source
                })
        
        # Add follow-up media for non-Claude models
        if self.follow_up_content:
            media.extend(self.follow_up_content)
            
        return media
    
    def get_images(self) -> List[Dict[str, Any]]:
        """Get all images from this tool result."""
        return [item for item in self.get_media() if item["type"] == "image"]
    
    def get_audio(self) -> List[Dict[str, Any]]:
        """Get all audio from this tool result."""
        return [item for item in self.get_media() if item["type"] == "audio"]
    
    def get_video(self) -> List[Dict[str, Any]]:
        """Get all video from this tool result.""" 
        return [item for item in self.get_media() if item["type"] == "video"]
    
    def get_documents(self) -> List[Dict[str, Any]]:
        """Get all documents from this tool result."""
        return [item for item in self.get_media() if item["type"] == "document"]
    
    
    def to_simple_format_with_placeholders(self) -> tuple[str, Dict[str, Any]]:
        """Convert to simple string format with overall placeholder for non-Claude models."""        
        if not isinstance(self.tool_output, ToolResult):
            return str(self.tool_output), {}
        
        # Check if there's any multimedia content
        has_multimedia = any(
            isinstance(block, (ImageBlock, AudioBlock, VideoBlock, DocumentBlock, FileBlock))
            for block in self.tool_output.content
        )
        
        if not has_multimedia:
            # No multimedia, return text as-is
            result_text = ""
            for block in self.tool_output.content:
                if isinstance(block, TextBlock):
                    result_text += block.text
                elif isinstance(block, ToolOutputPlaceholder):
                    media_type = block.media_type or "media"
                    result_text += f'<ToolOutput type="{media_type}" id="{block.placeholder_id}" doc="{block.doc}" />'
            return result_text, {}
        
        # Has multimedia - use overall placeholder approach
        placeholder_id = str(uuid.uuid4())
        placeholder_text = f"The tool output contains multimedia content. Due to API restrictions, the output will be supplied in the next user message, wrapped by <ToolOutput name=\"{self.tool_name}\" id=\"{placeholder_id}\">...</ToolOutput>."
        
        # Prepare the full content for the follow-up message
        follow_up_content = {
            "id": placeholder_id,
            "tool_name": self.tool_name,
            "full_content": list(self.tool_output.content)  # Preserve original interleaving order
        }
        
        return placeholder_text, follow_up_content

@dataclass
class UnifiedMessage:
    """A unified message structure for internal handling."""
    role: Role
    content: Optional[str] = None # Text content of the message
    tool_calls: Optional[List[UnifiedToolCall]] = None # If role is 'assistant'
    tool_results: Optional[List[UnifiedToolResult]] = None # If role is 'tool'

    # Optional metadata
    timestamp: float = field(default_factory=time.time)
    token_count: Optional[int] = None # If available from API response
    latency_ms: Optional[float] = None # Time taken for API call or tool execution

    def __str__(self) -> str:
        parts = [f"Role: {self.role}"]
        if self.content:
            parts.append(f"Content: '{self.content[:100]}{'...' if len(self.content) > 100 else ''}'")
        if self.tool_calls:
            parts.append(f"Tool Calls: {[tc.tool_name for tc in self.tool_calls]}")
        if self.tool_results:
            parts.append(f"Tool Results: {[tr.tool_name for tr in self.tool_results]}")
        return f"UnifiedMessage({', '.join(parts)})"

# --- Agent Configuration ---
class AgentConfig(BaseModel):
    """Base configuration for all agents."""
    model: str
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    debug_mode: bool = False
    max_tool_iterations: int = 128  # Maximum number of tool iterations per turn

class OpenAIAgentConfig(AgentConfig):
    """OpenAI specific configuration."""
    api_base: Optional[str] = None # For proxy services
    use_responses_api: bool = True # Default to the newer Responses API

class GeminiAgentConfig(AgentConfig):
    """Gemini specific configuration."""
    # Add Gemini specific params if any (e.g., safety settings)
    pass

class ClaudeAgentConfig(AgentConfig):
    """Claude specific configuration."""
    # Anthropic tool versioning might be relevant here
    tool_version: str = "2025-01-24" # Or "2024-10-22"
    thinking_budget: Optional[int] = None # For Claude 3.7 Sonnet

# --- Agent State ---
# Could be used to store conversation history summary, etc.
@dataclass
class AgentState:
    """Holds the state managed by the BaseAgent."""
    conversation_summary: str = ""
    turn_count: int = 0
    total_tokens_used: int = 0
    # Add other state variables as needed (e.g., performance metrics)


