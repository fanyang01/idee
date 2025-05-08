from dataclasses import dataclass, field
import time
from typing import List, Dict, Any, Literal, Optional, Union
from pydantic import BaseModel

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
    """Represents the result of executing a tool."""
    call_id: str # ID of the tool call this result corresponds to
    tool_name: str
    tool_output: Any # The data returned by the tool's run method
    is_error: bool = False # Flag to indicate if the tool execution failed

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


