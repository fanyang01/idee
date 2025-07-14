import asyncio
import base64
import logging
import os
import json
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union, cast

# Import Anthropic library
from anthropic import Anthropic, AsyncAnthropic
from anthropic.types import (
    MessageParam, # Type hint for messages list items
    ToolUnionParam, # Type hint for tools list items
    ToolParam, # Type hint for tool definitions
    TextBlock,
    ToolUseBlock,
    ToolResultBlockParam, # Type hint for formatting tool results
    Message, # Type hint for the response object
)
from anthropic.types.beta import (
    BetaToolUnionParam,
    BetaMessageParam,
    BetaMessage,
)

from .base import BaseAgent
from .templates import format_conversation_summary
from .types import (
    AgentConfig,
    ClaudeAgentConfig,
    UnifiedMessage,
    UnifiedToolCall,
    UnifiedToolResult,
    Role,
)
from ..tools.base import BaseTool, ToolResult, TextBlock, FileBlock, ImageBlock, AudioBlock, VideoBlock, DocumentBlock
from ..tools.media_utils import detect_media_type, read_file_as_base64

logger = logging.getLogger(__name__)

# Mapping from our roles to Claude roles
CLAUDE_ROLE_MAP = {
    "user": "user",
    "assistant": "assistant",
    "tool": "user", # Tool results are sent in a user message containing tool_result blocks
    "system": None # Handled by the 'system' parameter in the API call
}

class ClaudeAgent(BaseAgent):
    """
    Agent implementation for interacting with Anthropic's Claude models.
    """

    def __init__(
        self,
        config: ClaudeAgentConfig,
        tools: Optional[List[type[BaseTool]]] = None,
        history_db_path: str = None,
    ):
        super().__init__(config, tools, history_db_path)
        self.config: ClaudeAgentConfig = config

        # Configure Anthropic client
        api_key = self.config.api_key
        if not api_key:
            raise ValueError("Anthropic API key is required but not provided or found in environment variables.")

        self.client = AsyncAnthropic(api_key=api_key)
        logger.info(f"Anthropic client configured for model: {self.config.model}")
        logger.info(f"Using tool schema version: {self.config.tool_version}")

        # Define system prompt (can be customized)
        self.system_prompt = "You are Codemate, an AI coding assistant. Use the available tools to help the user with their requests. Think step-by-step if needed. When summarizing, be concise and capture the key outcomes and remaining tasks."

        # Configure vendor-specific tool behaviors
        self._configure_vendor_tools()
        
        # Precompute tool definitions
        self.tool_definitions = self._get_tool_definitions()
        self.has_vendor_tools = any(tool.has_vendor_spec("anthropic", self.config.model) for tool in self.tools.values())
        
        # Determine which beta flags are needed based on vendor specs
        self.beta_flags = self._get_beta_flags()
        
        if self.has_vendor_tools:
            logger.info(f"Using beta API with vendor tools (flags: {self.beta_flags})")
        else:
            logger.info(f"Using standard API")

    def _configure_vendor_tools(self):
        """Configure vendor-specific tool behaviors."""
        for tool in self.tools.values():
            # Enable persistent session for BashTool when using vendor specs
            if tool.name == "bash" and tool.has_vendor_spec("anthropic", self.config.model):
                if hasattr(tool, 'enable_persistent_session'):
                    tool.enable_persistent_session()
                    logger.debug("Enabled persistent session for bash tool")

    def _get_tool_definitions(self) -> List[Union[ToolUnionParam, BetaToolUnionParam]]:
        """
        Formats tool definitions for the Anthropic API.
        Uses vendor-specific format when available, otherwise uses regular format.
        Called once during initialization.

        Returns:
            List of tool definitions formatted for the Anthropic API.
        """
        formatted_tools: List[Union[ToolUnionParam, BetaToolUnionParam]] = []
        
        for tool in self.tools.values():
            # Check if this tool has Anthropic vendor specification for this model
            if tool.has_vendor_spec("anthropic", self.config.model):
                vendor_spec = tool.get_vendor_spec("anthropic", self.config.model)
                if vendor_spec:
                    formatted_tools.append(vendor_spec)
                    logger.debug(f"Added vendor tool: {tool.name} (spec: {vendor_spec})")
                    continue
            
            # Regular tool handling
            tool_def = tool.get_definition()
            formatted_tools.append(ToolParam(
                name=tool_def.name,
                description=tool_def.description,
                input_schema=tool_def.input_schema,
            ))

        return formatted_tools

    def _get_beta_flags(self) -> List[str]:
        """Determine which beta flags are needed based on vendor tool specs."""
        beta_flags = set()
        
        for tool in self.tools.values():
            if tool.has_vendor_spec("anthropic", self.config.model):
                vendor_spec = tool.get_vendor_spec("anthropic", self.config.model)
                if vendor_spec and "type" in vendor_spec:
                    spec_type = vendor_spec["type"]
                    
                    # Map spec types to beta flags
                    if spec_type == "text_editor_20250429":
                        beta_flags.add("str-replace-based-edit-tool")
                    elif spec_type == "text_editor_20250124":
                        beta_flags.add("text-editor-20250124")
                    elif spec_type == "text_editor_20241022":
                        beta_flags.add("text-editor-20241022")
                    elif spec_type == "computer_20241022":
                        beta_flags.add("computer-use-2024-10-22")
                    elif spec_type == "computer_20250124":
                        beta_flags.add("computer-use-2025-01-24")
                    elif spec_type == "bash_20250124":
                        beta_flags.add("bash-tool-2025-01-24")
                        
        return list(beta_flags)

    def _get_tool_config(
        self,
        force_tool_name: Optional[str] = None
    ) -> Optional[Any]:
        """
        Configures tool settings for the Anthropic API.
        Called for each API call to handle dynamic configs.

        Args:
            force_tool_name: If specified, configure tool_choice to force this tool.

        Returns:
            Tool choice parameter formatted for Claude.
        """
        # Claude requires a different format for tool_choice
        tool_choice = None
        if force_tool_name:
            if force_tool_name not in self.tools:
                logger.error(f"Attempted to force tool '{force_tool_name}', but it's not registered.")
            else:
                # Format for Claude's tool choice parameter
                tool_choice = {"type": "tool", "name": force_tool_name}
                logger.debug(f"Forcing tool choice: {tool_choice}")

        return cast(Optional[Any], tool_choice)

    async def _execute_provider_api_call(
        self,
        messages: List[MessageParam],
        tools: List[ToolUnionParam],
        tool_choice: Optional[Any] = None # Claude's tool_choice parameter
    ) -> Tuple[Any, Dict[str, int]]:
        """
        Executes the actual API call to Anthropic.
        Returns the response object and a dictionary with token usage.
        """
        # Prepare optional 'thinking' parameter for models like Claude 3.7 Sonnet
        thinking_param = None
        if self.config.thinking_budget and self.config.thinking_budget > 0:
             thinking_param = {"type": "enabled", "budget_tokens": self.config.thinking_budget}
             logger.debug(f"Enabling Claude 'thinking' with budget: {self.config.thinking_budget}")

        logger.debug(f"Calling Anthropic API with {len(messages)} messages.")

        # Decide whether to use beta API based on vendor tools
        if self.has_vendor_tools:
            # Use beta API for vendor tools
            beta_messages = cast(List[BetaMessageParam], messages)

            kwargs = {
                "model": self.config.model,
                "system": [self.system_prompt],
                "messages": beta_messages,
                "tools": self.tool_definitions,
                "max_tokens": self.config.max_tokens,
                "betas": self.beta_flags,
            }
            
            if tool_choice:
                kwargs["tool_choice"] = tool_choice
                
            if thinking_param:
                kwargs["extra_body"] = {"thinking": thinking_param}

            logger.debug("Using beta API for vendor tools")
            response: BetaMessage = await self.client.beta.messages.create(**kwargs)
        else:
            # Use regular API
            api_messages = cast(List[MessageParam], messages)
            api_tools = cast(List[ToolParam], tools)

            kwargs = {
                "model": self.config.model,
                "system": self.system_prompt,
                "messages": api_messages,
                "tools": api_tools,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
            }
            
            if tool_choice:
                kwargs["tool_choice"] = tool_choice
                
            if thinking_param:
                kwargs["thinking"] = thinking_param

            response: Message = await self.client.messages.create(**kwargs)

        # Extract token usage
        usage_info = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        if response.usage:
            usage_info["input_tokens"] = response.usage.input_tokens
            usage_info["output_tokens"] = response.usage.output_tokens
            usage_info["total_tokens"] = response.usage.input_tokens + response.usage.output_tokens
            
        return response, usage_info

    def _verify_tool_included(self, api_tools: List[Dict[str, Any]], summary_tool_name: str) -> bool:
        """
        Verify that the summary tool is included in the Claude tool definitions.
        """
        for tool in api_tools:
            if tool.get('name') == summary_tool_name:
                return True
        return False

    def _parse_api_response(
        self,
        response: Union[Message, BetaMessage] # Anthropic Message object
    ) -> Tuple[Optional[str], Optional[List[UnifiedToolCall]], Optional[str]]:
        """
        Parses the Anthropic API response (Message object).
        """
        try:
            stop_reason = response.stop_reason
            if stop_reason not in ["end_turn", "tool_use", "max_tokens"]:
                 logger.warning(f"Anthropic response stopped with reason: {stop_reason}")
                 if stop_reason == "stop_sequence":
                      pass

            assistant_text: Optional[str] = None
            tool_calls: Optional[List[UnifiedToolCall]] = None

            text_parts = []
            for block in response.content:
                if isinstance(block, TextBlock):
                    text_parts.append(block.text)
                elif isinstance(block, ToolUseBlock):
                    logger.debug(f"Parsed tool use block: {block.name}")
                    if tool_calls is None:
                        tool_calls = []

                    tool_input = dict(block.input) if block.input else {}

                    tool_calls.append(UnifiedToolCall(
                        id=block.id,
                        tool_name=block.name,
                        tool_input=tool_input
                    ))
                else:
                    logger.warning(f"Unhandled block type in Anthropic response: {type(block)}")

            if text_parts:
                assistant_text = "".join(text_parts)
                logger.debug("Parsed text content from Anthropic response.")

            if stop_reason == "tool_use" and not tool_calls:
                 logger.warning("Response stop_reason was 'tool_use', but no ToolUseBlocks found in content.")

            return assistant_text, tool_calls, None

        except Exception as e:
            logger.exception(f"Error parsing Anthropic response: {e}")
            return None, None, f"Error parsing response: {e}"

    async def _format_tool_results(
        self,
        tool_results: List[UnifiedToolResult]
    ) -> List[ToolResultBlockParam]:
        """
        Formats tool execution results into the Anthropic ToolResultBlockParam format.
        Claude supports multimodal content natively in tool results.
        """
        blocks: List[ToolResultBlockParam] = []
        for result in tool_results:
            try:
                # Handle multimodal content with Claude's native support
                if result.has_multimodal_content():
                    # Convert to Claude's native multimodal format
                    processed_content = await self._process_multimodal_blocks(result.tool_output)
                    
                    result_block: ToolResultBlockParam = {
                        "type": "tool_result",
                        "tool_use_id": result.call_id,
                        "content": processed_content,
                    }
                else:
                    # For simple tool output
                    output_content: Union[str, Dict[str, Any]]
                    if result.is_error:
                         output_content = {"error": str(result.tool_output)}
                    elif isinstance(result.tool_output, ToolResult):
                         output_content = result.tool_output.get_text_content()
                    elif isinstance(result.tool_output, dict):
                         output_content = result.tool_output
                    else:
                         output_content = str(result.tool_output)

                    result_block: ToolResultBlockParam = {
                        "type": "tool_result",
                        "tool_use_id": result.call_id,
                        "content": output_content,
                    }
                
                blocks.append(result_block)
                logger.debug(f"Formatted tool result for {result.tool_name} (ID: {result.call_id}) for Anthropic.")

            except Exception as e:
                 logger.exception(f"Failed to format tool result {result.tool_name} (ID: {result.call_id}) for Anthropic: {e}")
                 blocks.append({
                     "type": "tool_result",
                     "tool_use_id": result.call_id,
                     "content": f"Error formatting result: {e}",
                     "is_error": True,
                 })

        return blocks

    async def _process_multimodal_blocks(self, tool_output: ToolResult) -> List[Dict[str, Any]]:
        """Convert tool output to Claude API format."""
        result = []
        for block in tool_output.content:
            if isinstance(block, TextBlock):
                result.append({"type": "text", "text": block.text})
            elif isinstance(block, FileBlock):
                # Process FileBlock directly - try upload first, then fallback to base64
                processed_block = await self._process_file_block(block)
                result.append(processed_block)
            elif isinstance(block, (ImageBlock, AudioBlock, VideoBlock, DocumentBlock)):
                # Claude supports different media types with the same structure
                block_type_map = {
                    ImageBlock: "image",
                    AudioBlock: "audio", 
                    VideoBlock: "video",
                    DocumentBlock: "document"
                }
                
                # Use existing source format (base64 or file_path)
                result.append({
                    "type": block_type_map[type(block)],
                    "source": block.source
                })
        return result
    
    async def _process_file_block(self, file_block: FileBlock) -> Dict[str, Any]:
        """Process a FileBlock for Claude by uploading to Files API or converting to base64."""
        
        # Determine media type if not provided
        media_type = file_block.media_type
        if not media_type and file_block.file_path:
            media_type = detect_media_type(file_block.file_path)
        
        # Try to upload for documents
        if self._should_upload_file(file_block, media_type):
            file_id = await self._upload_file(file_block)
            if file_id:
                return {
                    "type": "document",
                    "source": {
                        "type": "file",
                        "file_id": file_id
                    }
                }
        
        # Fallback to base64 for images/other media
        data = file_block.get_base64_data()
        
        # Use appropriate format based on media type
        if media_type and media_type.startswith("image/"):
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": data
                }
            }
        else:
            # For other types, use text representation
            return {
                "type": "text",
                "text": f"[File: {file_block.filename or 'unnamed'} ({media_type or 'unknown type'})]"
            }

    async def _upload_file(self, file_block: FileBlock) -> Optional[str]:
        """Upload a file to Claude Files API and return file ID."""
        if not hasattr(self, 'client') or not self.client:
            logger.warning("Claude client not available for file upload")
            return None
            
        try:
            # Prepare file for upload - API supports multiple input types
            if file_block.file_path:
                # Use existing file path directly
                file_input = Path(file_block.file_path)
            else:
                # Use bytes content with optional filename and media type
                filename = file_block.filename or "file"
                content = file_block.get_bytes_content()
                if file_block.media_type:
                    file_input = (filename, content, file_block.media_type)
                else:
                    file_input = (filename, content)
                
            response = await self.client.beta.files.upload(file=file_input)
            return response.id
    
        except Exception as e:
            logger.error(f"Claude file upload failed: {e}")
            return None

    def _should_upload_file(self, file_block: FileBlock, media_type: str) -> bool:
        """Determine if a file should be uploaded vs embedded for Claude."""
        # Claude's Files API is best for documents and PDFs
        if media_type and media_type.startswith(("application/", "text/")):
            return True
        
        # Upload large files regardless of type
        if file_block.file_path:
            size = Path(file_block.file_path).stat().st_size
            if size > 1024 * 1024:  # > 1MB
                return True
        elif file_block.content and len(file_block.content) > 1024 * 1024:
            return True
            
        return False

    def _append_assistant_message(
        self, 
        current_native_messages: List[Dict[str, Any]], 
        api_response: Message
    ) -> None:
        """
        Appends the assistant's message from the API response to the native messages list.
        
        Args:
            current_native_messages: The list of native format messages to append to
            api_response: The raw API response object containing the assistant's message
        """
        assistant_api_message: MessageParam = {
            "role": "assistant",
            "content": api_response.content
        }
        current_native_messages.append(assistant_api_message)
        logger.debug("Appended assistant response message to native messages.")
        
    async def _append_tool_results(
        self,
        current_native_messages: List[Dict[str, Any]],
        tool_results: List[UnifiedToolResult]
    ) -> None:
        """
        Appends tool execution results to the native messages list.
        
        Args:
            current_native_messages: The list of native format messages to append to
            tool_results: The unified tool results to format and append
        """
        native_tool_result_blocks = await self._format_tool_results(tool_results)
        
        if native_tool_result_blocks:
            tool_results_message: MessageParam = {
                "role": "user",
                "content": native_tool_result_blocks
            }
            current_native_messages.append(tool_results_message)
            logger.debug("Appended tool results message (role 'user') to native messages.")
        
    def _append_warning_message(
        self,
        current_native_messages: List[Dict[str, Any]],
        warning_message: str
    ) -> None:
        """
        Appends a warning message to the native messages list.
        
        Args:
            current_native_messages: The list of native format messages to append to
            warning_message: The warning message to append
        """
        current_native_messages.append({
            "role": "assistant", 
            "content": f"[Warning: {warning_message}]"
        })

    def _format_initial_messages(
        self,
        user_input: str
    ) -> List[Dict[str, Any]]:
        """
        Formats the initial messages for a turn for the Anthropic API.
        """
        native_messages: List[Dict[str, Any]] = [] # Expects Anthropic MessageParam format

        # Add history summary as an user message before the user input
        if self.state.conversation_summary:
            logger.debug(f"Adding summary as preamble: {self.state.conversation_summary[:100]}...")
            native_messages.append({
                "role": "user",
                "content": format_conversation_summary(self.state.conversation_summary)
            })

        # Add current user input
        native_messages.append({"role": "user", "content": user_input})

        return native_messages
