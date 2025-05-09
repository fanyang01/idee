import asyncio
import logging
import os
import json
import time
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
from ..tools.base import BaseTool

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

        # Precompute tool definitions
        self.tool_definitions = self._get_tool_definitions()

    def _get_tool_definitions(self) -> List[ToolUnionParam]:
        """
        Formats tool definitions for the Anthropic API.
        Called once during initialization.

        Returns:
            List of tool definitions formatted for the Anthropic API.
        """
        formatted_tools: List[ToolUnionParam] = []
        for tool in self.tools.values():
            tool_def = tool.get_definition()

            # Anthropic's format
            formatted_tools.append(ToolParam(
                name=tool_def.name,
                description=tool_def.description,
                input_schema=tool_def.input_schema,
            ))

        return formatted_tools

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
        api_messages = cast(List[MessageParam], messages)
        api_tools = cast(List[ToolParam], tools)

        # Prepare optional 'thinking' parameter for models like Claude 3.7 Sonnet
        thinking_param = None
        if self.config.thinking_budget and self.config.thinking_budget > 0:
             thinking_param = {"type": "enabled", "budget_tokens": self.config.thinking_budget}
             logger.debug(f"Enabling Claude 'thinking' with budget: {self.config.thinking_budget}")

        logger.debug(f"Calling Anthropic API with {len(api_messages)} messages.")

        # Use the tool_choice parameter if provided
        kwargs = {
            "model": self.config.model,
            "system": self.system_prompt, # System prompt passed separately
            "messages": api_messages,
            "tools": api_tools,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }
        
        # Add tool_choice if specified
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
            
        # Add thinking parameter if enabled
        if thinking_param:
            kwargs["thinking"] = thinking_param

        response: Message = await self.client.messages.create(**kwargs)

        # Extract token usage
        usage_info = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        if response.usage:
            usage_info["input_tokens"] = response.usage.input_tokens
            usage_info["output_tokens"] = response.usage.output_tokens
            # Claude provides input/output, total can be derived
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
        response: Message # Anthropic Message object
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

    def _format_tool_results(
        self,
        tool_results: List[UnifiedToolResult]
    ) -> List[ToolResultBlockParam]:
        """
        Formats tool execution results into the Anthropic ToolResultBlockParam format.
        """
        blocks: List[ToolResultBlockParam] = []
        for result in tool_results:
            try:
                output_content: Union[str, Dict[str, Any]]
                if result.is_error:
                     output_content = {"error": str(result.tool_output)}
                elif isinstance(result.tool_output, (dict, list, str, int, float, bool, type(None))):
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
        
    def _append_tool_results(
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
        native_tool_result_blocks = self._format_tool_results(tool_results)
        
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
