import asyncio
import logging
import os
import json
import time
from typing import List, Dict, Any, Optional, Tuple, Union, cast

# Use v1 OpenAI library structure
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionMessage,
    ChatCompletionAssistantMessageParam,
    ChatCompletionToolMessageParam,
)

# Import types for the newer Responses API if available and chosen
from openai.types.responses import (
    Response,
    ResponseInputItemParam,
    ResponseOutputItem,
    ToolChoiceOptions,
    ToolChoiceFunctionParam,
    ToolParam,
    FunctionToolParam,
)
from openai.types.responses.response_create_params import (
    ToolChoice
)


from .base import BaseAgent
from .templates import format_conversation_summary
from .types import (
    AgentConfig,
    OpenAIAgentConfig,
    UnifiedMessage,
    UnifiedToolCall,
    UnifiedToolResult,
    Role,
)
from ..tools.base import BaseTool

logger = logging.getLogger(__name__)

class GPTAgent(BaseAgent):
    """
    Agent implementation for interacting with OpenAI's GPT models
    using the Chat Completions API or the newer Responses API.
    """

    def __init__(
        self,
        config: OpenAIAgentConfig,
        tools: Optional[List[type[BaseTool]]] = None,
        history_db_path: str = None,
    ):
        super().__init__(config, tools, history_db_path) # Pass config up
        self.config: OpenAIAgentConfig = config # Ensure type hint correctness

        # Initialize OpenAI client
        api_key = self.config.api_key
        if not api_key:
            raise ValueError("OpenAI API key is required but not provided or found in environment variables.")

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.config.api_base, # For proxies like OpenRouter
        )
        logger.info(f"OpenAI client initialized. Using Responses API: {self.config.use_responses_api}")
        logger.info(f"Target model: {self.config.model}")

        # Initialize tool definitions
        self.tool_definitions = self._get_tool_definitions()

    def _get_tool_definitions(self) -> Union[List[ChatCompletionToolParam], List[ToolParam]]:
        """
        Formats tool definitions for the OpenAI API.
        Called once during initialization.

        Returns:
            List of tool definitions formatted for the OpenAI API.
        """
        formatted_tools = []
        for tool in self.tools.values():
            tool_def = tool.get_definition()

            if self.config.use_responses_api:
                formatted_tools.append(FunctionToolParam(
                    type="function",
                    name=tool_def.name,
                    parameters=tool_def.parameters,
                    description=tool_def.description,
                    strict=tool_def.strict,
                ))
            else:
                formatted_tools.append(ChatCompletionToolParam(
                    type="function",
                    function=tool_def.to_dict()
                ))
        return formatted_tools

    def _get_tool_config(
        self,
        force_tool_name: Optional[str] = None
    ) -> Optional[Union[ChatCompletionToolChoiceOptionParam, ToolChoice]]:
        """
        Configures tool settings for the OpenAI API.
        Called for each API call to handle dynamic configs.

        Args:
            force_tool_name: If specified, configure tool_choice to force this tool.

        Returns:
            The OpenAI-specific tool_choice parameter.
        """
        tool_choice = "auto"
        if force_tool_name:
            if force_tool_name not in self.tools:
                 logger.error(f"Attempted to force tool '{force_tool_name}', but it's not registered. Falling back to auto.")
                 tool_choice = "auto"
            else:
                tool_choice = {"type": "function", "name": force_tool_name} if self.config.use_responses_api \
                    else {"type": "function", "function": {"name": force_tool_name}}
                logger.debug(f"Forcing tool choice: {tool_choice}")

        # Cast to expected types for the API call
        return cast(ToolChoice, tool_choice) if self.config.use_responses_api \
            else cast(ChatCompletionToolChoiceOptionParam, tool_choice)

    async def _execute_provider_api_call(
        self,
        messages: Union[List[ChatCompletionToolMessageParam], List[ResponseInputItemParam]],
        tools: Union[List[ChatCompletionToolParam], List[ToolParam]],
        tool_choice: Optional[Union[ChatCompletionToolChoiceOptionParam, ToolChoice]] = None,
    ) -> Tuple[Union[ChatCompletion, Response], Dict[str, int]]:
        """
        Executes the actual API call to OpenAI.
        Returns the response object and a dictionary with token usage.
        """
        response: Any
        if self.config.use_responses_api:
            # --- Using the new Responses API ---
            api_messages = cast(List[ResponseInputItemParam], messages)
            api_tools = cast(List[ToolParam], tools)
            api_tool_choice = cast(Optional[ToolChoice], tool_choice)
            
            logger.debug("Calling OpenAI Responses API...")
            response = await self.client.responses.create(
                input=api_messages,
                model=self.config.model,
                tools=api_tools,
                tool_choice=api_tool_choice,
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens,
            )
        else:
            # --- Using the standard Chat Completions API ---
            api_messages = cast(List[ChatCompletionMessageParam], messages)
            api_tools = cast(List[ChatCompletionToolParam], tools)
            api_tool_choice = cast(Optional[ChatCompletionToolChoiceOptionParam], tool_choice)

            logger.debug("Calling OpenAI Chat Completions API...")
            response = await self.client.chat.completions.create(
                messages=api_messages,
                model=self.config.model,
                tools=api_tools,
                tool_choice=api_tool_choice,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

        # Extract token usage
        usage_info = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        if response.usage:
            usage_info["input_tokens"] = response.usage.prompt_tokens or 0
            # Use completion_tokens for ChatCompletion, output_tokens for Responses API?
            # Assuming usage structure is consistent enough or check type.
            # Let's assume completion_tokens is the standard attribute name here.
            usage_info["output_tokens"] = response.usage.completion_tokens or 0 
            usage_info["total_tokens"] = response.usage.total_tokens or 0
            
        return response, usage_info

    def _parse_api_response(
        self,
        response: Union[ChatCompletion, Response]
    ) -> Tuple[Optional[str], Optional[List[UnifiedToolCall]], Optional[str]]:
        """
        Parses the OpenAI API response (ChatCompletion or Response format).
        """
        try:
            # Check for Response API format
            if hasattr(response, 'object') and response.object == 'response':
                return self._parse_responses_api_format(response)
            # Handle traditional ChatCompletion format
            elif isinstance(response, ChatCompletion):
                return self._parse_chat_completion_format(response)
            else:
                logger.error(f"Unexpected OpenAI response type: {type(response)}")
                return None, None, f"Unexpected response type: {type(response)}"

        except Exception as e:
            logger.exception(f"Error parsing OpenAI response: {e}")
            return None, None, f"Error parsing response: {e}"
            
    def _parse_responses_api_format(
        self,
        response: Response
    ) -> Tuple[Optional[str], Optional[List[UnifiedToolCall]], Optional[str]]:
        """
        Parses Response API format responses.
        """
        if response.error:
            error_msg = f"OpenAI Response error: {response.error.message}"
            logger.error(error_msg)
            return None, None, error_msg
        
        if response.status != 'completed':
            logger.warning(f"Response status is not 'completed': {response.status}")
            # Could still try to extract content, but with a warning
        
        if not response.output:
            logger.error("OpenAI Response missing 'output' field.")
            return None, None, "Invalid response: No output found."
        
        # Extract assistant message text
        assistant_text = ""
        for output_item in response.output:
            if output_item.type == "message" and output_item.role == "assistant":
                for content_item in output_item.content:
                    if content_item.type == "output_text":
                        assistant_text += content_item.text
        
        # Extract tool calls
        tool_calls: Optional[List[UnifiedToolCall]] = None
        for output_item in response.output:
            if output_item.type == "function_tool_call":
                if tool_calls is None:
                    tool_calls = []
                
                try:
                    args_str = output_item.arguments
                    if not isinstance(args_str, str):
                        logger.error(f"Tool call arguments are not a string: {args_str}")
                        continue
                            
                    tool_input = json.loads(args_str)
                    tool_calls.append(UnifiedToolCall(
                        id=output_item.id,
                        tool_name=output_item.name,
                        tool_input=tool_input
                    ))
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON arguments for tool {output_item.name}: {e}")
                    logger.error(f"Raw arguments string: {args_str}")
                except Exception as e:
                    logger.exception(f"Error processing tool call {output_item.name}: {e}")
        
        return assistant_text, tool_calls, None
            
    def _parse_chat_completion_format(
        self,
        response: ChatCompletion
    ) -> Tuple[Optional[str], Optional[List[UnifiedToolCall]], Optional[str]]:
        """
        Parses ChatCompletion format responses.
        """
        if not response.choices:
            logger.error("OpenAI response missing 'choices' field.")
            return None, None, "Invalid response: No choices found."

        message = response.choices[0].message

        assistant_text = message.content
        tool_calls: Optional[List[UnifiedToolCall]] = None

        if message.tool_calls:
            tool_calls = []
            for tc in message.tool_calls:
                if tc.type == "function":
                    try:
                        # Ensure arguments is a string before loading
                        args_str = tc.function.arguments
                        if not isinstance(args_str, str):
                            logger.error(f"Tool call arguments are not a string: {args_str}")
                            # Skip this tool call or raise error? Skipping for now.
                            continue

                        tool_input = json.loads(args_str)
                        tool_calls.append(UnifiedToolCall(
                            id=tc.id, # Use the tool_call_id provided by OpenAI
                            tool_name=tc.function.name,
                            tool_input=tool_input
                        ))
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON arguments for tool {tc.function.name}: {e}")
                        logger.error(f"Raw arguments string: {tc.function.arguments}")
                        # Decide how to handle - skip tool, return error? Skip for now.
                    except Exception as e:
                        logger.exception(f"Error processing tool call {tc.function.name}: {e}")

                else:
                    logger.warning(f"Unsupported tool call type: {tc.type}")

        return assistant_text, tool_calls, None

    def _format_tool_results(
        self,
        tool_results: List[UnifiedToolResult]
    ) -> List[Dict[str, Any]]: # Returns list of ChatCompletionToolMessageParam or Response formatted messages
        """
        Formats tool execution results into the appropriate message format.
        
        For ChatCompletion API: Uses "role": "tool" format
        For Response API: Uses "type": "function_call_output" format
        """
        formatted_results = []
        for result in tool_results:
            try:
                # Ensure output is JSON serializable string
                if isinstance(result.tool_output, (dict, list)):
                    content = json.dumps(result.tool_output)
                else:
                    content = str(result.tool_output) # Convert other types to string
                
                if self.config.use_responses_api:
                    # Format for Response API
                    formatted_results.append({
                        "type": "function_call_output",
                        "call_id": result.call_id,
                        "output": content
                    })
                else:
                    # Format for ChatCompletion API
                    formatted_results.append({
                        "role": "tool",
                        "tool_call_id": result.call_id,
                        "content": content,
                    })
            except Exception as e:
                 logger.exception(f"Error formatting tool result for call_id {result.call_id}: {e}")
                 
                 if self.config.use_responses_api:
                     formatted_results.append({
                         "type": "function_call_output",
                         "call_id": result.call_id,
                         "output": f"Error formatting tool result: {e}"
                     })
                 else:
                     formatted_results.append({
                         "role": "tool",
                         "tool_call_id": result.call_id,
                         "content": f"Error formatting tool result: {e}",
                     })

        return formatted_results

    def _verify_tool_included(self, api_tools: List[Dict[str, Any]], summary_tool_name: str) -> bool:
        """
        Verify that the summary tool is included in the OpenAI tool definitions.
        """
        for tool in api_tools:
            if (tool.get('type') == 'function' and 
                tool.get('function', {}).get('name') == summary_tool_name):
                return True
        return False

    def _append_assistant_message(
        self, 
        current_native_messages: List[Dict[str, Any]], 
        api_response: Any  # ChatCompletion or Response object
    ) -> None:
        """
        Appends the assistant's message from the API response to the native messages list.
        Handles both ChatCompletion and Responses API formats.
        
        Args:
            current_native_messages: The list of native format messages to append to
            api_response: The raw API response object containing the assistant's message
        """
        if self.config.use_responses_api:
            # Format for Response API
            if api_response.output:
                current_native_messages.extend([
                    {"role": el.role, "content": el.content} for el in api_response.output
                ])
                logger.debug("Appended assistant response message (Response API format) to native messages.")
            else:
                logger.warning("Could not find assistant message in Response API output to append.")
        else:
            # Format for ChatCompletion API
            if api_response.choices and len(api_response.choices) > 0:
                current_native_messages.append(api_response.choices[0].message)
                logger.debug("Appended assistant response message (ChatCompletion format) to native messages.")
            else:
                logger.warning("No choices found in ChatCompletion response to append.")

    def _append_tool_results(
        self,
        current_native_messages: List[Dict[str, Any]],
        tool_results: List[UnifiedToolResult]
    ) -> None:
        """
        Appends tool execution results to the native messages list.
        Handles both ChatCompletion and Responses API formats.
        
        Args:
            current_native_messages: The list of native format messages to append to
            tool_results: The unified tool results to format and append
        """
        formatted_results = self._format_tool_results(tool_results)
        
        if not formatted_results:
            logger.warning("No formatted tool results to append to native messages.")
            return
            
        if self.config.use_responses_api:
            # For Responses API, each function_call_output is appended directly
            for result in formatted_results:
                current_native_messages.append(result)
            logger.debug(f"Appended {len(formatted_results)} function_call_output items to native messages.")
        else:
            # For ChatCompletion API, each tool message is appended directly 
            for result in formatted_results:
                current_native_messages.append(result)
            logger.debug(f"Appended {len(formatted_results)} tool messages to native messages.")

    def _append_warning_message(
        self,
        current_native_messages: List[Dict[str, Any]],
        warning_message: str
    ) -> None:
        """
        Appends a warning message to the native messages list.
        Handles both ChatCompletion and Responses API formats.
        
        Args:
            current_native_messages: The list of native format messages to append to
            warning_message: The warning message to append
        """
        if self.config.use_responses_api:
            # Format for Response API - append as assistant message
            warning_item = {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": f"[Warning: {warning_message}]"
                    }
                ]
            }
            current_native_messages.append(warning_item)
        else:
            # Format for ChatCompletion API
            current_native_messages.append({
                "role": "assistant",
                "content": f"[Warning: {warning_message}]"
            })
            
        logger.debug(f"Appended warning message to native messages: {warning_message}")

    def _format_initial_messages(
        self,
        user_input: str
    ) -> List[Dict[str, Any]]:
        """
        Formats the initial messages for a turn for the OpenAI API.
        """
        native_messages: List[Dict[str, Any]] = []

        # Add system prompt (optional, configure if needed)
        # native_messages.append({"role": "system", "content": "You are Idee, a helpful AI assistant..."})

        # Add history summary
        if self.state.conversation_summary:
            logger.debug(f"Adding summary to messages: {self.state.conversation_summary[:100]}...")
            native_messages.append({
                "role": "user",
                "content": format_conversation_summary(self.state.conversation_summary)
            })

        # Add current user input
        native_messages.append({"role": "user", "content": user_input})

        return native_messages


