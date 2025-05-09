import asyncio
import logging
import os
import json
import time
from typing import List, Dict, Any, Optional, Tuple

from google import genai
from google.genai.types import (
    Content,
    ContentDict,
    ContentOrDict,
    UserContent,
    ModelContent,
    Tool,
    FunctionDeclaration,
    Part,
    FunctionCall,
    FunctionResponse,
    ToolConfig,
    FunctionCallingConfig,
    GenerateContentConfig,
    GenerateContentResponse,
)

from .base import BaseAgent
from .templates import format_conversation_summary
from .types import (
    GeminiAgentConfig,
    UnifiedToolCall,
    UnifiedToolResult,
)
from ..tools.base import BaseTool

logger = logging.getLogger(__name__)

class GeminiAgent(BaseAgent):
    """
    Agent implementation for interacting with Google's Gemini models.
    """

    def __init__(
        self,
        config: GeminiAgentConfig,
        tools: Optional[List[type[BaseTool]]] = None,
        history_db_path: str = None,
    ):
        super().__init__(config, tools, history_db_path)
        self.config: GeminiAgentConfig = config

        # Configure Gemini client
        api_key = self.config.api_key
        if not api_key:
            raise ValueError("Gemini API key is required but not provided or found in environment variables.")

        self.client = genai.Client(api_key=api_key)
        self.tool_definitions = self._get_tool_definitions()

    def _get_tool_definitions(self) -> Optional[Tool]:
        """
        Formats tool definitions for the Gemini API.
        Called once during initialization.

        Returns:
            A GeminiTool object containing function declarations.
        """
        functions: List[FunctionDeclaration] = []
        for name, tool in self.tools.items():
            tool_def = tool.get_definition()

            functions.append(FunctionDeclaration(
                name=name,
                description=tool_def.description,
                parameters=tool_def.input_schema,
            ))

        if not functions:
            return None # No tools defined

        return Tool(function_declarations=functions)

    def _get_tool_config(
        self,
        force_tool_name: Optional[str] = None
    ) -> Optional[ToolConfig]:
        """
        Configures tool settings for the Gemini API.
        Called for each API call to handle dynamic configs.

        Args:
            force_tool_name: If specified, configure tool_config to force this tool ('ANY' mode).

        Returns:
            The ToolConfig object for the API call.
        """
        # Configure tool calling mode
        tool_config_mode: FunctionCallingConfig.Mode = "AUTO"
        allowed_function_names: Optional[List[str]] = None

        if force_tool_name:
            if force_tool_name not in self.tools:
                logger.error(f"Attempted to force tool '{force_tool_name}', but it's not registered.")
                # Fallback to auto
            else:
                tool_config_mode = "ANY" # Force a tool call
                # Specify exactly which tool to call
                allowed_function_names = [force_tool_name]
                logger.debug(f"Forcing tool choice '{force_tool_name}' via allowed_function_names")

        tool_config = ToolConfig(
             function_calling_config=FunctionCallingConfig(
                 mode=tool_config_mode,
                 allowed_function_names=allowed_function_names 
             )
        )
        logger.debug(f"Gemini Tool Config Mode: {tool_config_mode}, Allowed functions: {allowed_function_names}")

        return tool_config

    async def _execute_provider_api_call(
        self,
        messages: List[ContentOrDict],
        tools: Optional[Tool],
        tool_config: Optional[ToolConfig]
    ) -> Tuple[GenerateContentResponse, Dict[str, int]]:
        """
        Executes the actual API call to Gemini.
        Returns the response object and a dictionary with token usage.
        """
        generate_content_config = GenerateContentConfig(
            tools=[tools],
            tool_config=tool_config,
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_tokens
        )

        response: GenerateContentResponse = await self.client.aio.models.generate_content(
            model=self.config.model,
            contents=messages,
            config=generate_content_config,
        )

        # Extract token usage
        usage_info = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        if response.usage_metadata:
            usage_info["input_tokens"] = response.usage_metadata.prompt_token_count
            usage_info["output_tokens"] = response.usage_metadata.candidates_token_count
            usage_info["total_tokens"] = response.usage_metadata.total_token_count
            
        return response, usage_info

    def _verify_tool_included(self, api_tools: Optional[Tool], summary_tool_name: str) -> bool:
        """
        Verify that the summary tool is included in the Gemini tool definitions.
        """
        if not api_tools or not isinstance(api_tools, Tool):
            return False
            
        for decl in api_tools.function_declarations:
            if decl.name == summary_tool_name:
                return True
        return False

    def _parse_api_response(
        self,
        response: GenerateContentResponse
    ) -> Tuple[Optional[str], Optional[List[UnifiedToolCall]], Optional[str]]:
        """
        Parses the Gemini API response (GenerateContentResponse).
        """
        try:
            if not response.candidates:
                # Check for prompt feedback if response was blocked
                block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else "Unknown"
                logger.error(f"Gemini response blocked or missing candidates. Reason: {block_reason}")
                return None, None, f"Response blocked or empty. Reason: {block_reason}"

            candidate = response.candidates[0]

            finish_reason = candidate.finish_reason.name
            if finish_reason not in ["STOP", "MAX_TOKENS"]:
                 logger.warning(f"Gemini response finished with reason: {finish_reason}")
                 if finish_reason == "SAFETY":
                      safety_ratings = candidate.safety_ratings
                      logger.error(f"Response blocked due to safety settings: {safety_ratings}")
                      return None, None, f"Response blocked by safety filters: {safety_ratings}"

            assistant_text: Optional[str] = None
            tool_calls: Optional[List[UnifiedToolCall]] = None

            for part in candidate.content.parts:
                if part.text:
                    assistant_text = part.text
                    logger.debug("Parsed text content from Gemini response.")
                elif part.function_call:
                    fc = part.function_call
                    logger.debug(f"Parsed function call: {fc.name}")
                    if tool_calls is None:
                        tool_calls = []

                    tool_input = dict(fc.args) if fc.args else {}

                    call_id = f"gemini_call_{fc.name}_{len(tool_calls)}"

                    tool_calls.append(UnifiedToolCall(
                        id=call_id,
                        tool_name=fc.name,
                        tool_input=tool_input
                    ))
                else:
                    logger.warning(f"Unhandled part type in Gemini response: {type(part)}")

            return assistant_text, tool_calls, None

        except Exception as e:
            logger.exception(f"Error parsing Gemini response: {e}")
            return None, None, f"Error parsing response: {e}"


    def _format_tool_results(
        self,
        tool_results: List[UnifiedToolResult]
    ) -> List[Part]:
        """
        Formats tool execution results into Gemini Part format containing FunctionResponse.
        These parts will be added to the *next* 'user' message sent to the API.
        """
        parts: List[Part] = []
        for result in tool_results:
            try:
                response_content: Dict[str, Any]
                if result.is_error:
                     response_content = {"error": str(result.tool_output)}
                elif isinstance(result.tool_output, (dict, list, str, int, float, bool, type(None))):
                     response_content = {"result": result.tool_output}
                else:
                     response_content = {"result": str(result.tool_output)}

                tool_name = result.tool_name

                parts.append(Part.from_function_response(
                    name=tool_name,
                    response=response_content
                ))
                logger.debug(f"Formatted tool result for {tool_name} into FunctionResponse part.")

            except Exception as e:
                 logger.exception(f"Error formatting tool result {result.tool_name} for Gemini: {e}")

        return parts

    def _append_assistant_message(
        self, 
        current_native_messages: List[ContentOrDict], 
        api_response: GenerateContentResponse
    ) -> None:
        """
        Appends the assistant's message from the API response to the native messages list.
        
        Args:
            current_native_messages: The list of native format messages to append to
            api_response: The raw API response object containing the assistant's message
        """
        if api_response.candidates:
            current_native_messages.append(api_response.candidates[0].content)
        else:
            logger.warning("Could not extract assistant message in native format to append (no candidates).")
        
    def _append_tool_results(
        self,
        current_native_messages: List[ContentOrDict],
        tool_results: List[UnifiedToolResult]
    ) -> None:
        """
        Appends tool execution results to the native messages list.
        
        Args:
            current_native_messages: The list of native format messages to append to
            tool_results: The unified tool results to format and append
        """
        native_tool_result_parts = self._format_tool_results(tool_results)
        if native_tool_result_parts:
            current_native_messages.append(UserContent(native_tool_result_parts))
            logger.debug("Appended tool results (FunctionResponse parts) as 'user' role to native messages.")
        
    def _append_warning_message(
        self,
        current_native_messages: List[ContentOrDict],
        warning_message: str
    ) -> None:
        """
        Appends a warning message to the native messages list.
        
        Args:
            current_native_messages: The list of native format messages to append to
            warning_message: The warning message to append
        """
        current_native_messages.append(UserContent([Part(text=f"[Warning: {warning_message}]")]))

    def _format_initial_messages(
        self,
        user_input: str
    ) -> List[Content]:
        """
        Formats the initial messages for a turn for the Gemini API.
        """
        native_messages: List[Content] = []

        # Add history summary
        if self.state.conversation_summary:
            logger.debug(f"Adding summary as preamble: {self.state.conversation_summary[:100]}...")
            native_messages.append(UserContent([
                Part(text=format_conversation_summary(self.state.conversation_summary))
            ]))

        # Add current user input
        native_messages.append(UserContent([Part(text=user_input)]))

        return native_messages

