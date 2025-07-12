import logging
import time
import asyncio
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Type, Tuple, Callable

from ..tools.base import BaseTool, ToolError, ToolResult
from ..tools.summary import ConversationSummaryTool
from ..agents.types import (
    UnifiedMessage,
    UnifiedToolCall,
    UnifiedToolResult,
    AgentConfig,
    AgentState,
)
from ..core.history_db import HistoryDB
from ..tools.bash import BashTool
from ..tools.editor import TextEditorTool
from ..tools.history import HistoryTool

logger = logging.getLogger(__name__)

# Define a type for the message observer callback
MessageObserverCallback = Callable[[UnifiedMessage], None]

class BaseAgent(ABC):
    """
    Abstract base class for LLM agents.

    Handles tool registration, dispatch, execution, message logging,
    state management, and defines the core interaction loop structure.
    """

    def __init__(
        self,
        config: AgentConfig,
        tools: Optional[List[Type[BaseTool]]] = None,
        history_db_path: str = None,
        message_observer: Optional[MessageObserverCallback] = None,
    ):
        self.config = config
        self.state = AgentState()
        self.tools: Dict[str, BaseTool] = {}
        self.history_db = HistoryDB(db_path=history_db_path) # HistoryDB is created here
        self.message_observer = message_observer

        # Instantiate and register tools, injecting dependencies as needed
        self._register_tools(tools)

        # Ensure the mandatory summary tool is registered
        if ConversationSummaryTool.name not in self.tools:
             summary_tool_instance = ConversationSummaryTool()
             self.tools[summary_tool_instance.name] = summary_tool_instance
             logger.info(f"Mandatory tool '{ConversationSummaryTool.name}' registered.")

        # Initialize tool definitions once
        self.tool_definitions = self._get_tool_definitions()

        logger.info(f"Agent initialized with model: {config.model}")
        logger.info(f"Registered tools: {list(self.tools.keys())}")
        logger.info(f"Max tool iterations: {config.max_tool_iterations}")

    def _register_tools(self, custom_tools: Optional[List[Type[BaseTool]]]) -> None:
        """Instantiates and registers default and custom tools."""
        # Define default tool classes
        default_tool_classes = [BashTool, TextEditorTool, HistoryTool]

        # Combine default and custom tools, ensuring no duplicates
        all_tool_classes = {tool_cls.name: tool_cls for tool_cls in default_tool_classes}
        if custom_tools:
            for tool_cls in custom_tools:
                if tool_cls.name in all_tool_classes:
                    logger.warning(f"Custom tool '{tool_cls.name}' overrides a default tool.")
                all_tool_classes[tool_cls.name] = tool_cls

        # Instantiate tools, injecting dependencies
        for tool_name, tool_cls in all_tool_classes.items():
            try:
                if tool_cls is HistoryTool:
                    # Inject the history_db instance
                    tool_instance = HistoryTool(history_db=self.history_db)
                # Add elif blocks here for other tools needing dependencies
                # elif tool_cls is SomeOtherTool:
                #    tool_instance = SomeOtherTool(dependency=self.some_dependency)
                else:
                    # Standard instantiation for tools without specific dependencies
                    tool_instance = tool_cls()

                # Register the instantiated tool
                self.tools[tool_instance.name] = tool_instance
                logger.debug(f"Registered tool: {tool_instance.name}")

            except Exception as e:
                logger.error(f"Failed to instantiate or register tool {tool_name}: {e}")

    async def _call_provider_api(
        self,
        messages: List[Any], # List of messages in the API's native format
        tools: List[Dict[str, Any]], # List of tool definitions in the API's native format
        tool_config: Optional[Any] = None # API-specific tool choice mechanism
    ) -> Tuple[Any, Dict[str, Any]]: # (API Response object, Performance metrics dict)
        """
        Calls the specific LLM API via _execute_provider_api_call and tracks metrics.

        Args:
            messages: Conversation history in the native format for the specific API.
            tools: Tool definitions formatted for the specific API.
            tool_config: API-specific mechanism to enforce or suggest tool use.

        Returns:
            A tuple containing:
            - The raw response object from the API library.
            - A dictionary with performance metrics (latency_ms, input_tokens, output_tokens, total_tokens).
        """
        start_time = time.monotonic()
        # Initialize metrics with defaults
        metrics = {"latency_ms": 0.0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        
        try:
            # The actual API call will be implemented by the derived class
            # It now returns the response and token usage info
            response, usage_info = await self._execute_provider_api_call(messages, tools, tool_config)
            
            # Calculate latency
            latency_ms = (time.monotonic() - start_time) * 1000
            metrics["latency_ms"] = latency_ms
            
            # Extract token counts from the usage_info returned by the specific implementation
            metrics["input_tokens"] = usage_info.get("input_tokens", 0)
            metrics["output_tokens"] = usage_info.get("output_tokens", 0)
            metrics["total_tokens"] = usage_info.get("total_tokens", metrics["input_tokens"] + metrics["output_tokens"]) # Calculate total if not provided
            
            logger.debug(f"API Token Usage - Input: {metrics['input_tokens']}, Output: {metrics['output_tokens']}, Total: {metrics['total_tokens']}")
            
            # Update agent's total token count state
            self.state.total_tokens_used += metrics["total_tokens"]
            
            return response, metrics
            
        except Exception as e:
            # Calculate latency even for errors
            latency_ms = (time.monotonic() - start_time) * 1000
            metrics["latency_ms"] = latency_ms
            logger.exception(f"Error calling LLM API after {latency_ms:.2f} ms: {e}")
            raise
    
    @abstractmethod
    async def _execute_provider_api_call(
        self,
        messages: List[Any],
        tools: List[Dict[str, Any]],
        tool_choice: Optional[Any] = None
    ) -> Tuple[Any, Dict[str, int]]:
        """
        Execute the actual API call to the LLM provider.
        This is implemented by each specific agent class.
        
        Returns:
            A tuple containing:
            - The raw response object from the API.
            - A dictionary with token usage: {"input_tokens": int, "output_tokens": int, "total_tokens": int}
              (total_tokens is optional, will be calculated if missing).
        """
        pass

    @abstractmethod
    def _parse_api_response(
        self,
        response: Any # The raw response object from _call_provider_api
    ) -> Tuple[Optional[str], Optional[List[UnifiedToolCall]], Optional[str]]:
        """
        Abstract method to parse the API response.

        Args:
            response: The raw response object from the API library.

        Returns:
            A tuple containing:
            - The assistant's textual response (if any).
            - A list of UnifiedToolCall objects (if any tools were called).
            - An error message string (if parsing failed).
        """
        pass

    @abstractmethod
    def _format_tool_results(
        self,
        tool_results: List[UnifiedToolResult]
    ) -> Any: # Returns tool results in the API's native format
        """
        Abstract method to format tool execution results for the specific LLM API.

        Args:
            tool_results: A list of UnifiedToolResult objects.

        Returns:
            The tool results formatted in the native structure expected by the API
            for the next call (e.g., a list of tool message dicts for OpenAI).
        """
        pass

    @abstractmethod
    def _get_tool_definitions(self) -> Any:
        """
        Create API-specific tool definitions.
        Called once during initialization to define the static tool schema.
        
        Returns:
            The API-specific tool definitions.
        """
        pass
    
    @abstractmethod
    def _get_tool_config(
        self,
        force_tool_name: Optional[str] = None
    ) -> Any:
        """
        Get API-specific tool configuration.
        Called for each API call to handle dynamic configs.
        
        Args:
            force_tool_name: If specified, configure tool_choice to force this tool.
            
        Returns:
            The API-specific tool configuration.
        """
        pass

    @abstractmethod
    def _append_assistant_message(
        self, 
        current_native_messages: List[Any], 
        api_response: Any
    ) -> None:
        """
        Appends the assistant's message from the API response to the native messages list.
        Each agent must implement this to correctly handle their API's specific format.

        Args:
            current_native_messages: The list of native format messages to append to
            api_response: The raw API response object containing the assistant's message
        """
        pass
        
    @abstractmethod
    def _append_tool_results(
        self,
        current_native_messages: List[Any],
        tool_results: List[UnifiedToolResult]
    ) -> None:
        """
        Appends tool execution results to the native messages list.
        Each agent must implement this to correctly format tool results for their API.

        Args:
            current_native_messages: The list of native format messages to append to
            tool_results: The unified tool results to format and append
        """
        pass
        
    @abstractmethod
    def _append_warning_message(
        self,
        current_native_messages: List[Any],
        warning_message: str
    ) -> None:
        """
        Appends a warning message to the native messages list.
        Used for cases like reaching maximum tool iterations.

        Args:
            current_native_messages: The list of native format messages to append to
            warning_message: The warning message to append
        """
        pass

    @abstractmethod
    def _format_initial_messages(
        self,
        user_input: str
    ) -> List[Any]:
        """
        Formats the initial messages for a turn, including the optional
        conversation summary and the user's input, into the provider's
        native message format.

        Args:
            user_input: The user's message for this turn.

        Returns:
            A list of messages formatted for the specific provider's API.
        """
        pass

    async def _execute_tool(self, tool_call: UnifiedToolCall) -> UnifiedToolResult:
        """Executes a single tool call."""
        tool_name = tool_call.tool_name
        tool_input = tool_call.tool_input
        call_id = tool_call.id
        start_time = time.monotonic()
        output: Any = None
        is_error = False

        logger.info(f"Executing tool '{tool_name}' with ID '{call_id}' and input: {tool_input}")
        # Add to TUI: Display "Thinking: Executing tool X..."

        if tool_name not in self.tools:
            logger.error(f"Tool '{tool_name}' not found.")
            output = f"Error: Tool '{tool_name}' is not available."
            is_error = True
        else:
            # Get the already instantiated tool
            tool = self.tools[tool_name]
            try:
                # TODO: Add input validation against tool's args_schema if implemented
                # The tool instance already has its dependencies (like history_db)
                tool_result = await tool.run(**tool_input)
                
                if tool_result.error:
                    logger.error(f"Tool '{tool_name}' (ID: {call_id}) execution error: {tool_result.error}")
                    output = tool_result.error
                    is_error = True
                else:
                    output = tool_result.output
                    logger.info(f"Tool '{tool_name}' (ID: {call_id}) executed successfully.")

            except ToolError as e:
                logger.error(f"Tool '{tool_name}' (ID: {call_id}) execution error: {e}")
                output = f"Error executing tool '{tool_name}': {e}"
                is_error = True
            except Exception as e:
                logger.exception(f"Unexpected error executing tool '{tool_name}' (ID: {call_id}): {e}")
                output = f"Unexpected error executing tool '{tool_name}': {e}"
                is_error = True

        latency_ms = (time.monotonic() - start_time) * 1000
        logger.debug(f"Tool '{tool_name}' (ID: {call_id}) execution time: {latency_ms:.2f} ms")
        # Add to TUI: Display tool result or error

        # Record tool execution in history DB? Maybe optional.

        # Create UnifiedToolResult - if we got a base64_image or system message from a ToolResult,
        # we might want to include these in the UnifiedToolResult in future iterations
        return UnifiedToolResult(
            call_id=call_id,
            tool_name=tool_name,
            tool_output=output,
            is_error=is_error,
        )

    async def _run_agent_loop(
        self,
        native_messages: List[Any] # Messages in API-native format
    ) -> Tuple[List[UnifiedMessage], List[Any]]:
        """
        Runs the core agent loop for a single turn of interaction
        (API call -> Tool Execution -> API call ... -> Final Response).

        Args:
            native_messages: The current conversation history in the API's native format.

        Returns:
            A tuple containing:
            - A list of UnifiedMessage objects generated during this loop (for logging/TUI).
            - The final state of the native_messages list after the loop completes.
        """
        turn_messages: List[UnifiedMessage] = []
        max_tool_iterations = self.config.max_tool_iterations
        iteration = 0

        current_native_messages = list(native_messages)

        while iteration < max_tool_iterations:
            iteration += 1
            logger.debug(f"Agent loop iteration {iteration}/{max_tool_iterations}")

            # 1. Call LLM API
            start_time = time.monotonic()
            api_tool_config = self._get_tool_config()
            try:
                api_response, metrics = await self._call_provider_api(
                    messages=current_native_messages,
                    tools=self.tool_definitions,
                    tool_config=api_tool_config,
                )
                latency_ms = metrics["latency_ms"]
                logger.info(f"LLM API call completed in {latency_ms:.2f} ms")

            except Exception as e:
                logger.exception("Error calling LLM API")
                error_msg = UnifiedMessage(role="assistant", content=f"Error calling API: {e}")
                turn_messages.append(error_msg)
                # Notify observer of error message
                if self.message_observer:
                    self.message_observer(error_msg)
                break

            # 2. Parse Response
            assistant_text, tool_calls, parse_error = self._parse_api_response(api_response)

            if parse_error:
                logger.error(f"Error parsing LLM response: {parse_error}")
                error_msg = UnifiedMessage(role="assistant", content=f"Error parsing response: {parse_error}")
                turn_messages.append(error_msg)
                # Notify observer of error message
                if self.message_observer:
                    self.message_observer(error_msg)
                break

            # 3. Update native messages with assistant response (API-specific implementation)
            self._append_assistant_message(current_native_messages, api_response)

            # Create UnifiedMessage for the assistant's turn
            assistant_message = UnifiedMessage(
                role="assistant",
                content=assistant_text,
                tool_calls=tool_calls,
                latency_ms=latency_ms,
                token_count=metrics.get("output_tokens")
            )
            turn_messages.append(assistant_message)
            
            # Notify observer of assistant message immediately
            if self.message_observer:
                self.message_observer(assistant_message)

            logger.debug(f"Assistant response: {assistant_text}")
            if tool_calls:
                logger.info(f"Assistant requested {len(tool_calls)} tool calls: {[tc.tool_name for tc in tool_calls]}")
            else:
                logger.info("No tool calls requested. Agent loop concluding.")
                break # Exit loop if no tools are called

            # 4. Execute Tools
            tool_results = await asyncio.gather(
                *(self._execute_tool(call) for call in tool_calls)
            )

            # 5. Format tool results for next API call (API-specific implementation)
            self._append_tool_results(current_native_messages, tool_results)

            # Add UnifiedMessage for tool results
            tool_result_message = UnifiedMessage(
                role="tool",
                tool_results=tool_results
            )
            turn_messages.append(tool_result_message)
            
            # Notify observer of tool results immediately
            if self.message_observer:
                self.message_observer(tool_result_message)

        if iteration == max_tool_iterations:
            logger.warning(f"Reached maximum tool execution iterations ({max_tool_iterations}).")
            warning_msg = UnifiedMessage(role="user", content=f"Reached maximum tool iterations ({max_tool_iterations}).")
            turn_messages.append(warning_msg)
            
            # Notify observer of warning message
            if self.message_observer:
                self.message_observer(warning_msg)
            
            # Add warning message to native_messages (API-specific implementation)
            self._append_warning_message(current_native_messages, f"Reached maximum tool iterations ({max_tool_iterations}).")

        return turn_messages, current_native_messages

    async def _summarize_conversation(self, native_messages: List[Any]) -> str:
        """
        Calls the LLM with a mandatory tool call to summarize the conversation.

        Args:
            native_messages: The conversation history leading up to the summary.

        Returns:
            The generated summary string, or an error message.
        """
        logger.info("Initiating end-of-turn conversation summary.")
        summary_tool_name = ConversationSummaryTool.name
        summary = f"Error: Could not generate summary." # Default error summary

        try:
            # Get tool configuration that forces the summary tool
            api_tool_config = self._get_tool_config(force_tool_name=summary_tool_name)

            # Ensure the summary tool definition is actually included
            if not self._verify_tool_included(self.tool_definitions, summary_tool_name):
                logger.error(f"Summary tool '{summary_tool_name}' definition not found for API.")
                return summary + " (Tool definition missing)"

            # Call LLM API, enforcing the summary tool
            api_response, metrics = await self._call_provider_api(
                messages=native_messages,
                tools=self.tool_definitions,
                tool_config=api_tool_config,
            )
            logger.info(f"Summary LLM call completed in {metrics['latency_ms']:.2f} ms")

            # Parse Response to find the summary tool call
            _, tool_calls, parse_error = self._parse_api_response(api_response)

            if parse_error:
                logger.error(f"Error parsing summary response: {parse_error}")
                return summary + f" (Parse Error: {parse_error})"

            if not tool_calls:
                logger.error("LLM did not call the required summary tool.")
                # Extract text response as fallback?
                text_response, _, _ = self._parse_api_response(api_response)
                if text_response:
                    logger.warning(f"Using text response as fallback summary: {text_response[:100]}...")
                    return text_response
                return summary + " (Tool not called)"

            # Find the specific summary tool call
            summary_call = next((tc for tc in tool_calls if tc.tool_name == summary_tool_name), None)

            if not summary_call:
                logger.error(f"Summary tool '{summary_tool_name}' was not called by the LLM, despite being required.")
                return summary + " (Tool call missing in response)"

            # Extract the summary argument from the tool call input
            summary = summary_call.tool_input.get("summary_text", summary + " (Argument 'summary_text' missing)")
            logger.info(f"Conversation summary generated: {summary[:100]}...")

        except Exception as e:
            logger.exception("Failed to summarize conversation")
            summary = summary + f" (Exception: {e})"

        return summary

    def _verify_tool_included(self, api_tools: List[Dict[str, Any]], tool_name: str) -> bool:
        """
        Verify that the specific tool is included in the tool definitions.
        Different APIs format tool definitions differently, so this needs to be flexible.
        
        Returns:
            True if the tool is included, False otherwise.
        """
        # Basic verification that works for most API formats
        for tool in api_tools:
            # Handle OpenAI format with nested function
            if 'function' in tool and tool.get('function', {}).get('name') == tool_name:
                return True
            # Handle simpler formats like Claude/Gemini
            if tool.get('name') == tool_name:
                return True
        return False

    async def start_turn(self, user_input: str) -> List[UnifiedMessage]:
        """
        Handles a single turn of conversation initiated by the user.

        - Retrieves history summary.
        - Formats the new user input and summary into the native message format.
        - Runs the agent loop (_run_agent_loop).
        - Triggers conversation summarization (_summarize_conversation).
        - Updates the agent state (summary, turn count).
        - Records messages to the history database.
        - Returns unified messages for display.

        Args:
            user_input: The user's message for this turn.

        Returns:
            A list of UnifiedMessage objects representing the conversation
            during this turn (user input, assistant responses, tool calls/results).
        """
        self.state.turn_count += 1
        turn_start_time = time.time()
        logger.info(f"--- Starting Turn {self.state.turn_count} ---")

        # 1. Prepare messages for API call using provider-specific formatting
        native_messages = self._format_initial_messages(user_input)

        # Keep track of unified messages for logging/display
        user_message_unified = UnifiedMessage(role="user", content=user_input)
        all_turn_messages: List[UnifiedMessage] = [user_message_unified]

        try:
            # 2. Run the main agent loop
            loop_unified_messages, final_native_messages = await self._run_agent_loop(native_messages)
            all_turn_messages.extend(loop_unified_messages)
            native_messages = final_native_messages # Use the final state for summarization

        except Exception as e:
            logger.exception("Error during agent loop execution.")
            error_msg = UnifiedMessage(role="assistant", content=f"An error occurred: {e}")
            all_turn_messages.append(error_msg)
            # Ensure native_messages is updated even on error if possible, or handle appropriately
            # For now, we proceed to try and log what we have.

        # 3. Summarize the conversation (unless the last message was an error)
        if not (all_turn_messages[-1].role == "assistant" and "error" in all_turn_messages[-1].content.lower()):
            try:
                current_summary = await self._summarize_conversation(native_messages)
                self.state.conversation_summary = current_summary
                self.history_db.add_summary(self.state.turn_count, current_summary)
            except Exception as e:
                 logger.exception("Failed to generate or record conversation summary.")
                 self.state.conversation_summary = "Error: Failed to generate summary for this turn."

        # 4. Record messages to history DB
        try:
            self.history_db.add_messages(self.state.turn_count, all_turn_messages)
        except Exception as e:
            logger.exception("Failed to record turn messages to history database.")

        turn_duration = time.time() - turn_start_time
        logger.info(f"--- Ending Turn {self.state.turn_count} (Duration: {turn_duration:.2f}s) ---")

        return all_turn_messages

    async def close(self) -> None:
        """Clean up resources like the history database connection."""
        logger.info("Closing agent resources.")
        await self.history_db.close()

