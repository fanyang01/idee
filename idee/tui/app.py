import json
import logging
import time
from typing import Optional, List, Type, cast
import asyncio

from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from rich.style import Style

from textual.app import App, ComposeResult, Binding
from textual.containers import Container, VerticalScroll
from textual.reactive import var
from textual.widgets import Header, Footer, Input, Static, RichLog # Use Log for conversation

# Import agent base and types
from ..agents.base import BaseAgent
from ..agents.types import UnifiedMessage, UnifiedToolCall, UnifiedToolResult

logger = logging.getLogger(__name__)

# --- Styles ---
USER_STYLE = Style(color="bright_blue", bold=True)
ASSISTANT_STYLE = Style(color="bright_green")
TOOL_CALL_STYLE = Style(color="yellow")
TOOL_RESULT_STYLE = Style(color="magenta")
ERROR_STYLE = Style(color="red", bold=True)
SYSTEM_STYLE = Style(color="grey70", italic=True) # For status messages

class ConversationLog(RichLog):
    """Custom Log widget to display conversation messages."""

    def clear_last_line(self) -> None:
        """Removes the last line from the log, useful for replacing temporary messages."""
        if not self._lines:
            return
        
        # Remove the last line
        self._lines.pop()
        # Update the widget
        self.refresh()
        
    def add_message(self, message: UnifiedMessage):
        """Formats and adds a UnifiedMessage to the log."""
        if message.role == "user":
            prefix = Text("You", style=USER_STYLE)
            content = Text(f": {message.content}") if message.content else Text("")
            self.write(prefix + content)

        elif message.role == "assistant":
            prefix = Text("Assistant", style=ASSISTANT_STYLE)
            content = Text(f": {message.content}") if message.content else Text("")
            self.write(prefix + content)
            if message.tool_calls:
                self.add_tool_calls(message.tool_calls)

        elif message.role == "tool":
             if message.tool_results:
                 self.add_tool_results(message.tool_results)

        elif message.role == "system": # For internal status/errors
             prefix = Text("System", style=SYSTEM_STYLE)
             content = Text(f": {message.content}") if message.content else Text("")
             self.write(prefix + content)

        # Add a blank line for spacing after assistant/tool results
        if message.role in ["assistant", "tool"]:
             self.write("")


    def add_tool_calls(self, tool_calls: List[UnifiedToolCall]):
        """Formats and displays tool calls."""
        prefix = Text(" > Tool Call:", style=TOOL_CALL_STYLE)
        for call in tool_calls:
            try:
                # Pretty print JSON arguments
                args_json = json.dumps(call.tool_input, indent=2)
                syntax = Syntax(args_json, "json", theme="github-dark", line_numbers=False, word_wrap=True) # Requires pygments
                panel = Panel(syntax, title=f"{call.tool_name} (ID: {call.id})", border_style=TOOL_CALL_STYLE, title_align="left")
                self.write(prefix)
                self.write(panel)
            except Exception as e:
                logger.error(f"Error formatting tool call {call.tool_name}: {e}")
                self.write(prefix + Text(f" {call.tool_name}(...) - Error formatting args: {e}", style=ERROR_STYLE))


    def add_tool_results(self, tool_results: List[UnifiedToolResult]):
        """Formats and displays tool results."""
        prefix = Text(" < Tool Result:", style=TOOL_RESULT_STYLE)
        for result in tool_results:
            try:
                # Format output based on type
                if isinstance(result.tool_output, str):
                    output_display = Text(result.tool_output, no_wrap=False) # Allow wrapping
                elif isinstance(result.tool_output, (dict, list)):
                     output_display = Syntax(json.dumps(result.tool_output, indent=2), "json", theme="github-dark", line_numbers=False, word_wrap=True)
                else:
                     output_display = Text(str(result.tool_output))

                status_style = ERROR_STYLE if result.is_error else TOOL_RESULT_STYLE
                panel_title = f"{result.tool_name} (Call ID: {result.call_id})"
                panel_border_style = ERROR_STYLE if result.is_error else TOOL_RESULT_STYLE

                panel = Panel(output_display, title=panel_title, border_style=panel_border_style, title_align="left")
                self.write(prefix)
                self.write(panel)

            except Exception as e:
                 logger.error(f"Error formatting tool result {result.tool_name}: {e}")
                 self.write(prefix + Text(f" {result.tool_name} - Error formatting result: {e}", style=ERROR_STYLE))

class MainApp(App[None]):
    """The main Textual application."""

    CSS_PATH = "app.css" # Load CSS from a file (optional)
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+c", "quit", "Quit", show=False), # Allow Ctrl+C too
        Binding("ctrl+l", "clear_log", "Clear Log"),
    ]

    agent: Optional[BaseAgent] = None
    is_processing = var(False) # Reactive variable for status updates

    def __init__(self, agent_instance: BaseAgent, *args, **kwargs):
        self.agent = agent_instance
        self.agent.message_observer = self.handle_message
        super().__init__(*args, **kwargs)
        
    def handle_message(self, message: UnifiedMessage) -> None:
        """Callback function to handle new messages from the agent."""
        # Check if we're in the same thread as the app
        import threading
        current_thread = threading.current_thread()
        app_thread = self._thread_id if hasattr(self, "_thread_id") else None
        
        if current_thread.ident == app_thread:
            # If we're in the app's thread, directly update the UI
            self.add_message_to_log(message)
        else:
            # If we're in a different thread, use call_from_thread
            try:
                self.call_from_thread(self.add_message_to_log, message)
            except RuntimeError as e:
                # In case we still get an error, log it
                logger.warning(f"RuntimeError in call_from_thread: {e}")
                # Try using async scheduling instead
                asyncio.run_coroutine_threadsafe(self._async_add_message(message), self.run_worker.app_loop)
    
    async def _async_add_message(self, message: UnifiedMessage) -> None:
        """Async helper to add message from another thread."""
        self.add_message_to_log(message)

    def add_message_to_log(self, message: UnifiedMessage) -> None:
        """Adds a message to the conversation log."""
        log = self.query_one(ConversationLog)
        # Remove the "Thinking..." message when we get a new message
        if message.role == "assistant" and message.content:
            # Check if the last message was the "Thinking..." system message
            if hasattr(log, "_last_message") and log._last_message == "thinking":
                # If we were showing "Thinking...", we can replace it
                log.clear_last_line()
            
        # Add the message to the log
        log.add_message(message)
        
        # Track the last message type for potential removal
        if message.role == "system" and message.content == "Thinking...":
            log._last_message = "thinking"
        else:
            log._last_message = message.role
            
        # Scroll to the bottom to show latest messages
        log.scroll_end(animate=False)

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="main-container"):
            yield ConversationLog(id="conversation-log", wrap=True, highlight=True, markup=True)
            yield Input(id="user-input", placeholder="Enter your message...")
        yield Footer()

    def on_mount(self) -> None:
        """Called when the app is mounted."""
        # Store the app's thread ID
        import threading
        self._thread_id = threading.current_thread().ident
        
        self.query_one("#user-input", Input).focus()
        log = self.query_one(ConversationLog)
        log.add_message(UnifiedMessage(role="system", content=f"Welcome to Idee! Using {self.agent.config.model}. Type your request and press Enter."))
        log.add_message(UnifiedMessage(role="system", content="Tip: Use Ctrl+L to clear the log, Ctrl+Q to quit."))


    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle user input submission."""
        user_input = event.value.strip()
        if not user_input or self.is_processing:
            return

        input_widget = event.input
        log = self.query_one(ConversationLog)

        # Clear input and display user message
        input_widget.clear()
        log.add_message(UnifiedMessage(role="user", content=user_input))

        # Set processing state and show Thinking message
        self.is_processing = True
        log.add_message(UnifiedMessage(role="system", content="Thinking..."))
        input_widget.disabled = True # Disable input while processing

        try:
            # Run the agent loop - messages will be displayed via the observer callback
            await self.agent.start_turn(user_input)
            # No need to display messages here, they're displayed in real-time by the observer

        except Exception as e:
            logger.exception("Error during agent loop.")
            log.add_message(UnifiedMessage(role="system", content=f"Error: {e}", timestamp=time.time()))

        finally:
            # Reset processing state
            self.is_processing = False
            input_widget.disabled = False
            input_widget.focus() # Refocus input

    def action_clear_log(self) -> None:
        """Clears the conversation log."""
        log = self.query_one(ConversationLog)
        log.clear()
        log.add_message(UnifiedMessage(role="system", content="Log cleared."))

    async def action_quit(self) -> None:
        """Called when the user quits."""
        log = self.query_one(ConversationLog)
        log.add_message(UnifiedMessage(role="system", content="Shutting down..."))
        if self.agent:
            await self.agent.close() # Close DB connection gracefully
        self.exit()


