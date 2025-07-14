# Idee - Multimodal Agent Framework

A powerful, extensible Python framework for building AI agents that support multiple LLM providers (OpenAI GPT, Google Gemini, Anthropic Claude) with rich multimodal capabilities.

## Features

### 🤖 Multi-Provider Support
- **OpenAI GPT**: Chat Completions API and Responses API support
- **Google Gemini**: Native Gemini API integration
- **Anthropic Claude**: Full Claude API support with beta features

### 🎯 Multimodal Tool Output
- **Rich Content Types**: Text, images, audio, video, documents, and files
- **Native Multimodal Support**: Claude handles multimodal tool output natively
- **Placeholder mechanism**: GPT and Gemini do not allow tools to return multimodal content. As a workaround, this framework supplies multimodal content in a follow-up user message
- **File Upload**: Automatic file upload to respective vendor APIs

### 🛠️ Extensible Tool System
- **Built-in Tools**: Bash execution, file editing, computer control, conversation history
- **Custom Tools**: Easy-to-implement tool interface for extending functionality
- **Vendor-Specific Tools**: Support for vendor-specific tool implementations
- **Async Support**: Full async/await support throughout the tool execution chain

### 📊 Advanced Features
- **Conversation History**: Persistent conversation storage and summarization
- **Token Usage Tracking**: Comprehensive token usage monitoring across all providers
- **Error Handling**: Robust error handling and recovery mechanisms

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd idee

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Quick Start

### CLI

```bash
idee
```

or

```bash
python -m idee
```

### Basic Agent Setup

```python
from idee.agents.gpt import GPTAgent
from idee.agents.gemini import GeminiAgent
from idee.agents.claude import ClaudeAgent
from idee.agents.types import OpenAIAgentConfig, GeminiAgentConfig, ClaudeAgentConfig

# GPT Agent
gpt_config = OpenAIAgentConfig(
    model="gpt-4.1",
    api_key="your-openai-api-key",
    use_responses_api=True
)
gpt_agent = GPTAgent(gpt_config)

# Gemini Agent
gemini_config = GeminiAgentConfig(
    model="gemini-2.5-pro",
    api_key="your-gemini-api-key"
)
gemini_agent = GeminiAgent(gemini_config)

# Claude Agent
claude_config = ClaudeAgentConfig(
    model="claude-sonnet-4",
    api_key="your-anthropic-api-key"
)
claude_agent = ClaudeAgent(claude_config)
```

### Running a Conversation

```python
async def main():
    # Start a conversation turn
    messages = await agent.start_turn("What does this project do?")
    
    # Process the response
    for message in messages:
        if message.role == "assistant":
            print(f"Assistant: {message.content}")
        elif message.role == "tool":
            print(f"Tool Results: {len(message.tool_results)} tools executed")

# Run the async function
import asyncio
asyncio.run(main())
```

### Creating Multimodal Tool Results

```python
from idee.tools.base import ToolResult, TextBlock, ImageBlock, FileBlock

# Create a tool result with mixed content
tool_result = ToolResult(content=[
    TextBlock(text="Analysis complete. Generated visualization:"),
    ImageBlock.from_base64(image_data, "image/png"),
    TextBlock(text="Summary report:"),
    FileBlock.from_path("/path/to/report.pdf", "application/pdf", "analysis_report.pdf")
])

# The framework automatically handles vendor-specific formatting
```

## Architecture

### Agent Architecture

```
BaseAgent (Abstract)
├── GPTAgent (OpenAI GPT Models)
├── GeminiAgent (Google Gemini Models)
└── ClaudeAgent (Anthropic Claude Models)
```

### Tool System

```
BaseTool (Abstract)
├── BashTool (System commands)
├── StrReplaceEditorTool (File editing)
├── ComputerTool (Screen interaction)
├── HistoryTool (Conversation history)
└── ConversationSummaryTool (Auto-summarization)
```

### Multimodal Content Flow

```
ToolResult → UnifiedToolResult → Vendor-Specific Formatting
```

- **Claude**: Native multimodal support in tool results
- **GPT** and **Gemini**: Placeholder-based approach with follow-up messages

## Configuration

### Environment Variables

```bash
# API Keys
OPENAI_API_KEY=your-openai-api-key
GOOGLE_API_KEY=your-gemini-api-key
ANTHROPIC_API_KEY=your-claude-api-key

# Optional: Custom API bases
OPENAI_API_BASE=https://api.openai.com/v1  # For proxies like OpenRouter
```

### Agent Configuration

```python
# Advanced GPT Configuration
gpt_config = OpenAIAgentConfig(
    model="gpt-4",
    api_key="your-key",
    api_base="https://api.openai.com/v1",
    use_responses_api=True,
    temperature=0.7,
    max_tokens=4096,
    max_tool_iterations=5
)

# Advanced Claude Configuration
claude_config = ClaudeAgentConfig(
    model="claude-sonnet-4-20250514",
    api_key="your-key",
    thinking_budget=10000,  # Enable thinking mode
    temperature=0.7,
    max_tokens=4096
)
```

## Custom Tools

### Creating a Custom Tool

```python
from idee.tools.base import BaseTool, ToolResult, TextBlock
from typing import Dict, Any

class WeatherTool(BaseTool):
    name = "get_weather"
    description = "Get current weather for a location"
    
    def get_definition(self):
        return ToolDefinition(
            name=self.name,
            description=self.description,
            input_schema={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name or coordinates"
                    }
                },
                "required": ["location"]
            }
        )
    
    async def run(self, location: str) -> ToolResult:
        # Your weather API integration here
        weather_data = await get_weather_data(location)
        
        return ToolResult(content=[
            TextBlock(text=f"Weather in {location}: {weather_data}")
        ])

# Register with agent
agent = GPTAgent(config, tools=[WeatherTool])
```

### Vendor-Specific Tool Implementations

```python
class MyTool(BaseTool):
    def has_vendor_spec(self, vendor: str, model: str) -> bool:
        return vendor == "anthropic" and "claude" in model
    
    def get_vendor_spec(self, vendor: str, model: str) -> Dict[str, Any]:
        if vendor == "anthropic":
            return {
                "type": "computer_20241022",
                "name": "computer",
                "display_width_px": 1024,
                "display_height_px": 768
            }
        return None
```

## Testing

```bash
# Run all tests
python -m pytest

# Run specific test suite
python -m pytest tests/test_multimodal_tool_output.py -v

# Run with coverage
python -m pytest --cov=idee tests/
```

`test_multimodal_tool_output.py` includes comprehensive test coverage for the multimodal tool output.


## Recent Updates

- Add support for multimodal tool output

## API Reference

### Core Classes

#### `BaseAgent`
Abstract base class for all AI agents.

**Methods:**
- `async start_turn(user_input: str) -> List[UnifiedMessage]`
- `async close() -> None`

#### `ToolResult`
Container for multimodal tool output.

**Methods:**
- `add_text(text: str) -> ToolResult`
- `add_image(data: str, media_type: str) -> ToolResult`
- `add_file(file_path: str, media_type: str, filename: str) -> ToolResult`
- `get_text_content() -> str`
- `has_media() -> bool`

#### `UnifiedToolResult`
Cross-vendor tool result abstraction.

**Methods:**
- `has_multimodal_content() -> bool`
- `to_simple_format_with_placeholders() -> Tuple[str, Dict]`

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
python -m pytest

# Run linting
ruff check .
black .
```
