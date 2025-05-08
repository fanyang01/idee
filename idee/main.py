import argparse
import logging
from pathlib import Path
import sys
import asyncio
from typing import Any, Dict, Optional, Type

# Import configuration, logging setup, agents, and TUI
from .core.config import load_config, get_config
from .core.logging import setup_logging
from .agents.base import BaseAgent
from .agents.gpt import GPTAgent, OpenAIAgentConfig
from .agents.gemini import GeminiAgent, GeminiAgentConfig
from .agents.claude import ClaudeAgent, ClaudeAgentConfig
from .agents.types import AgentConfig # Base config type
from .tui.app import MainApp

logger = logging.getLogger(__name__)

def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Idee - An Agentic CLI Assistant")
    parser.add_argument(
        "--agent",
        type=str,
        choices=["gpt", "gemini", "claude"],
        help="The type of agent to use (loads corresponding config section). Overrides config file/env var.",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Override the model name for the selected agent.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML configuration file.",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        help="Path to the conversation history database file.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level.",
    )
    # Add arguments for API keys? Generally prefer env vars or config file.
    # parser.add_argument("--openai-api-key", type=str, help="OpenAI API Key")
    # ... other keys

    return parser.parse_args()

def get_agent_class(agent_type: str) -> Type[BaseAgent]:
    """Returns the agent class based on the type string."""
    if agent_type == "gpt":
        return GPTAgent
    elif agent_type == "gemini":
        return GeminiAgent
    elif agent_type == "claude":
        return ClaudeAgent
    else:
        # Should be caught by argparse choices, but defensive check
        raise ValueError(f"Unknown agent type: {agent_type}")

def create_agent_config(agent_type: str, args: argparse.Namespace, cfg: Dict[str, Any]) -> AgentConfig:
    """Creates the specific agent configuration object."""
    agent_cfg_data = cfg.get(agent_type, {})

    # Override specific fields from args if provided
    if args.model:
        agent_cfg_data["model"] = args.model

    # Ensure required fields like model_name are present
    if not agent_cfg_data.get("model"):
         raise ValueError(f"Model name for agent type '{agent_type}' is not defined in config or arguments.")


    # Populate API keys from loaded config (which includes env vars)
    agent_cfg_data["api_key"] = agent_cfg_data.get("api_key")

    # Create the specific config dataclass
    if agent_type == "gpt":
        # Add OpenAI specific fields if they exist in loaded config
        agent_cfg_data["api_base"] = agent_cfg_data.get("api_base")
        agent_cfg_data["use_responses_api"] = agent_cfg_data.get("use_responses_api", False)
        return OpenAIAgentConfig(**agent_cfg_data)
    elif agent_type == "gemini":
        # Add Gemini specific fields if any
        return GeminiAgentConfig(**agent_cfg_data)
    elif agent_type == "claude":
        # Add Claude specific fields
        agent_cfg_data["tool_version"] = agent_cfg_data.get("tool_version", "2024-04-04")
        agent_cfg_data["thinking_budget"] = agent_cfg_data.get("thinking_budget", 0)
        return ClaudeAgentConfig(**agent_cfg_data)
    else:
        raise ValueError(f"Cannot create config for unknown agent type: {agent_type}")


async def main():
    """Main asynchronous entry point."""
    args = parse_arguments()

    # Load configuration from file and environment variables
    # Pass args.config path if provided
    cfg = load_config(config_path=args.config)

    # Determine final configuration values, potentially overridden by args
    log_level = args.log_level or cfg.get("log_level", "INFO")
    db_path = args.db_path or cfg.get("db_path", "~/.idee/history.db")
    agent_type = args.agent or cfg.get("agent_type", "gpt")

    # Expand user directory in db_path and mkdir if needed
    _db_path = Path(db_path).expanduser()
    if not _db_path.parent.exists():
        _db_path.parent.mkdir(parents=True, exist_ok=True)

    # Setup logging
    setup_logging(level=log_level)
    logger.info("Starting Idee...")
    # logger.debug(f"Arguments: {args}")
    # logger.debug(f"Effective Config: {json.dumps(cfg, indent=2)}")


    try:
        # Get agent class and create specific config
        AgentClass = get_agent_class(agent_type)
        agent_config = create_agent_config(agent_type, args, cfg)

        # Instantiate the agent
        logger.info(f"Initializing {agent_type.capitalize()} agent...")
        agent_instance = AgentClass(
            config=agent_config,
            history_db_path=db_path,
            # tools=[] # Add custom tools here if needed
        )

        # Instantiate and run the TUI app
        logger.info("Starting Textual TUI...")
        app = MainApp(agent_instance=agent_instance)
        await app.run_async() # Use run_async for awaitable shutdown

    except ValueError as e:
         logger.error(f"Configuration error: {e}")
         print(f"Error: {e}", file=sys.stderr)
         sys.exit(1)
    except ImportError as e:
         logger.error(f"Import error: {e}. Have you installed all dependencies from pyproject.toml?")
         print(f"Import Error: {e}. Please ensure all dependencies are installed (`pip install .`)", file=sys.stderr)
         sys.exit(1)
    except Exception as e:
        logger.exception("An unexpected error occurred during startup or runtime.")
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        logger.info("Codemate finished.")


def run():
    """Synchronous wrapper for the main async function."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Codemate interrupted by user.")
        print("\nExiting...")
    except Exception as e:
        # Catch errors during asyncio.run itself if any
        logger.critical(f"Critical error during asyncio execution: {e}", exc_info=True)
        print(f"\nA critical error occurred: {e}", file=sys.stderr)


if __name__ == "__main__":
    # This allows running the script directly for testing,
    # although the entry point in pyproject.toml is preferred for installation.
    run()


