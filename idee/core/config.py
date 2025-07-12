import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, Dict, Any
import yaml # Added pyyaml dependency earlier

logger = logging.getLogger(__name__)

# Default configuration values
DEFAULT_CONFIG = {
    "agent_type": "gpt",
    "db_path": "~/.idee/history.db",
    "log_level": "INFO",
    "gpt": {
        "model": "gpt-4.1",
        "api_key": None,
        "api_base": None,
        "use_responses_api": False,
        "max_tokens": 4096,
    },
    "gemini": {
        "model": "gemini-2.5-flash",
        "api_key": None,
        "max_tokens": 4096,
    },
    "claude": {
        "model": "claude-3-7-sonnet-20250219",
        "api_key": None,
        "tool_version": "2025-01-24",
        "thinking_budget": 0,
        "max_tokens": 4096,
    },
    "tui": {
        "theme": "dark",
    }
}

# Environment variable mapping
ENV_VAR_MAP = {
    "AGENT_TYPE": "agent_type",
    "DB_PATH": "db_path",
    "LOG_LEVEL": "log_level",
    "OPENAI_API_KEY": "gpt.api_key",
    "OPENAI_MODEL": "gpt.model",
    "OPENAI_API_BASE": "gpt.api_base",
    "GEMINI_API_KEY": "gemini.api_key",
    "GEMINI_MODEL": "gemini.model",
    "ANTHROPIC_API_KEY": "claude.api_key",
    "ANTHROPIC_MODEL": "claude.model",
}

_config: Optional[Dict[str, Any]] = None

def _merge_configs(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merges overlay dict into base dict."""
    merged = base.copy()
    for key, value in overlay.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = _merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged

def _set_nested_value(d: Dict[str, Any], keys: str, value: Any) -> None:
    """Sets a value in a nested dictionary using dot notation."""
    key_list = keys.split('.')
    for i, key in enumerate(key_list[:-1]):
        if key not in d or not isinstance(d[key], dict):
            # If intermediate key doesn't exist or isn't a dict, stop
            path = '.'.join(key_list[:i+1])
            logger.warning(f"Cannot set nested config key '{keys}': Path '{path}' not found or not a dictionary.")
            return
        d = d[key]
    final_key = key_list[-1]
    if final_key in d: # Only overwrite if key exists
         d[final_key] = value
    else:
         logger.warning(f"Cannot set nested config key '{keys}': Final key '{final_key}' not found.")


def load_config(config_path: Optional[str] = None, load_dotenv_file: bool = True) -> Dict[str, Any]:
    """
    Loads configuration from defaults, optional YAML file, and environment variables.

    Args:
        config_path: Path to an optional YAML configuration file.
        load_dotenv_file: Whether to load a .env file (defaults to True).

    Returns:
        The loaded configuration dictionary.
    """
    global _config
    if _config is not None:
        return _config

    config = DEFAULT_CONFIG.copy()

    # 1. Load from YAML file if provided
    if config_path:
        try:
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
                if yaml_config and isinstance(yaml_config, dict):
                    config = _merge_configs(config, yaml_config)
                    logger.info(f"Loaded configuration from YAML file: {config_path}")
                else:
                    logger.warning(f"YAML config file '{config_path}' is empty or invalid.")
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}")
        except Exception as e:
            logger.error(f"Error loading config file {config_path}: {e}")

    # 2. Load from .env file
    if load_dotenv_file:
        dotenv_path = find_dotenv()
        if dotenv_path:
            loaded = load_dotenv(dotenv_path=dotenv_path, override=True)
            if loaded:
                 logger.info(f"Loaded environment variables from: {dotenv_path}")
            else:
                 logger.debug("No .env file found or it was empty.")
        else:
            logger.debug("No .env file found.")


    # 3. Override with environment variables using the map
    for env_var, config_key in ENV_VAR_MAP.items():
        value = os.environ.get(env_var)
        if value is not None:
            # Handle nested keys
            if '.' in config_key:
                 _set_nested_value(config, config_key, value)
            elif config_key in config:
                 config[config_key] = value
            logger.debug(f"Overriding config '{config_key}' with environment variable '{env_var}'.")

    # Type conversions for specific keys if needed (e.g., from string env vars)
    if isinstance(config.get("openai", {}).get("use_responses_api"), str):
        config["openai"]["use_responses_api"] = config["openai"]["use_responses_api"].lower() in ['true', '1', 'yes']
    if isinstance(config.get("claude", {}).get("thinking_budget"), str):
        try:
            config["claude"]["thinking_budget"] = int(config["claude"]["thinking_budget"])
        except ValueError:
            logger.warning("Invalid value for CLAUDE_THINKING_BUDGET, using default.")
            config["claude"]["thinking_budget"] = DEFAULT_CONFIG["claude"]["thinking_budget"]


    _config = config
    logger.info("Configuration loaded.")
    # logger.debug(f"Final config: {json.dumps(_config, indent=2)}") # Can be verbose
    return _config

def get_config() -> Dict[str, Any]:
    """Returns the loaded configuration."""
    if _config is None:
        return load_config()
    return _config

def find_dotenv() -> Optional[str]:
    """Finds the .env file by searching current and parent directories."""
    cwd = Path.cwd()
    for directory in [cwd] + list(cwd.parents):
        dotenv_path = directory / ".env"
        if dotenv_path.exists():
            return str(dotenv_path)
    return None
