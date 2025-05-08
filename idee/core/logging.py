import logging
import sys
from rich.logging import RichHandler

# --- Configuration ---
LOG_LEVEL = "INFO"  # Set desired log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# --- Setup ---
def setup_logging(level: str = LOG_LEVEL) -> None:
    """
    Configures the root logger for the application.

    Args:
        level: The minimum logging level to output.
    """
    log_level_enum = getattr(logging, level.upper(), logging.INFO)

    # Use RichHandler for pretty console logging
    rich_handler = RichHandler(
        rich_tracebacks=True,
        markup=True,
        show_path=False # Don't show path to avoid clutter
    )
    rich_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))

    # Get the root logger and configure it
    root_logger = logging.getLogger()
    if not root_logger.hasHandlers(): # Avoid adding handlers multiple times
        root_logger.setLevel(log_level_enum)
        root_logger.addHandler(rich_handler)

        # Optionally, add a file handler
        # file_handler = logging.FileHandler("codemate.log")
        # file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
        # root_logger.addHandler(file_handler)

        logging.getLogger("httpx").setLevel(logging.WARNING) # Quiet down noisy libraries
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("google").setLevel(logging.WARNING)
        logging.getLogger("anthropic").setLevel(logging.WARNING)

        logging.info(f"Logging configured with level {level.upper()}")
    else:
        # If already configured, just ensure the level is set
        root_logger.setLevel(log_level_enum)
        logging.debug("Logger already configured. Setting level.")

# Call setup_logging() when this module is imported?
# Generally better to call it explicitly from the main entry point.
# setup_logging()

# Example usage (in other modules):
# import logging
# logger = logging.getLogger(__name__)
# logger.info("This is an info message.")
# logger.debug("This is a debug message.")

