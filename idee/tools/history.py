import asyncio
import logging
from typing import Dict, Any, List

from .base import BaseTool, ToolError
from ..core.history_db import HistoryDB

logger = logging.getLogger(__name__)


class HistoryTool(BaseTool):
    """
    A tool to query the conversation history stored in the database.
    Requires a HistoryDB instance to be injected during initialization.
    """
    name: str = "query_conversation_history"
    description: str = (
        "Searches the conversation history for past interactions based on keywords. "
        "Returns relevant messages including user inputs, assistant responses, and tool usage."
    )
    input_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "search_query": {
                "type": "string",
                "description": "Keywords or phrases to search for in the conversation history.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of history entries to return (optional, default 10).",
                "default": 10,
            }
        },
        "required": ["search_query"],
    }

    # Store the injected HistoryDB instance
    history_db: HistoryDB

    def __init__(self, history_db: HistoryDB):
        """
        Initializes the HistoryTool with a HistoryDB instance.

        Args:
            history_db: The HistoryDB instance to use for querying.
        """
        if not history_db or not history_db.conn:
             # Ensure a valid, connected HistoryDB instance is provided
             raise ValueError("A valid and connected HistoryDB instance is required.")
        self.history_db = history_db
        super().__init__() # Call parent __init__ if it exists/is needed

    async def run(self, search_query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Queries the history database using the injected instance.

        Args:
            search_query: The keyword(s) to search for.
            max_results: The maximum number of results to return.

        Returns:
            A list of dictionaries, each representing a historical message entry.
            Returns an empty list if the query fails or no results are found.
        """
        if not search_query:
            raise ToolError("Search query cannot be empty.")
        if max_results <= 0:
             raise ToolError("Max results must be a positive integer.")

        # Use the injected history_db instance directly
        if not self.history_db or not self.history_db.conn:
             # This check might be redundant if the __init__ validation is robust,
             # but good for safety if the connection could be closed externally.
             raise ToolError("History database connection is not available.")

        try:
            # Run the synchronous query method in a thread pool executor
            loop = asyncio.get_running_loop()
            results = await loop.run_in_executor(
                None, # Use default executor (ThreadPoolExecutor)
                self.history_db.query_history, # Use the injected instance's method
                search_query,
                'default', # Assuming single session for now
                max_results
            )
            # No need to close the connection here - the Agent manages it.

            if not results:
                 # Return a more informative message instead of just an empty list
                 return [{"message": f"No history found matching query: '{search_query}'"}]

            return results

        except Exception as e:
            logger.exception(f"Error querying history with query '{search_query}': {e}")
            # Do not close the connection here.
            raise ToolError(f"Failed to query history: {e}")


