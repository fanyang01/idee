import logging
import duckdb
import time
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from ..agents.types import UnifiedMessage # Use the unified type

logger = logging.getLogger(__name__)

# Define the schema for the history table
HISTORY_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversation_history (
    turn_id INTEGER,
    message_id INTEGER,
    session_id VARCHAR DEFAULT 'default', -- For potential multi-session support
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    role VARCHAR CHECK (role IN ('user', 'assistant', 'system', 'tool')),
    content TEXT,
    tool_calls TEXT,          -- JSON string representation of List[UnifiedToolCall]
    tool_results TEXT,        -- JSON string representation of List[UnifiedToolResult]
    latency_ms DOUBLE,
    token_count INTEGER,
    PRIMARY KEY (session_id, turn_id, message_id)
);
"""

SUMMARY_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversation_summaries (
    session_id VARCHAR DEFAULT 'default',
    turn_id INTEGER,
    summary TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (session_id, turn_id)
);
"""


class HistoryDB:
    """Handles persistence of conversation history using DuckDB."""

    def __init__(self, db_path: str = None):
        self.db_path: Path = Path(db_path) if db_path \
            else (Path.home() / ".idee" / "history.db")
        self.conn: Optional[duckdb.DuckDBPyConnection] = None
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Connects to the database and creates tables if they don't exist."""
        try:
            logger.info(f"Initializing history database at: {self.db_path}")
            # Ensure the directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = duckdb.connect(str(self.db_path))
            self.conn.execute(HISTORY_TABLE_SCHEMA)
            self.conn.execute(SUMMARY_TABLE_SCHEMA)
            self.conn.commit()
            logger.info("History database initialized successfully.")
        except Exception as e:
            logger.exception(f"Failed to initialize history database: {e}")
            self.conn = None # Ensure conn is None if init fails

    def add_messages(self, turn_id: int, messages: List[UnifiedMessage], session_id: str = 'default') -> None:
        """Adds a list of messages from a turn to the database."""
        if not self.conn:
            logger.error("Database connection is not available. Cannot add messages.")
            return

        try:
            records = []
            for i, msg in enumerate(messages):
                # Serialize tool calls/results to JSON strings
                tool_calls_json = json.dumps([vars(tc) for tc in msg.tool_calls]) if msg.tool_calls else None
                tool_results_json = json.dumps([vars(tr) for tr in msg.tool_results]) if msg.tool_results else None

                records.append((
                    turn_id,
                    i, # message_id within the turn
                    session_id,
                    # msg.timestamp, # Let DB handle timestamp default or convert Python float ts
                    msg.role,
                    msg.content,
                    tool_calls_json,
                    tool_results_json,
                    msg.latency_ms,
                    msg.token_count,
                ))

            # Use executemany for potentially better performance
            self.conn.executemany(
                """
                INSERT INTO conversation_history (
                    turn_id, message_id, session_id, role, content,
                    tool_calls, tool_results, latency_ms, token_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                records
            )
            self.conn.commit()
            logger.debug(f"Added {len(messages)} messages for turn {turn_id} to history DB.")
        except Exception as e:
            logger.exception(f"Failed to add messages to history database for turn {turn_id}: {e}")
            self.conn.rollback() # Rollback on error

    def add_summary(self, turn_id: int, summary: str, session_id: str = 'default') -> None:
        """Adds a conversation summary for a specific turn."""
        if not self.conn:
            logger.error("Database connection is not available. Cannot add summary.")
            return
        try:
            self.conn.execute(
                "INSERT INTO conversation_summaries (turn_id, session_id, summary) VALUES (?, ?, ?)",
                (turn_id, session_id, summary)
            )
            self.conn.commit()
            logger.debug(f"Added summary for turn {turn_id} to history DB.")
        except Exception as e:
            logger.exception(f"Failed to add summary for turn {turn_id}: {e}")
            self.conn.rollback()

    def get_latest_summary(self, session_id: str = 'default') -> Optional[str]:
        """Retrieves the most recent conversation summary."""
        if not self.conn:
            logger.error("Database connection is not available. Cannot get summary.")
            return None
        try:
            result = self.conn.execute(
                "SELECT summary FROM conversation_summaries WHERE session_id = ? ORDER BY turn_id DESC LIMIT 1",
                (session_id,)
            ).fetchone()
            if result:
                logger.debug("Retrieved latest summary from DB.")
                return result[0]
            else:
                logger.debug("No summary found in DB for this session.")
                return None
        except Exception as e:
            logger.exception("Failed to retrieve latest summary from database.")
            return None

    def query_history(self, query: str, session_id: str = 'default', limit: int = 20) -> List[Dict[str, Any]]:
        """
        Queries the conversation history using a simple keyword search
        or potentially more complex SQL via the query string.
        NOTE: Directly executing user/LLM provided SQL is dangerous.
        This implementation focuses on simple keyword search for safety.
        """
        if not self.conn:
            logger.error("Database connection is not available. Cannot query history.")
            return []

        logger.info(f"Querying history with keyword: '{query}' (limit: {limit})")
        try:
            # Simple keyword search across relevant text fields
            # Using FTS extension would be much better for real search.
            # For now, simple LIKE matching.
            # Escape '%' and '_' in the query to treat them literally in LIKE
            safe_query = query.replace('%', '\\%').replace('_', '\\_')
            search_term = f"%{safe_query}%"

            results = self.conn.execute(
                f"""
                SELECT turn_id, message_id, timestamp, role, content, tool_calls, tool_results
                FROM conversation_history
                WHERE session_id = ? AND (
                    content LIKE ?
                    -- Optionally search in tool calls/results JSON? Careful with performance.
                    -- OR tool_calls LIKE ?
                    -- OR tool_results LIKE ?
                )
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (session_id, search_term, limit) # Pass search_term multiple times if searching more fields
            ).fetchall()

            # Convert results to a list of dictionaries
            history_entries = []
            if results:
                column_names = [desc[0] for desc in self.conn.description]
                for row in results:
                    entry = dict(zip(column_names, row))
                    # Optionally parse JSON back into dicts/lists here if needed by the LLM
                    # entry['tool_calls'] = json.loads(entry['tool_calls']) if entry['tool_calls'] else None
                    # entry['tool_results'] = json.loads(entry['tool_results']) if entry['tool_results'] else None
                    history_entries.append(entry)

            logger.info(f"History query returned {len(history_entries)} results.")
            return history_entries

        except Exception as e:
            logger.exception(f"Failed to query history database with query '{query}': {e}")
            return []

    async def close(self) -> None:
        """Closes the database connection."""
        if self.conn:
            try:
                self.conn.close()
                logger.info("History database connection closed.")
                self.conn = None
            except Exception as e:
                logger.exception("Error closing history database connection.")


