def format_conversation_summary(summary: str) -> str:
    """
    Formats the conversation summary as preamble.

    Args:
        summary: The summary text to format.

    Returns:
        A formatted string containing the summary.
    """
    return f"""[Earlier conversation history has been truncated. Here is the summary you previously recorded.]
<ConversationHistorySummary>
{summary}
</ConversationHistorySummary>
"""