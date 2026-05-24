from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from src.graph.state import AgentState
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.config.settings import settings
from src.utils.logger import setup_logger
from src.utils.retry import retry_with_backoff
import re
import uuid

logger = setup_logger(__name__)


def _invoke_model_safely(model, messages):
    """Invokes the model, handling rate limits and empty responses gracefully."""
    try:
        response = model.invoke(messages)
    except Exception as e:
        error_msg = str(e).lower()
        if "429" in error_msg or "resource_exhausted" in error_msg or "quota" in error_msg:
            logger.warning("Rate limit hit (%s). Raising to trigger robust retry...", e)
            raise
        if "output text or tool calls" in error_msg or "cannot both be empty" in error_msg:
            logger.warning("Model failed with expected error: %s. Preparing fallback...", e)
            return None
        logger.exception("Model invocation failed with critical error")
        raise
    return response


def _extract_symbol_from_messages(messages):
    """Extracts a stock symbol from recent human messages for fallback logic."""
    for m in reversed(messages):
        if not isinstance(m, HumanMessage):
            continue
        content = m.content
        if "analyze" in content.lower() or "timeframe" in content.lower():
            match = re.search(r"\b[A-Z]{2,5}\b", content)
            if match:
                return match.group(0)
    return None


def _build_fallback_response(symbol):
    """Builds a manual tool call response for a given symbol."""
    call_id = str(uuid.uuid4())
    manual_call = {
        "name": "micro_analysis",
        "args": {"symbol": symbol},
        "id": call_id,
        "type": "tool_call"
    }
    return AIMessage(content="", tool_calls=[manual_call])


def _needs_fallback(response, messages):
    """Determines if the response needs fallback logic."""
    has_micro_tool_run = any(
        isinstance(m, ToolMessage) and m.name == 'micro_analysis'
        for m in messages
    )
    is_empty = response is None or (not response.tool_calls and not response.content)
    missed_tool = response and not response.tool_calls and not has_micro_tool_run
    return is_empty or missed_tool


@retry_with_backoff(max_retries=5)
def call_model(state: AgentState, model):
    logger.info("DEBUG: Entering call_model node.")
    messages = state['messages']
    logger.info("DEBUG: Invoking model with %d messages.", len(messages))

    response = _invoke_model_safely(model, messages)

    if _needs_fallback(response, messages):
        symbol = _extract_symbol_from_messages(messages)
        if symbol:
            logger.warning("Model missed micro-analysis for %s. Forcing fallback logic...", symbol)
            response = _build_fallback_response(symbol)
        elif response is None:
            logger.error("Could not recover from empty response (Symbol not found).")
            raise ValueError("Model failed and fallback logic could not determine symbol.")

    logger.info("DEBUG: Final Response content: %s", response.content)
    if response.tool_calls:
        logger.info("DEBUG: Tool calls: %s", response.tool_calls)
        
    return {"messages": [response]}
