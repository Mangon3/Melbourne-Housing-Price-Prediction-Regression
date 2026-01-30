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

@retry_with_backoff(max_retries=5)
def call_model(state: AgentState, model):
    logger.info("DEBUG: Entering call_model node.")
    messages = state['messages']
    logger.info(f"DEBUG: Invoking model with {len(messages)} messages.")
    try:
        response = model.invoke(messages)
    except Exception as e:
        error_msg = str(e).lower()
        if ("429" in error_msg or 
            "resource_exhausted" in error_msg or 
            "quota" in error_msg):
             logger.warning(f"Rate limit hit ({e}). Raising to trigger robust retry...")
             raise 
        if "output text or tool calls" in error_msg or "cannot both be empty" in error_msg:
             logger.warning(f"Model failed with expected error: {e}. Preparing fallback...")
             response = None
        else:
             logger.error(f"DEBUG: Model invocation failed with critical error: {e}")
             raise

    has_micro_tool_run = any(
        isinstance(m, ToolMessage) and m.name == 'micro_analysis' 
        for m in messages
    )

    is_empty_response = response is None or (not response.tool_calls and not response.content)
    missed_tool_call = (response and not response.tool_calls and not has_micro_tool_run)

    if is_empty_response or missed_tool_call:
        symbol = None
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                content = m.content
                if "analyze" in content.lower() or "timeframe" in content.lower():
                     match = re.search(r"\b[A-Z]{2,5}\b", content)
                     if match:
                         symbol = match.group(0)
                         break

        if symbol:
            logger.warning(f"Model missed micro-analysis for {symbol}. Forcing fallback logic...")
            call_id = str(uuid.uuid4())
            manual_call = {
                "name": "micro_analysis",
                "args": {"symbol": symbol},
                "id": call_id,
                "type": "tool_call"
            }
            response = AIMessage(
                content="", 
                tool_calls=[manual_call]
            )
        else:
             if is_empty_response:
                 logger.error("Could not recover from empty response (Symbol not found).")
                 if response is None:
                     raise ValueError("Model failed and fallback logic could not determine symbol.")

    logger.info(f"DEBUG: Final Response content: {response.content}")
    if response.tool_calls:
        logger.info(f"DEBUG: Tool calls: {response.tool_calls}")
        
    return {"messages": [response]}
