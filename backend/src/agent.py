import json
from typing import List, Dict, Any, Generator
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.config.prompts import StockAgentPrompts
from src.config.settings import settings
from src.utils.retry import retry_with_backoff
from src.pipeline import StockAnalysisPipeline
from src.utils.logger import setup_logger
from src.memory.store import memory_store
from src.memory.short_term import stm

logger = setup_logger(__name__)
class Agent:

    def __init__(self, api_key: str):
        self.GEMINI_API_KEY = api_key
        self.AGENT_MODEL = settings.MODEL
        self.llm = ChatGoogleGenerativeAI(
            model=self.AGENT_MODEL,
            api_key=self.GEMINI_API_KEY,
            temperature=0.2,
            max_retries=0
        )
        self.pipeline = StockAnalysisPipeline(self.llm)

    @retry_with_backoff(max_retries=3)
    def analyze(self, symbol: str, tools: List[str] = None) -> Generator[Dict[str, Any], None, None]:
        """
        Delegates to pipeline stream with selected tools.
        """
        logger.info(f"Tool Selection: StockAnalysisPipeline for {symbol} with tools {tools}")
        yield from self.pipeline.run_analysis(symbol, tools)

    @retry_with_backoff(max_retries=2)
    def parse_intent(self, query: str) -> Dict[str, Any]:
        """
        Classifies user intent and plans tool usage.
        Returns: { 'intent': str, 'symbol': str|None, 'tools': List[str] }
        """
        if not query:
            return {'intent': 'UNKNOWN', 'symbol': None, 'tools': []}
        history_context = stm.get_context_string()
        formatted_prompt = StockAgentPrompts.PLANNING_SYSTEM_PROMPT.format(history_context=history_context)
        messages = [
            SystemMessage(content=formatted_prompt),
            HumanMessage(content=f"User Query: {query}")
        ]
        response = self.llm.invoke(messages)
        content = response.content.strip()
        if content.startswith("```"):
            content = content.replace("```json", "").replace("```", "").strip()
        logger.info(f"Intent Planning Raw Output: {content}")
        try:
            plan = json.loads(content)
            if "tools" not in plan:
                plan["tools"] = ["macro", "micro"]
            if "symbol" in plan and plan["symbol"]:
                plan["symbol"] = plan["symbol"].upper()
            return plan
        except json.JSONDecodeError:
            logger.error("Failed to parse planning JSON. Fallback to Unknown.")
            return {'intent': 'UNKNOWN', 'symbol': None, 'tools': []}

    def respond_conversational(self, query: str) -> Generator[Dict[str, Any], None, None]:
        """
        Generates a conversational response. Yields progress then result.
        """
        logger.info(f"Tool Selection: Conversational Response for query: {query}")
        yield {"type": "progress", "step": "think", "message": "Thinking...", "percent": 20}
        messages = [
            SystemMessage(content=StockAgentPrompts.MAIN_AGENT_PERSONA),
            HumanMessage(content=query)
        ]
        yield {"type": "progress", "step": "generate", "message": "Typing response...", "percent": 60}
        response = self.llm.invoke(messages)
        reply_text = response.content
        logger.info("Generated conversational response.")
        yield {"type": "progress", "step": "complete", "message": "Done.", "percent": 100}
        yield {
            "type": "result",
            "symbol": "AI_AGENT",
            "final_report": reply_text
        }
        
    def parse_symbol(self, query: str) -> str:
        result = self.parse_intent(query)
        if result['intent'] == 'STOCK_QUERY':
            return result['symbol']
        return "UNKNOWN"
