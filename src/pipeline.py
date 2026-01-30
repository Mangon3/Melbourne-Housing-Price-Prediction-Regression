from typing import Dict, Any, List
import json
from src.utils.errors import format_error
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from src.tools.news import news_fetcher
from src.tools.micro import micro_model
from src.rag.core import rag_system
from src.utils.logger import setup_logger
from src.config.prompts import StockAgentPrompts
logger = setup_logger(__name__)
class StockAnalysisPipeline:
    """
    1. Macro News Analysis
    2. Micro Model Training & Inference
    3. Final Synthesis
    """

    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm

    def run_analysis(self, symbol: str, tools: List[str] = None):  
        """
        Orchestrates the analysis based on the selected tools plan.
        tools: List of strings, e.g. ["macro"], ["micro"], or ["macro", "micro"]
        """
        logger.info(f"--- STARTING PIPELINE ANALYSIS FOR {symbol} (Tools: {tools}) ---")
        if not tools:
            tools = ["macro", "micro"] 
        yield {"type": "progress", "step": "init", "message": f"Initializing {', '.join(tools)} Analysis...", "percent": 5}
        macro_analysis_text = "Not requested."
        micro_data = {"status": "skipped"}
        if "macro" in tools:
            yield {"type": "progress", "step": "news", "message": "Fetching & Analyzing Macro News...", "percent": 20}
            logger.info("[Step 1] Running Macro Analysis...")
            try:
                news_data = news_fetcher.fetch_stock_news(symbol=symbol, limit=10, timeframe_days=7)
                rag_system.ingest_news_documents(news_data)
                yield {"type": "progress", "step": "rag", "message": "Retrieving Context...", "percent": 35}
                query = f"Analyze the macro outlook for {symbol} based on recent news."
                context, _ = rag_system.retrieve_context(query)
                if not context or "No relevant" in context:
                    macro_analysis_text = "No significant macro news found for this period."
                else:
                    macro_analysis_text = context
            except Exception as e:
                logger.exception("Macro analysis failed") 
                error_payload = format_error(e)
                macro_analysis_text = "Macro analysis unavailable due to a provider error."
                yield error_payload
            logger.info("[Step 1] Macro Data Ready.")
        if "micro" in tools:
            yield {"type": "progress", "step": "model", "message": "Training Micro-Model (LSTM)...", "percent": 60}
            logger.info("[Step 2] Running Micro Analysis...")
            try:
                micro_data = micro_model.execute_model_training(symbols_list=symbol, num_epochs=50)
            except Exception as e:
                logger.exception("Micro analysis failed") 
                error_payload = format_error(e)
                micro_data = {"error": "Model training failed.", "status": "failed"}
                yield error_payload
            logger.info("[Step 2] Micro Data Ready.")
        yield {"type": "progress", "step": "synthesis", "message": "Synthesizing Final Report...", "percent": 85}
        logger.info("[Step 3] Synthesizing Final Report...")
        final_report = self._synthesize_report(
            symbol=symbol,
            macro_text=macro_analysis_text,
            micro_data=micro_data,
            tools_used=tools
        )
        yield {"type": "progress", "step": "complete", "message": "Finalizing...", "percent": 98}
        logger.info(f"--- PIPELINE COMPLETE FOR {symbol} ---")
        yield {
            "type": "result", 
            "final_report": final_report,
            "macro_analysis": macro_analysis_text,
            "micro_analysis": micro_data
        }
        
    def _synthesize_report(self, symbol: str, macro_text: str, micro_data: Dict[str, Any], tools_used: List[str]) -> str:
        micro_json = json.dumps(micro_data, indent=2)
        instruction_note = ""
        if "macro" not in tools_used:
             instruction_note += "\nNOTE: Macro Analysis was NOT requested. Do not hallucinate news. Focus on Technicals."
        if "micro" not in tools_used:
             instruction_note += "\nNOTE: Micro Analysis was NOT requested. Focus on News and Sentiment."
        formatted_system_prompt = StockAgentPrompts.REPORT_SYNTHESIS_SYSTEM.format(
            symbol=symbol,
            macro_text=macro_text,
            micro_json=micro_json
        ) + instruction_note
        messages = [
            SystemMessage(content=formatted_system_prompt),
            HumanMessage(content=StockAgentPrompts.get_report_synthesis_user_msg(symbol))
        ]
        return self.llm.invoke(messages).content
