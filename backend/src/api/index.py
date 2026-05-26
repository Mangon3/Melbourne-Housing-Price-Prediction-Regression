import json
import logging
from typing import Optional, Annotated
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, model_validator
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from src.agent import Agent
from src.utils.logger import setup_logger
from src.utils.errors import format_error
from src.config.settings import settings
from src.memory.store import memory_store
from src.memory.short_term import stm

logger = setup_logger(__name__)

app = FastAPI(
    title="StockAgent API",
    description=settings.API_DESCRIPTION,
    version="1.0.0"
)

Instrumentator().instrument(app).expose(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class AnalyzeRequest(BaseModel):
    symbol: Optional[str] = None
    timeframe_days: Optional[int] = 7
    query: Optional[str] = None
    @model_validator(mode='after')

    def check_symbol_or_query(self) -> 'AnalyzeRequest':
        if not self.symbol and not self.query:
            raise ValueError('Either symbol or query must be provided.')
        return self


def _parse_intent(current_agent: Agent, query: str):
    """Parses user intent, returning (intent, symbol, intent_data)."""
    sanitized_query = query.replace('\n', ' ').replace('\r', '')
    logger.info("Analyzing intent for query: %s", sanitized_query)
    try:
        intent_data = current_agent.parse_intent(query)
        return intent_data['intent'], intent_data.get('symbol'), intent_data
    except Exception:
        logger.exception("Intent parsing failed")
        return "UNKNOWN", None, {'intent': 'UNKNOWN', 'symbol': None, 'tools': []}


def _build_unknown_response():
    """Generates a response stream for unknown intents."""
    yield {"type": "progress", "step": "error", "message": "Analyzing...", "percent": 0}
    yield {
        "type": "result",
        "final_report": "I apologize, but I couldn't understand your request. Could you please specify a stock symbol or ask a financial question?",
        "symbol": "UNKNOWN"
    }


def _get_stream_iterator(intent, symbol, intent_data, current_agent, request):
    """Returns the appropriate stream iterator based on intent."""
    if intent == "STOCK_QUERY" and symbol:
        tools_to_use = intent_data.get('tools') if intent_data else None
        return current_agent.analyze(symbol.upper(), tools=tools_to_use)
    if intent == "GENERAL_CHAT":
        return current_agent.respond_conversational(request.query)
    if intent == "UNKNOWN":
        return _build_unknown_response()
    return None


def _save_memory_if_final_result(chunk: dict, query_text: str, intent: str) -> None:
    """Persists the final result chunk to long-term and short-term memory."""
    if chunk.get("type") == "result" and 'final_report' in chunk:
        memory_store.save_turn(
            user_input=query_text,
            model_output=chunk['final_report'],
            intent=intent
        )
        stm.add_turn(query_text, chunk['final_report'])


@app.get("/")
async def root():
    return {"message": "StockAgent API is running. Use /analyze to generate reports."}

@app.post("/analyze", responses={401: {"description": "Missing API Key"}})
async def analyze_stock(
    request: AnalyzeRequest,
    x_gemini_api_key: Annotated[str | None, Header()] = None
):
    """
    Streaming Endpoint (SSE).
    """

    api_key = x_gemini_api_key or settings.GOOGLE_API_KEY
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API Key. Provide 'X-Gemini-API-Key' header.")
    current_agent = Agent(api_key=api_key)
    intent = "STOCK_QUERY"
    symbol = request.symbol
    query_text = request.query or (f"Analyze {symbol}" if symbol else "")
    intent_data = None

    if not symbol and request.query:
        intent, symbol, intent_data = _parse_intent(current_agent, request.query)

    async def event_generator():
        try:
            stream_iterator = _get_stream_iterator(intent, symbol, intent_data, current_agent, request)
            if stream_iterator is None:
                yield f"data: {json.dumps({'error': 'Invalid Intent'})}\n\n"
                return
            for chunk in stream_iterator:
                _save_memory_if_final_result(chunk, query_text, intent)
                yield f"data: {json.dumps(chunk)}\n\n"
        except Exception:
            logger.exception("Analysis Stream Crash")
            error_payload = format_error(RuntimeError("Analysis stream error"))
            yield f"data: {json.dumps(error_payload)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no" 
        }
    )
