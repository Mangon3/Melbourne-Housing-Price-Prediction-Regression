from typing import List, Dict, Any
from src.config.settings import settings
from langchain_core.tools import tool
from src.tools.news import news_fetcher
from src.tools.micro import micro_model
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

@tool(description=settings.NEWS_TOOL_DESCRIPTION)
def get_recent_news(
    symbol: str,
    limit: int = 5,
    timeframe_days: int = 7
) -> List[Dict[str, Any]]:
    """
    Implementation of news fetching. Description is provided via settings.
    """
    logger.info(f"Calling Finnhub news tool for: {symbol}...")
    return news_fetcher.fetch_stock_news(symbol, limit, timeframe_days)

@tool(description=settings.MICRO_TOOL_DESCRIPTION)
def micro_analysis(
    symbol: str,
    num_epochs: int = 50,
    timeframe_days: int = 150
) -> Dict[str, Any]:
    """
    Implementation of micro-model training. Description is provided via settings.
    """
    logger.info(f"Tool entered: micro_analysis for {symbol}")
    logger.info(f"Calling Micro-Model Training tool for: {symbol}...")
    return micro_model.execute_model_training(
        symbols_list=symbol, 
        num_epochs=num_epochs
    )

tools = [get_recent_news, micro_analysis]
