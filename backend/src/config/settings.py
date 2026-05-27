from pydantic_settings import BaseSettings
from typing import Optional, List
from pathlib import Path

class Settings(BaseSettings):

    # External API Keys
    GOOGLE_API_KEY: Optional[str] = None
    MODEL: str = "gemini-2.5-flash"
    TEXT_EMBEDDING: str = "models/gemini-embedding-001"

    # Redis Configuration
    REDIS_URL: Optional[str] = None
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0

    # Financial Data Providers
    FINNHUB_API_KEY: Optional[str] = None

    # Micro Model Hyperparameters
    SEQ_LEN: int = 10
    HIDDEN_DIM: int = 32
    NUM_LAYERS: int = 2
    DROPOUT: float = 0.1
    
    # Features
    FEATURE_COLUMNS: List[str] = [
        'Dist_EMA_10', 'Dist_EMA_50', 
        'MACD', 'Signal_Line', 'RSI', 
        'Stochastic_K', 'Stochastic_D',
        'ATR_Ratio', 'BB_Width', 'Vol_Ratio', 'OBV_Slope', 
        'News_Sentiment_Score',
        'Log_Return_1d', 'Log_Return_5d',
        'dow_sin', 'dow_cos',
        'hour_sin', 'hour_cos'
    ]
    
    # OHLCV dataframe
    REQUIRED_OHLCV_COLS: List[str] = ['open', 'high', 'low', 'close', 'volume', 'News_Sentiment_Score']
    DATA_TIMEFRAME_DAYS: int = 730
    DATA_INTERVAL: str = "1h"
    CONVERSATION_MEMORY_LIMIT: int = 10
    
    # API Definitions
    API_DESCRIPTION: str = "API for StockAgent - Financial Analysis and Prediction"
    NEWS_TOOL_DESCRIPTION: str = """Grab news on stock for the given symbol, limiting the number of articles and the timeframe (in days). Args: symbol (str): The stock ticker symbol (e.g., 'MSFT'). limit (int): Maximum number of articles to return (default: 5). timeframe_days (int): How many days back to search for news (default: 7)."""
    MICRO_TOOL_DESCRIPTION: str = """Initiates the micro-model training pipeline on the specified stock symbol. It fetches historical data, calculates technical features, and trains a LSTM neural network model for prediction. Use this tool when the user asks to 'train the model', 'retrain the micro-model', or 'update the prediction model'. Args: symbol (str): The stock ticker symbol to train on (e.g., 'AAPL'). num_epochs (int): The number of training iterations to run (default: 50). timeframe_days (int): The number of days of historical data to use for training (default: 150)."""
    
    @property
    def INPUT_SIZE(self) -> int:
        return len(self.FEATURE_COLUMNS)

    @property
    def PROJECT_ROOT(self) -> Path:
        return Path(__file__).resolve().parent.parent.parent

    @property
    def MODEL_PATH(self) -> Path:
        if Path("/data").exists():
            return Path("/data/datasets/models/micro.pth")
        return self.PROJECT_ROOT / "data" / "datasets" / "models" / "micro.pth"

    @property
    def VECTOR_STORE_PATH(self) -> str:
        if Path("/data").exists():
            return "/data/vector_stores/conversation_history"
        return str(self.PROJECT_ROOT / "data" / "vector_stores" / "conversation_history")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
        
settings = Settings()
