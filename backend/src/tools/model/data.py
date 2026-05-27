import pandas as pd
import numpy as np
from typing import Dict, Union
from tvDatafeed import TvDatafeed, Interval
from dotenv import load_dotenv
from src.utils.logger import setup_logger

logger = setup_logger(__name__)
load_dotenv()

class TvDataFetcher:

    def __init__(self):
        self.tv = self._initialize_tv_datafeed()

    def _initialize_tv_datafeed(self):
        try:
            tv_instance = TvDatafeed()
            return tv_instance
        except Exception as e:
            if "driver" in str(e).lower() or "selenium" in str(e).lower():
                 logger.critical("FATAL ERROR: Failed to initialize TvDatafeed due to web driver issues.")
                 logger.critical("The library still requires a functioning Chromedriver/Selenium setup, even in anonymous mode.")
                 logger.critical("Original error: %s", e)
            else:
                 logger.exception("Failed to initialize TvDatafeed")
            return None

    def fetch_historical_data(self, symbol: str, timeframe_days: int, exchange: str = "NASDAQ", interval: str = None) -> Union[pd.DataFrame, Dict[str, str]]:
        if self.tv is None:
            return {"error": "TvDatafeed is not initialized. Cannot fetch data."}
        from src.config.settings import settings
        interval_str = interval or settings.DATA_INTERVAL
        tv_interval = Interval.in_daily
        if interval_str == "1h":
            tv_interval = Interval.in_1_hour
            n_bars = int(timeframe_days * 24)
        else:
            tv_interval = Interval.in_daily
            n_bars = int(timeframe_days * 1.5)
        max_retries = 3
        last_error = None
        for attempt in range(max_retries):
            try:
                data = self.tv.get_hist(
                    symbol=symbol,
                    exchange=exchange,
                    interval=tv_interval,
                    n_bars=n_bars
                )
                if data is None or data.empty:
                    last_error = f"No historical data found for {symbol} on {exchange}."
                    if attempt < max_retries - 1:
                        logger.warning(f"No data for {symbol}, retrying ({attempt+1}/{max_retries})...")
                        self.tv = self._initialize_tv_datafeed()
                        import time
                        time.sleep(2)
                        continue
                    return {"error": last_error}
                
                data.columns = [col.lower() for col in data.columns]
                returns_5d = data['close'].pct_change(5).fillna(0)
                proxy_sentiment = 1 / (1 + np.exp(-returns_5d * 10))
                rng = np.random.default_rng(42)
                noise = rng.normal(0, 0.05, len(data))
                data['News_Sentiment_Score'] = (proxy_sentiment + noise).clip(0, 1)
                data = data[['open', 'high', 'low', 'close', 'volume', 'News_Sentiment_Score']]
                return data
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    logger.warning(f"Error fetching data for {symbol}, retrying ({attempt+1}/{max_retries}). Error: {e}")
                    self.tv = self._initialize_tv_datafeed()
                    import time
                    time.sleep(2)
                    continue
                return {"error": f"Data fetching error for {symbol} after {max_retries} attempts: {last_error}"}
        return {"error": str(last_error)}
            
tv_data_fetcher = TvDataFetcher()
