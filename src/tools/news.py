from src.config.settings import settings
import finnhub
import joblib
from pathlib import Path
from typing import Dict, Any, List
from src.tools.sentiment.train import SentimentTrainer
from src.utils.logger import setup_logger
from datetime import datetime, timedelta

logger = setup_logger(__name__)
class NewsFetcher:
    
    def __init__(self):
        self.FINNHUB_API_KEY = settings.FINNHUB_API_KEY
        self.finnhub_client = None
        if self.FINNHUB_API_KEY:
            self.finnhub_client = finnhub.Client(api_key=self.FINNHUB_API_KEY)
        else:
            logger.warning("FINNHUB_API_KEY not found in environment.")
        self.root_dir = Path(__file__).resolve().parent.parent.parent
        if Path("/data").exists():
            self.model_dir = Path("/data/models")
        else:
            self.model_dir = self.root_dir / "data" / "datasets" / "models"
        self.model_path = self.model_dir / "sentiment_pipeline.pkl"
        self.encoder_path = self.model_dir / "sentiment_label_encoder.pkl"
        self.pipeline = None
        self.label_encoder = None
        if not self.model_path.exists() or not self.encoder_path.exists():
            logger.info("Sentiment model missing at %s. Initiating auto-training...", self.model_path)
            self.train_model()
        else:
            self._load_model()

    def _load_model(self):
        """Attempts to assign self.pipeline and self.label_encoder if files exist."""
        try:
            if self.model_path.exists() and self.encoder_path.exists():
                self.pipeline = joblib.load(self.model_path)
                self.label_encoder = joblib.load(self.encoder_path)
            else:
                logger.info("Sentiment model not found at %s. Training might be required.", self.model_path)
        except Exception as e:
            logger.warning("Failed to load sentiment model: %s", e)

    def train_model(self):
        """Triggers the sentiment model training pipeline."""
        logger.info("Triggering sentiment model training from NewsFetcher...")
        trainer = SentimentTrainer()
        result = trainer.train()
        self._load_model()
        return result

    def predict_sentiment(self, text: str) -> Dict[str, Any]:
        """Predicts sentiment for a given text."""
        if not self.pipeline or not self.label_encoder:
            return {"label": "N/A", "score": 0.0}
        try:
            prediction_idx = self.pipeline.predict([text])[0]
            sentiment_label = self.label_encoder.inverse_transform([prediction_idx])[0]
            probs = self.pipeline.predict_proba([text])[0]
            confidence = max(probs)
            return {
                "label": sentiment_label,
                "score": float(f"{confidence:.2f}")
            }
        except Exception:
            logger.exception("Sentiment prediction error")
            return {"label": "Error", "score": 0.0}

    def fetch_stock_news(
        self,
        symbol: str, 
        limit: int = 10,
        timeframe_days: int = 7
    ) -> List[Dict[str, Any]]:
        if self.finnhub_client is None:
            logger.error("Finnhub client is not initialized.")
            return [{"error": "Finnhub client not initialized."}]
        end_date = datetime.now()
        start_date = end_date - timedelta(days=timeframe_days)
        to_timestamp = end_date.strftime("%Y-%m-%d")
        from_timestamp = start_date.strftime("%Y-%m-%d")
        try:
            news_data = self.finnhub_client.company_news(
                symbol=symbol.upper(), 
                _from=from_timestamp, 
                to=to_timestamp
            )
            if not news_data:
                logger.info("No news found for %s.", symbol)
                return [{"warning": f"No news found for {symbol}."}]
            cleaned_articles = []
            for article in news_data:
                publish_timestamp = article.get('datetime')
                try:
                    created_at_dt = datetime.fromtimestamp(publish_timestamp)
                    created_at_str = created_at_dt.strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    created_at_str = "N/A"
                headline = article.get('headline')
                if not headline:
                    continue
                sentiment_result = self.predict_sentiment(headline)
                cleaned_articles.append({
                    "id": article.get('id'),
                    "symbols": [symbol],
                    "created_at": created_at_str,
                    "headline": headline,
                    "summary": article.get('summary', 'No summary available.'),
                    "source": article.get('source', 'N/A'),
                    "url": article.get('url', 'N/A'),
                    "sentiment_label": sentiment_result['label'],
                    "sentiment_score": sentiment_result['score']
                })
                if len(cleaned_articles) >= limit:
                    break
            if not cleaned_articles:
                logger.warning("No valid news articles found for %s after filtering.", symbol)
                return [{"warning": f"No valid news found for {symbol}."}]
            return cleaned_articles
        except finnhub.FinnhubAPIException as e:
            if "API limit reached" in str(e):
                error_message = f"Finnhub API limit reached for {symbol}."
                logger.warning(error_message)
            else:
                error_message = f"Finnhub API Error: {e}"
                logger.error(error_message)
            return [{"error": error_message}]
        except Exception as e:
            logger.exception("Failed to retrieve news due to unexpected issue: %s", e)
            return [{"error": f"Failed to retrieve news due to unexpected issue: {e}"}]

news_fetcher = NewsFetcher()
