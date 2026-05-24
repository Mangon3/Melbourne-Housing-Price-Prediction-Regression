import pytest
from unittest.mock import patch, MagicMock
from src.tools.registry import get_recent_news, micro_analysis
from src.tools.news import NewsFetcher
from src.tools.micro import MicroModel
import finnhub

def test_registry_get_recent_news():
    with patch("src.tools.registry.news_fetcher.fetch_stock_news") as mock_fetch:
        mock_fetch.return_value = [{"headline": "Test News"}]
        res = get_recent_news.invoke({"symbol": "AAPL", "limit": 1, "timeframe_days": 1})
        assert len(res) == 1
        assert res[0]["headline"] == "Test News"

def test_registry_micro_analysis():
    with patch("src.tools.registry.micro_model.execute_model_training") as mock_exec:
        mock_exec.return_value = {"status": "success", "loss": 0.1}
        res = micro_analysis.invoke({"symbol": "AAPL", "num_epochs": 1, "timeframe_days": 1})
        assert res["status"] == "success"

@patch("src.tools.news.finnhub.Client")
@patch("src.tools.news.SentimentTrainer")
@patch("src.tools.news.joblib")
def test_news_fetcher_init_and_train(mock_joblib, mock_trainer_cls, mock_finnhub):
    # Test initialization when model files don't exist (triggers training)
    with patch("src.tools.news.Path.exists", return_value=False):
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {"status": "success"}
        mock_trainer_cls.return_value = mock_trainer
        fetcher = NewsFetcher()
        mock_trainer.train.assert_called_once()
        
@patch("src.tools.news.finnhub.Client")
def test_news_fetcher_predict(mock_finnhub):
    fetcher = NewsFetcher()
    fetcher.pipeline = MagicMock()
    fetcher.label_encoder = MagicMock()
    
    fetcher.pipeline.predict.return_value = [1]
    fetcher.pipeline.predict_proba.return_value = [[0.1, 0.9, 0.0]]
    fetcher.label_encoder.inverse_transform.return_value = ["Positive"]
    
    res = fetcher.predict_sentiment("Good news")
    assert res["label"] == "Positive"
    assert res["score"] == 0.9

@patch("src.tools.news.finnhub.Client")
def test_news_fetcher_predict_error(mock_finnhub):
    fetcher = NewsFetcher()
    fetcher.pipeline = MagicMock()
    fetcher.label_encoder = MagicMock()
    fetcher.pipeline.predict.side_effect = Exception("Model error")
    
    res = fetcher.predict_sentiment("Bad news")
    assert res["label"] == "Error"

@patch("src.tools.news.finnhub.Client")
def test_news_fetcher_fetch_stock_news(mock_finnhub):
    fetcher = NewsFetcher()
    mock_client = MagicMock()
    fetcher.finnhub_client = mock_client
    
    mock_client.company_news.return_value = [
        {"datetime": 1600000000, "headline": "Apple is doing great!", "summary": "...", "source": "Bloomberg", "url": "http://x", "id": "1"}
    ]
    
    fetcher.predict_sentiment = MagicMock(return_value={"label": "Positive", "score": 0.99})
    
    res = fetcher.fetch_stock_news("AAPL", limit=1)
    assert len(res) == 1
    assert res[0]["headline"] == "Apple is doing great!"
    assert res[0]["sentiment_label"] == "Positive"

@patch("src.tools.news.finnhub.Client")
def test_news_fetcher_fetch_stock_news_api_error(mock_finnhub):
    fetcher = NewsFetcher()
    mock_client = MagicMock()
    fetcher.finnhub_client = mock_client
    mock_client.company_news.side_effect = Exception("API limit reached")
    
    res = fetcher.fetch_stock_news("AAPL")
    assert "error" in res[0]
    assert "limit reached" in res[0]["error"]

@patch("src.tools.micro.trainer")
@patch("src.tools.micro.micro_model_predictor")
def test_micro_model_execute_training(mock_predictor, mock_trainer):
    micro = MicroModel()
    
    # Empty symbols list
    res_empty = micro.execute_model_training("")
    assert res_empty["status"] == "error"
    
    # Successful training
    mock_trainer.train.return_value = {"status": "success", "test_accuracy": 0.85}
    mock_predictor.predict_price_outlook.return_value = {"prediction": "UP"}
    res = micro.execute_model_training("AAPL")
    assert res["training_status"] == "success"
    assert res["inference"]["prediction"] == "UP"
    mock_trainer.train.assert_called_once()
    
    # Failed training
    mock_trainer.train.return_value = {"status": "error", "message": "OOM"}
    res_err = micro.execute_model_training("MSFT")
    assert res_err["status"] == "error"
    assert "OOM" in res_err["message"]

@patch("src.tools.micro.trainer")
def test_micro_model_exception(mock_trainer):
    micro = MicroModel()
    mock_trainer.train.side_effect = Exception("Unknown Exception")
    res = micro.execute_model_training("AAPL")
    assert res["status"] == "error"
    assert "Unknown Exception" in res["message"]
