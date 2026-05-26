import pytest
from unittest.mock import patch, MagicMock
from src.tools.sentiment.train import SentimentTrainer
import pandas as pd

@patch("src.tools.sentiment.train.Path")
def test_sentiment_trainer_init_data_path(mock_path):
    # Test line 19 branch
    mock_path.return_value.exists.return_value = True
    trainer = SentimentTrainer()
    assert mock_path.return_value.exists.called

@patch("src.tools.sentiment.train.Path.exists")
def test_load_data_not_found(mock_exists):
    mock_exists.return_value = False
    trainer = SentimentTrainer()
    with pytest.raises(FileNotFoundError):
        trainer.load_data()

@patch("src.tools.sentiment.train.Path.mkdir")
@patch("src.tools.sentiment.train.pd.read_csv")
@patch("src.tools.sentiment.train.joblib.dump")
@patch("src.tools.sentiment.train.Path.exists")
def test_load_data_success(mock_exists, mock_dump, mock_read_csv, mock_mkdir):
    mock_exists.return_value = True
    df_mock = pd.DataFrame({
        "text": ["good", "bad"],
        "sentiment": ["positive", "negative"]
    })
    mock_read_csv.return_value = df_mock
    
    trainer = SentimentTrainer()
    df, le = trainer.load_data()
    
    assert len(df) == 2
    assert "label" in df.columns
    mock_dump.assert_called_once()

@patch("src.tools.sentiment.train.SentimentTrainer.load_data")
@patch("src.tools.sentiment.train.joblib.dump")
@patch("src.tools.sentiment.train.train_test_split")
@patch("src.tools.sentiment.train.Pipeline")
def test_train_and_call(mock_pipeline_cls, mock_split, mock_dump, mock_load_data):
    # Mock load_data
    df_mock = pd.DataFrame({
        "text": ["good text", "bad text"],
        "label": [1, 0]
    })
    mock_load_data.return_value = (df_mock, MagicMock())
    
    # Mock split
    mock_split.return_value = (["good text"], ["bad text"], [1], [0])
    
    # Mock pipeline
    mock_pipeline = MagicMock()
    mock_pipeline.score.return_value = 0.95
    mock_pipeline_cls.return_value = mock_pipeline
    
    trainer = SentimentTrainer()
    
    # Test __call__ which calls train()
    res = trainer()
    
    assert res["status"] == "success"
    assert res["accuracy"] == 0.95
    mock_pipeline.fit.assert_called_once()
    mock_pipeline.score.assert_called_once()
    mock_dump.assert_called_once()
