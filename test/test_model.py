import pytest
import pandas as pd
import numpy as np
import torch
from unittest.mock import patch, MagicMock

from src.tools.model.data import TvDataFetcher
from src.tools.model.feature import FeatureCalculator
from src.tools.model.neural import HybridStockNet
from src.tools.model.infer import MicroModelPredictor, inference_results
from src.tools.model.train import StockDataset, DirectionalMSELoss, StockModelTrainer
from src.config.settings import settings

# --- data.py Tests ---
@patch("src.tools.model.data.TvDatafeed")
def test_tv_data_fetcher_init_success(mock_tv):
    fetcher = TvDataFetcher()
    assert fetcher.tv is not None

@patch("src.tools.model.data.TvDatafeed")
def test_tv_data_fetcher_init_driver_error(mock_tv):
    mock_tv.side_effect = Exception("driver issue")
    fetcher = TvDataFetcher()
    assert fetcher.tv is None

@patch("src.tools.model.data.TvDatafeed")
def test_tv_data_fetcher_init_other_error(mock_tv):
    mock_tv.side_effect = Exception("some issue")
    fetcher = TvDataFetcher()
    assert fetcher.tv is None

def test_fetch_historical_data_no_tv():
    fetcher = TvDataFetcher()
    fetcher.tv = None
    res = fetcher.fetch_historical_data("AAPL", 5)
    assert "error" in res

@patch("src.tools.model.data.TvDatafeed")
def test_fetch_historical_data_success(mock_tv):
    fetcher = TvDataFetcher()
    mock_data = pd.DataFrame({
        "open": [100]*10, "high": [105]*10, "low": [95]*10, "close": [102]*10, "volume": [1000]*10
    })
    fetcher.tv.get_hist.return_value = mock_data
    
    # default daily interval
    res = fetcher.fetch_historical_data("AAPL", 5)
    assert isinstance(res, pd.DataFrame)
    assert "News_Sentiment_Score" in res.columns
    
    # 1h interval
    res_1h = fetcher.fetch_historical_data("AAPL", 5, interval="1h")
    assert isinstance(res_1h, pd.DataFrame)
    
    # daily interval
    res_daily = fetcher.fetch_historical_data("AAPL", 5, interval="1d")
    assert isinstance(res_daily, pd.DataFrame)

@patch("src.tools.model.data.TvDatafeed")
def test_fetch_historical_data_empty_and_error(mock_tv):
    fetcher = TvDataFetcher()
    
    fetcher.tv.get_hist.return_value = pd.DataFrame()
    res = fetcher.fetch_historical_data("AAPL", 5)
    assert "error" in res
    
    fetcher.tv.get_hist.side_effect = Exception("Fetch err")
    res2 = fetcher.fetch_historical_data("AAPL", 5)
    assert "error" in res2

# --- feature.py Tests ---
def test_feature_calculator_missing_cols():
    df = pd.DataFrame({"close": [100]})
    with pytest.raises(ValueError):
        FeatureCalculator.calculate_features(df)

def test_feature_calculator_success():
    df = pd.DataFrame({
        "open": np.random.rand(50),
        "high": np.random.rand(50),
        "low": np.random.rand(50),
        "close": np.random.rand(50),
        "volume": np.random.rand(50),
        "News_Sentiment_Score": np.random.rand(50)
    })
    # Without datetime index
    res = FeatureCalculator.calculate_features(df)
    assert not res.empty
    
    # With datetime index
    df.index = pd.date_range("2020-01-01", periods=50)
    res2 = FeatureCalculator.calculate_features(df)
    assert not res2.empty
    
    # With invalid datetime string index
    df.index = ["invalid"] * 50
    res3 = FeatureCalculator.calculate_features(df)
    assert not res3.empty

# --- neural.py Tests ---
def test_hybrid_stock_net():
    model = HybridStockNet(input_size=10, hidden_dim=16, num_layers=1)
    dummy_input = torch.randn(2, 5, 10) # batch=2, seq=5, features=10
    price, prob = model(dummy_input)
    assert price.shape == (2, 1)
    assert prob.shape == (2, 1)

# --- infer.py Tests ---
def test_inference_results():
    assert inference_results(0.7)[0] == "STRONG_BULLISH"
    assert inference_results(0.55)[0] == "BULLISH"
    assert inference_results(0.5)[0] == "NEUTRAL"
    assert inference_results(0.45)[0] == "BEARISH"
    assert inference_results(0.3)[0] == "STRONG_BEARISH"

@patch("pathlib.Path.exists", return_value=False)
def test_micromodel_load_model_not_found(mock_exists):
    predictor = MicroModelPredictor()
    res = predictor.predict_price_outlook("AAPL")
    assert "error" in res

@patch("pathlib.Path.exists", return_value=True)
@patch("src.tools.model.infer.torch.load")
@patch("src.tools.model.infer.HybridStockNet.load_state_dict")
def test_micromodel_predict_success_and_errors(mock_load_state, mock_load, mock_exists):
    predictor = MicroModelPredictor()
    predictor.data_fetcher = MagicMock()
    
    # Data fetch error
    predictor.data_fetcher.fetch_historical_data.return_value = {"error": "fetch error"}
    res1 = predictor.predict_price_outlook("AAPL")
    assert "error" in res1
    
    # Insufficient data
    df_short = pd.DataFrame({col: np.random.rand(2) for col in settings.REQUIRED_OHLCV_COLS})
    predictor.data_fetcher.fetch_historical_data.return_value = df_short
    res2 = predictor.predict_price_outlook("AAPL")
    assert "error" in res2
    
    # Success
    df_valid = pd.DataFrame({col: np.random.rand(60) for col in settings.REQUIRED_OHLCV_COLS})
    predictor.data_fetcher.fetch_historical_data.return_value = df_valid
    
    with patch("src.tools.model.infer.HybridStockNet.forward") as mock_forward:
        mock_forward.return_value = (torch.tensor([[5.0]]), torch.tensor([[0.8]]))
        res3 = predictor.predict_price_outlook("AAPL")
        assert res3["signal"] == "STRONG_BULLISH"

# --- train.py Tests ---
def test_stock_dataset_missing_close():
    df = pd.DataFrame({"open": [1]})
    with pytest.raises(ValueError):
        StockDataset.create_sequences(df, 5, ["open"])

def test_stock_dataset_create_sequences():
    ds = StockDataset(np.array([[1]]), np.array([1]))
    assert len(ds) == 1
    assert ds[0][0] == torch.tensor([1.])
    assert ds[0][1] == torch.tensor([1.])
    
    df = pd.DataFrame({"close": np.random.rand(10), "feature": np.random.rand(10)})
    X, y = StockDataset.create_sequences(df, 3, ["feature"])
    assert X.shape[0] > 0
    assert y.shape[0] > 0
    
    # Test insufficient data
    df2 = pd.DataFrame({"close": [1, 2], "feature": [1, 2]})
    X2, y2 = StockDataset.create_sequences(df2, 5, ["feature"])
    assert len(X2) == 0

def test_directional_mse_loss():
    loss_fn = DirectionalMSELoss()
    pred = torch.tensor([1.0, -1.0])
    target = torch.tensor([1.0, 1.0])
    loss = loss_fn(pred, target)
    assert loss.item() > 0

@patch("src.tools.model.train.tv_data_fetcher.fetch_historical_data")
def test_stock_model_trainer_fetch(mock_fetch):
    trainer = StockModelTrainer()
    
    # Success on first exchange
    mock_fetch.return_value = pd.DataFrame()
    res = trainer._fetch_from_exchanges("AAPL")
    assert isinstance(res, pd.DataFrame)
    
    # All fail
    mock_fetch.return_value = {"error": "fail"}
    res2 = trainer._fetch_from_exchanges("AAPL")
    assert "error" in res2

@patch("src.tools.model.train.FeatureCalculator.calculate_features")
def test_stock_model_trainer_process_symbol(mock_calc):
    trainer = StockModelTrainer()
    df_raw = pd.DataFrame({col: np.random.rand(100) for col in settings.REQUIRED_OHLCV_COLS})
    df_features = pd.DataFrame({col: np.random.rand(100) for col in settings.FEATURE_COLUMNS + ['close']})
    mock_calc.return_value = df_features.copy()
    res = trainer._process_symbol_data("AAPL", df_raw)
    assert res is not None
    assert len(res) == 4
    
    # Test x_data.size == 0
    with patch("src.tools.model.train.StockDataset.create_sequences") as mock_seq:
        mock_seq.return_value = (np.array([]), np.array([]))
        res_empty = trainer._process_symbol_data("AAPL", df_raw)
        assert res_empty is None
    
    # Force failure via ValueError
    mock_calc.side_effect = ValueError("calc error")
    res_err = trainer._process_symbol_data("AAPL", df_raw)
    assert res_err is None

    # Force failure via general Exception
    mock_calc.side_effect = Exception("gen error")
    res_err2 = trainer._process_symbol_data("AAPL", df_raw)
    assert res_err2 is None
    
    # Test insufficient data size after seq creation
    df_short = pd.DataFrame({col: np.random.rand(2) for col in settings.REQUIRED_OHLCV_COLS})
    mock_calc.side_effect = None
    mock_calc.return_value = df_short.copy()
    res_short = trainer._process_symbol_data("AAPL", df_short)
    assert res_short is None

@patch("src.tools.model.train.StockModelTrainer._load_and_prepare_data")
def test_stock_model_trainer_train(mock_load_prep):
    trainer = StockModelTrainer()
    
    # Mock mismatch to test RuntimeError
    mock_dataset_err = MagicMock()
    mock_dataset_err.X.shape = (10, settings.INPUT_SIZE, 5) 
    mock_load_prep.return_value = (mock_dataset_err, mock_dataset_err)
    
    res_err = trainer.train(num_epochs=1)
    assert res_err["status"] == "error"
    
    # Proper shape
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size):
            self.X = torch.randn(size, 5, settings.INPUT_SIZE)
            self.y = torch.randn(size, 1)
        def __len__(self): return len(self.X)
        def __getitem__(self, i): return self.X[i], self.y[i]
        
    ds_train = DummyDataset(10)
    ds_test = DummyDataset(10)
    mock_load_prep.return_value = (ds_train, ds_test)
    
    with patch("pathlib.Path.mkdir"):
        with patch("src.tools.model.train.torch.save"):
            # Trigger early stopping with patience
            res = trainer.train(symbols=["AAPL"], num_epochs=20, batch_size=2)
            assert res["status"] == "success"

@patch("src.tools.model.train.StockModelTrainer._process_symbol_data")
@patch("src.tools.model.train.StockModelTrainer._fetch_from_exchanges")
def test_load_and_prepare_data_success_branches(mock_fetch, mock_process):
    trainer = StockModelTrainer()
    mock_fetch.return_value = pd.DataFrame()
    
    # Test skipping a symbol that fails process_symbol_data
    mock_process.side_effect = [
        None, 
        (np.ones((2,5)), np.ones(2), np.ones((2,5)), np.ones(2)),
        (np.ones((2,5)), np.ones(2), np.empty((0,5)), np.empty((0,)))
    ]
    ds_train, ds_test = trainer._load_and_prepare_data(["AAPL", "MSFT", "GOOG"])
    assert len(ds_train) == 4
    assert len(ds_test) == 2
    
    # Test empty test_x_list
    mock_process.side_effect = [
        (np.ones((2,5)), np.ones(2), np.empty((0,5)), np.empty((0,)))
    ]
    ds_train2, ds_test2 = trainer._load_and_prepare_data(["AAPL"])
    assert len(ds_train2) == 2
    assert len(ds_test2) == 0

@patch("src.tools.model.train.StockModelTrainer._fetch_from_exchanges")
def test_load_and_prepare_data_exceptions(mock_fetch):
    trainer = StockModelTrainer()
    mock_fetch.return_value = {"error": "err"}
    with patch("src.tools.model.train.time.sleep"):
        with pytest.raises(RuntimeError):
            trainer._load_and_prepare_data(["AAPL"])
