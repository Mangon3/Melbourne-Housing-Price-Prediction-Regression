import torch
torch.set_num_threads(1)
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
import time 
import random 
from src.config.settings import settings
from src.tools.model.data import tv_data_fetcher
from src.tools.model.neural import HybridStockNet
from src.tools.model.feature import FeatureCalculator
from src.utils.device import get_device
from src.utils.logger import setup_logger

logger = setup_logger(__name__)
class StockDataset(Dataset):
    """
    Custom PyTorch Dataset for handling time-series sequence data.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    @staticmethod
    def create_sequences(df: pd.DataFrame, seq_len: int, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        if 'close' not in df.columns:
             raise ValueError("DataFrame must contain a 'close' column to calculate the target.")
        df['target'] = np.log(df['close'].shift(-1) / df['close']) * 100.0
        df = df.iloc[:-1]
        df = df.dropna(subset=feature_cols + ['target'])
        feature_data = df[feature_cols].values
        target_data = df['target'].values
        for i in range(len(feature_data) - seq_len):
            seq = feature_data[i : (i + seq_len)]
            label = target_data[i + seq_len - 1] 
            X.append(seq)
            y.append(label)
        if not X:
            logger.warning(f"Could not create any sequences. Data length: {len(df)}, SEQ_LEN: {seq_len}")
            return np.array([]), np.array([])
        return np.array(X), np.array(y)
class DirectionalMSELoss(nn.Module):

    def __init__(self, penalty_factor: float = 5.0):
        super(DirectionalMSELoss, self).__init__()
        self.mse = nn.MSELoss()
        self.penalty_factor = penalty_factor

    def forward(self, pred, target):
        loss_mse = self.mse(pred, target)
        interaction = -1 * (pred * target)
        directional_penalty = torch.relu(interaction).mean()
        total_loss = loss_mse + (self.penalty_factor * directional_penalty)
        return total_loss
class StockModelTrainer:
    DEFAULT_SYMBOLS: List[str] = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    DEFAULT_EPOCHS: int = 50
    DEFAULT_BATCH_SIZE: int = 32
    LEARNING_RATE: float = 0.001
    TEST_SIZE_RATIO: float = 0.2
    DEVICE = get_device()

    def __init__(self):
        self.best_test_accuracy = 0.0
        logger.info(f"Trainer initialized. Target device: {self.DEVICE}.")

    def _load_and_prepare_data(self, symbols: List[str]) -> Tuple[StockDataset, StockDataset]:
        """Fetches data, calculates features, and creates the final sequence datasets with temporal splitting."""
        train_X_list, train_y_list = [], []
        test_X_list, test_y_list = [], []
        logger.info(f"Starting data fetch and feature engineering for {len(symbols)} symbols...")
        for symbol in symbols:
            logger.info(f"Fetching data for {symbol}...")
            df_raw = tv_data_fetcher.fetch_historical_data(
                symbol, 
                timeframe_days=settings.DATA_TIMEFRAME_DAYS, 
                exchange="NASDAQ"
            )

            if isinstance(df_raw, dict) and "error" in df_raw:
                logger.info(f"NASDAQ fetch failed for {symbol}, trying NYSE...")
                df_raw = tv_data_fetcher.fetch_historical_data(
                    symbol, 
                    timeframe_days=settings.DATA_TIMEFRAME_DAYS, 
                    exchange="NYSE"
                )

            if isinstance(df_raw, dict) and "error" in df_raw:
                logger.info(f"NYSE fetch failed for {symbol}, trying BINANCE...")
                df_raw = tv_data_fetcher.fetch_historical_data(
                    symbol, 
                    timeframe_days=settings.DATA_TIMEFRAME_DAYS, 
                    exchange="BINANCE"
                )

            if isinstance(df_raw, dict) and "error" in df_raw:
                logger.error(f"Skipping {symbol}: Data fetch failed on NASDAQ, NYSE, and BINANCE: {df_raw['error']}")
                time.sleep(random.uniform(1, 3))
                continue
            try:
                df_features = FeatureCalculator.calculate_features(df_raw.copy())
                df_features['close'] = df_raw['close']
                X, y = StockDataset.create_sequences(
                    df_features,
                    settings.SEQ_LEN, 
                    settings.FEATURE_COLUMNS
                )
                if X.size > 0:
                    split_index = int(len(X) * (1 - self.TEST_SIZE_RATIO))
                    if split_index >= len(X): split_index = len(X) - 1
                    if split_index <= 0: split_index = 1
                    X_train, X_test = X[:split_index], X[split_index:]
                    y_train, y_test = y[:split_index], y[split_index:]
                    train_X_list.append(X_train)
                    train_y_list.append(y_train)
                    test_X_list.append(X_test)
                    test_y_list.append(y_test)
                    logger.info(f"Success for {symbol}: Created {len(X)} sequences (Train: {len(X_train)}, Test: {len(X_test)}).")
                else:
                    logger.warning(f"Skipping {symbol}: Insufficient data after sequence creation.")
            except ValueError as e:
                logger.error(f"Skipping {symbol} due to feature calculation error: {e}")
            except Exception as e:
                logger.error(f"An unexpected error occurred for {symbol}: {str(e)}")

        if not train_X_list:
            raise RuntimeError("No training data could be generated for any symbol.")
        X_train_combined = np.concatenate(train_X_list, axis=0)
        y_train_combined = np.concatenate(train_y_list, axis=0)

        if test_X_list:
            X_test_combined = np.concatenate(test_X_list, axis=0)
            y_test_combined = np.concatenate(test_y_list, axis=0)
        else:
            X_test_combined = np.array([])
            y_test_combined = np.array([])

        return StockDataset(X_train_combined, y_train_combined), StockDataset(X_test_combined, y_test_combined)

    def _evaluate_model(self, model: nn.Module, dataloader: DataLoader) -> Tuple[float, float]:
        """Evaluates loss and accuracy on a given dataset (train or test)."""
        model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        criterion_reg = nn.MSELoss()
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.DEVICE), labels.to(self.DEVICE)
                price_pred, prob_pred = model(inputs)
                loss = criterion_reg(price_pred, labels)
                running_loss += loss.item()
                target_sign = (labels > 0).float()
                pred_sign = (prob_pred > 0.5).float()
                correct_predictions += (pred_sign == target_sign).sum().item()
                total_samples += labels.size(0)
        avg_loss = running_loss / len(dataloader)
        if total_samples > 0:
            directional_accuracy = correct_predictions / total_samples
        else:
            directional_accuracy = 0.0
        return avg_loss, directional_accuracy

    def train(self, symbols: List[str] = None, num_epochs: int = None, batch_size: int = None) -> Dict[str, Any]:
        logger.info(f"Trainer.train started. Device: {self.DEVICE}")
        """
        The main public method to start and run the model training process.
        Args:
            symbols (List[str], optional): List of stock symbols to train on. 
                Defaults to internal list if None.
            num_epochs (int, optional): Number of training epochs. Defaults to 50 if None.
            batch_size (int, optional): Batch size for DataLoader. Defaults to 32 if None.
        Returns:
            Dict[str, Any]: Final training metrics and status.
        """

        symbols = symbols if symbols is not None else self.DEFAULT_SYMBOLS
        num_epochs = num_epochs if num_epochs is not None else self.DEFAULT_EPOCHS
        batch_size = batch_size if batch_size is not None else self.DEFAULT_BATCH_SIZE

        try:
            train_dataset, test_dataset = self._load_and_prepare_data(symbols)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            logger.info(f"\nTotal training samples available: {len(train_dataset)}")
            logger.info(f"Total testing samples available: {len(test_dataset)}")

            if train_dataset.X.shape[2] != settings.INPUT_SIZE:
                 raise RuntimeError(f"Feature count mismatch! Expected {settings.INPUT_SIZE}, got {train_dataset.X.shape[2]}.")

            model = HybridStockNet(
                input_size=settings.INPUT_SIZE,
                hidden_dim=settings.HIDDEN_DIM,
                num_layers=settings.NUM_LAYERS,
                dropout=settings.DROPOUT
            ).to(self.DEVICE)

            criterion_reg = DirectionalMSELoss(penalty_factor=5.0)
            criterion_cls = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=self.LEARNING_RATE)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
            self.best_test_accuracy = 0.0
            best_test_loss = float('inf')
            early_stop_counter = 0
            patience = 12
            logger.info(f"Training started for {num_epochs} epochs on symbols: {', '.join(symbols)}")

            for epoch in range(num_epochs):
                model.train()

                running_loss = 0.0

                for i, (inputs, labels) in enumerate(train_dataloader):
                    inputs, labels = inputs.to(self.DEVICE), labels.to(self.DEVICE)
                    optimizer.zero_grad()
                    price_pred, prob_pred = model(inputs)
                    target_price = labels
                    target_prob = (labels > 0).float()
                    loss_reg = criterion_reg(price_pred, target_price)
                    loss_cls = criterion_cls(prob_pred, target_prob)
                    loss = loss_reg + 0.5 * loss_cls
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                train_loss = running_loss / len(train_dataloader)
                test_loss, test_accuracy = self._evaluate_model(model, test_dataloader)
                logger.info(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.4f}")

                if test_accuracy > self.best_test_accuracy:
                    self.best_test_accuracy = test_accuracy
                    settings.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), settings.MODEL_PATH)
                    logger.info(f"--> Model saved! New BEST Test Accuracy: {self.best_test_accuracy:.4f}")

                scheduler.step(test_loss)
                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= patience:
                        logger.info(f"Early stopping triggered at epoch {epoch+1}")
                        break
            logger.info("Training complete.")

            return {
                "status": "success",
                "final_accuracy": self.best_test_accuracy,
                "epochs_run": num_epochs,
                "model_path": str(settings.MODEL_PATH)
            }
            
        except RuntimeError as e:
            logger.error(f"Training failed: {e}")
            return {"status": "error", "message": f"Training failed: {e}"}
        except Exception as e:
            logger.error(f"An unexpected error occurred during training: {e}")
            return {"status": "error", "message": f"Unexpected error: {e}"}

trainer = StockModelTrainer()
