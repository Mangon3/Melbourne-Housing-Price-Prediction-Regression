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
            logger.warning("Could not create any sequences. Data length: %d, SEQ_LEN: %d", len(df), seq_len)
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
        logger.info("Trainer initialized. Target device: %s.", self.DEVICE)

    def _fetch_from_exchanges(self, symbol: str) -> Any:
        """Tries to fetch data from multiple exchanges, returning the first success."""
        exchanges = ["NASDAQ", "NYSE", "BINANCE"]
        for exchange in exchanges:
            df_raw = tv_data_fetcher.fetch_historical_data(
                symbol,
                timeframe_days=settings.DATA_TIMEFRAME_DAYS,
                exchange=exchange
            )
            if not (isinstance(df_raw, dict) and "error" in df_raw):
                return df_raw
            if exchange != exchanges[-1]:
                logger.info("%s fetch failed for %s, trying next exchange...", exchange, symbol)
        return df_raw

    def _process_symbol_data(self, symbol: str, df_raw):
        """Processes raw data into train/test sequences. Returns (X_train, y_train, X_test, y_test) or None."""
        try:
            df_features = FeatureCalculator.calculate_features(df_raw.copy())
            df_features['close'] = df_raw['close']
            x_data, y_data = StockDataset.create_sequences(
                df_features,
                settings.SEQ_LEN,
                settings.FEATURE_COLUMNS
            )
            if x_data.size == 0:
                logger.warning("Skipping %s: Insufficient data after sequence creation.", symbol)
                return None
            split_index = int(len(x_data) * (1 - self.TEST_SIZE_RATIO))
            split_index = max(1, min(split_index, len(x_data) - 1))
            x_train, x_test = x_data[:split_index], x_data[split_index:]
            y_train, y_test = y_data[:split_index], y_data[split_index:]
            logger.info("Success for %s: Created %d sequences (Train: %d, Test: %d).", symbol, len(x_data), len(x_train), len(x_test))
            return x_train, y_train, x_test, y_test
        except ValueError as e:
            logger.exception("Skipping %s due to feature calculation error", symbol)
            return None
        except Exception:
            logger.exception("An unexpected error occurred for %s", symbol)
            return None

    def _load_and_prepare_data(self, symbols: List[str]) -> Tuple[StockDataset, StockDataset]:
        """Fetches data, calculates features, and creates the final sequence datasets with temporal splitting."""
        train_x_list, train_y_list = [], []
        test_x_list, test_y_list = [], []
        logger.info("Starting data fetch and feature engineering for %d symbols...", len(symbols))
        for symbol in symbols:
            logger.info("Fetching data for %s...", symbol)
            df_raw = self._fetch_from_exchanges(symbol)

            if isinstance(df_raw, dict) and "error" in df_raw:
                logger.error("Skipping %s: Data fetch failed on all exchanges: %s", symbol, df_raw['error'])
                time.sleep(random.uniform(1, 3))
                continue

            result = self._process_symbol_data(symbol, df_raw)
            if result is not None:
                x_train, y_train, x_test, y_test = result
                train_x_list.append(x_train)
                train_y_list.append(y_train)
                test_x_list.append(x_test)
                test_y_list.append(y_test)

        if not train_x_list:
            raise RuntimeError("No training data could be generated for any symbol.")
        x_train_combined = np.concatenate(train_x_list, axis=0)
        y_train_combined = np.concatenate(train_y_list, axis=0)

        x_test_combined = np.concatenate(test_x_list, axis=0)
        y_test_combined = np.concatenate(test_y_list, axis=0)

        return StockDataset(x_train_combined, y_train_combined), StockDataset(x_test_combined, y_test_combined)

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
        avg_loss = running_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        if total_samples > 0:
            directional_accuracy = correct_predictions / total_samples
        else:
            directional_accuracy = 0.0
        return avg_loss, directional_accuracy

    def _run_training_epoch(self, model, train_dataloader, criterion_reg, criterion_cls, optimizer):
        """Runs a single training epoch and returns the average loss."""
        model.train()
        running_loss = 0.0
        for inputs, labels in train_dataloader:
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
        return running_loss / len(train_dataloader)

    def train(self, symbols: List[str] = None, num_epochs: int = None, batch_size: int = None) -> Dict[str, Any]:
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
        logger.info("Trainer.train started. Device: %s", self.DEVICE)

        symbols = symbols if symbols is not None else self.DEFAULT_SYMBOLS
        num_epochs = num_epochs if num_epochs is not None else self.DEFAULT_EPOCHS
        batch_size = batch_size if batch_size is not None else self.DEFAULT_BATCH_SIZE

        try:
            train_dataset, test_dataset = self._load_and_prepare_data(symbols)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            logger.info("Total training samples available: %d", len(train_dataset))
            logger.info("Total testing samples available: %d", len(test_dataset))

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
            optimizer = optim.Adam(model.parameters(), lr=self.LEARNING_RATE, weight_decay=0)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
            self.best_test_accuracy = 0.0
            best_test_loss = float('inf')
            early_stop_counter = 0
            patience = 12
            logger.info("Training started for %d epochs on symbols: %s", num_epochs, ', '.join(symbols))

            for epoch in range(num_epochs):
                train_loss = self._run_training_epoch(model, train_dataloader, criterion_reg, criterion_cls, optimizer)
                test_loss, test_accuracy = self._evaluate_model(model, test_dataloader)
                logger.info("Epoch %d/%d | Train Loss: %.4f | Test Loss: %.4f | Test Accuracy: %.4f", epoch + 1, num_epochs, train_loss, test_loss, test_accuracy)

                if test_accuracy > self.best_test_accuracy:
                    self.best_test_accuracy = test_accuracy
                    settings.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), settings.MODEL_PATH)
                    logger.info("--> Model saved! New BEST Test Accuracy: %.4f", self.best_test_accuracy)

                scheduler.step(test_loss)
                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= patience:
                        logger.info("Early stopping triggered at epoch %d", epoch + 1)
                        break
            logger.info("Training complete.")

            return {
                "status": "success",
                "final_accuracy": self.best_test_accuracy,
                "epochs_run": num_epochs,
                "model_path": str(settings.MODEL_PATH)
            }
            
        except RuntimeError:
            logger.exception("Training failed")
            return {"status": "error", "message": "Training failed"}
        except Exception:
            logger.exception("An unexpected error occurred during training")
            return {"status": "error", "message": "Unexpected training error"}

trainer = StockModelTrainer()
