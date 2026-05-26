from typing import Dict, Any, List
from src.tools.model.train import trainer
from src.tools.model.infer import micro_model_predictor
from src.utils.logger import setup_logger

logger = setup_logger(__name__)
class MicroModel:
    """
    The tool interface for managing the Micro Stock Prediction Model, 
    including training and future prediction/backtesting functions.
    """
    
    def execute_model_training(
        self,
        symbols_list: str = "AAPL,MSFT,GOOGL,AMZN,TSLA",
        num_epochs: int = 100
    ) -> Dict[str, Any]:
        logger.info(f"MicroModel.execute_model_training called with symbols={symbols_list}")
        DIVERSE_TICKERS = ["NVDA", "INTC", "KO"]
        symbols_parsed: List[str] = [s.strip().upper() for s in symbols_list.split(',') if s.strip()]
        training_symbols = list(dict.fromkeys(symbols_parsed + DIVERSE_TICKERS))
        if not symbols_parsed:
             return {
                "status": "error",
                "message": "Symbol list is empty. Please provide one or more comma-separated stock symbols for training."
            }
        primary_symbol = symbols_parsed[0]
        logger.info(f"Starting direct training call for micro-model on symbols: {symbols_parsed} for {num_epochs} epochs.")
        try:
            training_results = trainer.train(
                symbols=training_symbols,
                num_epochs=num_epochs
            )
            if training_results.get("status") != "success":
                logger.error(f"Training failed for {primary_symbol}. Aborting inference.")
                return {
                    "status": "error",
                    "message": f"Training failed: {training_results.get('message')}. Micro-model could not be updated."
                }
            logger.info(f"Training successful. Running immediate inference on primary symbol: {primary_symbol}")
            inference_results = micro_model_predictor.predict_price_outlook(
                symbol=primary_symbol
            )
            final_output = {
                "training_status": training_results.get("status", "completed"),
                "training_message": training_results.get("message", "Model successfully trained and saved."),
                "training_accuracy": training_results.get("test_accuracy", "N/A"),
                "inference": inference_results
            }
            return final_output
        except Exception as e:
            return {
                "status": "error",
                "message": f"A critical error occurred during training or inference: {str(e)}"
            }

micro_model = MicroModel()
