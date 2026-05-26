import torch
import logging
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def get_device(device_id: int = 0) -> torch.device:
    """
    Detects and returns the best available device (CUDA > CPU).
    Logs the detected device on first use.
    """

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{device_id}")
        logger.info(f"Device detected: NVIDIA CUDA ({torch.cuda.get_device_name(device_id)})")
        return device

    logger.info("Device detected: CPU")
    return torch.device("cpu")
