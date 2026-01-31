import torch
import logging
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def get_device(device_id: int = 0) -> torch.device:
    """
    Detects and returns the best available device (CUDA > XPU > CPU).
    Logs the detected device on first use.
    """

    device = torch.device("cpu")
    
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{device_id}")
        logger.info(f"Device detected: NVIDIA CUDA ({torch.cuda.get_device_name(device_id)})")
        return device
        logger.error(f"IPEX check failed with unexpected error: {e}")
        pass
    
    logger.info("Device detected: CPU")
    return device
