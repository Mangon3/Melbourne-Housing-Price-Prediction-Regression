import time
import re
import functools
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable, InternalServerError
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def retry_with_backoff(max_retries=5, initial_delay=5.0, backoff_factor=2.0):
    """
    Decorator to retry a function call upon encountering Google API rate limit errors.
    It attempts to parse the 'Please retry in X seconds' message. 
    Otherwise, it uses exponential backoff.
    """
    
    def decorator(func):
        @functools.wraps(func)

        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_str = str(e)
                    is_rate_limit = (
                        "429" in error_str or 
                        "RESOURCE_EXHAUSTED" in error_str or
                        isinstance(e, (ResourceExhausted, ServiceUnavailable, InternalServerError))
                    )
                    if not is_rate_limit:
                        raise e
                    if attempt == max_retries:
                        logger.critical(f"Max retries ({max_retries}) exceeded for {func.__name__}. Last error: {e}")
                        raise e
                    wait_time = delay
                    try:
                        match = re.search(r'retry in (\d+(\.\d+)?)\s*s', error_str, re.IGNORECASE)
                        if match:
                             wait_time = float(match.group(1)) + 1.0 
                             logger.warning(f"Rate limit hit in {func.__name__}. API requested wait of {wait_time:.2f}s.")
                        else:
                             wait_time = delay
                             logger.warning(f"Rate limit hit in {func.__name__}. Backing off for {wait_time:.2f}s (Attempt {attempt+1}/{max_retries}). Error: {e}")
                    except Exception:
                        wait_time = delay
                        logger.warning(f"Rate limit hit in {func.__name__}. Default backoff {wait_time:.2f}s.")
                    time.sleep(wait_time)
                    if not match:
                        delay *= backoff_factor
            return None 

        return wrapper

    return decorator