import time
import re
import functools
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable, InternalServerError
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def _is_rate_limit_error(error):
    """Checks if an exception is a rate limit / transient error."""
    error_str = str(error)
    return (
        "429" in error_str or
        "RESOURCE_EXHAUSTED" in error_str or
        isinstance(error, (ResourceExhausted, ServiceUnavailable, InternalServerError))
    )


def _calculate_wait_time(error_str, default_delay):
    """Parses API retry header or returns default delay. Returns (wait_time, used_api_hint)."""
    match = re.search(r'retry in (\d+(\.\d+)?)\s*s', error_str, re.IGNORECASE)
    if match:
        wait_time = float(match.group(1)) + 1.0
        return wait_time, True
    return default_delay, False


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
                    if not _is_rate_limit_error(e):
                        raise
                    if attempt == max_retries:
                        logger.critical("Max retries (%d) exceeded for %s. Last error: %s", max_retries, func.__name__, e)
                        raise
                    wait_time, used_api_hint = _calculate_wait_time(str(e), delay)
                    if used_api_hint:
                        logger.warning("Rate limit hit in %s. API requested wait of %.2fs.", func.__name__, wait_time)
                    else:
                        logger.warning("Rate limit hit in %s. Backing off for %.2fs (Attempt %d/%d). Error: %s", func.__name__, wait_time, attempt + 1, max_retries, e)
                    time.sleep(wait_time)
                    if not used_api_hint:
                        delay *= backoff_factor
            return None 

        return wrapper

    return decorator