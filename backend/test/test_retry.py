import pytest
from unittest.mock import patch
from src.utils.retry import _is_rate_limit_error, _calculate_wait_time, retry_with_backoff
from google.api_core.exceptions import ResourceExhausted

def test_is_rate_limit_error():
    assert _is_rate_limit_error(Exception("HTTP 429 Too Many Requests"))
    assert _is_rate_limit_error(Exception("RESOURCE_EXHAUSTED"))
    assert _is_rate_limit_error(ResourceExhausted("API limits"))
    assert not _is_rate_limit_error(Exception("General Error"))

def test_calculate_wait_time():
    # Test api hint regex
    wait_time, used_api = _calculate_wait_time("Please retry in 5.5s.", 10.0)
    assert wait_time == 6.5 # 5.5 + 1.0
    assert used_api is True
    
    wait_time, used_api = _calculate_wait_time("Please retry in 2 s", 10.0)
    assert wait_time == 3.0
    assert used_api is True
    
    # Test default
    wait_time, used_api = _calculate_wait_time("Some other error", 10.0)
    assert wait_time == 10.0
    assert used_api is False

@patch("src.utils.retry.time.sleep")
def test_retry_with_backoff_success_after_failure(mock_sleep):
    call_count = 0
    @retry_with_backoff(max_retries=3, initial_delay=1.0)
    def flaky_func():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise Exception("429 error")
        return "success"
    
    res = flaky_func()
    assert res == "success"
    assert call_count == 3
    assert mock_sleep.call_count == 2
    # First sleep 1.0, second 2.0
    mock_sleep.assert_any_call(1.0)
    mock_sleep.assert_any_call(2.0)

@patch("src.utils.retry.time.sleep")
def test_retry_with_backoff_api_hint(mock_sleep):
    call_count = 0
    @retry_with_backoff(max_retries=2, initial_delay=1.0)
    def api_hint_func():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise Exception("Please retry in 3.0s")
        return "success"
        
    # We must mock _is_rate_limit_error or add 429 to exception so it catches it
    with patch("src.utils.retry._is_rate_limit_error", return_value=True):
        res = api_hint_func()
    assert res == "success"
    mock_sleep.assert_called_once_with(4.0)

@patch("src.utils.retry.time.sleep")
def test_retry_with_backoff_max_retries_exceeded(mock_sleep):
    @retry_with_backoff(max_retries=2, initial_delay=1.0)
    def failing_func():
        raise Exception("429 error")
        
    with pytest.raises(Exception, match="429 error"):
        failing_func()
    assert mock_sleep.call_count == 2
    
def test_retry_with_backoff_non_rate_limit():
    @retry_with_backoff(max_retries=2)
    def func():
        raise ValueError("General failure")
        
    with pytest.raises(ValueError, match="General failure"):
        func()
