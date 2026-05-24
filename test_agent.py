import pytest
from src.utils.errors import format_error, StockAgentError, ProviderError, ModelError
from src.config.settings import settings

def test_format_error_stock_agent_exception():
    exc = StockAgentError("Test error message", code="ERR_TEST", details="Details here")
    formatted = format_error(exc)
    assert formatted["type"] == "error"
    assert formatted["code"] == "ERR_TEST"
    assert formatted["message"] == "Test error message"
    assert formatted["details"] == "Details here"

def test_format_error_provider_exception():
    exc = ProviderError("Provider failed", details="Timeout")
    formatted = format_error(exc)
    assert formatted["type"] == "error"
    assert formatted["code"] == "ERR_PROVIDER"
    assert formatted["message"] == "Provider failed"
    assert formatted["details"] == "Timeout"

def test_format_error_model_exception():
    exc = ModelError("Model prediction failed", details="Inference error")
    formatted = format_error(exc)
    assert formatted["type"] == "error"
    assert formatted["code"] == "ERR_MODEL"
    assert formatted["message"] == "Model prediction failed"
    assert formatted["details"] == "Inference error"

def test_format_error_generic_exception():
    exc = ValueError("Invalid value")
    formatted = format_error(exc)
    assert formatted["type"] == "error"
    assert formatted["code"] == "ERR_UNKNOWN"
    assert formatted["message"] == "An unexpected system error occurred. Please try again later."
    assert formatted["details"] == "Invalid value"

def test_settings_default_values():
    assert "gemini-2.5-flash" in settings.MODEL
    assert settings.SEQ_LEN == 10
    assert settings.HIDDEN_DIM == 32
    assert settings.DROPOUT == pytest.approx(0.1)
    assert "open" in settings.REQUIRED_OHLCV_COLS
