from typing import Dict, Optional, Any

class StockAgentError(Exception):
    """Base class for all application errors."""
    def __init__(self, message: str, code: str = "ERR_INTERNAL", details: Optional[str] = None):
        self.message = message
        self.code = code
        self.details = details
        super().__init__(self.message)

class ProviderError(StockAgentError):
    """Errors related to external APIs (Finnhub, Gemini, etc.)."""
    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(message, code="ERR_PROVIDER", details=details)

class ModelError(StockAgentError):
    """Errors related to internal model inference or training."""
    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(message, code="ERR_MODEL", details=details)

def format_error(e: Exception) -> Dict[str, Any]:
    """
    Standardizes exception output for the API/User.
    """
    if isinstance(e, StockAgentError):
        return {
            "type": "error",
            "code": e.code,
            "message": e.message,
            "details": e.details
        }

    return {
        "type": "error",
        "code": "ERR_UNKNOWN",
        "message": "An unexpected system error occurred. Please try again later.",
        "details": str(e)
    }
