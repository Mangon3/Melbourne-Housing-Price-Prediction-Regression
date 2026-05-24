import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock
from src.main import _process_sse_chunk, stream_response, main

def test_process_sse_chunk():
    # Test DONE
    assert _process_sse_chunk("[DONE]") is True
    
    # Test PROGRESS
    prog_chunk = json.dumps({"type": "progress", "message": "Loading"})
    assert _process_sse_chunk(prog_chunk) is False
    
    # Test RESULT
    res_chunk = json.dumps({"type": "result", "final_report": "Output"})
    assert _process_sse_chunk(res_chunk) is False
    
    # Test ERROR
    err_chunk = json.dumps({"type": "error", "code": "500", "message": "Fail"})
    assert _process_sse_chunk(err_chunk) is False
    
    # Test generic error
    gen_err_chunk = json.dumps({"error": "Unknown"})
    assert _process_sse_chunk(gen_err_chunk) is False

@pytest.mark.anyio
@patch("src.main.httpx.AsyncClient")
async def test_stream_response(mock_client_cls):
    mock_client = AsyncMock()
    mock_response = AsyncMock()
    mock_response.status_code = 200
    
    async def mock_aiter_lines():
        yield "data: {\"type\": \"progress\", \"message\": \"test\"}"
        yield "invalid json"
        yield "data: [DONE]"
        
    mock_response.aiter_lines = mock_aiter_lines
    mock_client.stream.return_value.__aenter__.return_value = mock_response
    mock_client_cls.return_value.__aenter__.return_value = mock_client
    
    await stream_response("test query")
    mock_client.stream.assert_called_once()

@pytest.mark.anyio
@patch("src.main.httpx.AsyncClient")
async def test_stream_response_error(mock_client_cls):
    mock_client = AsyncMock()
    mock_response = AsyncMock()
    mock_response.status_code = 500
    mock_response.read.return_value = b"Internal Error"
    
    mock_client.stream.return_value.__aenter__.return_value = mock_response
    mock_client_cls.return_value.__aenter__.return_value = mock_client
    
    await stream_response("test query")

@pytest.mark.anyio
@patch("src.main.httpx.AsyncClient")
async def test_stream_response_connect_error(mock_client_cls):
    import httpx
    mock_client_cls.return_value.__aenter__.side_effect = httpx.ConnectError("Failed")
    await stream_response("test query")

@pytest.mark.anyio
@patch("src.main.asyncio.to_thread")
@patch("src.main.stream_response")
async def test_main_loop(mock_stream, mock_input):
    # Simulate user typing "test", then "exit"
    mock_input.side_effect = ["test", "", "exit"]
    await main()
    mock_stream.assert_called_once_with("test")
