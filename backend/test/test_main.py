import pytest
import asyncio
import json
from unittest.mock import patch, MagicMock, AsyncMock
import httpx

from src.main import _process_sse_chunk, stream_response, main, cli_entry

@pytest.mark.asyncio
async def test_stream_response_success():
    class MockResponse:
        status_code = 200
        async def aiter_lines(self):
            yield "data: {\"type\": \"progress\", \"message\": \"hi\"}"
            yield "data: {\"type\": \"result\", \"final_report\": \"done\"}"
            yield "data: {\"type\": \"error\", \"code\": \"ERR\", \"message\": \"bad\"}"
            yield "data: {\"error\": \"Server Error\"}"
            yield "data: {\"invalid_json"
            yield ""
            yield "nodata"
            yield "data: [DONE]"
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
            
    class MockClient:
        def stream(self, method, url, json, headers):
            return MockResponse()
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    with patch("src.main.httpx.AsyncClient", return_value=MockClient()):
        await stream_response("test query")

@pytest.mark.asyncio
async def test_stream_response_error_status():
    class MockResponseError:
        status_code = 400
        async def aread(self):
            return b"Bad Request"
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
            
    class MockClientError:
        def stream(self, method, url, json, headers):
            return MockResponseError()
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    with patch("src.main.httpx.AsyncClient", return_value=MockClientError()):
        await stream_response("test query")
        
    with patch("src.main.httpx.AsyncClient", side_effect=Exception("GenErr")):
        await stream_response("test query")

@pytest.mark.asyncio
async def test_stream_response_connect_error():
    with patch("src.main.httpx.AsyncClient", side_effect=httpx.ConnectError("Connection failed")):
        await stream_response("test query")

@pytest.mark.asyncio
async def test_main_loop():
    # Test clear, normal query, exit, and KeyboardInterrupt
    inputs = ["", "clear", "hello", "exit"]
    input_idx = 0
    
    def mock_input(*args, **kwargs):
        nonlocal input_idx
        val = inputs[input_idx]
        input_idx += 1
        return val

    with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
        mock_thread.side_effect = mock_input
        with patch("src.main.stream_response", new_callable=AsyncMock) as mock_stream:
            with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
                await main()
                assert mock_exec.call_count == 1
                assert mock_stream.call_count == 1

@pytest.mark.asyncio
async def test_main_keyboard_interrupt():
    def mock_input_kb(prompt):
        raise KeyboardInterrupt()

    with patch("asyncio.to_thread", side_effect=lambda func, prompt: mock_input_kb(prompt)):
        # Should catch KeyboardInterrupt and break
        await main()

def test_cli_entry():
    with patch("src.main.asyncio.run", side_effect=KeyboardInterrupt):
        cli_entry()
    
    with patch("src.main.asyncio.run") as mock_run:
        cli_entry()
        assert mock_run.call_count == 1
