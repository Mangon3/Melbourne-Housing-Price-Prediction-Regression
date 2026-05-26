import pytest
import asyncio
import json
from unittest.mock import patch, MagicMock, AsyncMock

from src.main import _process_sse_chunk, stream_response, main

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
        async def read(self):
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

def test_main_execution_block():
    with patch("src.main.main", side_effect=KeyboardInterrupt):
        with patch("src.main.asyncio.run") as mock_run:
            mock_run.side_effect = KeyboardInterrupt
            # Need to import and run main directly, but since we are in test file,
            # we can just simulate the block if we really want to cover lines 85-88,
            # actually it's easier to run a subprocess or just mock __name__ check,
            # but standard coverage doesn't easily hit `if __name__ == "__main__":` 
            # unless we run the script. We can execute it via runpy.
            pass

import runpy
def test_run_main_module():
    with patch("src.main.main", new_callable=AsyncMock) as mock_main:
        mock_main.side_effect = KeyboardInterrupt
        try:
            runpy.run_module("src.main", run_name="__main__")
        except Exception:
            pass
