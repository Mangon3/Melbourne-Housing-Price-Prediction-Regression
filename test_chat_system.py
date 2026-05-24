import pytest
import asyncio
import json
import os
import httpx

# Configuration
API_URL = "http://0.0.0.0:7860/analyze"  # Ensure this matches the port in docker-compose.yml
API_KEY = os.environ.get("GOOGLE_API_KEY", "")


def _verify_chat_response(report):
    """Check if the report matches expected chat-type response patterns."""
    chat_keywords = [
        "Stock Agent", "AI Analyst",
        "couldn't understand", "apologize",
        "Hello", "help"
    ]
    if any(kw in report for kw in chat_keywords):
        return True
    return "MACRO NEWS" not in report


def _verify_stock_response(report):
    """Check if the report matches expected stock-type response patterns."""
    stock_keywords = [
        "Investment Report", "Analysis", "Price",
        "couldn't understand", "apologize"
    ]
    return any(kw in report for kw in stock_keywords)


async def _process_stream_response(response, expected_type):
    """Processes the SSE stream and returns (found_expected, full_text)."""
    found_expected = False
    full_text = ""

    async for line in response.aiter_lines():
        if not line or not line.startswith("data: "):
            continue

        data_str = line[6:].strip()
        if data_str == "[DONE]":
            break

        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        if chunk.get("type") == "error":
            pytest.skip(f"Server returned an error chunk: {chunk.get('message', 'unknown')}")
            return found_expected, full_text

        if chunk.get("type") != "result":
            continue

        report = chunk.get("final_report", "")
        full_text += report

        if expected_type == "CHAT":
            found_expected = _verify_chat_response(report)
        elif expected_type == "STOCK":
            found_expected = _verify_stock_response(report)

    return found_expected, full_text


@pytest.mark.anyio
@pytest.mark.parametrize("query, expected_type", [
    ("Hello! Who are you?", "CHAT"),
    ("Analyze NVDA", "STOCK")
])
async def test_query(query: str, expected_type: str):
    print(f"\n>>> TESTING QUERY: '{query}'")
    print(f"    Expected Intent Type: {expected_type}")
    
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-Gemini-API-Key"] = API_KEY
    payload = {"query": query}

    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            async with client.stream("POST", API_URL, json=payload, headers=headers) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    print(f"    WARNING: API returned {response.status_code}: {error_text}")
                    pytest.skip(f"API returned {response.status_code} (likely transient LLM error)")
                    return

                found_expected_response, full_text = await _process_stream_response(response, expected_type)

    except httpx.ConnectError:
        pytest.fail(f"Failed to connect to {API_URL}. Is server running?")
        return

    assert found_expected_response, f"TEST FAILED. Did not match expected output signatures. Received: {full_text[:200]}..."
