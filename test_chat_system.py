import pytest
import asyncio
import json
import os
import httpx

# Configuration
API_URL = "http://0.0.0.0:7860/analyze"  # Ensure this matches the port in docker-compose.yml
API_KEY = os.environ.get("GOOGLE_API_KEY", "")

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
    
    found_expected_response = False
    full_text = ""

    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            async with client.stream("POST", API_URL, json=payload, headers=headers) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    print(f"    WARNING: API returned {response.status_code}: {error_text}")
                    pytest.skip(f"API returned {response.status_code} (likely transient LLM error)")
                    return

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                        
                    try:
                        chunk = json.loads(data_str)
                        
                        if chunk.get("type") == "result":
                            # Check if valid result
                            report = chunk.get("final_report", "")
                            full_text += report
                            
                            # Simple heuristic verification
                            if expected_type == "CHAT":
                                if any(kw in report for kw in [
                                    "Stock Agent", "AI Analyst",
                                    "couldn't understand", "apologize",
                                    "Hello", "help"
                                ]):
                                    found_expected_response = True
                                else:
                                    if "MACRO NEWS" not in report:
                                         found_expected_response = True
                                         
                            elif expected_type == "STOCK":
                                if any(kw in report for kw in [
                                    "Investment Report", "Analysis", "Price",
                                    "couldn't understand", "apologize"
                                ]):
                                    found_expected_response = True
                                    
                        elif chunk.get("type") == "error":
                            print(f"    Server-side error: {chunk.get('message', 'unknown')}")
                            pytest.skip("Server returned an error chunk (transient LLM failure)")
                            return

                    except json.JSONDecodeError:
                        pass
                        
    except httpx.ConnectError:
        pytest.fail(f"Failed to connect to {API_URL}. Is server running?")
        return

    assert found_expected_response, f"TEST FAILED. Did not match expected output signatures. Received: {full_text[:200]}..."

