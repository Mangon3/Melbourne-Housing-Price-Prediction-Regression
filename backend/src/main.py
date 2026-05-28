import asyncio
import json
import os
import sys
import httpx
from dotenv import load_dotenv

load_dotenv()
DEFAULT_PORT = 7861
API_URL = os.getenv("API_URL", f"http://0.0.0.0:{DEFAULT_PORT}/analyze")  # nosonar
API_KEY = os.getenv("GOOGLE_API_KEY", "")


def _process_sse_chunk(data_str: str) -> bool:
    """Processes a single SSE data chunk. Returns True if stream should stop."""
    if data_str.strip() == "[DONE]":
        return True
    chunk = json.loads(data_str)
    if chunk.get("type") == "progress":
        sys.stdout.write(f"\n   -> [PROGRESS] {chunk.get('message')}")
    elif chunk.get("type") == "result":
        sys.stdout.write("\n\n")
        report = chunk.get("final_report", "")
        print(report)
    elif chunk.get("type") == "error":
        code = chunk.get("code", "ERR")
        msg = chunk.get("message", "Unknown Error")
        print(f"\n[!] Error ({code}): {msg}")
    elif "error" in chunk:
        print(f"\nServer Error: {chunk['error']}")
    return False


async def stream_response(query: str):
    
    headers = {
        "Content-Type": "application/json",
        "X-Gemini-API-Key": API_KEY
    }

    payload = {"query": query}
    print(f"\n[Connecting to {API_URL} map...]")

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream("POST", API_URL, json=payload, headers=headers) as response:
                if response.status_code != 200:
                    error_msg = (await response.aread()).decode("utf-8")
                    print(f"Error {response.status_code}: {error_msg}")
                    return
                print("Agent: ", end="", flush=True)
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    try:
                        if _process_sse_chunk(data_str):
                            break
                    except json.JSONDecodeError:
                        pass
    except httpx.ConnectError:
        print(f"\nCould not connect to API at {API_URL}. Is the server running? (uvicorn src.api.index:app --reload)")
    except Exception as e:
        print(f"\nError: {e}")

async def main():
    print("==================================================")
    print("       Stock Agent CLI (API Client)            ")
    print("==================================================")
    print(f"Target: {API_URL}")
    print("Commands: 'exit', 'quit', 'clear'")
    while True:
        try:
            query = (await asyncio.to_thread(input, "\nYou: ")).strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit"]:
                break
            if query.lower() == "clear":
                await asyncio.create_subprocess_exec('clear')
                continue
            await stream_response(query)
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        
def cli_entry():
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    cli_entry()
