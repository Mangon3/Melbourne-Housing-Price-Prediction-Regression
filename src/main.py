import asyncio
import json
import os
import sys
import httpx

DEFAULT_PORT = 7861
API_URL = os.getenv("API_URL", f"http://0.0.0.0:{DEFAULT_PORT}/analyze")
API_KEY = os.getenv("GOOGLE_API_KEY", "")

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
                    print(f"Error {response.status_code}: {await response.read()}")
                    return
                buffer = ""
                print("Agent: ", end="", flush=True)
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:]
                        try:
                            if data_str.strip() == "[DONE]":
                                break
                            chunk = json.loads(data_str)
                            if chunk.get("type") == "progress":
                                sys.stdout.write(f"\n   ↳ [PROGRESS] {chunk.get('message')}")
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
            query = input("\nYou: ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit"]:
                break
            if query.lower() == "clear":
                os.system('clear')
                continue
            await stream_response(query)
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
