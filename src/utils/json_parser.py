import re
import ast
from typing import Any, List, Dict

def parse_agent_output(raw_output: str) -> List[Dict[str, Any]]:
    cleaned_str = re.sub(r'^\s*```[a-zA-Z]*\n?|```\s*$', '', raw_output, flags=re.MULTILINE).strip()
    literal_like_str = cleaned_str.replace(r'<\ctrl46>', "'") 
    key_regex = re.compile(r'([{,]\s*)(\w+)(\s*:\s*)')
    python_literal_str = key_regex.sub(r"\1'\2'\3", literal_like_str)
    final_literal_str = python_literal_str.replace('\n', '').replace('\r', '')

    try:
        parsed_data = ast.literal_eval(final_literal_str)
    except Exception as e:
        print(f"[DEBUG] Failed to parse string using ast.literal_eval. Final literal string segment: {final_literal_str[:200]}...")
        raise ValueError(f"AST Parsing Error: {e}") from e
    if not isinstance(parsed_data, list):
        if isinstance(parsed_data, dict):
            return [parsed_data]
        else:
            raise ValueError(f"Parsed data is not a list or dictionary: {type(parsed_data)}")
            
    return parsed_data
