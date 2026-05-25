import pytest
from src.utils.json_parser import parse_agent_output

def test_parse_agent_output_valid_list():
    raw = """```json
    [
        {"name": "AAPL", "price": 150}
    ]
    ```"""
    res = parse_agent_output(raw)
    assert isinstance(res, list)
    assert len(res) == 1
    assert res[0]["name"] == "AAPL"

def test_parse_agent_output_valid_dict():
    raw = """
    {"key": "value", "num": 1}
    """
    res = parse_agent_output(raw)
    assert isinstance(res, list)
    assert len(res) == 1
    assert res[0]["key"] == "value"

def test_parse_agent_output_unquoted_keys():
    raw = """[{key: 'value'}]"""
    res = parse_agent_output(raw)
    assert res[0]["key"] == "value"

def test_parse_agent_output_ctrl46():
    raw = """[{"key": <\ctrl46>value<\ctrl46>}]"""
    res = parse_agent_output(raw)
    assert res[0]["key"] == "value"

def test_parse_agent_output_invalid_type():
    raw = """ "just a string" """
    with pytest.raises(ValueError):
        parse_agent_output(raw)

def test_parse_agent_output_malformed():
    raw = """ [{"key": "value" """ # missing closing brackets
    with pytest.raises(ValueError):
        parse_agent_output(raw)
