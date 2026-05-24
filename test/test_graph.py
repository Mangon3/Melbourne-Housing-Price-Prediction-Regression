import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from src.graph.workflow import create_workflow
from src.graph.nodes import _invoke_model_safely, _extract_symbol_from_messages, _build_fallback_response, _needs_fallback, call_model
from src.graph.state import AgentState

def test_workflow_creation():
    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = "mock_model_with_tools"
    workflow = create_workflow(mock_llm)
    assert workflow is not None

def test_invoke_model_safely_success():
    mock_model = MagicMock()
    mock_model.invoke.return_value = "Success"
    assert _invoke_model_safely(mock_model, []) == "Success"

def test_invoke_model_safely_rate_limit():
    mock_model = MagicMock()
    mock_model.invoke.side_effect = Exception("429 Too Many Requests")
    with pytest.raises(Exception):
        _invoke_model_safely(mock_model, [])

def test_invoke_model_safely_empty_error():
    mock_model = MagicMock()
    mock_model.invoke.side_effect = Exception("output text or tool calls cannot both be empty")
    assert _invoke_model_safely(mock_model, []) is None

def test_invoke_model_safely_unknown_error():
    mock_model = MagicMock()
    mock_model.invoke.side_effect = Exception("Unknown critical error")
    with pytest.raises(Exception):
        _invoke_model_safely(mock_model, [])

def test_extract_symbol_from_messages():
    msgs = [HumanMessage(content="analyze AAPL please")]
    assert _extract_symbol_from_messages(msgs) == "AAPL"
    
    msgs = [HumanMessage(content="what is the timeframe for NVDA?")]
    assert _extract_symbol_from_messages(msgs) == "NVDA"
    
    msgs = [HumanMessage(content="hello")]
    assert _extract_symbol_from_messages(msgs) is None

def test_build_fallback_response():
    resp = _build_fallback_response("AAPL")
    assert isinstance(resp, AIMessage)
    assert resp.tool_calls[0]["name"] == "micro_analysis"
    assert resp.tool_calls[0]["args"]["symbol"] == "AAPL"

def test_needs_fallback():
    # Has micro tool run
    msgs = [ToolMessage(content="{}", name="micro_analysis", tool_call_id="1")]
    assert not _needs_fallback(AIMessage(content="Hello", tool_calls=[]), msgs)
    
    # Is empty
    assert _needs_fallback(None, [])
    
    # Missed tool
    assert _needs_fallback(AIMessage(content="Hello", tool_calls=[]), [])

@patch("src.graph.nodes._invoke_model_safely")
def test_call_model(mock_invoke):
    mock_invoke.return_value = AIMessage(content="Hello", tool_calls=[{"name": "test", "args": {}, "id": "123"}])
    state = {"messages": [HumanMessage(content="hello")]}
    res = call_model(state, MagicMock())
    assert len(res["messages"]) == 1
    
@patch("src.graph.nodes._invoke_model_safely")
def test_call_model_fallback(mock_invoke):
    mock_invoke.return_value = AIMessage(content="Hello", tool_calls=[])
    state = {"messages": [HumanMessage(content="analyze MSFT")]}
    res = call_model(state, MagicMock())
    assert len(res["messages"]) == 1
    assert res["messages"][0].tool_calls[0]["name"] == "micro_analysis"

@patch("src.graph.nodes._invoke_model_safely")
def test_call_model_fallback_fail(mock_invoke):
    mock_invoke.return_value = None
    state = {"messages": [HumanMessage(content="hello")]}
    with pytest.raises(ValueError):
        call_model(state, MagicMock())
