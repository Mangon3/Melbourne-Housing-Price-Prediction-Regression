import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from src.api.index import app, _parse_intent, _get_stream_iterator, AnalyzeRequest

client = TestClient(app)

def test_root():
    res = client.get("/")
    assert res.status_code == 200
    assert "message" in res.json()

def test_missing_api_key():
    with patch("src.api.index.settings.GOOGLE_API_KEY", ""):
        res = client.post("/analyze", json={"symbol": "AAPL"})
        assert res.status_code == 401

def test_parse_intent_success():
    agent = MagicMock()
    agent.parse_intent.return_value = {"intent": "STOCK_QUERY", "symbol": "AAPL", "tools": []}
    intent, sym, data = _parse_intent(agent, "query")
    assert intent == "STOCK_QUERY"
    assert sym == "AAPL"

def test_parse_intent_exception():
    agent = MagicMock()
    agent.parse_intent.side_effect = Exception("err")
    intent, sym, data = _parse_intent(agent, "query")
    assert intent == "UNKNOWN"
    assert sym is None

def test_get_stream_iterator():
    agent = MagicMock()
    req = AnalyzeRequest(symbol="AAPL")
    
    # STOCK_QUERY
    _get_stream_iterator("STOCK_QUERY", "AAPL", {"tools": []}, agent, req)
    agent.analyze.assert_called_with("AAPL", tools=[])
    
    # GENERAL_CHAT
    _get_stream_iterator("GENERAL_CHAT", None, None, agent, AnalyzeRequest(query="hi"))
    agent.respond_conversational.assert_called_with("hi")
    
    # UNKNOWN
    it = _get_stream_iterator("UNKNOWN", None, None, agent, req)
    res_list = list(it)
    assert len(res_list) == 2
    assert res_list[0]["type"] == "progress"
    assert res_list[1]["type"] == "result"
    
    # Invalid
    assert _get_stream_iterator("INVALID", None, None, agent, req) is None

def test_analyze_stream_invalid_intent():
    # To hit stream_iterator is None branch
    with patch("src.api.index.Agent") as MockAgent:
        agent_inst = MockAgent.return_value
        agent_inst.parse_intent.return_value = {"intent": "INVALID", "symbol": None}
        res = client.post("/analyze", json={"query": "hi"}, headers={"X-Gemini-API-Key": "test"})
        assert res.status_code == 200
        text = res.text
        assert "Invalid Intent" in text

def test_analyze_stream_exception():
    # To hit exception block in event_generator
    with patch("src.api.index.Agent") as MockAgent:
        agent_inst = MockAgent.return_value
        agent_inst.parse_intent.return_value = {"intent": "STOCK_QUERY", "symbol": "AAPL"}
        def bad_stream(*args, **kwargs):
            raise Exception("Stream failed")
        agent_inst.analyze = bad_stream
        
        res = client.post("/analyze", json={"query": "hi"}, headers={"X-Gemini-API-Key": "test"})
        assert res.status_code == 200
        text = res.text
        assert "Analysis stream error" in text

def test_analyze_stream_success_result():
    # To hit memory_store save inside event_generator
    with patch("src.api.index.Agent") as MockAgent:
        agent_inst = MockAgent.return_value
        agent_inst.parse_intent.return_value = {"intent": "STOCK_QUERY", "symbol": "AAPL"}
        
        def good_stream(*args, **kwargs):
            yield {"type": "result", "final_report": "All good"}
        agent_inst.analyze = good_stream
        
        with patch("src.api.index.memory_store.save_turn") as mock_save:
            with patch("src.api.index.stm.add_turn") as mock_add:
                res = client.post("/analyze", json={"symbol": "AAPL"}, headers={"X-Gemini-API-Key": "test"})
                assert res.status_code == 200
                assert "All good" in res.text
                mock_save.assert_called_once()
                mock_add.assert_called_once()

def test_analyze_request_validation():
    with pytest.raises(ValueError):
        AnalyzeRequest(symbol=None, query=None)
