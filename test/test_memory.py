import pytest
from unittest.mock import patch, MagicMock
from src.memory.short_term import ShortTermMemory
from src.memory.store import ConversationStore

# --- ShortTermMemory Tests ---
def test_short_term_memory_add():
    stm = ShortTermMemory(limit=2)
    stm.add_turn("user1", "agent1")
    stm.add_turn("user2", "agent2")
    assert len(stm.history) == 4
    
    # Exceed limit
    stm.add_turn("user3", "agent3")
    assert len(stm.history) == 4
    assert stm.history[0]["content"] == "user2"
    
def test_short_term_memory_empty():
    stm = ShortTermMemory()
    stm.add_turn("", "")
    assert len(stm.history) == 0

def test_get_context_string():
    stm = ShortTermMemory(limit=2)
    assert stm.get_context_string() == "No previous context."
    stm.add_turn("Hello", "World")
    ctx = stm.get_context_string()
    assert "USER: Hello" in ctx
    assert "AGENT: World" in ctx

def test_clear():
    stm = ShortTermMemory(limit=2)
    stm.add_turn("A", "B")
    stm.clear()
    assert len(stm.history) == 0

# --- ConversationStore Tests ---
@patch("src.memory.store.chromadb.PersistentClient")
def test_get_client(mock_client_cls):
    store = ConversationStore()
    client = store._get_client()
    mock_client_cls.assert_called_once()
    assert client is not None
    
    # Call again to test caching
    store._get_client()
    mock_client_cls.assert_called_once()

@patch("src.memory.store.chromadb.PersistentClient")
def test_get_client_error(mock_client_cls):
    mock_client_cls.side_effect = Exception("DB Init Error")
    store = ConversationStore()
    with pytest.raises(Exception):
        store._get_client()

@patch("src.memory.store.ConversationStore._get_client")
def test_get_collection(mock_get_client):
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    store = ConversationStore()
    
    coll = store._get_collection()
    mock_client.get_or_create_collection.assert_called_once_with(name=store.COLLECTION_NAME)

@patch("src.memory.store.embedding")
@patch("src.memory.store.ConversationStore._get_collection")
def test_save_turn(mock_get_collection, mock_embedding):
    store = ConversationStore()
    mock_collection = MagicMock()
    mock_get_collection.return_value = mock_collection
    mock_embedding.embed_query.return_value = [0.1, 0.2]
    
    store.save_turn("hello", "hi there", "greet")
    mock_collection.add.assert_called_once()
    
    # Check that it handles exceptions gracefully
    mock_collection.add.side_effect = Exception("Write error")
    store.save_turn("hello", "hi there", "greet") # Should not raise

@patch("src.memory.store.embedding")
@patch("src.memory.store.ConversationStore._get_collection")
def test_retrieve_similar(mock_get_collection, mock_embedding):
    store = ConversationStore()
    mock_collection = MagicMock()
    mock_get_collection.return_value = mock_collection
    mock_embedding.embed_query.return_value = [0.1, 0.2]
    
    mock_collection.query.return_value = {
        "documents": [["User: hello\nAI: hi"]],
        "metadatas": [[{"intent": "greet"}]]
    }
    
    res = store.retrieve_similar("hi")
    assert len(res) == 1
    assert "User: hello" in res[0]["content"]
    assert res[0]["metadata"]["intent"] == "greet"

@patch("src.memory.store.ConversationStore._get_collection")
def test_retrieve_similar_error(mock_get_collection):
    store = ConversationStore()
    mock_get_collection.side_effect = Exception("Read error")
    
    res = store.retrieve_similar("hi")
    assert res == []
