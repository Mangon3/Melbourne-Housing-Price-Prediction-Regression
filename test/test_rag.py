import pytest
from unittest.mock import MagicMock, patch
from src.rag.core import ChromaRAG

@patch("src.rag.core.chromadb.PersistentClient")
def test_get_chroma_client(mock_client):
    rag = ChromaRAG()
    client = rag._get_chroma_client()
    mock_client.assert_called_once()
    assert client is not None
    
    # Call again to test caching
    rag._get_chroma_client()
    mock_client.assert_called_once()

@patch("src.rag.core.chromadb.PersistentClient")
def test_get_chroma_client_error(mock_client):
    mock_client.side_effect = Exception("DB Init Error")
    rag = ChromaRAG()
    with pytest.raises(Exception):
        rag._get_chroma_client()

@patch("src.rag.core.ChromaRAG._get_chroma_client")
def test_get_news_collection(mock_get_client):
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    rag = ChromaRAG()
    
    coll = rag._get_news_collection()
    mock_client.get_or_create_collection.assert_called_once_with(name=rag.COLLECTION_NAME)

@patch("src.rag.core.embedding")
@patch("src.rag.core.ChromaRAG._get_news_collection")
def test_ingest_news_documents(mock_get_collection, mock_embedding):
    rag = ChromaRAG()
    rag.chunker = MagicMock()
    rag.chunker.return_value = [
        {"content": "c1", "metadata": {"chunk_id": "1"}}
    ]
    mock_embedding.embed_texts.return_value = [[0.1]]
    mock_collection = MagicMock()
    mock_get_collection.return_value = mock_collection
    
    rag.ingest_news_documents([{"headline": "test"}])
    mock_collection.add.assert_called_once()

@patch("src.rag.core.ChromaRAG._get_news_collection")
def test_ingest_news_documents_empty(mock_get_collection):
    rag = ChromaRAG()
    rag.chunker = MagicMock(return_value=[])
    rag.ingest_news_documents([{"headline": "test"}])
    mock_get_collection.assert_called_once()

@patch("src.rag.core.embedding")
@patch("src.rag.core.ChromaRAG._get_news_collection")
def test_ingest_news_documents_error(mock_get_collection, mock_embedding):
    rag = ChromaRAG()
    rag.chunker = MagicMock(return_value=[{"content": "c1", "metadata": {"chunk_id": "1"}}])
    mock_embedding.embed_texts.return_value = [[0.1]]
    mock_collection = MagicMock()
    mock_collection.add.side_effect = Exception("DB Error")
    mock_get_collection.return_value = mock_collection
    
    with pytest.raises(Exception):
        rag.ingest_news_documents([{"headline": "test"}])

@patch("src.rag.core.embedding")
@patch("src.rag.core.ChromaRAG._get_news_collection")
def test_retrieve_context(mock_get_collection, mock_embedding):
    rag = ChromaRAG()
    mock_embedding.embed_query.return_value = [0.1]
    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "documents": [["doc1"]],
        "metadatas": [[{"headline": "H1", "source": "S1"}]]
    }
    mock_get_collection.return_value = mock_collection
    
    ctx, sources = rag.retrieve_context("query")
    assert "doc1" in ctx
    assert "H1" in ctx
    assert len(sources) == 1
    assert sources[0]["headline"] == "H1"
