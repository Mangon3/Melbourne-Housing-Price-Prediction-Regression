import chromadb
from typing import List, Dict, Any, Tuple
from pathlib import Path
from src.utils.logger import setup_logger
from src.rag.utils.chunk import chunker
from src.rag.utils.embed import embedding

project_root = Path(__file__).resolve().parent.parent.parent
logger = setup_logger(__name__)
class ChromaRAG:

    COLLECTION_NAME: str = "financial_news_collection"
    SEARCH_K: int = 3
    CHROMA_PERSIST_PATH: Path = Path("/data/vector_stores/news_store") if Path("/data").exists() else project_root / "data" / "vector_stores" / "news_store"

    def __init__(self):
        self._chroma_client: chromadb.PersistentClient = None
        self._chroma_collection: chromadb.Collection = None
        self.chunker = chunker

    def _get_chroma_client(self) -> chromadb.PersistentClient:
        if self._chroma_client is None:
            self.CHROMA_PERSIST_PATH.mkdir(parents=True, exist_ok=True)
            try:
                logger.info("Initializing ChromaDB at path: %s", self.CHROMA_PERSIST_PATH)
                self._chroma_client = chromadb.PersistentClient(path=str(self.CHROMA_PERSIST_PATH))
            except Exception:
                logger.exception("Failed to initialize ChromaDB client")
                raise
        return self._chroma_client

    def _get_news_collection(self) -> chromadb.Collection:
        if self._chroma_collection is None:
            client = self._get_chroma_client()
            self._chroma_collection = client.get_or_create_collection(
                name=self.COLLECTION_NAME,
            )
        return self._chroma_collection

    def ingest_news_documents(self, news_articles: List[Dict[str, Any]]):
        collection = self._get_news_collection()
        chunked_documents = self.chunker(news_articles)
        if not chunked_documents:
            return
        contents = [doc['content'] for doc in chunked_documents]
        metadatas = [doc['metadata'] for doc in chunked_documents]
        ids = [doc['metadata']['chunk_id'] for doc in chunked_documents]
        embeddings_list = embedding.embed_texts(contents)
        try:
            collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=contents
            )
        except Exception:
            logger.exception("Failed to ingest news documents into ChromaDB")
            raise

    def retrieve_context(self, query: str) -> Tuple[str, List[Dict[str, Any]]]:
        collection = self._get_news_collection()
        query_embedding = embedding.embed_query(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=self.SEARCH_K,
            include=['documents', 'metadatas']
        )
        retrieved_documents = results['documents'][0]
        retrieved_metadatas = results['metadatas'][0]
        context_list = []
        sources = []
        for doc, meta in zip(retrieved_documents, retrieved_metadatas):
            context_list.append(f"--- SOURCE: {meta.get('headline', 'N/A')} ({meta.get('source', 'N/A')}) ---\n{doc}")
            sources.append({
                'headline': meta.get('headline', 'N/A'),
                'source': meta.get('source', 'N/A'),
                'url': meta.get('url', 'N/A')
            })
        formatted_context = "\n\n".join(context_list)
        return formatted_context, sources
        
rag_system = ChromaRAG()
