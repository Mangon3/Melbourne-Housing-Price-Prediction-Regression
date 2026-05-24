import chromadb
import uuid
import time
from typing import List, Dict, Any
from pathlib import Path
from src.config.settings import settings
from src.rag.utils.embed import embedding
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class ConversationStore:

    COLLECTION_NAME: str = "conversation_history"

    def __init__(self):
        self._client: chromadb.PersistentClient = None
        self._collection: chromadb.Collection = None
        self.persist_path = Path(settings.VECTOR_STORE_PATH)

    def _get_client(self) -> chromadb.PersistentClient:
        if self._client is None:
            self.persist_path.mkdir(parents=True, exist_ok=True)
            try:
                self._client = chromadb.PersistentClient(path=str(self.persist_path))
            except Exception:
                logger.exception("Failed to initialize ChromaDB for memory")
                raise
        return self._client

    def _get_collection(self) -> chromadb.Collection:
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=self.COLLECTION_NAME
            )
        return self._collection

    def save_turn(self, user_input: str, model_output: str, intent: str):
        """
        Saves a conversation turn (User + AI) into ChromaDB.
        """
        try:
            collection = self._get_collection()
            combined_text = f"User: {user_input}\nAI: {model_output}"
            embed_vector = embedding.embed_query(combined_text)
            turn_id = str(uuid.uuid4())
            timestamp = time.time()
            collection.add(
                ids=[turn_id],
                embeddings=[embed_vector],
                documents=[combined_text],
                metadatas=[{
                    "timestamp": timestamp,
                    "user_input": user_input,
                    "model_output": model_output,
                    "intent": intent,
                    "type": "conversation_turn"
                }]
            )
            logger.info(f"Saved conversation turn {turn_id} to vector memory.")
        except Exception:
            logger.exception("Error saving conversation to memory")

    def retrieve_similar(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves similar past conversations.
        """
        try:
            collection = self._get_collection()
            query_embed = embedding.embed_query(query)
            results = collection.query(
                query_embeddings=[query_embed],
                n_results=limit
            )
            docs = results['documents'][0]
            metas = results['metadatas'][0]
            history = []
            for doc, meta in zip(docs, metas):
                history.append({
                    "content": doc,
                    "metadata": meta
                })
            return history
        except Exception:
            logger.exception("Error retrieving memory")
            return []
            
memory_store = ConversationStore()
