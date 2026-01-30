from typing import List
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.config.settings import settings
from src.utils.logger import setup_logger
from src.utils.retry import retry_with_backoff

logger = setup_logger(__name__)

class EmbeddingClient:

    def __init__(self):
        self.GEMINI_API_KEY = settings.GOOGLE_API_KEY
        self.GEMINI_EMBEDDING_MODEL = settings.TEXT_EMBEDDING
        logger.info(f"Using Embedding Model: {self.GEMINI_EMBEDDING_MODEL}")
        self._embedding_client = None
        
    def get_embedding_client(self) -> GoogleGenerativeAIEmbeddings:
        self._embedding_client = GoogleGenerativeAIEmbeddings(
            model=self.GEMINI_EMBEDDING_MODEL,
            api_key=self.GEMINI_API_KEY
        )
        return self._embedding_client

    @retry_with_backoff(max_retries=5)
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        client = self.get_embedding_client()
        embeddings = client.embed_documents(texts)
        return embeddings

    @retry_with_backoff(max_retries=5)
    def embed_query(self, query: str) -> List[float]:
        client = self.get_embedding_client()
        embedding = client.embed_query(query)
        return embedding
        
embedding = EmbeddingClient()
