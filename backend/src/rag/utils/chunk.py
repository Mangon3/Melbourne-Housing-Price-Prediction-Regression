import re
from typing import List, Dict, Any

class NewsChunker:

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        if chunk_overlap >= chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _split_text(self, text: str) -> List[str]:
        if not text:
            return []
        chunks = []
        start = 0
        text_length = len(text)
        stride = self.chunk_size - self.chunk_overlap
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            chunk = text[start:end]
            chunks.append(chunk)
            start += stride
            if start >= text_length and end < text_length:
                final_chunk_start = max(0, text_length - self.chunk_size)
                if final_chunk_start > start - stride:
                    chunks[-1] = text[final_chunk_start:text_length]
        return chunks

    def __call__(self, news_articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunked_documents = []
        for article in news_articles:
            summary = article.get('summary', '')
            metadata = {
                'source': article.get('source', 'N/A'),
                'date': article.get('date', 'N/A'),
                'url': article.get('url', 'N/A'),
                'headline': article.get('headline', 'N/A')
            }
            chunks = self._split_text(summary)
            for i, chunk in enumerate(chunks):
                chunked_documents.append({
                    'content': chunk,
                    'metadata': {
                        **metadata,
                        'chunk_id': f"{metadata['date']}_{metadata['headline'][:30]}_{i}"
                    }
                })
        return chunked_documents
        
chunker = NewsChunker()
