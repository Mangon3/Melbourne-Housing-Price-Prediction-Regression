import json
import redis
from src.config.settings import settings
from datetime import datetime, timedelta
from typing import Dict, Any, Callable
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class CacheManager:
    
    CACHE_KEY_PREFIX = "macro_analysis_cache" 
    TTL_SECONDS = 24 * 3600

    def __init__(self):
        self.redis_available = False
        self.redis_client = None
        try:
            if settings.REDIS_URL:
                logger.info(f"Connecting to Redis via URL...")
                self.redis_client = redis.from_url(
                    settings.REDIS_URL,
                    decode_responses=True,
                    socket_connect_timeout=1
                )
            else:
                host = settings.REDIS_HOST
                port = settings.REDIS_PORT
                db = settings.REDIS_DB
                logger.info(f"Connecting to Redis at {host}:{port}...")
                self.redis_client = redis.Redis(
                    host=host, 
                    port=port, 
                    db=db, 
                    decode_responses=True, 
                    socket_connect_timeout=1
                )
            self.redis_client.ping()
            self.redis_available = True
            logger.info(f"Redis client initialized successfully for caching.")
        except Exception as e:
            logger.critical(f"Failed to connect to Redis. Caching will be disabled. Error: {e}")
            
    def invoke_with_cache(
        self,
        worker_function: Callable[[], str],
        input_query: str, 
        symbol: str, 
        timeframe_days: int
    ) -> Dict[str, Any]:
        cache_key = f"{self.CACHE_KEY_PREFIX}:{symbol.upper()}_{timeframe_days}D_{hash(input_query)}"
        if self.redis_available:
            try:
                cached_response_json = self.redis_client.get(cache_key)
                if cached_response_json:
                    logger.info(f"[CACHE HIT] Returning cached analysis for {symbol}.")
                    response_output = json.loads(cached_response_json)
                    ttl_seconds = self.redis_client.ttl(cache_key)
                    if ttl_seconds > 0:
                         logger.debug(f"Time remaining in cache: {timedelta(seconds=ttl_seconds)}")
                    return {"output": response_output}
                else:
                    logger.info(f"[CACHE MISS] No existing cache entry found for {symbol}. Calculating.")
            except Exception as e:
                logger.error(f"Redis get failed: {e}")
        logger.info("--- Running full RAG Pipeline Worker (LLM/Tool/Ingestion) ---")
        response_string = worker_function()
        response = {"output": response_string}
        if self.redis_available:
            try:
                response_json = json.dumps(response_string)
                self.redis_client.setex(cache_key, self.TTL_SECONDS, response_json)
                expiry_dt = datetime.now() + timedelta(seconds=self.TTL_SECONDS)
                logger.info(f"[CACHE UPDATE] Saved new analysis to Redis. (Expires: {expiry_dt.strftime('%Y-%m-%d %H:%M:%S')})")
            except Exception as e:
                logger.error(f"Redis set failed: {e}")
        return response

cache_manager = CacheManager()
