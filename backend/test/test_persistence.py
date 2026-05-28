import pytest
import json
from unittest.mock import patch, MagicMock
from src.cache.persistence import CacheManager

def test_cache_manager_init_url():
    with patch("src.cache.persistence.settings") as mock_settings:
        mock_settings.REDIS_URL = "redis://localhost:6379"
        with patch("src.cache.persistence.redis.from_url") as mock_from_url:
            mock_redis = MagicMock()
            mock_from_url.return_value = mock_redis
            cm = CacheManager()
            assert cm.redis_available is True
            mock_redis.ping.assert_called_once()

def test_cache_manager_init_host_port():
    with patch("src.cache.persistence.settings") as mock_settings:
        mock_settings.REDIS_URL = None
        mock_settings.REDIS_HOST = "localhost"
        mock_settings.REDIS_PORT = 6379
        mock_settings.REDIS_DB = 0
        with patch("src.cache.persistence.redis.Redis") as mock_redis_cls:
            mock_redis = MagicMock()
            mock_redis_cls.return_value = mock_redis
            cm = CacheManager()
            assert cm.redis_available is True
            mock_redis.ping.assert_called_once()

def test_cache_manager_init_failure():
    with patch("src.cache.persistence.settings") as mock_settings:
        mock_settings.REDIS_URL = "redis://localhost:6379"
        with patch("src.cache.persistence.redis.from_url", side_effect=Exception("Redis dead")):
            cm = CacheManager()
            assert cm.redis_available is False

def test_invoke_with_cache_disabled():
    with patch("src.cache.persistence.settings"):
        with patch("src.cache.persistence.redis.from_url", side_effect=Exception):
            cm = CacheManager()
    
    worker = MagicMock(return_value="worker_output")
    res = cm.invoke_with_cache(worker, "query", "AAPL", 7)
    assert res == {"output": "worker_output"}
    worker.assert_called_once()

def test_invoke_with_cache_hit():
    with patch("src.cache.persistence.settings") as mock_settings:
        mock_settings.REDIS_URL = "redis://localhost:6379"
        with patch("src.cache.persistence.redis.from_url") as mock_from_url:
            mock_redis = MagicMock()
            mock_redis.get.return_value = json.dumps("cached_result")
            mock_redis.ttl.return_value = 100
            mock_from_url.return_value = mock_redis
            cm = CacheManager()
            
    worker = MagicMock()
    res = cm.invoke_with_cache(worker, "query", "AAPL", 7)
    assert res == {"output": "cached_result"}
    worker.assert_not_called()

def test_invoke_with_cache_get_fails():
    with patch("src.cache.persistence.settings") as mock_settings:
        mock_settings.REDIS_URL = "redis://localhost:6379"
        with patch("src.cache.persistence.redis.from_url") as mock_from_url:
            mock_redis = MagicMock()
            mock_redis.get.side_effect = Exception("Get failed")
            mock_from_url.return_value = mock_redis
            cm = CacheManager()
            
    worker = MagicMock(return_value="worker_result")
    res = cm.invoke_with_cache(worker, "query", "AAPL", 7)
    assert res == {"output": "worker_result"}
    worker.assert_called_once()

def test_invoke_with_cache_miss():
    with patch("src.cache.persistence.settings") as mock_settings:
        mock_settings.REDIS_URL = "redis://localhost:6379"
        with patch("src.cache.persistence.redis.from_url") as mock_from_url:
            mock_redis = MagicMock()
            mock_redis.get.return_value = None
            mock_from_url.return_value = mock_redis
            cm = CacheManager()
            
    worker = MagicMock(return_value="worker_result")
    res = cm.invoke_with_cache(worker, "query", "AAPL", 7)
    assert res == {"output": "worker_result"}
    worker.assert_called_once()
    mock_redis.setex.assert_called_once()

def test_invoke_with_cache_set_fails():
    with patch("src.cache.persistence.settings") as mock_settings:
        mock_settings.REDIS_URL = "redis://localhost:6379"
        with patch("src.cache.persistence.redis.from_url") as mock_from_url:
            mock_redis = MagicMock()
            mock_redis.get.return_value = None
            mock_redis.setex.side_effect = Exception("Set failed")
            mock_from_url.return_value = mock_redis
            cm = CacheManager()
            
    worker = MagicMock(return_value="worker_result")
    res = cm.invoke_with_cache(worker, "query", "AAPL", 7)
    assert res == {"output": "worker_result"}
    worker.assert_called_once()
