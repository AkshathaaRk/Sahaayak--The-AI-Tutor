#!/usr/bin/env python3
"""
Redis Setup for MCP Server
Sets up a simple in-memory Redis alternative for development
"""

import threading
import time
import json
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class SimpleRedis:
    """Simple in-memory Redis alternative for development"""
    
    def __init__(self):
        self.data = {}
        self.expiry = {}
        self.lock = threading.Lock()
        self.running = True
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_expired, daemon=True)
        self.cleanup_thread.start()
    
    def _cleanup_expired(self):
        """Clean up expired keys"""
        while self.running:
            try:
                current_time = time.time()
                with self.lock:
                    expired_keys = [
                        key for key, expiry_time in self.expiry.items()
                        if current_time > expiry_time
                    ]
                    
                    for key in expired_keys:
                        if key in self.data:
                            del self.data[key]
                        if key in self.expiry:
                            del self.expiry[key]
                
                time.sleep(1)  # Check every second
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
    
    def ping(self):
        """Test connection"""
        return True
    
    def get(self, key):
        """Get value by key"""
        with self.lock:
            if key in self.expiry and time.time() > self.expiry[key]:
                # Key expired
                if key in self.data:
                    del self.data[key]
                del self.expiry[key]
                return None
            
            return self.data.get(key)
    
    def set(self, key, value):
        """Set key-value pair"""
        with self.lock:
            self.data[key] = value
            # Remove expiry if exists
            if key in self.expiry:
                del self.expiry[key]
        return True
    
    def setex(self, key, seconds, value):
        """Set key-value pair with expiration"""
        with self.lock:
            self.data[key] = value
            self.expiry[key] = time.time() + seconds
        return True
    
    def delete(self, *keys):
        """Delete keys"""
        deleted = 0
        with self.lock:
            for key in keys:
                if key in self.data:
                    del self.data[key]
                    deleted += 1
                if key in self.expiry:
                    del self.expiry[key]
        return deleted
    
    def ttl(self, key):
        """Get time to live for key"""
        with self.lock:
            if key not in self.data:
                return -2  # Key doesn't exist
            
            if key not in self.expiry:
                return -1  # Key exists but no expiry
            
            remaining = self.expiry[key] - time.time()
            return max(0, int(remaining))
    
    def keys(self, pattern="*"):
        """Get keys matching pattern"""
        with self.lock:
            if pattern == "*":
                return list(self.data.keys())
            
            # Simple pattern matching
            if pattern.endswith("*"):
                prefix = pattern[:-1]
                return [key for key in self.data.keys() if key.startswith(prefix)]
            
            return [key for key in self.data.keys() if key == pattern]
    
    def info(self):
        """Get server info"""
        with self.lock:
            return {
                'connected_clients': 1,
                'used_memory': len(str(self.data)),
                'total_commands_processed': 0,
                'keyspace_hits': 0,
                'keyspace_misses': 0,
                'uptime_in_seconds': int(time.time())
            }
    
    def lpush(self, key, *values):
        """Left push to list"""
        with self.lock:
            if key not in self.data:
                self.data[key] = []
            
            for value in values:
                self.data[key].insert(0, value)
        return len(self.data[key])
    
    def lrange(self, key, start, end):
        """Get range from list"""
        with self.lock:
            if key not in self.data:
                return []
            
            lst = self.data[key]
            if end == -1:
                return lst[start:]
            else:
                return lst[start:end+1]
    
    def expire(self, key, seconds):
        """Set expiration for key"""
        with self.lock:
            if key in self.data:
                self.expiry[key] = time.time() + seconds
                return True
            return False
    
    def zadd(self, key, mapping):
        """Add to sorted set"""
        with self.lock:
            if key not in self.data:
                self.data[key] = {}
            
            self.data[key].update(mapping)
        return len(mapping)
    
    def zcard(self, key):
        """Get sorted set size"""
        with self.lock:
            if key not in self.data:
                return 0
            return len(self.data[key])
    
    def zremrangebyscore(self, key, min_score, max_score):
        """Remove by score range"""
        with self.lock:
            if key not in self.data:
                return 0
            
            removed = 0
            items_to_remove = []
            
            for member, score in self.data[key].items():
                if min_score <= score <= max_score:
                    items_to_remove.append(member)
            
            for member in items_to_remove:
                del self.data[key][member]
                removed += 1
            
            return removed

# Global Redis instance
_redis_instance = None

def get_redis_client():
    """Get Redis client (real or mock)"""
    global _redis_instance
    
    # Try real Redis first
    try:
        import redis
        client = redis.Redis(
            host='localhost',
            port=6379,
            db=0,
            decode_responses=True,
            socket_timeout=2,
            socket_connect_timeout=2
        )
        client.ping()
        logger.info("✅ Connected to real Redis server")
        return client
    except Exception as e:
        logger.warning(f"⚠️ Real Redis not available: {e}")
    
    # Use mock Redis
    if _redis_instance is None:
        _redis_instance = SimpleRedis()
        logger.info("✅ Using mock Redis for development")
    
    return _redis_instance

def test_redis_functionality():
    """Test Redis functionality"""
    print("🧪 Testing Redis functionality...")
    
    client = get_redis_client()
    
    try:
        # Test basic operations
        client.set("test_key", "test_value")
        value = client.get("test_key")
        assert value == "test_value", f"Expected 'test_value', got {value}"
        print("✅ Basic set/get works")
        
        # Test expiration
        client.setex("temp_key", 2, "temp_value")
        value = client.get("temp_key")
        assert value == "temp_value", "Temp key should exist"
        
        time.sleep(3)
        value = client.get("temp_key")
        assert value is None, "Temp key should have expired"
        print("✅ Expiration works")
        
        # Test TTL
        client.setex("ttl_key", 10, "ttl_value")
        ttl = client.ttl("ttl_key")
        assert 0 < ttl <= 10, f"TTL should be between 0 and 10, got {ttl}"
        print("✅ TTL works")
        
        # Test delete
        client.set("delete_key", "delete_value")
        deleted = client.delete("delete_key")
        assert deleted == 1, f"Should delete 1 key, deleted {deleted}"
        print("✅ Delete works")
        
        # Test keys
        client.set("mcp:test1", "value1")
        client.set("mcp:test2", "value2")
        keys = client.keys("mcp:*")
        assert len(keys) >= 2, f"Should find at least 2 keys, found {len(keys)}"
        print("✅ Keys pattern matching works")
        
        print("🎉 All Redis tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Redis test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Setting up Redis for MCP Server...")
    
    # Test Redis functionality
    if test_redis_functionality():
        print("✅ Redis setup complete and functional!")
    else:
        print("❌ Redis setup failed!")
        exit(1)
