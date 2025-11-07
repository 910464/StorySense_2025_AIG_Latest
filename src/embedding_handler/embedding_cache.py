import threading
import logging


class EmbeddingCache:
    """Cache for embeddings to reduce API calls"""

    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        self.lock = threading.Lock()

    def get(self, text):
        """Get embedding from cache"""
        # Use hash of text as key to handle large texts
        key = hash(text)
        with self.lock:
            if key in self.cache:
                self.hits += 1
                return self.cache[key]
            self.misses += 1
            return None

    def put(self, text, embedding):
        """Store embedding in cache"""
        key = hash(text)
        with self.lock:
            # Implement LRU eviction if needed
            if len(self.cache) >= self.max_size:
                # Remove random item (could be improved with LRU)
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = embedding

    def stats(self):
        """Return cache statistics"""
        with self.lock:
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_ratio": self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
            }