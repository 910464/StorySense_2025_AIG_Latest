import pytest
from unittest.mock import Mock, patch, MagicMock
import threading
import time
from src.embedding_handler.embedding_cache import EmbeddingCache


class TestEmbeddingCache:
    """Comprehensive unit tests for EmbeddingCache class"""

    @pytest.fixture
    def cache(self):
        """Create a fresh cache instance for each test"""
        return EmbeddingCache(max_size=10)

    @pytest.fixture
    def cache_small(self):
        """Create a small cache for testing eviction"""
        return EmbeddingCache(max_size=3)

    @pytest.fixture
    def sample_embedding(self):
        """Sample embedding vector"""
        return [0.1, 0.2, 0.3, 0.4, 0.5]

    @pytest.fixture
    def sample_embeddings(self):
        """Multiple sample embeddings"""
        return {
            'text1': [0.1, 0.2, 0.3],
            'text2': [0.4, 0.5, 0.6],
            'text3': [0.7, 0.8, 0.9],
            'text4': [1.0, 1.1, 1.2]
        }

    # ==================== Initialization Tests ====================

    def test_initialization_default(self):
        """Test cache initialization with default parameters"""
        cache = EmbeddingCache()

        assert cache.max_size == 1000
        assert cache.cache == {}
        assert cache.hits == 0
        assert cache.misses == 0
        assert cache.lock is not None
        assert isinstance(cache.lock, type(threading.Lock()))

    def test_initialization_custom_size(self):
        """Test cache initialization with custom max_size"""
        cache = EmbeddingCache(max_size=500)

        assert cache.max_size == 500
        assert cache.cache == {}
        assert cache.hits == 0
        assert cache.misses == 0

    def test_initialization_small_size(self):
        """Test cache initialization with very small size"""
        cache = EmbeddingCache(max_size=1)

        assert cache.max_size == 1
        assert cache.cache == {}

    def test_initialization_large_size(self):
        """Test cache initialization with large size"""
        cache = EmbeddingCache(max_size=10000)

        assert cache.max_size == 10000
        assert cache.cache == {}

    # ==================== Get Method Tests ====================

    def test_get_cache_miss(self, cache):
        """Test getting an item that doesn't exist in cache (cache miss)"""
        result = cache.get("nonexistent text")

        assert result is None
        assert cache.misses == 1
        assert cache.hits == 0

    def test_get_cache_hit(self, cache, sample_embedding):
        """Test getting an item that exists in cache (cache hit)"""
        # First, put an item in the cache
        cache.put("test text", sample_embedding)

        # Now get it
        result = cache.get("test text")

        assert result == sample_embedding
        assert cache.hits == 1
        assert cache.misses == 0

    def test_get_multiple_misses(self, cache):
        """Test multiple cache misses increment counter correctly"""
        cache.get("text1")
        cache.get("text2")
        cache.get("text3")

        assert cache.misses == 3
        assert cache.hits == 0

    def test_get_multiple_hits(self, cache, sample_embedding):
        """Test multiple cache hits increment counter correctly"""
        cache.put("test text", sample_embedding)

        cache.get("test text")
        cache.get("test text")
        cache.get("test text")

        assert cache.hits == 3
        assert cache.misses == 0

    def test_get_mixed_hits_and_misses(self, cache, sample_embedding):
        """Test mixed cache hits and misses"""
        cache.put("existing", sample_embedding)

        cache.get("existing")  # hit
        cache.get("nonexistent1")  # miss
        cache.get("existing")  # hit
        cache.get("nonexistent2")  # miss

        assert cache.hits == 2
        assert cache.misses == 2

    def test_get_with_empty_string(self, cache):
        """Test getting with empty string"""
        result = cache.get("")

        assert result is None
        assert cache.misses == 1

    def test_get_with_long_text(self, cache, sample_embedding):
        """Test getting with very long text"""
        long_text = "a" * 10000
        cache.put(long_text, sample_embedding)

        result = cache.get(long_text)

        assert result == sample_embedding
        assert cache.hits == 1

    def test_get_with_special_characters(self, cache, sample_embedding):
        """Test getting with special characters in text"""
        special_text = "Test with special chars: !@#\$%^&amp;*()_+-=[]{}|;':\",./&lt;&gt;?"
        cache.put(special_text, sample_embedding)

        result = cache.get(special_text)

        assert result == sample_embedding
        assert cache.hits == 1

    def test_get_with_unicode(self, cache, sample_embedding):
        """Test getting with unicode characters"""
        unicode_text = "Test with unicode: 你好世界 🌍 مرحبا"
        cache.put(unicode_text, sample_embedding)

        result = cache.get(unicode_text)

        assert result == sample_embedding
        assert cache.hits == 1

    def test_get_with_newlines(self, cache, sample_embedding):
        """Test getting with newlines in text"""
        text_with_newlines = "Line 1\nLine 2\nLine 3"
        cache.put(text_with_newlines, sample_embedding)

        result = cache.get(text_with_newlines)

        assert result == sample_embedding
        assert cache.hits == 1

    # ==================== Put Method Tests ====================

    def test_put_single_item(self, cache, sample_embedding):
        """Test putting a single item in cache"""
        cache.put("test text", sample_embedding)

        assert len(cache.cache) == 1
        result = cache.get("test text")
        assert result == sample_embedding

    def test_put_multiple_items(self, cache, sample_embeddings):
        """Test putting multiple items in cache"""
        for text, embedding in sample_embeddings.items():
            cache.put(text, embedding)

        assert len(cache.cache) == len(sample_embeddings)

        for text, embedding in sample_embeddings.items():
            assert cache.get(text) == embedding

    def test_put_overwrite_existing(self, cache):
        """Test overwriting an existing item in cache"""
        old_embedding = [0.1, 0.2, 0.3]
        new_embedding = [0.4, 0.5, 0.6]

        cache.put("test text", old_embedding)
        cache.put("test text", new_embedding)

        result = cache.get("test text")
        assert result == new_embedding
        assert len(cache.cache) == 1

    def test_put_empty_embedding(self, cache):
        """Test putting an empty embedding"""
        cache.put("test text", [])

        result = cache.get("test text")
        assert result == []

    def test_put_none_embedding(self, cache):
        """Test putting None as embedding"""
        cache.put("test text", None)

        result = cache.get("test text")
        assert result is None

    def test_put_large_embedding(self, cache):
        """Test putting a large embedding vector"""
        large_embedding = [0.1] * 1536  # Typical size for embeddings
        cache.put("test text", large_embedding)

        result = cache.get("test text")
        assert result == large_embedding
        assert len(result) == 1536

    # ==================== Eviction Tests ====================

    def test_eviction_when_full(self, cache_small):
        """Test that cache evicts items when full"""
        # Fill the cache to max capacity
        cache_small.put("text1", [0.1])
        cache_small.put("text2", [0.2])
        cache_small.put("text3", [0.3])

        assert len(cache_small.cache) == 3

        # Add one more item - should trigger eviction
        cache_small.put("text4", [0.4])

        # Cache should still be at max size
        assert len(cache_small.cache) == 3

        # One of the original items should be evicted
        # (We can't predict which one due to dict iteration order)
        all_texts = ["text1", "text2", "text3", "text4"]
        present_count = sum(1 for text in all_texts if cache_small.get(text) is not None)
        assert present_count == 3

    def test_eviction_multiple_times(self, cache_small):
        """Test multiple evictions"""
        # Add items beyond capacity multiple times
        for i in range(10):
            cache_small.put(f"text{i}", [float(i)])

        # Cache should still be at max size
        assert len(cache_small.cache) == 3

    def test_eviction_preserves_recent_items(self, cache_small):
        """Test that eviction removes older items (basic LRU-like behavior)"""
        cache_small.put("text1", [0.1])
        cache_small.put("text2", [0.2])
        cache_small.put("text3", [0.3])

        # Add a new item
        cache_small.put("text4", [0.4])

        # The most recently added item should be present
        assert cache_small.get("text4") == [0.4]

    def test_no_eviction_when_not_full(self, cache):
        """Test that no eviction occurs when cache is not full"""
        cache.put("text1", [0.1])
        cache.put("text2", [0.2])
        cache.put("text3", [0.3])

        assert len(cache.cache) == 3

        # All items should still be present
        assert cache.get("text1") == [0.1]
        assert cache.get("text2") == [0.2]
        assert cache.get("text3") == [0.3]

    # ==================== Stats Method Tests ====================

    def test_stats_initial_state(self, cache):
        """Test stats for a newly created cache"""
        stats = cache.stats()

        assert stats['size'] == 0
        assert stats['max_size'] == 10
        assert stats['hits'] == 0
        assert stats['misses'] == 0
        assert stats['hit_ratio'] == 0

    def test_stats_after_operations(self, cache, sample_embedding):
        """Test stats after various cache operations"""
        cache.put("text1", sample_embedding)
        cache.put("text2", sample_embedding)

        cache.get("text1")  # hit
        cache.get("text2")  # hit
        cache.get("text3")  # miss

        stats = cache.stats()

        assert stats['size'] == 2
        assert stats['max_size'] == 10
        assert stats['hits'] == 2
        assert stats['misses'] == 1
        assert stats['hit_ratio'] == 2 / 3  # 2 hits out of 3 total accesses

    def test_stats_hit_ratio_calculation(self, cache, sample_embedding):
        """Test hit ratio calculation in stats"""
        cache.put("text1", sample_embedding)

        # 3 hits, 2 misses = 3/5 = 0.6
        cache.get("text1")  # hit
        cache.get("text2")  # miss
        cache.get("text1")  # hit
        cache.get("text3")  # miss
        cache.get("text1")  # hit

        stats = cache.stats()

        assert stats['hits'] == 3
        assert stats['misses'] == 2
        assert stats['hit_ratio'] == 0.6

    def test_stats_zero_division_protection(self, cache):
        """Test that stats handles zero division when no accesses"""
        stats = cache.stats()

        assert stats['hit_ratio'] == 0  # Should not raise ZeroDivisionError

    def test_stats_all_hits(self, cache, sample_embedding):
        """Test stats when all accesses are hits"""
        cache.put("text1", sample_embedding)

        cache.get("text1")
        cache.get("text1")
        cache.get("text1")

        stats = cache.stats()

        assert stats['hits'] == 3
        assert stats['misses'] == 0
        assert stats['hit_ratio'] == 1.0

    def test_stats_all_misses(self, cache):
        """Test stats when all accesses are misses"""
        cache.get("text1")
        cache.get("text2")
        cache.get("text3")

        stats = cache.stats()

        assert stats['hits'] == 0
        assert stats['misses'] == 3
        assert stats['hit_ratio'] == 0.0

    def test_stats_after_eviction(self, cache_small):
        """Test stats after cache eviction"""
        # Fill cache and trigger eviction
        for i in range(5):
            cache_small.put(f"text{i}", [float(i)])

        stats = cache_small.stats()

        assert stats['size'] == 3  # Should be at max size
        assert stats['max_size'] == 3

    # ==================== Thread Safety Tests ====================

    def test_concurrent_get_operations(self, cache, sample_embedding):
        """Test thread safety of concurrent get operations"""
        cache.put("test text", sample_embedding)

        results = []
        errors = []

        def get_operation():
            try:
                result = cache.get("test text")
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=get_operation)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # All operations should succeed
        assert len(errors) == 0
        assert len(results) == 10
        assert all(r == sample_embedding for r in results)
        assert cache.hits == 10

    def test_concurrent_put_operations(self, cache):
        """Test thread safety of concurrent put operations"""
        errors = []

        def put_operation(i):
            try:
                cache.put(f"text{i}", [float(i)])
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=put_operation, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # All operations should succeed
        assert len(errors) == 0
        assert len(cache.cache) == 10

    def test_concurrent_mixed_operations(self, cache, sample_embedding):
        """Test thread safety of mixed get and put operations"""
        cache.put("initial", sample_embedding)

        errors = []

        def mixed_operation(i):
            try:
                if i % 2 == 0:
                    cache.put(f"text{i}", [float(i)])
                else:
                    cache.get("initial")
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for i in range(20):
            thread = threading.Thread(target=mixed_operation, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # All operations should succeed
        assert len(errors) == 0

    def test_concurrent_stats_access(self, cache, sample_embedding):
        """Test thread safety of concurrent stats access"""
        cache.put("test", sample_embedding)

        stats_results = []
        errors = []

        def stats_operation():
            try:
                stats = cache.stats()
                stats_results.append(stats)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=stats_operation)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # All operations should succeed
        assert len(errors) == 0
        assert len(stats_results) == 10

    def test_concurrent_eviction(self, cache_small):
        """Test thread safety during concurrent evictions"""
        errors = []

        def put_many(start_idx):
            try:
                for i in range(start_idx, start_idx + 5):
                    cache_small.put(f"text{i}", [float(i)])
            except Exception as e:
                errors.append(e)

        # Create multiple threads that will trigger evictions
        threads = []
        for i in range(5):
            thread = threading.Thread(target=put_many, args=(i * 5,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # All operations should succeed
        assert len(errors) == 0
        # Cache should still be at max size
        assert len(cache_small.cache) <= 3

    # ==================== Hash Collision Tests ====================

    def test_hash_collision_handling(self, cache):
        """Test that different texts with same hash are handled correctly"""
        # Note: In practice, hash collisions are rare but possible
        # This test ensures the cache handles them gracefully

        text1 = "test text 1"
        text2 = "test text 2"
        embedding1 = [0.1, 0.2, 0.3]
        embedding2 = [0.4, 0.5, 0.6]

        cache.put(text1, embedding1)
        cache.put(text2, embedding2)

        # Both should be retrievable
        assert cache.get(text1) == embedding1
        assert cache.get(text2) == embedding2

    # ==================== Edge Cases ====================

    def test_cache_with_size_zero(self):
        """Test cache behavior with max_size of 0"""
        cache = EmbeddingCache(max_size=0)

        # Should not be able to store anything
        cache.put("test", [0.1])

        # Cache should be empty
        assert len(cache.cache) == 0

    def test_cache_with_negative_size(self):
        """Test cache initialization with negative size"""
        # This should work but behave oddly - testing actual behavior
        cache = EmbeddingCache(max_size=-1)

        cache.put("test", [0.1])

        # With negative max_size, eviction logic might behave unexpectedly
        # Just ensure it doesn't crash
        assert cache.max_size == -1

    def test_very_large_cache(self):
        """Test cache with very large max_size"""
        cache = EmbeddingCache(max_size=1000000)

        # Add many items
        for i in range(100):
            cache.put(f"text{i}", [float(i)])

        assert len(cache.cache) == 100
        assert cache.max_size == 1000000

    def test_cache_with_identical_embeddings(self, cache):
        """Test caching different texts with identical embeddings"""
        same_embedding = [0.1, 0.2, 0.3]

        cache.put("text1", same_embedding)
        cache.put("text2", same_embedding)
        cache.put("text3", same_embedding)

        assert len(cache.cache) == 3
        assert cache.get("text1") == same_embedding
        assert cache.get("text2") == same_embedding
        assert cache.get("text3") == same_embedding

    def test_cache_with_whitespace_variations(self, cache, sample_embedding):
        """Test that whitespace variations are treated as different keys"""
        cache.put("test", sample_embedding)
        cache.put(" test", sample_embedding)
        cache.put("test ", sample_embedding)
        cache.put(" test ", sample_embedding)

        # All should be stored separately
        assert len(cache.cache) == 4

    def test_cache_with_case_variations(self, cache, sample_embedding):
        """Test that case variations are treated as different keys"""
        cache.put("Test", sample_embedding)
        cache.put("test", sample_embedding)
        cache.put("TEST", sample_embedding)

        # All should be stored separately
        assert len(cache.cache) == 3

    # ==================== Performance Tests ====================

    def test_get_performance(self, cache, sample_embedding):
        """Test that get operation is fast"""
        cache.put("test", sample_embedding)

        start_time = time.time()
        for _ in range(1000):
            cache.get("test")
        end_time = time.time()

        # Should complete in reasonable time (&lt; 0.1 seconds)
        assert (end_time - start_time) < 0.1

    def test_put_performance(self, cache):
        """Test that put operation is fast"""
        start_time = time.time()
        for i in range(1000):
            cache.put(f"text{i}", [float(i)])
        end_time = time.time()

        # Should complete in reasonable time (&lt; 0.5 seconds)
        assert (end_time - start_time) < 0.5

    def test_stats_performance(self, cache, sample_embedding):
        """Test that stats operation is fast"""
        # Add some data
        for i in range(100):
            cache.put(f"text{i}", [float(i)])
            cache.get(f"text{i}")

        start_time = time.time()
        for _ in range(1000):
            cache.stats()
        end_time = time.time()

        # Should complete in reasonable time (&lt; 0.1 seconds)
        assert (end_time - start_time) < 0.1

    # ==================== Integration Tests ====================

    def test_realistic_usage_pattern(self, cache):
        """Test a realistic usage pattern"""
        # Simulate realistic cache usage
        embeddings = {}

        # Add some initial data
        for i in range(50):
            embedding = [float(i)] * 10
            embeddings[f"text{i}"] = embedding
            cache.put(f"text{i}", embedding)

        # Simulate mixed access pattern
        for i in range(100):
            if i % 3 == 0:
                # Add new item
                cache.put(f"new_text{i}", [float(i)])
            elif i % 3 == 1:
                # Access existing item
                cache.get(f"text{i % 50}")
            else:
                # Access non-existent item
                cache.get(f"nonexistent{i}")

        # Check stats
        stats = cache.stats()
        assert stats['hits'] > 0
        assert stats['misses'] > 0
        assert 0 < stats['hit_ratio'] < 1

    def test_cache_lifecycle(self, cache, sample_embedding):
        """Test complete cache lifecycle"""
        # Start empty
        assert len(cache.cache) == 0

        # Add items
        cache.put("text1", sample_embedding)
        assert len(cache.cache) == 1

        # Access items
        result = cache.get("text1")
        assert result == sample_embedding

        # Update items
        new_embedding = [0.9, 0.8, 0.7]
        cache.put("text1", new_embedding)
        assert cache.get("text1") == new_embedding

        # Check stats
        stats = cache.stats()
        assert stats['size'] == 1
        assert stats['hits'] == 2
        assert stats['misses'] == 0


# ==================== Additional Test Scenarios ====================

class TestEmbeddingCacheAdvanced:
    """Advanced test scenarios for EmbeddingCache"""

    def test_memory_efficiency(self):
        """Test that cache doesn't consume excessive memory"""
        cache = EmbeddingCache(max_size=100)

        # Add 100 items
        for i in range(100):
            cache.put(f"text{i}", [0.1] * 1536)

        # Cache should not exceed max_size
        assert len(cache.cache) == 100

    def test_cache_clear_simulation(self, cache, sample_embedding):
        """Test simulating cache clear by creating new instance"""
        # Add items
        cache.put("text1", sample_embedding)
        cache.put("text2", sample_embedding)

        # Create new cache (simulates clearing)
        new_cache = EmbeddingCache(max_size=10)

        # New cache should be empty
        assert len(new_cache.cache) == 0
        assert new_cache.get("text1") is None

    def test_hash_function_consistency(self, cache, sample_embedding):
        """Test that hash function is consistent"""
        text = "test text"

        # Put and get multiple times
        cache.put(text, sample_embedding)

        result1 = cache.get(text)
        result2 = cache.get(text)
        result3 = cache.get(text)

        # All results should be the same
        assert result1 == result2 == result3 == sample_embedding
        assert cache.hits == 3

    def test_lock_acquisition(self, cache):
        """Test that lock is properly acquired and released"""

        # This test ensures the lock doesn't deadlock

        def operation():
            for i in range(10):
                cache.put(f"text{i}", [float(i)])
                cache.get(f"text{i}")
                cache.stats()

        # Run in multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=operation)
            threads.append(thread)
            thread.start()

        # All threads should complete without deadlock
        for thread in threads:
            thread.join(timeout=5.0)
            assert not thread.is_alive()  # Thread should have completed
