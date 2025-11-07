# tests/context_handler/test_pgvector_searcher.py

import pytest
from unittest.mock import Mock, patch, MagicMock, call
import time
import logging

from src.context_handler.context_storage_handler.pgvector_searcher import PGVectorSearcher
from src.context_handler.context_storage_handler.pgvector_orchestrator import PGVectorOrchestrator


class TestPGVectorSearcher:
    """Comprehensive test suite for PGVectorSearcher class"""

    @pytest.fixture
    def mock_orchestrator(self, metrics_manager_mock):
        """Create a mock orchestrator with all required attributes"""
        mock_orch = Mock(spec=PGVectorOrchestrator)
        mock_orch.metrics_manager = metrics_manager_mock
        mock_orch.collection_name = "test_collection"
        mock_orch.config_file_path = "../Config/Config.properties"
        mock_orch.reconnect_if_needed = Mock()
        mock_orch.threshold = 0.7
        return mock_orch

    @pytest.fixture
    def searcher(self, mock_orchestrator):
        """Create a PGVectorSearcher instance with mocked dependencies"""
        with patch(
                'src.context_handler.context_storage_handler.pgvector_searcher.PGVectorRetriever') as mock_retriever_class:
            mock_retriever = Mock()
            mock_retriever_class.return_value = mock_retriever

            searcher = PGVectorSearcher(mock_orchestrator)
            searcher.retriever = mock_retriever
            return searcher

    @pytest.fixture
    def mock_query_expander(self):
        """Create a mock QueryExpander"""
        mock_instance = Mock()
        mock_instance.expand_query = Mock()
        mock_instance.generate_search_queries = Mock()
        return mock_instance

    # ==================== Initialization Tests ====================

    def test_initialization_success(self, mock_orchestrator):
        """Test successful initialization of PGVectorSearcher"""
        with patch(
                'src.context_handler.context_storage_handler.pgvector_searcher.PGVectorRetriever') as mock_retriever_class:
            mock_retriever = Mock()
            mock_retriever_class.return_value = mock_retriever

            searcher = PGVectorSearcher(mock_orchestrator)

            assert searcher.orch == mock_orchestrator
            assert searcher.metrics_manager == mock_orchestrator.metrics_manager
            assert searcher.collection_name == "test_collection"
            assert searcher.retriever == mock_retriever
            mock_retriever_class.assert_called_once_with(mock_orchestrator)

    def test_initialization_with_none_orchestrator(self):
        """Test initialization with None orchestrator raises appropriate error"""
        with pytest.raises(AttributeError):
            PGVectorSearcher(None)

    # ==================== Search All Collections Tests ====================

    def test_search_all_collections_success(self, searcher, mock_orchestrator, mock_query_expander):
        """Test successful search across all collections - covers lines 24-90"""
        # Setup mock query expansion
        mock_query_expander.expand_query.return_value = {
            "original_query": "test query",
            "expanded_data": {
                "alternative_phrasings": ["query test", "test search"],
                "keywords": ["test", "query"],
                "broader_concepts": ["testing"]
            }
        }
        mock_query_expander.generate_search_queries.return_value = [
            "test query",
            "query test",
            "test search"
        ]

        # Setup mock retrieval results
        searcher.retriever.retrieval_context.return_value = (
            "Context from query",
            {0.9: "Document 1", 0.8: "Document 2"},
            {0.9: {"source": "doc1.pdf"}, 0.8: {"source": "doc2.pdf"}},
            0.7
        )

        # Mock the stories orchestrator and retriever
        with patch(
                'src.context_handler.context_storage_handler.pgvector_searcher.PGVectorOrchestrator') as mock_stories_orch_class:
            with patch(
                    'src.context_handler.context_storage_handler.pgvector_searcher.PGVectorRetriever') as mock_stories_retriever_class:
                with patch('src.prompt_layer.query_expander.QueryExpander') as mock_qe_class:
                    mock_qe_class.return_value = mock_query_expander

                    mock_stories_orch = Mock()
                    mock_stories_orch_class.return_value = mock_stories_orch

                    mock_stories_retriever = Mock()
                    mock_stories_retriever.retrieval_context.return_value = (
                        "Stories from query",
                        {0.85: "Story 1"},
                        {0.85: {"source": "story1.txt"}},
                        0.7
                    )
                    mock_stories_retriever_class.return_value = mock_stories_retriever

                    # Execute search
                    result = searcher.search_all_collections("test query", k=5)

                    # Verify results
                    assert isinstance(result, tuple)
                    assert len(result) == 4
                    combined_context, combined_docs, combined_metadata, threshold = result

                    assert "CONTEXT FROM QUERY" in combined_context
                    assert "STORIES FROM QUERY" in combined_context
                    assert len(combined_docs) > 0
                    assert threshold == 0.7

                    # Verify query expansion was called
                    mock_query_expander.expand_query.assert_called_once_with("test query")
                    mock_query_expander.generate_search_queries.assert_called_once()

    def test_search_all_collections_with_empty_results(self, searcher, mock_query_expander):
        """Test search with no results found - covers lines 24-90"""
        # Setup mock query expansion
        mock_query_expander.expand_query.return_value = {
            "original_query": "test query",
            "expanded_data": {}
        }
        mock_query_expander.generate_search_queries.return_value = ["test query"]

        # Setup empty retrieval results
        searcher.retriever.retrieval_context.return_value = (
            "",  # Empty context
            {},  # No documents
            {},  # No metadata
            0.7
        )

        with patch(
                'src.context_handler.context_storage_handler.pgvector_searcher.PGVectorOrchestrator') as mock_stories_orch_class:
            with patch(
                    'src.context_handler.context_storage_handler.pgvector_searcher.PGVectorRetriever') as mock_stories_retriever_class:
                with patch('src.prompt_layer.query_expander.QueryExpander') as mock_qe_class:
                    mock_qe_class.return_value = mock_query_expander

                    mock_stories_orch = Mock()
                    mock_stories_orch_class.return_value = mock_stories_orch

                    mock_stories_retriever = Mock()
                    mock_stories_retriever.retrieval_context.return_value = ("", {}, {}, 0.7)
                    mock_stories_retriever_class.return_value = mock_stories_retriever

                    result = searcher.search_all_collections("test query", k=5)

                    combined_context, combined_docs, combined_metadata, threshold = result

                    assert combined_context == ""
                    assert len(combined_docs) == 0
                    assert len(combined_metadata) == 0

    def test_search_all_collections_with_k_limit(self, searcher, mock_query_expander):
        """Test that results are limited to k documents - covers lines 77-84"""
        # Setup mock query expansion
        mock_query_expander.expand_query.return_value = {
            "original_query": "test query",
            "expanded_data": {}
        }
        mock_query_expander.generate_search_queries.return_value = ["test query"]

        # Setup retrieval results with more than k documents
        many_docs = {0.9: "Doc1", 0.85: "Doc2", 0.8: "Doc3", 0.75: "Doc4", 0.7: "Doc5", 0.65: "Doc6"}
        many_metadata = {score: {"source": f"doc{i}.pdf"} for i, score in enumerate(many_docs.keys())}

        searcher.retriever.retrieval_context.return_value = (
            "Context",
            many_docs,
            many_metadata,
            0.7
        )

        with patch(
                'src.context_handler.context_storage_handler.pgvector_searcher.PGVectorOrchestrator') as mock_stories_orch_class:
            with patch(
                    'src.context_handler.context_storage_handler.pgvector_searcher.PGVectorRetriever') as mock_stories_retriever_class:
                with patch('src.prompt_layer.query_expander.QueryExpander') as mock_qe_class:
                    mock_qe_class.return_value = mock_query_expander

                    mock_stories_orch = Mock()
                    mock_stories_orch_class.return_value = mock_stories_orch

                    mock_stories_retriever = Mock()
                    mock_stories_retriever.retrieval_context.return_value = ("", {}, {}, 0.7)
                    mock_stories_retriever_class.return_value = mock_stories_retriever

                    # Search with k=3
                    result = searcher.search_all_collections("test query", k=3)

                    combined_context, combined_docs, combined_metadata, threshold = result

                    # Should only return top 3 documents
                    assert len(combined_docs) == 3
                    assert len(combined_metadata) == 3

                    # Should be sorted by similarity (highest first)
                    scores = list(combined_docs.keys())
                    assert scores == sorted(scores, reverse=True)

    def test_search_all_collections_multiple_queries(self, searcher, mock_query_expander):
        """Test search with multiple expanded queries - covers lines 45-76"""
        # Setup mock query expansion with multiple queries
        mock_query_expander.expand_query.return_value = {
            "original_query": "test query",
            "expanded_data": {}
        }
        mock_query_expander.generate_search_queries.return_value = [
            "test query",
            "query test",
            "testing query"
        ]

        # Track how many times retrieval is called
        call_count = [0]

        def mock_retrieval(*args, **kwargs):
            call_count[0] += 1
            return (
                f"Context {call_count[0]}",
                {0.9 - (call_count[0] * 0.1): f"Doc {call_count[0]}"},
                {0.9 - (call_count[0] * 0.1): {"source": f"doc{call_count[0]}.pdf"}},
                0.7
            )

        searcher.retriever.retrieval_context = Mock(side_effect=mock_retrieval)

        with patch(
                'src.context_handler.context_storage_handler.pgvector_searcher.PGVectorOrchestrator') as mock_stories_orch_class:
            with patch(
                    'src.context_handler.context_storage_handler.pgvector_searcher.PGVectorRetriever') as mock_stories_retriever_class:
                with patch('src.prompt_layer.query_expander.QueryExpander') as mock_qe_class:
                    mock_qe_class.return_value = mock_query_expander

                    mock_stories_orch = Mock()
                    mock_stories_orch_class.return_value = mock_stories_orch

                    mock_stories_retriever = Mock()
                    mock_stories_retriever.retrieval_context = Mock(side_effect=mock_retrieval)
                    mock_stories_retriever_class.return_value = mock_stories_retriever

                    result = searcher.search_all_collections("test query", k=5)

                    combined_context, combined_docs, combined_metadata, threshold = result

                    # Should have called retrieval for each expanded query (3 queries * 2 collections = 6 calls)
                    assert call_count[0] == 6

                    # Should have context from all queries
                    assert "CONTEXT FROM QUERY" in combined_context
                    assert "STORIES FROM QUERY" in combined_context

    # ==================== Error Handling Tests ====================

    def test_search_all_collections_with_retry_on_error(self, searcher, mock_query_expander):
        """Test retry logic on database error - covers lines 86-103"""
        # Setup mock query expansion
        mock_query_expander.expand_query.return_value = {
            "original_query": "test query",
            "expanded_data": {}
        }
        mock_query_expander.generate_search_queries.return_value = ["test query"]

        # Make retrieval fail twice, then succeed
        call_count = [0]

        def mock_retrieval_with_error(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] >= 2:
                raise Exception("Database connection error")
            return ("Context", {0.9: "Doc1"}, {0.9: {"source": "doc1.pdf"}}, 0.7)

        searcher.retriever.retrieval_context = Mock(side_effect=mock_retrieval_with_error)

        with patch(
                'src.context_handler.context_storage_handler.pgvector_searcher.PGVectorOrchestrator') as mock_stories_orch_class:
            with patch(
                    'src.context_handler.context_storage_handler.pgvector_searcher.PGVectorRetriever') as mock_stories_retriever_class:
                with patch('src.prompt_layer.query_expander.QueryExpander') as mock_qe_class:
                    with patch('time.sleep') as mock_sleep:
                        mock_qe_class.return_value = mock_query_expander

                        mock_stories_orch = Mock()
                        mock_stories_orch_class.return_value = mock_stories_orch

                        mock_stories_retriever = Mock()
                        mock_stories_retriever.retrieval_context = Mock(side_effect=mock_retrieval_with_error)
                        mock_stories_retriever_class.return_value = mock_stories_retriever

                        result = searcher.search_all_collections("test query", k=5)

                        # Should eventually succeed after retries
                        combined_context, combined_docs, combined_metadata, threshold = result
                        assert "Context" in combined_context

                        # Should have called sleep for retries
                        assert mock_sleep.call_count >= 1

    def test_search_all_collections_max_retries_exceeded(self, searcher, mock_query_expander):
        """Test behavior when max retries are exceeded - covers lines 86-103"""
        # Setup mock query expansion
        mock_query_expander.expand_query.return_value = {
            "original_query": "test query",
            "expanded_data": {}
        }
        mock_query_expander.generate_search_queries.return_value = ["test query"]

        # Make retrieval always fail
        searcher.retriever.retrieval_context = Mock(side_effect=Exception("Persistent database error"))

        with patch(
                'src.context_handler.context_storage_handler.pgvector_searcher.PGVectorOrchestrator') as mock_stories_orch_class:
            with patch(
                    'src.context_handler.context_storage_handler.pgvector_searcher.PGVectorRetriever') as mock_stories_retriever_class:
                with patch('src.prompt_layer.query_expander.QueryExpander') as mock_qe_class:
                    with patch('time.sleep') as mock_sleep:
                        mock_qe_class.return_value = mock_query_expander

                        mock_stories_orch = Mock()
                        mock_stories_orch_class.return_value = mock_stories_orch

                        mock_stories_retriever = Mock()
                        mock_stories_retriever.retrieval_context = Mock(
                            side_effect=Exception("Persistent database error"))
                        mock_stories_retriever_class.return_value = mock_stories_retriever

                        result = searcher.search_all_collections("test query", k=5)

                        # Should return empty results after max retries
                        combined_context, combined_docs, combined_metadata, threshold = result
                        assert combined_context == ""
                        assert len(combined_docs) == 0
                        assert len(combined_metadata) == 0
                        assert threshold == searcher.orch.threshold

                        # Should have recorded error
                        searcher.metrics_manager.record_error.assert_called()

    def test_search_all_collections_reconnect_on_error(self, searcher, mock_orchestrator, mock_query_expander):
        """Test database reconnection on error - covers lines 99-100"""
        # Setup mock query expansion
        mock_query_expander.expand_query.return_value = {
            "original_query": "test query",
            "expanded_data": {}
        }
        mock_query_expander.generate_search_queries.return_value = ["test query"]

        # Make retrieval fail once
        call_count = [0]

        def mock_retrieval_with_error(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Connection lost")
            return ("Context", {0.9: "Doc1"}, {0.9: {"source": "doc1.pdf"}}, 0.7)

        searcher.retriever.retrieval_context = Mock(side_effect=mock_retrieval_with_error)

        with patch(
                'src.context_handler.context_storage_handler.pgvector_searcher.PGVectorOrchestrator') as mock_stories_orch_class:
            with patch(
                    'src.context_handler.context_storage_handler.pgvector_searcher.PGVectorRetriever') as mock_stories_retriever_class:
                with patch('src.prompt_layer.query_expander.QueryExpander') as mock_qe_class:
                    with patch('time.sleep'):
                        mock_qe_class.return_value = mock_query_expander

                        mock_stories_orch = Mock()
                        mock_stories_orch.db = Mock()
                        mock_stories_orch.db.connect_with_retry = Mock()
                        mock_stories_orch_class.return_value = mock_stories_orch

                        mock_stories_retriever = Mock()
                        mock_stories_retriever.retrieval_context = Mock(side_effect=mock_retrieval_with_error)
                        mock_stories_retriever_class.return_value = mock_stories_retriever

                        result = searcher.search_all_collections("test query", k=5)

                        # Should have called connect_with_retry
                        mock_stories_orch.db.connect_with_retry.assert_called()

    # ==================== Edge Cases Tests ====================

    def test_search_all_collections_with_empty_query(self, searcher, mock_query_expander):
        """Test search with empty query string"""
        mock_query_expander.expand_query.return_value = {
            "original_query": "",
            "expanded_data": {}
        }
        mock_query_expander.generate_search_queries.return_value = [""]

        searcher.retriever.retrieval_context.return_value = ("", {}, {}, 0.7)

        with patch(
                'src.context_handler.context_storage_handler.pgvector_searcher.PGVectorOrchestrator') as mock_stories_orch_class:
            with patch(
                    'src.context_handler.context_storage_handler.pgvector_searcher.PGVectorRetriever') as mock_stories_retriever_class:
                with patch('src.prompt_layer.query_expander.QueryExpander') as mock_qe_class:
                    mock_qe_class.return_value = mock_query_expander

                    mock_stories_orch = Mock()
                    mock_stories_orch_class.return_value = mock_stories_orch

                    mock_stories_retriever = Mock()
                    mock_stories_retriever.retrieval_context.return_value = ("", {}, {}, 0.7)
                    mock_stories_retriever_class.return_value = mock_stories_retriever

                    result = searcher.search_all_collections("", k=5)

                    combined_context, combined_docs, combined_metadata, threshold = result
                    assert combined_context == ""
                    assert len(combined_docs) == 0

    def test_search_all_collections_with_k_zero(self, searcher, mock_query_expander):
        """Test search with k=0"""
        mock_query_expander.expand_query.return_value = {
            "original_query": "test query",
            "expanded_data": {}
        }
        mock_query_expander.generate_search_queries.return_value = ["test query"]

        searcher.retriever.retrieval_context.return_value = (
            "Context",
            {0.9: "Doc1"},
            {0.9: {"source": "doc1.pdf"}},
            0.7
        )

        with patch(
                'src.context_handler.context_storage_handler.pgvector_searcher.PGVectorOrchestrator') as mock_stories_orch_class:
            with patch(
                    'src.context_handler.context_storage_handler.pgvector_searcher.PGVectorRetriever') as mock_stories_retriever_class:
                with patch('src.prompt_layer.query_expander.QueryExpander') as mock_qe_class:
                    mock_qe_class.return_value = mock_query_expander

                    mock_stories_orch = Mock()
                    mock_stories_orch_class.return_value = mock_stories_orch

                    mock_stories_retriever = Mock()
                    mock_stories_retriever.retrieval_context.return_value = ("", {}, {}, 0.7)
                    mock_stories_retriever_class.return_value = mock_stories_retriever

                    result = searcher.search_all_collections("test query", k=0)

                    combined_context, combined_docs, combined_metadata, threshold = result
                    # Should handle k=0 gracefully
                    assert isinstance(combined_docs, dict)

    def test_search_all_collections_with_large_k(self, searcher, mock_query_expander):
        """Test search with very large k value"""
        mock_query_expander.expand_query.return_value = {
            "original_query": "test query",
            "expanded_data": {}
        }
        mock_query_expander.generate_search_queries.return_value = ["test query"]

        # Return fewer documents than k
        searcher.retriever.retrieval_context.return_value = (
            "Context",
            {0.9: "Doc1", 0.8: "Doc2"},
            {0.9: {"source": "doc1.pdf"}, 0.8: {"source": "doc2.pdf"}},
            0.7
        )

        with patch(
                'src.context_handler.context_storage_handler.pgvector_searcher.PGVectorOrchestrator') as mock_stories_orch_class:
            with patch(
                    'src.context_handler.context_storage_handler.pgvector_searcher.PGVectorRetriever') as mock_stories_retriever_class:
                with patch('src.prompt_layer.query_expander.QueryExpander') as mock_qe_class:
                    mock_qe_class.return_value = mock_query_expander

                    mock_stories_orch = Mock()
                    mock_stories_orch_class.return_value = mock_stories_orch

                    mock_stories_retriever = Mock()
                    mock_stories_retriever.retrieval_context.return_value = ("", {}, {}, 0.7)
                    mock_stories_retriever_class.return_value = mock_stories_retriever

                    result = searcher.search_all_collections("test query", k=1000)

                    combined_context, combined_docs, combined_metadata, threshold = result
                    # Should return all available documents (not fail)
                    assert len(combined_docs) <= 1000

    # ==================== Integration Tests ====================

    def test_search_all_collections_logging(self, searcher, mock_query_expander):
        """Test that appropriate logging occurs during search"""
        mock_query_expander.expand_query.return_value = {
            "original_query": "test query",
            "expanded_data": {}
        }
        mock_query_expander.generate_search_queries.return_value = ["test query"]

        searcher.retriever.retrieval_context.return_value = (
            "Context",
            {0.9: "Doc1"},
            {0.9: {"source": "doc1.pdf"}},
            0.7
        )

        with patch(
                'src.context_handler.context_storage_handler.pgvector_searcher.PGVectorOrchestrator') as mock_stories_orch_class:
            with patch(
                    'src.context_handler.context_storage_handler.pgvector_searcher.PGVectorRetriever') as mock_stories_retriever_class:
                with patch('src.prompt_layer.query_expander.QueryExpander') as mock_qe_class:
                    with patch('logging.info') as mock_log_info:
                        mock_qe_class.return_value = mock_query_expander

                        mock_stories_orch = Mock()
                        mock_stories_orch_class.return_value = mock_stories_orch

                        mock_stories_retriever = Mock()
                        mock_stories_retriever.retrieval_context.return_value = ("", {}, {}, 0.7)
                        mock_stories_retriever_class.return_value = mock_stories_retriever

                        result = searcher.search_all_collections("test query", k=5)

                        # Should have logged query expansion info
                        assert mock_log_info.call_count > 0

    def test_search_all_collections_metrics_recording(self, searcher, mock_orchestrator, mock_query_expander):
        """Test that metrics are properly recorded during search"""
        mock_query_expander.expand_query.return_value = {
            "original_query": "test query",
            "expanded_data": {}
        }
        mock_query_expander.generate_search_queries.return_value = ["test query"]

        searcher.retriever.retrieval_context.return_value = (
            "Context",
            {0.9: "Doc1"},
            {0.9: {"source": "doc1.pdf"}},
            0.7
        )

        with patch(
                'src.context_handler.context_storage_handler.pgvector_searcher.PGVectorOrchestrator') as mock_stories_orch_class:
            with patch(
                    'src.context_handler.context_storage_handler.pgvector_searcher.PGVectorRetriever') as mock_stories_retriever_class:
                with patch('src.prompt_layer.query_expander.QueryExpander') as mock_qe_class:
                    mock_qe_class.return_value = mock_query_expander

                    mock_stories_orch = Mock()
                    mock_stories_orch_class.return_value = mock_stories_orch

                    mock_stories_retriever = Mock()
                    mock_stories_retriever.retrieval_context.return_value = ("", {}, {}, 0.7)
                    mock_stories_retriever_class.return_value = mock_stories_retriever

                    # Make retrieval fail to test error recording
                    searcher.retriever.retrieval_context = Mock(side_effect=Exception("Test error"))
                    mock_stories_retriever.retrieval_context = Mock(side_effect=Exception("Test error"))

                    with patch('time.sleep'):
                        result = searcher.search_all_collections("test query", k=5)

                    # Should have recorded error in metrics
                    mock_orchestrator.metrics_manager.record_error.assert_called()

    # ==================== Query Expansion Tests ====================

    def test_search_with_query_expansion_failure(self, searcher, mock_query_expander):
        """Test handling of query expansion failure"""
        # Make query expansion fail
        mock_query_expander.expand_query.side_effect = Exception("Query expansion error")
        mock_query_expander.generate_search_queries.return_value = ["test query"]

        searcher.retriever.retrieval_context.return_value = (
            "Context",
            {0.9: "Doc1"},
            {0.9: {"source": "doc1.pdf"}},
            0.7
        )

        with patch(
                'src.context_handler.context_storage_handler.pgvector_searcher.PGVectorOrchestrator') as mock_stories_orch_class:
            with patch(
                    'src.context_handler.context_storage_handler.pgvector_searcher.PGVectorRetriever') as mock_stories_retriever_class:
                with patch('src.prompt_layer.query_expander.QueryExpander') as mock_qe_class:
                    mock_qe_class.return_value = mock_query_expander

                    mock_stories_orch = Mock()
                    mock_stories_orch_class.return_value = mock_stories_orch

                    mock_stories_retriever = Mock()
                    mock_stories_retriever.retrieval_context.return_value = ("", {}, {}, 0.7)
                    mock_stories_retriever_class.return_value = mock_stories_retriever

                    # Should handle the error gracefully
                    with pytest.raises(Exception):
                        result = searcher.search_all_collections("test query", k=5)

    def test_search_with_single_query_expansion(self, searcher, mock_query_expander):
        """Test search with only original query (no expansion)"""
        mock_query_expander.expand_query.return_value = {
            "original_query": "test query",
            "expanded_data": {}
        }
        mock_query_expander.generate_search_queries.return_value = ["test query"]  # Only original

        searcher.retriever.retrieval_context.return_value = (
            "Context",
            {0.9: "Doc1"},
            {0.9: {"source": "doc1.pdf"}},
            0.7
        )

        with patch(
                'src.context_handler.context_storage_handler.pgvector_searcher.PGVectorOrchestrator') as mock_stories_orch_class:
            with patch(
                    'src.context_handler.context_storage_handler.pgvector_searcher.PGVectorRetriever') as mock_stories_retriever_class:
                with patch('src.prompt_layer.query_expander.QueryExpander') as mock_qe_class:
                    mock_qe_class.return_value = mock_query_expander

                    mock_stories_orch = Mock()
                    mock_stories_orch_class.return_value = mock_stories_orch

                    mock_stories_retriever = Mock()
                    mock_stories_retriever.retrieval_context.return_value = ("", {}, {}, 0.7)
                    mock_stories_retriever_class.return_value = mock_stories_retriever

                    result = searcher.search_all_collections("test query", k=5)

                    combined_context, combined_docs, combined_metadata, threshold = result

                    # Should still work with single query
                    assert isinstance(combined_context, str)
                    assert isinstance(combined_docs, dict)

    # ==================== Result Combination Tests ====================

    def test_search_result_deduplication(self, searcher, mock_query_expander):
        """Test that duplicate results from different queries are handled"""
        mock_query_expander.expand_query.return_value = {
            "original_query": "test query",
            "expanded_data": {}
        }
        mock_query_expander.generate_search_queries.return_value = ["test query", "query test"]

        # Return same document with different scores
        call_count = [0]

        def mock_retrieval(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return ("Context", {0.9: "Same Doc"}, {0.9: {"source": "doc1.pdf"}}, 0.7)
            else:
                return ("Context", {0.85: "Same Doc"}, {0.85: {"source": "doc1.pdf"}}, 0.7)

        searcher.retriever.retrieval_context = Mock(side_effect=mock_retrieval)

        with patch(
                'src.context_handler.context_storage_handler.pgvector_searcher.PGVectorOrchestrator') as mock_stories_orch_class:
            with patch(
                    'src.context_handler.context_storage_handler.pgvector_searcher.PGVectorRetriever') as mock_stories_retriever_class:
                with patch('src.prompt_layer.query_expander.QueryExpander') as mock_qe_class:
                    mock_qe_class.return_value = mock_query_expander

                    mock_stories_orch = Mock()
                    mock_stories_orch_class.return_value = mock_stories_orch

                    mock_stories_retriever = Mock()
                    mock_stories_retriever.retrieval_context = Mock(side_effect=mock_retrieval)
                    mock_stories_retriever_class.return_value = mock_stories_retriever

                    result = searcher.search_all_collections("test query", k=5)

                    combined_context, combined_docs, combined_metadata, threshold = result

                    # Should have both scores for the same document
                    assert len(combined_docs) >= 2

    def test_search_result_sorting(self, searcher, mock_query_expander):
        """Test that results are properly sorted by similarity score"""
        mock_query_expander.expand_query.return_value = {
            "original_query": "test query",
            "expanded_data": {}
        }
        mock_query_expander.generate_search_queries.return_value = ["test query"]

        # Return unsorted documents
        unsorted_docs = {0.7: "Doc3", 0.9: "Doc1", 0.8: "Doc2"}
        unsorted_metadata = {0.7: {"source": "doc3.pdf"}, 0.9: {"source": "doc1.pdf"}, 0.8: {"source": "doc2.pdf"}}

        searcher.retriever.retrieval_context.return_value = (
            "Context",
            unsorted_docs,
            unsorted_metadata,
            0.7
        )

        with patch(
                'src.context_handler.context_storage_handler.pgvector_searcher.PGVectorOrchestrator') as mock_stories_orch_class:
            with patch(
                    'src.context_handler.context_storage_handler.pgvector_searcher.PGVectorRetriever') as mock_stories_retriever_class:
                with patch('src.prompt_layer.query_expander.QueryExpander') as mock_qe_class:
                    mock_qe_class.return_value = mock_query_expander

                    mock_stories_orch = Mock()
                    mock_stories_orch_class.return_value = mock_stories_orch

                    mock_stories_retriever = Mock()
                    mock_stories_retriever.retrieval_context.return_value = ("", {}, {}, 0.7)
                    mock_stories_retriever_class.return_value = mock_stories_retriever

                    result = searcher.search_all_collections("test query", k=5)

                    combined_context, combined_docs, combined_metadata, threshold = result

                    # Check that results are sorted
                    scores = list(combined_docs.keys())
                    assert scores == sorted(scores, reverse=True)

    # ==================== Performance Tests ====================

    def test_search_performance_with_many_queries(self, searcher, mock_query_expander):
        """Test performance with many expanded queries"""
        # Generate many queries
        many_queries = [f"query {i}" for i in range(10)]

        mock_query_expander.expand_query.return_value = {
            "original_query": "test query",
            "expanded_data": {}
        }
        mock_query_expander.generate_search_queries.return_value = many_queries

        searcher.retriever.retrieval_context.return_value = (
            "Context",
            {0.9: "Doc1"},
            {0.9: {"source": "doc1.pdf"}},
            0.7
        )

        with patch(
                'src.context_handler.context_storage_handler.pgvector_searcher.PGVectorOrchestrator') as mock_stories_orch_class:
            with patch(
                    'src.context_handler.context_storage_handler.pgvector_searcher.PGVectorRetriever') as mock_stories_retriever_class:
                with patch('src.prompt_layer.query_expander.QueryExpander') as mock_qe_class:
                    mock_qe_class.return_value = mock_query_expander

                    mock_stories_orch = Mock()
                    mock_stories_orch_class.return_value = mock_stories_orch

                    mock_stories_retriever = Mock()
                    mock_stories_retriever.retrieval_context.return_value = ("", {}, {}, 0.7)
                    mock_stories_retriever_class.return_value = mock_stories_retriever

                    start_time = time.time()
                    result = searcher.search_all_collections("test query", k=5)
                    elapsed_time = time.time() - start_time

                    # Should complete in reasonable time
                    assert elapsed_time <5.0  # 5 seconds max

    # ==================== Context Formatting Tests ====================

    def test_context_formatting_with_multiple_sources(self, searcher, mock_query_expander):
        """Test that context from multiple queries is properly formatted"""
        mock_query_expander.expand_query.return_value = {
            "original_query": "test query",
            "expanded_data": {}
        }
        mock_query_expander.generate_search_queries.return_value = ["query1", "query2"]

        call_count = [0]

        def mock_retrieval(*args, **kwargs):
            call_count[0] += 1
            return (
                f"Context from query {call_count[0]}",
                {0.9: f"Doc {call_count[0]}"},
                {0.9: {"source": f"doc{call_count[0]}.pdf"}},
                0.7
            )

        searcher.retriever.retrieval_context = Mock(side_effect=mock_retrieval)

        with patch(
                'src.context_handler.context_storage_handler.pgvector_searcher.PGVectorOrchestrator') as mock_stories_orch_class:
            with patch(
                    'src.context_handler.context_storage_handler.pgvector_searcher.PGVectorRetriever') as mock_stories_retriever_class:
                with patch('src.prompt_layer.query_expander.QueryExpander') as mock_qe_class:
                    mock_qe_class.return_value = mock_query_expander

                    mock_stories_orch = Mock()
                    mock_stories_orch_class.return_value = mock_stories_orch

                    mock_stories_retriever = Mock()
                    mock_stories_retriever.retrieval_context = Mock(side_effect=mock_retrieval)
                    mock_stories_retriever_class.return_value = mock_stories_retriever

                    result = searcher.search_all_collections("test query", k=5)

                    combined_context, combined_docs, combined_metadata, threshold = result

                    # Should have formatted context from both queries
                    assert "CONTEXT FROM QUERY 'query1'" in combined_context
                    assert "CONTEXT FROM QUERY 'query2'" in combined_context
                    assert "STORIES FROM QUERY" in combined_context

    def test_search_with_no_context_but_stories(self, searcher, mock_query_expander):
        """Test search when context is empty but stories are found"""
        mock_query_expander.expand_query.return_value = {
            "original_query": "test query",
            "expanded_data": {}
        }
        mock_query_expander.generate_search_queries.return_value = ["test query"]

        # No context, but stories exist
        searcher.retriever.retrieval_context.return_value = ("", {}, {}, 0.7)

        with patch(
                'src.context_handler.context_storage_handler.pgvector_searcher.PGVectorOrchestrator') as mock_stories_orch_class:
            with patch(
                    'src.context_handler.context_storage_handler.pgvector_searcher.PGVectorRetriever') as mock_stories_retriever_class:
                with patch('src.prompt_layer.query_expander.QueryExpander') as mock_qe_class:
                    mock_qe_class.return_value = mock_query_expander

                    mock_stories_orch = Mock()
                    mock_stories_orch_class.return_value = mock_stories_orch

                    mock_stories_retriever = Mock()
                    mock_stories_retriever.retrieval_context.return_value = (
                        "Stories found",
                        {0.9: "Story1"},
                        {0.9: {"source": "story1.txt"}},
                        0.7
                    )
                    mock_stories_retriever_class.return_value = mock_stories_retriever

                    result = searcher.search_all_collections("test query", k=5)

                    combined_context, combined_docs, combined_metadata, threshold = result

                    # Should have stories but no context
                    assert "STORIES FROM QUERY" in combined_context
                    assert len(combined_docs) > 0

    def test_search_with_context_but_no_stories(self, searcher, mock_query_expander):
        """Test search when context is found but no stories"""
        mock_query_expander.expand_query.return_value = {
            "original_query": "test query",
            "expanded_data": {}
        }
        mock_query_expander.generate_search_queries.return_value = ["test query"]

        # Context exists, but no stories
        searcher.retriever.retrieval_context.return_value = (
            "Context found",
            {0.9: "Doc1"},
            {0.9: {"source": "doc1.pdf"}},
            0.7
        )

        with patch(
                'src.context_handler.context_storage_handler.pgvector_searcher.PGVectorOrchestrator') as mock_stories_orch_class:
            with patch(
                    'src.context_handler.context_storage_handler.pgvector_searcher.PGVectorRetriever') as mock_stories_retriever_class:
                with patch('src.prompt_layer.query_expander.QueryExpander') as mock_qe_class:
                    mock_qe_class.return_value = mock_query_expander

                    mock_stories_orch = Mock()
                    mock_stories_orch_class.return_value = mock_stories_orch

                    mock_stories_retriever = Mock()
                    mock_stories_retriever.retrieval_context.return_value = ("", {}, {}, 0.7)
                    mock_stories_retriever_class.return_value = mock_stories_retriever

                    result = searcher.search_all_collections("test query", k=5)

                    combined_context, combined_docs, combined_metadata, threshold = result

                    # Should have context but no stories
                    assert "CONTEXT FROM QUERY" in combined_context
                    assert len(combined_docs) > 0

    # ==================== Metadata Preservation Tests ====================

    def test_metadata_preservation(self, searcher, mock_query_expander):
        """Test that metadata is properly preserved through search"""
        mock_query_expander.expand_query.return_value = {
            "original_query": "test query",
            "expanded_data": {}
        }
        mock_query_expander.generate_search_queries.return_value = ["test query"]

        # Setup retrieval with detailed metadata
        detailed_metadata = {
            0.9: {
                "source": "doc1.pdf",
                "page": 5,
                "author": "Test Author",
                "date": "2024-01-01"
            }
        }

        searcher.retriever.retrieval_context.return_value = (
            "Context",
            {0.9: "Doc1"},
            detailed_metadata,
            0.7
        )

        with patch(
                'src.context_handler.context_storage_handler.pgvector_searcher.PGVectorOrchestrator') as mock_stories_orch_class:
            with patch(
                    'src.context_handler.context_storage_handler.pgvector_searcher.PGVectorRetriever') as mock_stories_retriever_class:
                with patch('src.prompt_layer.query_expander.QueryExpander') as mock_qe_class:
                    mock_qe_class.return_value = mock_query_expander

                    mock_stories_orch = Mock()
                    mock_stories_orch_class.return_value = mock_stories_orch

                    mock_stories_retriever = Mock()
                    mock_stories_retriever.retrieval_context.return_value = ("", {}, {}, 0.7)
                    mock_stories_retriever_class.return_value = mock_stories_retriever

                    result = searcher.search_all_collections("test query", k=5)

                    combined_context, combined_docs, combined_metadata, threshold = result

                    # Metadata should be preserved
                    assert 0.9 in combined_metadata
                    assert combined_metadata[0.9]["source"] == "doc1.pdf"
                    assert combined_metadata[0.9]["page"] == 5
                    assert combined_metadata[0.9]["author"] == "Test Author"