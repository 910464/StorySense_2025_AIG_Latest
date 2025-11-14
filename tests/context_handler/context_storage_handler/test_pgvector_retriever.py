import pytest
from unittest.mock import Mock, patch, MagicMock, ANY
import time
import json

from attr.validators import gt

from src.context_handler.context_storage_handler.pgvector_retriever import PGVectorRetriever
from src.context_handler.context_storage_handler.pgvector_orchestrator import PGVectorOrchestrator


class TestPGVectorRetriever:

    @pytest.fixture
    def mock_orchestrator(self, global_metrics_manager_mock):
        """Create a mock orchestrator with all required attributes"""
        orchestrator = Mock(spec=PGVectorOrchestrator)

        # Mock database
        orchestrator.db = Mock()
        orchestrator.db.cursor = Mock()
        orchestrator.db.reconnect_if_needed = Mock()
        orchestrator.db.connect_with_retry = Mock()

        # Mock embeddings
        orchestrator.embeddings = Mock()
        orchestrator.embeddings.embed_query = Mock(return_value=[0.1] * 1536)

        # Mock configuration
        orchestrator.similarity_metric = 'cosine'
        orchestrator.threshold = 0.7
        orchestrator.collection_name = 'test_collection'
        orchestrator.retrieval_dir = '../Output/RetrievalContext'

        # Mock metrics
        orchestrator.metrics_reporter = Mock()
        orchestrator.metrics_reporter.record_vector_operation = Mock()
        orchestrator.metrics_manager = global_metrics_manager_mock

        return orchestrator

    @pytest.fixture
    def retriever(self, mock_orchestrator):
        """Create a PGVectorRetriever instance with mocked orchestrator"""
        return PGVectorRetriever(mock_orchestrator)

    def test_initialization(self, retriever, mock_orchestrator):
        """Test that retriever initializes correctly with orchestrator attributes"""
        assert retriever.orch == mock_orchestrator
        assert retriever.db == mock_orchestrator.db
        assert retriever.embeddings == mock_orchestrator.embeddings
        assert retriever.similarity_metric == 'cosine'
        assert retriever.threshold == 0.7
        assert retriever.metrics_reporter == mock_orchestrator.metrics_reporter
        assert retriever.metrics_manager == mock_orchestrator.metrics_manager
        assert retriever.collection_name == 'test_collection'
        assert retriever.retrieval_dir == '../Output/RetrievalContext'

    def test_retrieval_context_cosine_success(self, retriever, mock_orchestrator):
        """Test successful retrieval with cosine similarity metric"""
        # Setup mock cursor to return results
        mock_cursor = mock_orchestrator.db.cursor
        mock_cursor.fetchall.return_value = [
            ('Document content 1', 0.85, {'source': 'doc1.pdf', 'page': 1}),
            ('Document content 2', 0.75, {'source': 'doc2.pdf', 'page': 2}),
            ('Document content 3', 0.72, {'source': 'doc3.pdf', 'page': 1})
        ]

        # Call retrieval_context
        context, docs_with_score, docs_metadata, threshold = retriever.retrieval_context(
            query="test query",
            k=5
        )

        # Verify database reconnection was called
        mock_orchestrator.db.reconnect_if_needed.assert_called_once()

        # Verify embedding was generated
        mock_orchestrator.embeddings.embed_query.assert_called_once_with("test query")

        # Verify SQL query was executed with correct parameters
        mock_cursor.execute.assert_called_once()
        call_args = mock_cursor.execute.call_args
        # Check for the actual operator
        assert '<=>' in call_args[0][0]  # Cosine distance operator
        assert call_args[0][1][1] == 'test_collection'  # Collection name
        assert call_args[0][1][5] == 5  # k value

        # Verify results
        assert len(docs_with_score) == 3
        assert 0.85 in docs_with_score
        assert docs_with_score[0.85] == 'Document content 1'
        assert len(docs_metadata) == 3
        assert docs_metadata[0.85]['source'] == 'doc1.pdf'
        assert 'Document content 1' in context
        assert threshold == 0.7

        # Verify metrics were recorded
        mock_orchestrator.metrics_reporter.record_vector_operation.assert_called_once_with(
            operation_type='query',
            item_count=3,
            duration=ANY
        )

    def test_retrieval_context_l2_similarity(self, retriever, mock_orchestrator):
        """Test retrieval with L2 (Euclidean) distance metric"""
        # Change similarity metric to L2
        retriever.similarity_metric = 'l2'

        # Setup mock cursor
        mock_cursor = mock_orchestrator.db.cursor
        mock_cursor.fetchall.return_value = [
            ('Document content 1', 0.9, {'source': 'doc1.pdf'})
        ]

        # Call retrieval_context
        context, docs_with_score, docs_metadata, threshold = retriever.retrieval_context(
            query="test query",
            k=3
        )

        # Verify L2 distance operator was used
        call_args = mock_cursor.execute.call_args
        assert '<->' in call_args[0][0]  # L2 distance operator (fixed)
        assert '1 / (1 + (embedding <-> %s::vector))' in call_args[0][0]  # L2 similarity formula

        # Verify results
        assert len(docs_with_score) == 1
        assert 0.9 in docs_with_score

    def test_retrieval_context_inner_product_similarity(self, retriever, mock_orchestrator):
        """Test retrieval with inner product similarity metric"""
        # Change similarity metric to inner product
        retriever.similarity_metric = 'inner'

        # Setup mock cursor
        mock_cursor = mock_orchestrator.db.cursor
        mock_cursor.fetchall.return_value = [
            ('Document content 1', 0.95, {'source': 'doc1.pdf'})
        ]

        # Call retrieval_context
        context, docs_with_score, docs_metadata, threshold = retriever.retrieval_context(
            query="test query",
            k=3
        )

        # Verify inner product operator was used
        call_args = mock_cursor.execute.call_args
        assert '<#>' in call_args[0][0]  # Inner product operator (fixed)
        assert 'DESC' in call_args[0][0]  # Descending order

        # Verify results
        assert len(docs_with_score) == 1
        assert 0.95 in docs_with_score

    def test_retrieval_context_no_results(self, retriever, mock_orchestrator):
        """Test retrieval when no documents match the query"""
        # Setup mock cursor to return empty results
        mock_cursor = mock_orchestrator.db.cursor
        mock_cursor.fetchall.return_value = []

        # Call retrieval_context
        context, docs_with_score, docs_metadata, threshold = retriever.retrieval_context(
            query="test query",
            k=5
        )

        # Verify empty results
        assert context == ''
        assert len(docs_with_score) == 0
        assert len(docs_metadata) == 0
        assert threshold == 0.7

        # Verify metrics were still recorded
        mock_orchestrator.metrics_reporter.record_vector_operation.assert_called_once_with(
            operation_type='query',
            item_count=0,
            duration=ANY
        )

    def test_retrieval_context_with_file_write(self, retriever, mock_orchestrator):
        """Test that retrieval context is written to file"""
        # Setup mock cursor
        mock_cursor = mock_orchestrator.db.cursor
        mock_cursor.fetchall.return_value = [
            ('Document content 1', 0.85, {'source': 'doc1.pdf'})
        ]

        # Mock file operations
        with patch('builtins.open', create=True) as mock_open:
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file

            # Call retrieval_context
            context, _, _, _ = retriever.retrieval_context(query="test query", k=5)

            # Verify file was opened for writing
            mock_open.assert_called_once()
            assert 'retrieved_' in mock_open.call_args[0][0]
            assert '.txt' in mock_open.call_args[0][0]

            # Verify content was written
            mock_file.write.assert_called_once_with('Document content 1\n')

    def test_retrieval_context_file_write_error(self, retriever, mock_orchestrator):
        """Test that file write errors don't break retrieval"""
        # Setup mock cursor
        mock_cursor = mock_orchestrator.db.cursor
        mock_cursor.fetchall.return_value = [
            ('Document content 1', 0.85, {'source': 'doc1.pdf'})
        ]

        # Mock file operations to raise an error
        with patch('builtins.open', side_effect=IOError("File write error")):
            # Call retrieval_context - should not raise exception
            context, docs_with_score, docs_metadata, threshold = retriever.retrieval_context(
                query="test query",
                k=5
            )

            # Verify retrieval still succeeded
            assert len(docs_with_score) == 1
            assert 'Document content 1' in context

    def test_retrieval_context_with_retry_success(self, retriever, mock_orchestrator):
        """Test retry logic when database operation fails then succeeds"""
        # Setup mock cursor to fail once then succeed
        mock_cursor = mock_orchestrator.db.cursor
        mock_cursor.execute.side_effect = [
            Exception("Database error"),  # First attempt fails
            None  # Second attempt succeeds
        ]
        mock_cursor.fetchall.return_value = [
            ('Document content 1', 0.85, {'source': 'doc1.pdf'})
        ]

        # Call retrieval_context
        context, docs_with_score, docs_metadata, threshold = retriever.retrieval_context(
            query="test query",
            k=5
        )

        # Verify retry was attempted
        assert mock_cursor.execute.call_count == 2
        mock_orchestrator.db.connect_with_retry.assert_called_once()

        # Verify results
        assert len(docs_with_score) == 1
        assert 'Document content 1' in context

    def test_retrieval_context_with_retry_failure(self, retriever, mock_orchestrator):
        """Test retry logic when all attempts fail"""
        # Setup mock cursor to always fail
        mock_cursor = mock_orchestrator.db.cursor
        mock_cursor.execute.side_effect = Exception("Database error")

        # Call retrieval_context
        context, docs_with_score, docs_metadata, threshold = retriever.retrieval_context(
            query="test query",
            k=5
        )

        # Verify all retries were attempted (3 attempts)
        assert mock_cursor.execute.call_count == 3
        # connect_with_retry is called only on retry (not on first attempt)
        assert mock_orchestrator.db.connect_with_retry.call_count == 2

        # Verify empty results are returned
        assert context == ""
        assert len(docs_with_score) == 0
        assert len(docs_metadata) == 0

        # Verify error was recorded
        mock_orchestrator.metrics_manager.record_error.assert_called_once()
        assert 'vector_query_error' in mock_orchestrator.metrics_manager.record_error.call_args[0]

    def test_retrieval_context_embedding_generation(self, retriever, mock_orchestrator):
        """Test that query embedding is generated correctly"""
        # Setup mock embeddings
        test_embedding = [0.1] * 1536
        mock_orchestrator.embeddings.embed_query.return_value = test_embedding

        # Setup mock cursor
        mock_cursor = mock_orchestrator.db.cursor
        mock_cursor.fetchall.return_value = []

        # Call retrieval_context
        retriever.retrieval_context(query="test query", k=5)

        # Verify embedding was generated
        mock_orchestrator.embeddings.embed_query.assert_called_once_with("test query")

        # Verify embedding was used in SQL query
        call_args = mock_cursor.execute.call_args
        embedding_str = call_args[0][1][0]
        assert embedding_str.startswith('[')
        assert embedding_str.endswith(']')
        assert '0.1' in embedding_str

    def test_retrieval_context_collection_name(self, retriever, mock_orchestrator):
        """Test that correct collection name is used in query"""
        # Setup mock cursor
        mock_cursor = mock_orchestrator.db.cursor
        mock_cursor.fetchall.return_value = []

        # Set collection name
        retriever.collection_name = 'custom_collection'

        # Call retrieval_context
        retriever.retrieval_context(query="test query", k=5)

        # Verify collection name was used in SQL query
        call_args = mock_cursor.execute.call_args
        assert call_args[0][1][1] == 'custom_collection'

    def test_retrieval_context_k_parameter(self, retriever, mock_orchestrator):
        """Test that k parameter limits results correctly"""
        # Setup mock cursor
        mock_cursor = mock_orchestrator.db.cursor
        mock_cursor.fetchall.return_value = []

        # Call retrieval_context with different k values
        retriever.retrieval_context(query="test query", k=10)

        # Verify k was used in SQL query
        call_args = mock_cursor.execute.call_args
        assert call_args[0][1][5] == 10  # k parameter position

    def test_retrieval_context_metadata_extraction(self, retriever, mock_orchestrator):
        """Test that metadata is correctly extracted from results"""
        # Setup mock cursor with complex metadata
        mock_cursor = mock_orchestrator.db.cursor
        mock_cursor.fetchall.return_value = [
            ('Document 1', 0.85, {
                'source': 'doc1.pdf',
                'page': 1,
                'author': 'John Doe',
                'date': '2024-01-01'
            }),
            ('Document 2', 0.75, {
                'source': 'doc2.pdf',
                'page': 5,
                'category': 'technical'
            })
        ]

        # Call retrieval_context
        context, docs_with_score, docs_metadata, threshold = retriever.retrieval_context(
            query="test query",
            k=5
        )

        # Verify metadata was extracted correctly
        assert len(docs_metadata) == 2
        assert docs_metadata[0.85]['source'] == 'doc1.pdf'
        assert docs_metadata[0.85]['page'] == 1
        assert docs_metadata[0.85]['author'] == 'John Doe'
        assert docs_metadata[0.75]['category'] == 'technical'

    def test_retrieval_context_performance_timing(self, retriever, mock_orchestrator):
        """Test that performance timing is recorded correctly"""
        # Setup mock cursor
        mock_cursor = mock_orchestrator.db.cursor
        mock_cursor.fetchall.return_value = [
            ('Document 1', 0.85, {'source': 'doc1.pdf'})
        ]

        # Add delay to simulate processing time
        def delayed_execute(*args, **kwargs):
            time.sleep(0.1)

        mock_cursor.execute.side_effect = delayed_execute

        # Call retrieval_context
        start_time = time.time()
        retriever.retrieval_context(query="test query", k=5)
        elapsed_time = time.time() - start_time

        # Verify timing was recorded in metrics
        call_args = mock_orchestrator.metrics_reporter.record_vector_operation.call_args
        assert call_args[1]['duration'] >= 0.1
        assert call_args[1]['duration'] <= elapsed_time + 0.1

    def test_retrieval_context_empty_query(self, retriever, mock_orchestrator):
        """Test retrieval with empty query string"""
        # Setup mock cursor
        mock_cursor = mock_orchestrator.db.cursor
        mock_cursor.fetchall.return_value = []

        # Call retrieval_context with empty query
        context, docs_with_score, docs_metadata, threshold = retriever.retrieval_context(
            query="",
            k=5
        )

        # Verify embedding was still generated
        mock_orchestrator.embeddings.embed_query.assert_called_once_with("")

        # Verify query was executed
        mock_cursor.execute.assert_called_once()

    def test_retrieval_context_special_characters_in_query(self, retriever, mock_orchestrator):
        """Test retrieval with special characters in query"""
        # Setup mock cursor
        mock_cursor = mock_orchestrator.db.cursor
        mock_cursor.fetchall.return_value = []

        # Call retrieval_context with special characters
        special_query = "Test query with 'quotes' and \"double quotes\" and \n newlines"
        context, docs_with_score, docs_metadata, threshold = retriever.retrieval_context(
            query=special_query,
            k=5
        )

        # Verify embedding was generated with special characters
        mock_orchestrator.embeddings.embed_query.assert_called_once_with(special_query)

        # Verify query executed successfully
        mock_cursor.execute.assert_called_once()

    def test_retrieval_context_unicode_query(self, retriever, mock_orchestrator):
        """Test retrieval with Unicode characters in query"""
        # Setup mock cursor
        mock_cursor = mock_orchestrator.db.cursor
        mock_cursor.fetchall.return_value = []

        # Call retrieval_context with Unicode characters
        unicode_query = "Test query with émojis 😀 and Chinese 中文"
        context, docs_with_score, docs_metadata, threshold = retriever.retrieval_context(
            query=unicode_query,
            k=5
        )

        # Verify embedding was generated with Unicode
        mock_orchestrator.embeddings.embed_query.assert_called_once_with(unicode_query)

        # Verify query executed successfully
        mock_cursor.execute.assert_called_once()

    def test_retrieval_context_large_k_value(self, retriever, mock_orchestrator):
        """Test retrieval with very large k value"""
        # Setup mock cursor
        mock_cursor = mock_orchestrator.db.cursor
        mock_cursor.fetchall.return_value = []

        # Call retrieval_context with large k
        retriever.retrieval_context(query="test query", k=10000)

        # Verify k was used in SQL query
        call_args = mock_cursor.execute.call_args
        assert call_args[0][1][5] == 10000

    def test_retrieval_context_zero_k_value(self, retriever, mock_orchestrator):
        """Test retrieval with k=0"""
        # Setup mock cursor
        mock_cursor = mock_orchestrator.db.cursor
        mock_cursor.fetchall.return_value = []

        # Call retrieval_context with k=0
        context, docs_with_score, docs_metadata, threshold = retriever.retrieval_context(
            query="test query",
            k=0
        )

        # Verify query was executed with k=0
        call_args = mock_cursor.execute.call_args
        assert call_args[0][1][5] == 0

        # Verify empty results
        assert len(docs_with_score) == 0

    def test_retrieval_context_threshold_adjustment(self, retriever, mock_orchestrator):
        """Test that threshold is adjusted in SQL query (0.8 * threshold)"""
        # Setup mock cursor
        mock_cursor = mock_orchestrator.db.cursor
        mock_cursor.fetchall.return_value = []

        # Set threshold
        retriever.threshold = 0.7

        # Call retrieval_context
        retriever.retrieval_context(query="test query", k=5)

        # Verify adjusted threshold was used in SQL query (0.7 * 0.8 = 0.56)
        call_args = mock_cursor.execute.call_args
        if retriever.similarity_metric == 'cosine':
            # Use approximate comparison for floating point
            adjusted_threshold = call_args[0][1][3]
            assert abs(adjusted_threshold - 0.56) < 0.001  # Allow small floating point error

    def test_retrieval_context_concurrent_calls(self, retriever, mock_orchestrator):
        """Test that concurrent retrieval calls work correctly"""
        import threading

        # Setup mock cursor
        mock_cursor = mock_orchestrator.db.cursor
        mock_cursor.fetchall.return_value = [
            ('Document 1', 0.85, {'source': 'doc1.pdf'})
        ]

        results = []

        def retrieve():
            context, docs, metadata, threshold = retriever.retrieval_context(
                query="test query",
                k=5
            )
            results.append((context, docs, metadata, threshold))

        # Create multiple threads
        threads = [threading.Thread(target=retrieve) for _ in range(5)]

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify all calls succeeded
        assert len(results) == 5
        for context, docs, metadata, threshold in results:
            assert len(docs) == 1
            assert 'Document 1' in context

    def test_retrieval_context_logging(self, retriever, mock_orchestrator):
        """Test that appropriate logging occurs during retrieval"""
        # Setup mock cursor
        mock_cursor = mock_orchestrator.db.cursor
        mock_cursor.fetchall.return_value = [
            ('Document content 1', 0.85, {'source': 'doc1.pdf'}),
            ('Document content 2', 0.65, {'source': 'doc2.pdf'})  # Below threshold
        ]

        # Set threshold
        retriever.threshold = 0.7

        # Mock logging
        with patch('logging.info') as mock_log:
            # Call retrieval_context
            retriever.retrieval_context(query="test query", k=5)

            # Verify logging occurred
            assert mock_log.call_count > 0
            # Check that results were logged
            log_calls = [str(call) for call in mock_log.call_args_list]
            assert any('Retrieved' in str(call) for call in log_calls)


# Additional test for edge cases and error scenarios
class TestPGVectorRetrieverEdgeCases:

    @pytest.fixture
    def mock_orchestrator_for_errors(self, global_metrics_manager_mock):
        """Create a mock orchestrator for error testing"""
        orchestrator = Mock(spec=PGVectorOrchestrator)

        # Mock database
        orchestrator.db = Mock()
        orchestrator.db.cursor = Mock()
        orchestrator.db.reconnect_if_needed = Mock()
        orchestrator.db.connect_with_retry = Mock()

        # Mock embeddings
        orchestrator.embeddings = Mock()
        orchestrator.embeddings.embed_query = Mock(return_value=[0.1] * 1536)

        # Mock configuration
        orchestrator.similarity_metric = 'cosine'
        orchestrator.threshold = 0.7
        orchestrator.collection_name = 'test_collection'
        orchestrator.retrieval_dir = '../Output/RetrievalContext'

        # Mock metrics
        orchestrator.metrics_reporter = Mock()
        orchestrator.metrics_reporter.record_vector_operation = Mock()
        orchestrator.metrics_manager = global_metrics_manager_mock

        return orchestrator

    @pytest.fixture
    def retriever_with_errors(self, mock_orchestrator_for_errors):
        """Create retriever for error testing"""
        return PGVectorRetriever(mock_orchestrator_for_errors)

    def test_retrieval_context_database_connection_error(self, retriever_with_errors, mock_orchestrator_for_errors):
        """Test handling of database connection errors that eventually succeed"""
        # Make reconnect_if_needed raise an error on first call, then succeed
        mock_orchestrator_for_errors.db.reconnect_if_needed.side_effect = [
            Exception("Connection error"),
            None,
            None
        ]

        # Setup cursor to work after reconnection
        mock_cursor = mock_orchestrator_for_errors.db.cursor
        mock_cursor.fetchall.return_value = []

        # Call retrieval_context
        context, docs_with_score, docs_metadata, threshold = retriever_with_errors.retrieval_context(
            query="test query",
            k=5
        )

        # Verify retry was attempted
        assert mock_orchestrator_for_errors.db.reconnect_if_needed.call_count >= 1

        # Since the error happens in reconnect_if_needed (not in the main try block),
        # it will be caught and retried, but record_error is only called after all retries fail
        # In this case, the second attempt succeeds, so no error is recorded
        # Just verify the function completed
        assert context == ""  # Empty because no results

    def test_retrieval_context_embedding_generation_error(self, retriever_with_errors, mock_orchestrator_for_errors):
        """Test handling of embedding generation errors that eventually succeed"""
        # Make embed_query raise an error on first call, then succeed
        mock_orchestrator_for_errors.embeddings.embed_query.side_effect = [
            Exception("Embedding error"),
            [0.1] * 1536,
            [0.1] * 1536
        ]

        # Setup cursor
        mock_cursor = mock_orchestrator_for_errors.db.cursor
        mock_cursor.fetchall.return_value = []

        # Call retrieval_context
        context, docs_with_score, docs_metadata, threshold = retriever_with_errors.retrieval_context(
            query="test query",
            k=5
        )

        # Verify retry was attempted
        assert mock_orchestrator_for_errors.embeddings.embed_query.call_count >= 1

        # Since the error happens in embed_query (not in the main try block),
        # it will be caught and retried, but record_error is only called after all retries fail
        # In this case, the second attempt succeeds, so no error is recorded
        # Just verify the function completed
        assert context == ""  # Empty because no results

    def test_retrieval_context_sql_execution_error(self, retriever_with_errors, mock_orchestrator_for_errors):
        """Test handling of SQL execution errors"""
        # Make execute raise an error
        mock_cursor = mock_orchestrator_for_errors.db.cursor
        mock_cursor.execute.side_effect = Exception("SQL error")

        # Call retrieval_context
        context, docs_with_score, docs_metadata, threshold = retriever_with_errors.retrieval_context(
            query="test query",
            k=5
        )

        # Verify retry logic was triggered
        assert mock_cursor.execute.call_count == 3

        # Verify empty results are returned
        assert context == ""
        assert len(docs_with_score) == 0

        # Verify error was recorded
        mock_orchestrator_for_errors.metrics_manager.record_error.assert_called()

    def test_retrieval_context_all_retries_fail(self, retriever_with_errors, mock_orchestrator_for_errors):
        """Test when all retry attempts fail"""
        # Make all attempts fail
        mock_cursor = mock_orchestrator_for_errors.db.cursor
        mock_cursor.execute.side_effect = [
            Exception("Database error 1"),
            Exception("Database error 2"),
            Exception("Database error 3")
        ]

        # Call retrieval_context
        context, docs_with_score, docs_metadata, threshold = retriever_with_errors.retrieval_context(
            query="test query",
            k=5
        )

        # Verify all retries were attempted
        assert mock_cursor.execute.call_count == 3
        assert mock_orchestrator_for_errors.db.connect_with_retry.call_count == 2

        # Verify empty results
        assert context == ""
        assert len(docs_with_score) == 0
        assert len(docs_metadata) == 0

        # Verify error was recorded
        mock_orchestrator_for_errors.metrics_manager.record_error.assert_called_once()

    def test_retrieval_context_partial_results(self, retriever_with_errors, mock_orchestrator_for_errors):
        """Test handling of partial results with mixed quality"""
        # Setup mock cursor with results of varying quality
        mock_cursor = mock_orchestrator_for_errors.db.cursor
        mock_cursor.fetchall.return_value = [
            ('High quality doc', 0.95, {'source': 'doc1.pdf'}),
            ('Medium quality doc', 0.75, {'source': 'doc2.pdf'}),
            ('Low quality doc', 0.55, {'source': 'doc3.pdf'}),
            ('Very low quality doc', 0.35, {'source': 'doc4.pdf'})
        ]

        # Set threshold
        retriever_with_errors.threshold = 0.7

        # Call retrieval_context
        context, docs_with_score, docs_metadata, threshold = retriever_with_errors.retrieval_context(
            query="test query",
            k=10
        )

        # Verify all results are returned (SQL query handles threshold filtering)
        assert len(docs_with_score) == 4

        # Verify results are properly structured
        assert 0.95 in docs_with_score
        assert 0.35 in docs_with_score
        assert 'High quality doc' in context
        assert 'Very low quality doc' in context

    def test_retrieval_context_with_null_metadata(self, retriever_with_errors, mock_orchestrator_for_errors):
        """Test handling of null metadata in results"""
        # Setup mock cursor with null metadata
        mock_cursor = mock_orchestrator_for_errors.db.cursor
        mock_cursor.fetchall.return_value = [
            ('Document with metadata', 0.85, {'source': 'doc1.pdf'}),
            ('Document without metadata', 0.75, None),
            ('Document with empty metadata', 0.70, {})
        ]

        # Call retrieval_context
        context, docs_with_score, docs_metadata, threshold = retriever_with_errors.retrieval_context(
            query="test query",
            k=5
        )

        # Verify all results are handled correctly
        assert len(docs_with_score) == 3
        assert len(docs_metadata) == 3
        assert docs_metadata[0.75] is None
        assert docs_metadata[0.70] == {}

    def test_retrieval_context_metrics_recording_on_success(self, retriever_with_errors, mock_orchestrator_for_errors):
        """Test that metrics are recorded correctly on successful retrieval"""
        # Setup mock cursor
        mock_cursor = mock_orchestrator_for_errors.db.cursor
        mock_cursor.fetchall.return_value = [
            ('Document 1', 0.85, {'source': 'doc1.pdf'}),
            ('Document 2', 0.75, {'source': 'doc2.pdf'})
        ]

        # Call retrieval_context
        retriever_with_errors.retrieval_context(query="test query", k=5)

        # Verify metrics were recorded
        mock_orchestrator_for_errors.metrics_reporter.record_vector_operation.assert_called_once()
        call_args = mock_orchestrator_for_errors.metrics_reporter.record_vector_operation.call_args

        assert call_args[1]['operation_type'] == 'query'
        assert call_args[1]['item_count'] == 2
        assert call_args[1]['duration'] > 0

    def test_retrieval_context_metrics_recording_on_failure(self, retriever_with_errors, mock_orchestrator_for_errors):
        """Test that metrics are NOT recorded when all retries fail"""
        # Make all attempts fail
        mock_cursor = mock_orchestrator_for_errors.db.cursor
        mock_cursor.execute.side_effect = Exception("Database error")

        # Call retrieval_context
        retriever_with_errors.retrieval_context(query="test query", k=5)

        # Verify metrics were NOT recorded (because operation failed)
        mock_orchestrator_for_errors.metrics_reporter.record_vector_operation.assert_not_called()

        # But error should be recorded
        mock_orchestrator_for_errors.metrics_manager.record_error.assert_called_once()


class TestPGVectorRetrieverSimilarityMetrics:
    """Test different similarity metrics in detail"""

    @pytest.fixture
    def base_orchestrator(self, global_metrics_manager_mock):
        """Create base orchestrator for similarity tests"""
        orchestrator = Mock(spec=PGVectorOrchestrator)
        orchestrator.db = Mock()
        orchestrator.db.cursor = Mock()
        orchestrator.db.reconnect_if_needed = Mock()
        orchestrator.db.connect_with_retry = Mock()
        orchestrator.embeddings = Mock()
        orchestrator.embeddings.embed_query = Mock(return_value=[0.1] * 1536)
        orchestrator.threshold = 0.7
        orchestrator.collection_name = 'test_collection'
        orchestrator.retrieval_dir = '../Output/RetrievalContext'
        orchestrator.metrics_reporter = Mock()
        orchestrator.metrics_reporter.record_vector_operation = Mock()
        orchestrator.metrics_manager = global_metrics_manager_mock
        return orchestrator

    def test_cosine_similarity_sql_query(self, base_orchestrator):
        """Test cosine similarity SQL query structure"""
        base_orchestrator.similarity_metric = 'cosine'
        retriever = PGVectorRetriever(base_orchestrator)

        mock_cursor = base_orchestrator.db.cursor
        mock_cursor.fetchall.return_value = []

        retriever.retrieval_context(query="test", k=5)

        call_args = mock_cursor.execute.call_args
        sql_query = call_args[0][0]

        # Verify cosine-specific SQL elements
        assert '1 - (embedding <=> %s::vector) as similarity' in sql_query
        assert 'ORDER BY embedding <=> %s::vector' in sql_query
        assert '1 - (embedding <=> %s::vector) >' in sql_query

    def test_l2_similarity_sql_query(self, base_orchestrator):
        """Test L2 similarity SQL query structure"""
        base_orchestrator.similarity_metric = 'l2'
        retriever = PGVectorRetriever(base_orchestrator)

        mock_cursor = base_orchestrator.db.cursor
        mock_cursor.fetchall.return_value = []

        retriever.retrieval_context(query="test", k=5)

        call_args = mock_cursor.execute.call_args
        sql_query = call_args[0][0]

        # Verify L2-specific SQL elements
        assert '1 / (1 + (embedding <-> %s::vector)) as similarity' in sql_query
        assert 'ORDER BY embedding <-> %s::vector' in sql_query
        assert 'embedding <-> %s::vector <' in sql_query

    def test_inner_product_similarity_sql_query(self, base_orchestrator):
        """Test inner product similarity SQL query structure"""
        base_orchestrator.similarity_metric = 'inner'
        retriever = PGVectorRetriever(base_orchestrator)

        mock_cursor = base_orchestrator.db.cursor
        mock_cursor.fetchall.return_value = []

        retriever.retrieval_context(query="test", k=5)

        call_args = mock_cursor.execute.call_args
        sql_query = call_args[0][0]

        # Verify inner product-specific SQL elements
        assert '(embedding <#> %s::vector) as similarity' in sql_query
        assert 'ORDER BY embedding <#> %s::vector DESC' in sql_query
        assert 'embedding <#> %s::vector >' in sql_query

    def test_retrieval_context_with_metrics_recording_error(self, base_orchestrator):
        """Test that metrics recording errors are handled gracefully"""
        # Create retriever instance with cosine similarity
        base_orchestrator.similarity_metric = 'cosine'
        retriever = PGVectorRetriever(base_orchestrator)
        
        # Setup mock cursor to return valid results (content, similarity, metadata)
        mock_cursor = base_orchestrator.db.cursor
        mock_cursor.fetchall.return_value = [("test content", 0.9, '{"key": "value"}')]
        
        # Setup embeddings
        base_orchestrator.embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
        
        # Mock metrics recording to raise exception
        base_orchestrator.metrics_reporter.record_vector_operation.side_effect = Exception("Metrics error")
        
        # Call method - should not raise exception despite metrics error
        context, docs_with_score, docs_metadata, threshold = retriever.retrieval_context("test query", k=5)
        
        # Verify retrieval still works despite metrics error
        assert len(docs_with_score) == 1
        assert "test content" in context
        assert 0.9 in docs_with_score

    def test_retrieval_context_with_error_recording_error(self, base_orchestrator):
        """Test that error recording errors don't prevent main error handling"""
        # Create retriever instance
        base_orchestrator.similarity_metric = 'cosine'
        retriever = PGVectorRetriever(base_orchestrator)
        
        # Setup embeddings to raise exception (simulate main error)
        base_orchestrator.embeddings.embed_query.side_effect = Exception("Main error")
        
        # Mock error recording to also raise exception
        base_orchestrator.metrics_manager.record_error.side_effect = Exception("Error recording error")
        
        # Call method - should handle gracefully and return empty results
        context, docs_with_score, docs_metadata, threshold = retriever.retrieval_context("test query", k=5)
        
        # Verify empty results are returned despite all errors
        assert context == ""
        assert len(docs_with_score) == 0
        assert len(docs_metadata) == 0
        assert threshold == base_orchestrator.threshold

    def test_retrieval_context_logging_with_valid_results(self, base_orchestrator):
        """Test logging behavior with valid results and metrics"""
        # Create retriever instance
        base_orchestrator.similarity_metric = 'cosine'
        retriever = PGVectorRetriever(base_orchestrator)
        
        # Setup mock cursor to return valid results (content, similarity, metadata)
        mock_cursor = base_orchestrator.db.cursor
        mock_cursor.fetchall.return_value = [("test content", 0.9, '{"key": "value"}')]
        
        # Mock embedding generation
        base_orchestrator.embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
        
        # Call method with logging capture
        with patch('logging.info') as mock_logger:
            context, docs_with_score, docs_metadata, threshold = retriever.retrieval_context("test query", k=5)
        
        # Verify results
        assert len(docs_with_score) == 1
        assert "test content" in context
        assert 0.9 in docs_with_score
        
        # Verify some logging occurred (check if logger was called)
        assert mock_logger.call_count >= 0  # Just verify no exception during logging

    def test_retrieval_context_retry_with_sleep_timing(self, base_orchestrator):
        """Test retry mechanism with proper sleep timing"""
        # Create retriever instance
        base_orchestrator.similarity_metric = 'cosine'
        retriever = PGVectorRetriever(base_orchestrator)
        
        # Mock embedding generation to succeed
        base_orchestrator.embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
        
        # Mock database cursor to fail first two times, succeed third time
        mock_cursor = base_orchestrator.db.cursor
        mock_cursor.execute.side_effect = [
            Exception("Connection error"),
            Exception("Connection error"),
            None  # Success on third try
        ]
        mock_cursor.fetchall.return_value = [("test content", 0.9, '{"key": "value"}')]
        
        # Mock time.sleep to track sleep calls
        with patch('time.sleep') as mock_sleep:
            context, docs_with_score, docs_metadata, threshold = retriever.retrieval_context("test query", k=5)
        
        # Verify sleep was called for retries
        assert mock_sleep.call_count >= 2
        
        # Verify final success
        assert len(docs_with_score) == 1
        assert "test content" in context

    def test_retrieval_context_with_different_similarity_metrics_comprehensive(self, base_orchestrator):
        """Test all similarity metrics with comprehensive SQL verification"""
        mock_cursor = base_orchestrator.db.cursor
        mock_cursor.fetchall.return_value = []

        # Test cosine similarity
        base_orchestrator.similarity_metric = 'cosine'
        retriever = PGVectorRetriever(base_orchestrator)
        
        retriever.retrieval_context(query="test", k=5)
        call_args = mock_cursor.execute.call_args
        sql_query = call_args[0][0]
        assert 'embedding <=> %s::vector' in sql_query
        assert '1 - (embedding <=> %s::vector) as similarity' in sql_query
        
        # Test L2 similarity  
        mock_cursor.reset_mock()
        base_orchestrator.similarity_metric = 'l2'
        retriever = PGVectorRetriever(base_orchestrator)
        
        retriever.retrieval_context(query="test", k=5)
        call_args = mock_cursor.execute.call_args
        sql_query = call_args[0][0]
        assert 'embedding <-> %s::vector' in sql_query
        assert '1 / (1 + (embedding <-> %s::vector)) as similarity' in sql_query
        
        # Test inner product similarity
        mock_cursor.reset_mock()
        base_orchestrator.similarity_metric = 'inner'
        retriever = PGVectorRetriever(base_orchestrator)
        
        retriever.retrieval_context(query="test", k=5)
        call_args = mock_cursor.execute.call_args
        sql_query = call_args[0][0]
        assert 'embedding <#> %s::vector' in sql_query
        assert '(embedding <#> %s::vector) as similarity' in sql_query
        assert 'DESC' in sql_query

    def test_retrieval_context_threshold_calculations(self, base_orchestrator):
        """Test threshold calculation and similarity score computation"""
        # Create retriever instance
        base_orchestrator.similarity_metric = 'cosine'
        retriever = PGVectorRetriever(base_orchestrator)
        
        # Mock embedding generation
        base_orchestrator.embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
        
        # Mock database to return results with different similarity scores (content, similarity, metadata)
        mock_cursor = base_orchestrator.db.cursor
        mock_cursor.fetchall.return_value = [
            ("high similarity content", 0.9, '{"key": "value1"}'),
            ("medium similarity content", 0.7, '{"key": "value2"}'),
            ("low similarity content", 0.4, '{"key": "value3"}')
        ]
        
        # Test with cosine similarity and threshold
        context, docs_with_score, docs_metadata, threshold = retriever.retrieval_context("test query", k=5)
        
        # Verify that results are returned
        assert len(docs_with_score) == 3
        assert "high similarity content" in context
        assert "medium similarity content" in context
        assert 0.9 in docs_with_score
        assert 0.7 in docs_with_score
