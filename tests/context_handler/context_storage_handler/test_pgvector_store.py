import pytest
from unittest.mock import Mock, patch, MagicMock, call, ANY

from sqlalchemy.exc import OperationalError

from src.context_handler.context_storage_handler.pgvector_store import PGVectorStore


class TestPGVectorStore:
    @pytest.fixture
    def mock_orchestrator(self, metrics_manager_mock, mock_db_connector):
        """Create a mock orchestrator for testing"""
        mock_orch = Mock()
        mock_orch.db = mock_db_connector
        mock_orch.embeddings = Mock()
        mock_orch.metrics_reporter = Mock()
        mock_orch.metrics_manager = metrics_manager_mock
        mock_orch.collection_name = "test_collection"
        return mock_orch

    @pytest.fixture
    def vector_store(self, mock_orchestrator):
        """Create a PGVectorStore instance for testing"""
        return PGVectorStore(mock_orchestrator)

    def test_initialization(self, vector_store, mock_orchestrator):
        """Test proper initialization of PGVectorStore"""
        assert vector_store.orch == mock_orchestrator
        assert vector_store.db == mock_orchestrator.db
        assert vector_store.embeddings == mock_orchestrator.embeddings
        assert vector_store.metrics_reporter == mock_orchestrator.metrics_reporter
        assert vector_store.metrics_manager == mock_orchestrator.metrics_manager
        assert vector_store.collection_name == "test_collection"

    def test_vector_store_success(self, vector_store):
        """Test successful vector_store method with CSV file"""
        # Mock pandas read_csv
        mock_df = Mock()
        mock_df.columns = ['text', 'category', 'source']
        mock_row1 = {'text': 'Test document 1', 'category': 'test', 'source': 'file1'}
        mock_row2 = {'text': 'Test document 2', 'category': 'test', 'source': 'file2'}
        mock_df.iterrows.return_value = [
            (0, mock_row1),
            (1, mock_row2)
        ]

        with patch('pandas.read_csv', return_value=mock_df), \
                patch.object(vector_store, 'vector_store_documents', return_value=True) as mock_store_docs:
            result = vector_store.vector_store("test.csv")

            # Check that vector_store_documents was called with correct documents
            assert mock_store_docs.call_count == 1
            called_docs = mock_store_docs.call_args[0][0]
            assert len(called_docs) == 2
            assert called_docs[0]['text'] == 'Test document 1'
            assert called_docs[0]['metadata'] == {'category': 'test', 'source': 'file1'}
            assert called_docs[1]['text'] == 'Test document 2'
            assert called_docs[1]['metadata'] == {'category': 'test', 'source': 'file2'}

            # Check that metrics were recorded
            vector_store.metrics_reporter.record_vector_operation.assert_called_once()

            # Check result
            assert result is True

    def test_vector_store_missing_text_column(self, vector_store):
        """Test vector_store method with CSV missing 'text' column"""
        # Mock pandas read_csv to return DataFrame without 'text' column
        mock_df = Mock()
        mock_df.columns = ['content', 'category']  # No 'text' column

        with patch('pandas.read_csv', return_value=mock_df):
            result = vector_store.vector_store("test.csv")

            # Should fail and record error
            assert result is False
            vector_store.metrics_manager.record_error.assert_called_once_with(
                'vector_store_error', "CSV file must contain a 'text' column"
            )

    def test_vector_store_with_retry(self, vector_store):
        """Test retry logic in vector_store method"""
        # Mock pandas read_csv
        mock_df = Mock()
        mock_df.columns = ['text']
        mock_df.iterrows.return_value = [(0, {'text': 'Test document'})]

        # Create a database exception for testing
        db_error = Exception("Connection lost")

        # Mock db to fail once then succeed
        vector_store.db.reconnect_if_needed.side_effect = [
            db_error,  # First call fails
            None  # Second call succeeds
        ]

        with patch('pandas.read_csv', return_value=mock_df), \
                patch.object(vector_store, 'vector_store_documents', return_value=True) as mock_store_docs, \
                patch('time.sleep') as mock_sleep:
            result = vector_store.vector_store("test.csv")

            # Check that reconnect was called twice
            assert vector_store.db.reconnect_if_needed.call_count == 2
            # Check that sleep was called once for retry
            mock_sleep.assert_called_once()
            # Check final result
            assert result is True

    def test_vector_store_max_retries_exceeded(self, vector_store):
        """Test vector_store when max retries are exceeded"""
        # Mock pandas read_csv
        mock_df = Mock()
        mock_df.columns = ['text']
        mock_df.iterrows.return_value = [(0, {'text': 'Test document'})]

        # Create a database exception for testing
        db_error = Exception("Connection lost")

        # Mock db to always fail
        vector_store.db.reconnect_if_needed.side_effect = db_error

        with patch('pandas.read_csv', return_value=mock_df), \
                patch('time.sleep') as mock_sleep:
            result = vector_store.vector_store("test.csv")

            # Check that reconnect was called 3 times (max retries)
            assert vector_store.db.reconnect_if_needed.call_count == 3
            # Check that sleep was called twice for retries
            assert mock_sleep.call_count == 2
            # Check that error was recorded
            vector_store.metrics_manager.record_error.assert_called_once()
            # Check final result
            assert result is False

    def test_vector_store_documents_success(self, vector_store):
        """Test successful vector_store_documents method"""
        # Mock documents
        documents = [
            {'text': 'Test document 1', 'metadata': {'source': 'file1'}},
            {'text': 'Test document 2', 'metadata': {'source': 'file2'}}
        ]

        # Mock chunker
        mock_chunker = Mock()
        mock_chunker.chunk_document.side_effect = lambda doc: [doc]  # Return document unchanged

        # Mock embeddings
        vector_store.embeddings.embed_query.return_value = [0.1] * 1536

        # Mock cursor
        vector_store.db.cursor.fetchone.return_value = None  # No duplicates

        with patch('src.metrics.semantic_chunker.SemanticChunker', return_value=mock_chunker), \
                patch('psycopg2.extras.execute_values') as mock_execute_values, \
                patch.dict('os.environ', {'CHUNK_SIZE': '512', 'CHUNK_OVERLAP': '50'}):
            result = vector_store.vector_store_documents(documents)

            # Check that chunker was used
            assert mock_chunker.chunk_document.call_count == 2

            # Check that embeddings were generated
            assert vector_store.embeddings.embed_query.call_count == 2

            # Check that execute_values was called
            mock_execute_values.assert_called_once()

            # Check that metrics were recorded
            vector_store.metrics_reporter.record_vector_operation.assert_called_once_with(
                operation_type='store',
                item_count=2,
                duration=ANY
            )

            # Check result
            assert result is True

    def test_vector_store_missing_text_column(self, vector_store):
        """Test vector_store method with CSV missing 'text' column"""
        # Mock pandas read_csv to return DataFrame without 'text' column
        mock_df = Mock()
        mock_df.columns = ['content', 'category']  # No 'text' column

        with patch('pandas.read_csv', return_value=mock_df):
            result = vector_store.vector_store("test.csv")

            # Should fail and record error
            assert result is False
            vector_store.metrics_manager.record_error.assert_called_once_with(
                'vector_store_error', "CSV file must contain a 'text' column"
            )

    def test_vector_store_with_retry(self, vector_store):
        """Test retry logic in vector_store method"""
        # Mock pandas read_csv
        mock_df = Mock()
        mock_df.columns = ['text']
        mock_df.iterrows.return_value = [(0, {'text': 'Test document'})]

        # Mock db to fail once then succeed
        vector_store.db.reconnect_if_needed.side_effect = [
            OperationalError("Connection lost"),  # First call fails
            None  # Second call succeeds
        ]

        with patch('pandas.read_csv', return_value=mock_df), \
                patch.object(vector_store, 'vector_store_documents', return_value=True) as mock_store_docs, \
                patch('time.sleep') as mock_sleep:
            result = vector_store.vector_store("test.csv")

            # Check that reconnect was called twice
            assert vector_store.db.reconnect_if_needed.call_count == 2
            # Check that sleep was called once for retry
            mock_sleep.assert_called_once()
            # Check final result
            assert result is True

    def test_vector_store_max_retries_exceeded(self, vector_store):
        """Test vector_store when max retries are exceeded"""
        # Mock pandas read_csv
        mock_df = Mock()
        mock_df.columns = ['text']
        mock_df.iterrows.return_value = [(0, {'text': 'Test document'})]

        # Mock db to always fail
        vector_store.db.reconnect_if_needed.side_effect = OperationalError("Connection lost")

        with patch('pandas.read_csv', return_value=mock_df), \
                patch('time.sleep') as mock_sleep:
            result = vector_store.vector_store("test.csv")

            # Check that reconnect was called 3 times (max retries)
            assert vector_store.db.reconnect_if_needed.call_count == 3
            # Check that sleep was called twice for retries
            assert mock_sleep.call_count == 2
            # Check that error was recorded
            vector_store.metrics_manager.record_error.assert_called_once()
            # Check final result
            assert result is False

    def test_vector_store_documents_success(self, vector_store):
        """Test successful vector_store_documents method"""
        # Mock documents
        documents = [
            {'text': 'Test document 1', 'metadata': {'source': 'file1'}},
            {'text': 'Test document 2', 'metadata': {'source': 'file2'}}
        ]

        # Mock chunker
        mock_chunker = Mock()
        mock_chunker.chunk_document.side_effect = lambda doc: [doc]  # Return document unchanged

        # Mock embeddings
        vector_store.embeddings.embed_query.return_value = [0.1] * 1536

        # Mock cursor
        vector_store.db.cursor.fetchone.return_value = None  # No duplicates

        with patch('src.metrics.semantic_chunker.SemanticChunker', return_value=mock_chunker), \
                patch('psycopg2.extras.execute_values') as mock_execute_values, \
                patch.dict('os.environ', {'CHUNK_SIZE': '512', 'CHUNK_OVERLAP': '50'}):
            result = vector_store.vector_store_documents(documents)

            # Check that chunker was used
            assert mock_chunker.chunk_document.call_count == 2

            # Check that embeddings were generated
            assert vector_store.embeddings.embed_query.call_count == 2

            # Check that execute_values was called
            mock_execute_values.assert_called_once()

            # Check that metrics were recorded
            vector_store.metrics_reporter.record_vector_operation.assert_called_once_with(
                operation_type='store',
                item_count=2,
                duration=ANY
            )

            # Check result
            assert result is True

    def test_vector_store_documents_with_duplicates(self, vector_store):
        """Test vector_store_documents with duplicate detection"""
        # Mock documents
        documents = [
            {'text': 'Test document 1', 'metadata': {'source': 'file1'}},
            {'text': 'Test document 2', 'metadata': {'source': 'file2'}}
        ]

        # Mock chunker
        mock_chunker = Mock()
        mock_chunker.chunk_document.side_effect = lambda doc: [doc]  # Return document unchanged

        # Mock cursor to indicate first document is a duplicate
        vector_store.db.cursor.fetchone.side_effect = [
            (1,),  # First document is a duplicate
            None  # Second document is not a duplicate
        ]

        # Mock embeddings
        vector_store.embeddings.embed_query.return_value = [0.1] * 1536

        with patch('src.metrics.semantic_chunker.SemanticChunker', return_value=mock_chunker), \
                patch('psycopg2.extras.execute_values') as mock_execute_values, \
                patch.dict('os.environ', {'CHUNK_SIZE': '512', 'CHUNK_OVERLAP': '50'}), \
                patch('logging.info') as mock_log:
            result = vector_store.vector_store_documents(documents)

            # Check that duplicate was logged
            mock_log.assert_any_call(
                f"Document already exists in collection '{vector_store.collection_name}', skipping")

            # Check that execute_values was called with only one document
            mock_execute_values.assert_called_once()
            # Get the data passed to execute_values
            data = mock_execute_values.call_args[0][2]
            assert len(data) == 1
            assert data[0][1] == 'Test document 2'  # Only second document should be stored

            # Check result
            assert result is True

    def test_vector_store_documents_all_duplicates(self, vector_store):
        """Test vector_store_documents when all documents are duplicates"""
        # Mock documents
        documents = [
            {'text': 'Test document 1', 'metadata': {'source': 'file1'}},
            {'text': 'Test document 2', 'metadata': {'source': 'file2'}}
        ]

        # Mock chunker
        mock_chunker = Mock()
        mock_chunker.chunk_document.side_effect = lambda doc: [doc]  # Return document unchanged

        # Mock cursor to indicate all documents are duplicates
        vector_store.db.cursor.fetchone.return_value = (1,)  # All documents are duplicates

        with patch('src.metrics.semantic_chunker.SemanticChunker', return_value=mock_chunker), \
                patch('psycopg2.extras.execute_values') as mock_execute_values, \
                patch.dict('os.environ', {'CHUNK_SIZE': '512', 'CHUNK_OVERLAP': '50'}), \
                patch('logging.info') as mock_log:
            result = vector_store.vector_store_documents(documents)

            # Check that duplicates were logged
            assert mock_log.call_count >= 2

            # Check that execute_values was not called (no documents to store)
            mock_execute_values.assert_not_called()

            # Check result - should still be True as operation completed successfully
            assert result is True

    def test_vector_store_documents_with_retry(self, vector_store):
        """Test retry logic in vector_store_documents method"""
        # Mock documents
        documents = [{'text': 'Test document', 'metadata': {'source': 'file1'}}]

        # Mock chunker
        mock_chunker = Mock()
        mock_chunker.chunk_document.return_value = documents

        # Mock db to fail once then succeed
        vector_store.db.reconnect_if_needed.side_effect = [
            OperationalError("Connection lost"),  # First call fails
            None  # Second call succeeds
        ]

        # Mock cursor
        vector_store.db.cursor.fetchone.return_value = None  # No duplicates

        # Mock embeddings
        vector_store.embeddings.embed_query.return_value = [0.1] * 1536

        with patch('src.metrics.semantic_chunker.SemanticChunker', return_value=mock_chunker), \
                patch('psycopg2.extras.execute_values') as mock_execute_values, \
                patch('time.sleep') as mock_sleep, \
                patch.dict('os.environ', {'CHUNK_SIZE': '512', 'CHUNK_OVERLAP': '50'}):
            result = vector_store.vector_store_documents(documents)

            # Check that reconnect was called twice
            assert vector_store.db.reconnect_if_needed.call_count == 2
            # Check that sleep was called once for retry
            mock_sleep.assert_called_once()
            # Check that execute_values was called
            mock_execute_values.assert_called_once()
            # Check result
            assert result is True

    def test_vector_store_documents_max_retries_exceeded(self, vector_store):
        """Test vector_store_documents when max retries are exceeded"""
        # Mock documents
        documents = [{'text': 'Test document', 'metadata': {'source': 'file1'}}]

        # Mock chunker
        mock_chunker = Mock()
        mock_chunker.chunk_document.return_value = documents

        # Mock db to always fail
        vector_store.db.reconnect_if_needed.side_effect = OperationalError("Connection lost")

        with patch('src.metrics.semantic_chunker.SemanticChunker', return_value=mock_chunker), \
                patch('time.sleep') as mock_sleep, \
                patch.dict('os.environ', {'CHUNK_SIZE': '512', 'CHUNK_OVERLAP': '50'}):
            result = vector_store.vector_store_documents(documents)

            # Check that reconnect was called 3 times (max retries)
            assert vector_store.db.reconnect_if_needed.call_count == 3
            # Check that sleep was called twice for retries
            assert mock_sleep.call_count == 2
            # Check that error was recorded
            vector_store.metrics_manager.record_error.assert_called_once()
            # Check result
            assert result is False

    def test_vector_store_documents_with_large_batch(self, vector_store):
        """Test vector_store_documents with a large batch that gets processed in chunks"""
        # Create 150 documents (more than the batch size of 100)
        documents = [{'text': f'Test document {i}', 'metadata': {'source': f'file{i}'}} for i in range(150)]

        # Mock chunker to return documents unchanged
        mock_chunker = Mock()
        mock_chunker.chunk_document.side_effect = lambda doc: [doc]

        # Mock cursor to indicate no duplicates
        vector_store.db.cursor.fetchone.return_value = None

        # Mock embeddings
        vector_store.embeddings.embed_query.return_value = [0.1] * 1536

        with patch('src.metrics.semantic_chunker.SemanticChunker', return_value=mock_chunker), \
                patch('psycopg2.extras.execute_values') as mock_execute_values, \
                patch.dict('os.environ', {'CHUNK_SIZE': '512', 'CHUNK_OVERLAP': '50'}), \
                patch('logging.info') as mock_log:
            result = vector_store.vector_store_documents(documents)

            # Check that execute_values was called twice (once for each batch)
            assert mock_execute_values.call_count == 2

            # First batch should have 100 documents
            first_batch_data = mock_execute_values.call_args_list[0][0][2]
            assert len(first_batch_data) == 100

            # Second batch should have 50 documents
            second_batch_data = mock_execute_values.call_args_list[1][0][2]
            assert len(second_batch_data) == 50

            # Check that batch processing was logged
            mock_log.assert_any_call("Stored batch 1 with 100 documents")
            mock_log.assert_any_call("Stored batch 2 with 50 documents")

            # Check result
            assert result is True

            # Check that metrics were recorded with total count
            vector_store.metrics_reporter.record_vector_operation.assert_called_once_with(
                operation_type='store',
                item_count=150,
                duration=ANY
            )

    def test_vector_store_documents_with_chunking(self, vector_store):
        """Test that documents are properly chunked"""
        # Mock documents
        documents = [
            {'text': 'Test document 1', 'metadata': {'source': 'file1'}},
            {'text': 'Test document 2', 'metadata': {'source': 'file2'}}
        ]

        # Mock chunker to split each document into two chunks
        mock_chunker = Mock()
        mock_chunker.chunk_document.side_effect = lambda doc: [
            {'text': doc['text'] + ' (chunk 1)', 'metadata': doc['metadata']},
            {'text': doc['text'] + ' (chunk 2)', 'metadata': doc['metadata']}
        ]

        # Mock cursor to indicate no duplicates
        vector_store.db.cursor.fetchone.return_value = None

        # Mock embeddings
        vector_store.embeddings.embed_query.return_value = [0.1] * 1536

        with patch('src.metrics.semantic_chunker.SemanticChunker', return_value=mock_chunker), \
                patch('psycopg2.extras.execute_values') as mock_execute_values, \
                patch.dict('os.environ', {'CHUNK_SIZE': '512', 'CHUNK_OVERLAP': '50'}):
            result = vector_store.vector_store_documents(documents)

            # Check that chunker was called for each document
            assert mock_chunker.chunk_document.call_count == 2

            # Check that execute_values was called with 4 chunks
            mock_execute_values.assert_called_once()
            data = mock_execute_values.call_args[0][2]
            assert len(data) == 4

            # Check the content of the chunks
            texts = [item[1] for item in data]
            assert 'Test document 1 (chunk 1)' in texts
            assert 'Test document 1 (chunk 2)' in texts
            assert 'Test document 2 (chunk 1)' in texts
            assert 'Test document 2 (chunk 2)' in texts

            # Check result
            assert result is True

    def test_vector_store_documents_empty_input(self, vector_store):
        """Test vector_store_documents with empty input"""
        # Empty documents list
        documents = []

        with patch('src.metrics.semantic_chunker.SemanticChunker') as mock_chunker_class, \
                patch('psycopg2.extras.execute_values') as mock_execute_values:
            result = vector_store.vector_store_documents(documents)

            # Chunker should not be instantiated
            mock_chunker_class.assert_not_called()

            # execute_values should not be called
            mock_execute_values.assert_not_called()

            # Should still return True as operation completed successfully
            assert result is True

    def test_vector_store_documents_with_error_handling(self, vector_store):
        """Test error handling in vector_store_documents"""
        # Mock documents
        documents = [{'text': 'Test document', 'metadata': {'source': 'file1'}}]

        # Mock chunker to raise an exception
        mock_chunker = Mock()
        mock_chunker.chunk_document.side_effect = Exception("Chunking error")

        with patch('src.metrics.semantic_chunker.SemanticChunker', return_value=mock_chunker), \
                patch('time.sleep') as mock_sleep:
            result = vector_store.vector_store_documents(documents)

            # Check that error was recorded
            vector_store.metrics_manager.record_error.assert_called_once_with(
                'vector_store_error', ANY
            )

            # Check result
            assert result is False

    def test_vector_store_documents_with_custom_chunk_settings(self, vector_store):
        """Test vector_store_documents with custom chunk settings from environment"""
        # Mock documents
        documents = [{'text': 'Test document', 'metadata': {'source': 'file1'}}]

        # Mock chunker
        mock_chunker = Mock()
        mock_chunker.chunk_document.return_value = documents

        # Mock cursor
        vector_store.db.cursor.fetchone.return_value = None  # No duplicates

        # Mock embeddings
        vector_store.embeddings.embed_query.return_value = [0.1] * 1536

        with patch('src.metrics.semantic_chunker.SemanticChunker') as mock_chunker_class, \
                patch('psycopg2.extras.execute_values'), \
                patch.dict('os.environ', {'CHUNK_SIZE': '1024', 'CHUNK_OVERLAP': '100'}):
            mock_chunker_class.return_value = mock_chunker

            result = vector_store.vector_store_documents(documents)

            # Check that chunker was created with custom settings
            mock_chunker_class.assert_called_once_with(chunk_size=1024, chunk_overlap=100)

            # Check result
            assert result is True