import pytest
from unittest.mock import Mock, patch, MagicMock, ANY
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import tempfile
import shutil

from src.context_handler.context_storage_handler.pgvector_connector import PGVectorConnector


class TestPGVectorConnector:
    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock orchestrator with all required attributes"""
        mock = Mock()
        mock.collection_name = "test_collection"
        mock.config_file_path = "../Config/Config.properties"
        mock.metrics_manager = Mock()
        mock.embeddings = Mock()
        mock.similarity_metric = "cosine"
        mock.threshold = 0.7
        mock.model_name = "test_model"
        mock.local_storage_path = "../Data/LocalEmbeddings"
        mock.retrieval_dir = "../Output/RetrievalContext"
        mock.db = Mock()
        mock.reconnect_if_needed = Mock()
        mock.diagnose_database = Mock(return_value=True)
        mock.close = Mock()
        mock.get_metrics = Mock(return_value={"test": "metrics"})
        mock.save_metrics = Mock(return_value="metrics_file.json")
        return mock

    @pytest.fixture
    def mock_retriever(self):
        """Create a mock retriever"""
        mock = Mock()
        mock.retrieval_context = Mock(return_value=("context", {"0.8": "doc1"}, {"0.8": {"source": "test.pdf"}}, 0.7))
        return mock

    @pytest.fixture
    def mock_store(self):
        """Create a mock store"""
        mock = Mock()
        mock.vector_store = Mock(return_value=True)
        mock.vector_store_documents = Mock(return_value=True)
        return mock

    @pytest.fixture
    def mock_searcher(self):
        """Create a mock searcher"""
        mock = Mock()
        mock.search_all_collections = Mock(
            return_value=("combined_context", {"0.8": "doc1"}, {"0.8": {"source": "test.pdf"}}, 0.7))
        return mock

    @pytest.fixture
    def connector(self):
        """Create a PGVectorConnector with mocked components"""
        # Create mock components
        mock_orchestrator = Mock()
        mock_orchestrator.collection_name = "test_collection"
        mock_orchestrator.embeddings = Mock()
        mock_orchestrator.similarity_metric = "cosine"
        mock_orchestrator.threshold = 0.7
        mock_orchestrator.model_name = "test_model"
        mock_orchestrator.local_storage_path = "../Data/LocalEmbeddings"
        mock_orchestrator.get_metrics.return_value = {"test": "metrics"}
        mock_orchestrator.save_metrics.return_value = "metrics_file.json"
        mock_orchestrator.diagnose_database.return_value = True

        mock_retriever = Mock()
        mock_retriever.retrieval_context.return_value = ("context", {"0.8": "doc1"}, {"0.8": {"source": "test.pdf"}},
                                                         0.7)

        mock_store = Mock()
        mock_store.vector_store.return_value = True
        mock_store.vector_store_documents.return_value = True

        mock_searcher = Mock()
        mock_searcher.search_all_collections.return_value = ("combined_context", {"0.8": "doc1"},
                                                             {"0.8": {"source": "test.pdf"}}, 0.7)

        # Patch the component classes
        with patch('src.context_handler.context_storage_handler.pgvector_orchestrator.PGVectorOrchestrator',
                   return_value=mock_orchestrator):
            with patch('src.context_handler.context_storage_handler.pgvector_retriever.PGVectorRetriever',
                       return_value=mock_retriever):
                with patch('src.context_handler.context_storage_handler.pgvector_store.PGVectorStore',
                           return_value=mock_store):
                    with patch('src.context_handler.context_storage_handler.pgvector_searcher.PGVectorSearcher',
                               return_value=mock_searcher):
                        # Create the connector
                        connector = PGVectorConnector(collection_name="test_collection")

                        # Set the mocked components
                        connector.orch = mock_orchestrator
                        connector.retriever = mock_retriever
                        connector.store = mock_store
                        connector.searcher = mock_searcher

                        yield connector

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_initialization(self):
        """Test PGVectorConnector initialization"""
        metrics_manager = Mock()

        with patch(
                'src.context_handler.context_storage_handler.pgvector_orchestrator.PGVectorOrchestrator') as mock_orch_class:
            mock_orch = Mock()
            mock_orch.embeddings = Mock()
            mock_orch.similarity_metric = "cosine"
            mock_orch.threshold = 0.7
            mock_orch.model_name = "test_model"
            mock_orch.local_storage_path = "../Data/LocalEmbeddings"
            mock_orch_class.return_value = mock_orch

            with patch(
                    'src.context_handler.context_storage_handler.pgvector_retriever.PGVectorRetriever') as mock_retriever_class:
                with patch(
                        'src.context_handler.context_storage_handler.pgvector_store.PGVectorStore') as mock_store_class:
                    with patch(
                            'src.context_handler.context_storage_handler.pgvector_searcher.PGVectorSearcher') as mock_searcher_class:
                        # Create the connector
                        connector = PGVectorConnector(
                            collection_name="test_collection",
                            config_file_path="../Config/test_config.properties",
                            metrics_manager=metrics_manager
                        )

                        # Check that orchestrator was initialized correctly
                        mock_orch_class.assert_called_once_with(
                            collection_name="test_collection",
                            config_file_path="../Config/test_config.properties",
                            metrics_manager=metrics_manager
                        )

                        # Check that components were initialized with the orchestrator
                        mock_retriever_class.assert_called_once()
                        mock_store_class.assert_called_once()
                        mock_searcher_class.assert_called_once()

                        # Check that attributes were set correctly
                        assert connector.collection_name == "test_collection"
                        assert connector.config_file_path == "../Config/test_config.properties"
                        assert connector.metrics_manager == metrics_manager
                        assert connector.embeddings == mock_orch.embeddings
                        assert connector.similarity_metric == mock_orch.similarity_metric
                        assert connector.threshold == mock_orch.threshold
                        assert connector.model_name == mock_orch.model_name
                        assert connector.local_storage_path == mock_orch.local_storage_path

    def test_initialization_with_defaults(self):
        """Test PGVectorConnector initialization with default values"""
        with patch(
                'src.context_handler.context_storage_handler.pgvector_orchestrator.PGVectorOrchestrator') as mock_orch_class:
            mock_orch = Mock()
            mock_orch.embeddings = Mock()
            mock_orch.similarity_metric = "cosine"
            mock_orch.threshold = 0.7
            mock_orch.model_name = "test_model"
            mock_orch.local_storage_path = "../Data/LocalEmbeddings"
            mock_orch_class.return_value = mock_orch

            with patch('src.context_handler.context_storage_handler.pgvector_retriever.PGVectorRetriever'):
                with patch('src.context_handler.context_storage_handler.pgvector_store.PGVectorStore'):
                    with patch('src.context_handler.context_storage_handler.pgvector_searcher.PGVectorSearcher'):
                        # Create the connector with default values
                        connector = PGVectorConnector()

                        # Check that orchestrator was initialized with defaults
                        mock_orch_class.assert_called_once_with(
                            collection_name="default",
                            config_file_path="../Config/Config.properties",
                            metrics_manager=ANY
                        )

    def test_diagnose_database(self, connector):
        """Test diagnose_database method"""
        result = connector.diagnose_database()

        # Verify the method delegates to orchestrator
        connector.orch.diagnose_database.assert_called_once()
        assert result == True  # Based on mock return value

    def test_reconnect_if_needed(self, connector):
        """Test reconnect_if_needed method"""
        connector.reconnect_if_needed()

        # Verify the method delegates to orchestrator
        connector.orch.reconnect_if_needed.assert_called_once()

    def test_search_all_collections(self, connector):
        """Test search_all_collections method"""
        result = connector.search_all_collections("test query", k=5)

        # Verify the method delegates to searcher
        connector.searcher.search_all_collections.assert_called_once_with("test query", k=5)

        # Check return values
        context, docs, metadata, threshold = result
        assert context == "combined_context"
        assert docs == {"0.8": "doc1"}
        assert metadata == {"0.8": {"source": "test.pdf"}}
        assert threshold == 0.7

    def test_retrieval_context(self, connector):
        """Test retrieval_context method"""
        result = connector.retrieval_context("test query", k=5)

        # Verify the method delegates to retriever
        connector.retriever.retrieval_context.assert_called_once_with("test query", k=5)

        # Check return values
        context, docs, metadata, threshold = result
        assert context == "context"
        assert docs == {"0.8": "doc1"}
        assert metadata == {"0.8": {"source": "test.pdf"}}
        assert threshold == 0.7

    def test_vector_store(self, connector):
        """Test vector_store method"""
        result = connector.vector_store("test_file.csv")

        # Verify the method delegates to store
        connector.store.vector_store.assert_called_once_with("test_file.csv")
        assert result == True  # Based on mock return value

    def test_vector_store_documents(self, connector):
        """Test vector_store_documents method"""
        documents = [{"text": "test doc", "metadata": {"source": "test"}}]
        result = connector.vector_store_documents(documents)

        # Verify the method delegates to store
        connector.store.vector_store_documents.assert_called_once_with(documents)
        assert result == True  # Based on mock return value

    def test_get_metrics(self, connector):
        """Test get_metrics method"""
        result = connector.get_metrics()

        # Verify the method delegates to orchestrator
        connector.orch.get_metrics.assert_called_once()
        assert result == {"test": "metrics"}  # Based on mock return value

    def test_save_metrics(self, connector):
        """Test save_metrics method"""
        result = connector.save_metrics(output_dir="../Output/TestMetrics")

        # Verify the method delegates to orchestrator
        connector.orch.save_metrics.assert_called_once_with(output_dir="../Output/TestMetrics")
        assert result == "metrics_file.json"  # Based on mock return value

    def test_close(self, connector):
        """Test __del__ method (close)"""
        # Manually call __del__ since it's normally called by garbage collection
        connector.__del__()

        # Verify the method delegates to orchestrator
        connector.orch.close.assert_called_once()

    def test_close_with_exception(self, connector):
        """Test __del__ method with exception"""
        # Make close throw an exception
        connector.orch.close.side_effect = Exception("Test error")

        # Should not raise exception
        with patch('logging.error') as mock_log:
            connector.__del__()
            mock_log.assert_called_once()

    @patch('os.makedirs')
    def test_directory_creation(self, mock_makedirs):
        """Test directory creation during initialization"""
        with patch('src.context_handler.context_storage_handler.pgvector_orchestrator.PGVectorOrchestrator'):
            with patch('src.context_handler.context_storage_handler.pgvector_retriever.PGVectorRetriever'):
                with patch('src.context_handler.context_storage_handler.pgvector_store.PGVectorStore'):
                    with patch('src.context_handler.context_storage_handler.pgvector_searcher.PGVectorSearcher'):
                        # Create the connector
                        connector = PGVectorConnector()

                        # Check that directory creation was attempted
                        mock_makedirs.assert_called()

    def test_error_handling_in_search_all_collections(self, connector):
        """Test error handling in search_all_collections"""
        # Make searcher throw an exception
        connector.searcher.search_all_collections.side_effect = Exception("Test error")

        # Should handle the error and return empty results
        with patch('logging.error') as mock_log:
            result = connector.search_all_collections("test query")
            mock_log.assert_called_once()

            # Check default return values on error
            context, docs, metadata, threshold = result
            assert context == ""
            assert docs == {}
            assert metadata == {}
            assert threshold == connector.threshold

    def test_retrieval_context_with_empty_results(self, connector):
        """Test retrieval_context with empty results"""
        # Mock empty results
        connector.retriever.retrieval_context.return_value = ("", {}, {}, 0.7)

        result = connector.retrieval_context("test query", k=5)

        # Check empty return values
        context, docs, metadata, threshold = result
        assert context == ""
        assert docs == {}
        assert metadata == {}
        assert threshold == 0.7

    def test_vector_store_with_nonexistent_file(self, connector):
        """Test vector_store with nonexistent file"""
        # Make store return False (failure)
        connector.store.vector_store.return_value = False

        result = connector.vector_store("nonexistent_file.csv")

        # Should return False
        assert result == False

    def test_vector_store_documents_with_empty_list(self, connector):
        """Test vector_store_documents with empty list"""
        result = connector.vector_store_documents([])

        # Verify the method delegates to store with empty list
        connector.store.vector_store_documents.assert_called_once_with([])

    def test_search_all_collections_with_large_k(self, connector):
        """Test search_all_collections with large k value"""
        connector.search_all_collections("test query", k=100)

        # Verify the method delegates to searcher with large k
        connector.searcher.search_all_collections.assert_called_once_with("test query", k=100)

    def test_retrieval_context_with_special_characters(self, connector):
        """Test retrieval_context with special characters in query"""
        special_query = "test query with special chars: !@#\$%^&*()"
        connector.retrieval_context(special_query, k=5)

        # Verify the method delegates to retriever with special characters
        connector.retriever.retrieval_context.assert_called_once_with(special_query, k=5)

    @patch('logging.error')
    def test_error_in_vector_store(self, mock_log, connector):
        """Test error handling in vector_store"""
        # Make store throw an exception
        connector.store.vector_store.side_effect = Exception("Test error")

        result = connector.vector_store("test_file.csv")

        # Should handle the error and return False
        mock_log.assert_called_once()
        assert result == False

    @patch('logging.error')
    def test_error_in_vector_store_documents(self, mock_log, connector):
        """Test error handling in vector_store_documents"""
        # Make store throw an exception
        connector.store.vector_store_documents.side_effect = Exception("Test error")

        documents = [{"text": "test doc", "metadata": {"source": "test"}}]
        result = connector.vector_store_documents(documents)

        # Should handle the error and return False
        mock_log.assert_called_once()
        assert result == False

    @pytest.mark.integration
    def test_integration_search_and_store(self, connector):
        """Test integration between search and store operations"""
        # First store some documents
        documents = [{"text": "test doc", "metadata": {"source": "test"}}]
        store_result = connector.vector_store_documents(documents)

        # Then search for them
        search_result = connector.search_all_collections("test doc")

        # Verify both operations were called
        connector.store.vector_store_documents.assert_called_once_with(documents)
        connector.searcher.search_all_collections.assert_called_once_with("test doc", k=5)

        # Check results
        assert store_result == True
        assert search_result[0] == "combined_context"  # From mock

    @pytest.mark.integration
    def test_integration_with_metrics(self, connector):
        """Test integration with metrics collection"""
        # Perform an operation that should record metrics
        connector.retrieval_context("test query", k=5)

        # Get and save metrics
        metrics = connector.get_metrics()
        metrics_file = connector.save_metrics()

        # Verify metrics operations were called
        connector.orch.get_metrics.assert_called_once()
        connector.orch.save_metrics.assert_called_once()

        # Verify retrieval operation was called
        connector.retriever.retrieval_context.assert_called_once()

    def test_vector_store_with_real_file(self, connector, temp_dir):
        """Test vector_store with a real CSV file"""
        # Create a test CSV file
        csv_path = os.path.join(temp_dir, "test.csv")
        df = pd.DataFrame({"text": ["test document 1", "test document 2"]})
        df.to_csv(csv_path, index=False)

        # Call vector_store with the real file
        result = connector.vector_store(csv_path)

        # Verify the method was called with the file path
        connector.store.vector_store.assert_called_once_with(csv_path)
        assert result == True

    def test_error_in_retrieval_context(self, connector):
        """Test error handling in retrieval_context"""
        # Make retriever throw an exception
        connector.retriever.retrieval_context.side_effect = Exception("Test error")

        # Should handle the error and return empty results
        with patch('logging.error') as mock_log:
            result = connector.retrieval_context("test query", k=5)
            mock_log.assert_called_once()

            # Check default return values on error
            context, docs, metadata, threshold = result
            assert context == ""
            assert docs == {}
            assert metadata == {}
            assert threshold == connector.threshold

    def test_attribute_passthrough(self, connector):
        """Test that attributes are correctly passed through from orchestrator"""
        # Verify that connector attributes match orchestrator attributes
        assert connector.embeddings == connector.orch.embeddings
        assert connector.similarity_metric == connector.orch.similarity_metric
        assert connector.threshold == connector.orch.threshold
        assert connector.model_name == connector.orch.model_name
        assert connector.local_storage_path == connector.orch.local_storage_path

    def test_default_output_dir_for_save_metrics(self, connector):
        """Test save_metrics with default output directory"""
        connector.save_metrics()

        # Verify the method delegates to orchestrator with default path
        connector.orch.save_metrics.assert_called_once_with(output_dir="../Output/Metrics")

    def test_custom_output_dir_for_save_metrics(self, connector):
        """Test save_metrics with custom output directory"""
        custom_dir = "../Custom/Metrics/Path"
        connector.save_metrics(output_dir=custom_dir)

        # Verify the method delegates to orchestrator with custom path
        connector.orch.save_metrics.assert_called_once_with(output_dir=custom_dir)

    def test_search_all_collections_default_k(self, connector):
        """Test search_all_collections with default k value"""
        connector.search_all_collections("test query")

        # Verify the method delegates to searcher with default k=5
        connector.searcher.search_all_collections.assert_called_once_with("test query", k=5)

    def test_retrieval_context_with_unicode(self, connector):
        """Test retrieval_context with unicode characters"""
        unicode_query = "test query with unicode: 你好, こんにちは, привет"
        connector.retrieval_context(unicode_query, k=5)

        # Verify the method delegates to retriever with unicode characters
        connector.retriever.retrieval_context.assert_called_once_with(unicode_query, k=5)

    def test_vector_store_documents_with_complex_metadata(self, connector):
        """Test vector_store_documents with complex metadata"""
        documents = [
            {
                "text": "test doc",
                "metadata": {
                    "source": "test",
                    "date": "2023-01-01",
                    "author": "Test Author",
                    "tags": ["tag1", "tag2"],
                    "nested": {"key": "value"}
                }
            }
        ]
        result = connector.vector_store_documents(documents)

        # Verify the method delegates to store with complex metadata
        connector.store.vector_store_documents.assert_called_once_with(documents)
        assert result == True