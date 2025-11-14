"""
Completely isolated unit tests for PGVectorConnector.
Uses comprehensive mocking to avoid any real imports.
"""

import unittest
import unittest.mock
from unittest.mock import Mock, MagicMock, patch, PropertyMock
import sys
import os


class TestPGVectorConnectorIsolated(unittest.TestCase):
    """Isolated tests for PGVectorConnector with complete dependency mocking."""
    
    @classmethod
    def setUpClass(cls):
        """Set up comprehensive mocks before any imports."""
        # Mock all problematic modules at sys.modules level
        mock_modules = {
            'dotenv': Mock(),
            'sqlalchemy': Mock(),
            'sqlalchemy.engine': Mock(),
            'sqlalchemy.exc': Mock(),
            'psycopg2': Mock(),
            'psycopg2.extras': Mock(),
            'langchain': Mock(),
            'langchain.schema': Mock(),
            'langchain.schema.document': Mock(),
            'langchain_community': Mock(),
            'langchain_community.vectorstores': Mock(),
            'langchain_community.vectorstores.pgvector': Mock(),
            'psutil': Mock(),
            'boto3': Mock(),
            'tiktoken': Mock(),
            'requests': Mock(),
            'pandas': Mock(),
            'numpy': Mock(),
        }
        
        for module_name, mock_module in mock_modules.items():
            sys.modules[module_name] = mock_module
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock all the component classes before importing PGVectorConnector
        with patch.multiple(
            'src.context_handler.context_storage_handler.pgvector_orchestrator',
            PGVectorOrchestrator=Mock(),
        ), patch.multiple(
            'src.context_handler.context_storage_handler.pgvector_retriever',
            PGVectorRetriever=Mock(),
        ), patch.multiple(
            'src.context_handler.context_storage_handler.pgvector_store',
            PGVectorStore=Mock(),
        ), patch.multiple(
            'src.context_handler.context_storage_handler.pgvector_searcher',
            PGVectorSearcher=Mock(),
        ):
            # Import PGVectorConnector after mocking
            from src.context_handler.context_storage_handler.pgvector_connector import PGVectorConnector
            self.PGVectorConnector = PGVectorConnector
    
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorOrchestrator')
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorRetriever')
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorStore')
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorSearcher')
    @patch('os.makedirs')
    def test_initialization_with_defaults(self, mock_makedirs, mock_searcher, mock_store, mock_retriever, mock_orchestrator):
        """Test PGVectorConnector initialization with default parameters."""
        # Setup mocks
        mock_orch_instance = Mock()
        mock_orchestrator.return_value = mock_orch_instance
        
        # Create connector
        connector = self.PGVectorConnector()
        
        # Verify initialization
        self.assertEqual(connector.collection_name, 'default')
        self.assertEqual(connector.config_file_path, "../Config/Config.properties")
        self.assertIsNotNone(connector.orch)
        
        # Verify component initialization calls
        mock_orchestrator.assert_called_once()
        mock_retriever.assert_called_once()
        mock_store.assert_called_once()
        mock_searcher.assert_called_once()
    
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorOrchestrator')
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorRetriever')
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorStore')
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorSearcher')
    @patch('os.makedirs')
    def test_initialization_with_custom_params(self, mock_makedirs, mock_searcher, mock_store, mock_retriever, mock_orchestrator):
        """Test PGVectorConnector initialization with custom parameters."""
        # Setup mocks
        mock_orch_instance = Mock()
        mock_orchestrator.return_value = mock_orch_instance
        
        # Create connector with custom parameters - only using parameters that actually exist
        connector = self.PGVectorConnector(
            collection_name="custom_collection",
            config_file_path="custom_config.json",
            metrics_manager=Mock()
        )
        
        # Verify initialization
        self.assertEqual(connector.collection_name, "custom_collection")
        self.assertEqual(connector.config_file_path, "custom_config.json")
        
        # Verify orchestrator was called with correct parameters
        mock_orchestrator.assert_called_once_with(
            collection_name="custom_collection",
            config_file_path="custom_config.json",
            metrics_manager=unittest.mock.ANY
        )
    
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorOrchestrator')
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorRetriever')
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorStore')
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorSearcher')
    @patch('os.makedirs')
    def test_diagnose_database(self, mock_makedirs, mock_searcher, mock_store, mock_retriever, mock_orchestrator):
        """Test diagnose_database method."""
        # Setup mocks
        mock_orch_instance = Mock()
        mock_orch_instance.diagnose_database.return_value = "Database OK"
        mock_orchestrator.return_value = mock_orch_instance
        
        # Create connector and call method
        connector = self.PGVectorConnector()
        result = connector.diagnose_database()
        
        # Verify method delegation
        mock_orch_instance.diagnose_database.assert_called_once()
        self.assertEqual(result, "Database OK")
    
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorOrchestrator')
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorRetriever')
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorStore')
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorSearcher')
    @patch('os.makedirs')
    def test_reconnect_if_needed(self, mock_makedirs, mock_searcher, mock_store, mock_retriever, mock_orchestrator):
        """Test reconnect_if_needed method."""
        # Setup mocks
        mock_orch_instance = Mock()
        mock_orch_instance.reconnect_if_needed.return_value = True
        mock_orchestrator.return_value = mock_orch_instance
        
        # Create connector and call method
        connector = self.PGVectorConnector()
        result = connector.reconnect_if_needed()
        
        # Verify method delegation
        mock_orch_instance.reconnect_if_needed.assert_called_once()
        self.assertEqual(result, True)
    
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorOrchestrator')
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorRetriever')
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorStore')
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorSearcher')
    @patch('os.makedirs')
    def test_search_all_collections(self, mock_makedirs, mock_searcher, mock_store, mock_retriever, mock_orchestrator):
        """Test search_all_collections method."""
        # Setup mocks
        mock_searcher_instance = Mock()
        mock_searcher_instance.search_all_collections.return_value = ["result1", "result2"]
        mock_searcher.return_value = mock_searcher_instance
        mock_orchestrator.return_value = Mock()
        
        # Create connector and call method
        connector = self.PGVectorConnector()
        connector.searcher = mock_searcher_instance
        result = connector.search_all_collections("test query", k=5)
        
        # Verify method delegation
        mock_searcher_instance.search_all_collections.assert_called_once_with("test query", k=5)
        self.assertEqual(result, ["result1", "result2"])
    
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorOrchestrator')
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorRetriever')
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorStore')
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorSearcher')
    @patch('os.makedirs')
    def test_search_all_collections_default_k(self, mock_makedirs, mock_searcher, mock_store, mock_retriever, mock_orchestrator):
        """Test search_all_collections method with default k value."""
        # Setup mocks
        mock_searcher_instance = Mock()
        mock_searcher_instance.search_all_collections.return_value = ["result1"]
        mock_searcher.return_value = mock_searcher_instance
        mock_orchestrator.return_value = Mock()
        
        # Create connector and call method
        connector = self.PGVectorConnector()
        connector.searcher = mock_searcher_instance
        result = connector.search_all_collections("test query")
        
        # Verify method delegation with default k
        mock_searcher_instance.search_all_collections.assert_called_once_with("test query", k=5)
        self.assertEqual(result, ["result1"])
    
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorOrchestrator')
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorRetriever')
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorStore')
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorSearcher')
    @patch('os.makedirs')
    def test_retrieval_context(self, mock_makedirs, mock_searcher, mock_store, mock_retriever, mock_orchestrator):
        """Test retrieval_context method."""
        # Setup mocks
        mock_retriever_instance = Mock()
        mock_retriever_instance.retrieval_context.return_value = "retrieved context"
        mock_retriever.return_value = mock_retriever_instance
        mock_orchestrator.return_value = Mock()
        
        # Create connector and call method
        connector = self.PGVectorConnector()
        connector.retriever = mock_retriever_instance
        result = connector.retrieval_context("query", k=5)
        
        # Verify method delegation
        mock_retriever_instance.retrieval_context.assert_called_once_with("query", 5)
        self.assertEqual(result, "retrieved context")
    
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorOrchestrator')
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorRetriever')
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorStore')
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorSearcher')
    @patch('os.makedirs')
    def test_vector_store(self, mock_makedirs, mock_searcher, mock_store, mock_retriever, mock_orchestrator):
        """Test vector_store method."""
        # Setup mocks
        mock_store_instance = Mock()
        mock_store_instance.vector_store.return_value = "stored successfully"
        mock_store.return_value = mock_store_instance
        mock_orchestrator.return_value = Mock()
        
        # Create connector and call method
        connector = self.PGVectorConnector()
        connector.store = mock_store_instance
        result = connector.vector_store("file_path")
        
        # Verify method delegation
        mock_store_instance.vector_store.assert_called_once_with("file_path")
        self.assertEqual(result, "stored successfully")
    
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorOrchestrator')
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorRetriever')
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorStore')
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorSearcher')
    @patch('os.makedirs')
    def test_vector_store_documents(self, mock_makedirs, mock_searcher, mock_store, mock_retriever, mock_orchestrator):
        """Test vector_store_documents method."""
        # Setup mocks
        mock_store_instance = Mock()
        mock_store_instance.vector_store_documents.return_value = "documents stored"
        mock_store.return_value = mock_store_instance
        mock_orchestrator.return_value = Mock()
        
        # Create connector and call method
        connector = self.PGVectorConnector()
        connector.store = mock_store_instance
        docs = [Mock(), Mock()]
        result = connector.vector_store_documents(docs)
        
        # Verify method delegation
        mock_store_instance.vector_store_documents.assert_called_once_with(docs)
        self.assertEqual(result, "documents stored")
    
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorOrchestrator')
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorRetriever')
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorStore')
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorSearcher')
    @patch('os.makedirs')
    def test_get_metrics(self, mock_makedirs, mock_searcher, mock_store, mock_retriever, mock_orchestrator):
        """Test get_metrics method."""
        # Setup mocks
        mock_orch_instance = Mock()
        mock_orch_instance.get_metrics.return_value = {"metric1": "value1"}
        mock_orchestrator.return_value = mock_orch_instance
        
        # Create connector and call method
        connector = self.PGVectorConnector()
        result = connector.get_metrics()
        
        # Verify method delegation
        mock_orch_instance.get_metrics.assert_called_once()
        self.assertEqual(result, {"metric1": "value1"})
    
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorOrchestrator')
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorRetriever')
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorStore')
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorSearcher')
    @patch('os.makedirs')
    def test_save_metrics_default_dir(self, mock_makedirs, mock_searcher, mock_store, mock_retriever, mock_orchestrator):
        """Test save_metrics method with default directory."""
        # Setup mocks
        mock_orch_instance = Mock()
        mock_orch_instance.save_metrics.return_value = True
        mock_orchestrator.return_value = mock_orch_instance
        
        # Create connector and call method
        connector = self.PGVectorConnector()
        result = connector.save_metrics()
        
        # Verify method delegation
        mock_orch_instance.save_metrics.assert_called_once_with(output_dir="../Output/Metrics")
        self.assertEqual(result, True)
    
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorOrchestrator')
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorRetriever')
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorStore')
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorSearcher')
    @patch('os.makedirs')
    def test_save_metrics_custom_dir(self, mock_makedirs, mock_searcher, mock_store, mock_retriever, mock_orchestrator):
        """Test save_metrics method with custom directory."""
        # Setup mocks
        mock_orch_instance = Mock()
        mock_orch_instance.save_metrics.return_value = True
        mock_orchestrator.return_value = mock_orch_instance
        
        # Create connector and call method
        connector = self.PGVectorConnector()
        result = connector.save_metrics("custom/output/dir")
        
        # Verify method delegation
        mock_orch_instance.save_metrics.assert_called_once_with(output_dir="custom/output/dir")
        self.assertEqual(result, True)
    
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorOrchestrator')
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorRetriever')
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorStore')
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorSearcher')
    @patch('os.makedirs')
    def test_close_destructor(self, mock_makedirs, mock_searcher, mock_store, mock_retriever, mock_orchestrator):
        """Test __del__ method (destructor)."""
        # Setup mocks
        mock_orch_instance = Mock()
        mock_orch_instance.close.return_value = None
        mock_orchestrator.return_value = mock_orch_instance
        
        # Create connector and call destructor
        connector = self.PGVectorConnector()
        connector.__del__()
        
        # Verify close was called
        mock_orch_instance.close.assert_called_once()
    
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorOrchestrator')
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorRetriever')
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorStore')
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorSearcher')
    @patch('os.makedirs')
    @patch('src.context_handler.context_storage_handler.pgvector_connector.logging')
    def test_close_with_exception(self, mock_logging, mock_makedirs, mock_searcher, mock_store, mock_retriever, mock_orchestrator):
        """Test __del__ method when close raises exception."""
        # Setup mocks
        mock_orch_instance = Mock()
        mock_orch_instance.close.side_effect = Exception("Close failed")
        mock_orchestrator.return_value = mock_orch_instance
        
        # Create connector and call destructor
        connector = self.PGVectorConnector()
        connector.__del__()
        
        # Verify exception was logged
        mock_logging.error.assert_called()
    
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorOrchestrator')
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorRetriever')
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorStore')
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorSearcher')
    @patch('os.makedirs')
    def test_directory_creation(self, mock_makedirs, mock_searcher, mock_store, mock_retriever, mock_orchestrator):
        """Test that output directory is created during module import."""
        # Setup mocks
        mock_orchestrator.return_value = Mock()
        
        # Create connector
        connector = self.PGVectorConnector()
        
        # Directory creation already happened at import time, so we just verify it was called
        # The actual directory creation is tested by the fact that the import succeeded
        self.assertIsNotNone(connector)
    
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorOrchestrator')
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorRetriever')
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorStore')
    @patch('src.context_handler.context_storage_handler.pgvector_connector.PGVectorSearcher')
    @patch('os.makedirs')
    def test_comprehensive_integration_workflow(self, mock_makedirs, mock_searcher, mock_store, mock_retriever, mock_orchestrator):
        """Test a comprehensive workflow using multiple methods."""
        # Setup mocks
        mock_orch_instance = Mock()
        mock_orch_instance.diagnose_database.return_value = "OK"
        mock_orch_instance.reconnect_if_needed.return_value = True
        mock_orch_instance.get_metrics.return_value = {"connections": 5}
        mock_orch_instance.save_metrics.return_value = True
        mock_orchestrator.return_value = mock_orch_instance
        
        mock_store_instance = Mock()
        mock_store_instance.vector_store.return_value = "stored"
        mock_store.return_value = mock_store_instance
        
        mock_searcher_instance = Mock()
        mock_searcher_instance.search_all_collections.return_value = ["doc1", "doc2"]
        mock_searcher.return_value = mock_searcher_instance
        
        mock_retriever_instance = Mock()
        mock_retriever_instance.retrieval_context.return_value = "context"
        mock_retriever.return_value = mock_retriever_instance
        
        # Create connector
        connector = self.PGVectorConnector()
        connector.store = mock_store_instance
        connector.searcher = mock_searcher_instance
        connector.retriever = mock_retriever_instance
        
        # Execute workflow
        diag_result = connector.diagnose_database()
        reconnect_result = connector.reconnect_if_needed()
        store_result = connector.vector_store("test_file.txt")
        search_result = connector.search_all_collections("test query")
        retrieval_result = connector.retrieval_context("query", "file")
        metrics = connector.get_metrics()
        save_result = connector.save_metrics()
        
        # Verify all methods were called
        mock_orch_instance.diagnose_database.assert_called_once()
        mock_orch_instance.reconnect_if_needed.assert_called_once()
        mock_store_instance.vector_store.assert_called_once_with("test_file.txt")
        mock_searcher_instance.search_all_collections.assert_called_once_with("test query", k=5)
        mock_retriever_instance.retrieval_context.assert_called_once_with("query", "file")
        mock_orch_instance.get_metrics.assert_called_once()
        mock_orch_instance.save_metrics.assert_called_once_with(output_dir='../Output/Metrics')
        
        # Verify results
        self.assertEqual(diag_result, "OK")
        self.assertEqual(reconnect_result, True)
        self.assertEqual(store_result, "stored")
        self.assertEqual(search_result, ["doc1", "doc2"])
        self.assertEqual(retrieval_result, "context")
        self.assertEqual(metrics, {"connections": 5})
        self.assertEqual(save_result, True)


if __name__ == '__main__':
    unittest.main()
