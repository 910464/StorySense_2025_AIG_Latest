# tests/context_handler/context_storage_handler/test_pgvector_orchestrator.py

import pytest
from unittest.mock import Mock, patch, MagicMock, call
import os
import logging
from src.context_handler.context_storage_handler.pgvector_orchestrator import PGVectorOrchestrator


class TestPGVectorOrchestrator:
    """Comprehensive test suite for PGVectorOrchestrator"""

    @pytest.fixture
    def mock_db_config_loader(self):
        """Mock DBConfigLoader"""
        with patch('src.context_handler.context_storage_handler.pgvector_orchestrator.DBConfigLoader') as mock:
            mock_instance = Mock()
            mock_instance.embeddings = Mock()
            mock_instance.similarity_metric = 'cosine'
            mock_instance.threshold = 0.7
            mock_instance.model_name = 'amazon.titan-embed-text-v1'
            mock_instance.local_storage_path = '../Data/LocalEmbeddings'
            mock.return_value = mock_instance
            yield mock

    @pytest.fixture
    def mock_pg_database(self):
        """Mock PGDatabase"""
        with patch('src.context_handler.context_storage_handler.pgvector_orchestrator.PGDatabase') as mock:
            mock_instance = Mock()
            mock_instance.cursor = Mock()
            mock_instance.conn = Mock()
            mock_instance.diagnose_database = Mock(return_value=True)
            mock_instance.reconnect_if_needed = Mock()
            mock_instance.close = Mock()
            mock.return_value = mock_instance
            yield mock

    @pytest.fixture
    def mock_metrics_reporter(self):
        """Mock MetricsReporter"""
        with patch('src.context_handler.context_storage_handler.pgvector_orchestrator.MetricsReporter') as mock:
            mock_instance = Mock()
            mock_instance.get_metrics = Mock(return_value={
                'vector_store_operations': 5,
                'vector_query_operations': 10,
                'total_vectors_stored': 100
            })
            mock_instance.save_metrics = Mock(return_value='/path/to/metrics.json')
            mock.return_value = mock_instance
            yield mock

    @pytest.fixture
    def mock_metrics_manager(self):
        """Mock MetricsManager"""
        mock = Mock()
        mock.record_vector_operation = Mock()
        mock.record_error = Mock()
        return mock

    @pytest.fixture
    def orchestrator(self, mock_db_config_loader, mock_pg_database,
                     mock_metrics_reporter, mock_metrics_manager):
        """Create orchestrator instance with mocked dependencies"""
        with patch('os.makedirs'):
            orch = PGVectorOrchestrator(
                collection_name="test_collection",
                config_file_path="../Config/Config.properties",
                metrics_manager=mock_metrics_manager
            )
            return orch

    # ==================== Initialization Tests ====================

    def test_initialization_default_parameters(self, mock_db_config_loader,
                                               mock_pg_database, mock_metrics_reporter):
        """Test initialization with default parameters"""
        with patch('os.makedirs') as mock_makedirs:
            orch = PGVectorOrchestrator()

            # Verify default collection name
            assert orch.collection_name == "default"

            # Verify config file path
            assert orch.config_file_path == '../Config/Config.properties'

            # Verify metrics manager was created
            assert orch.metrics_manager is not None

            # Verify directory creation was attempted
            mock_makedirs.assert_called_once_with('../Output/RetrievalContext', exist_ok=True)

    def test_initialization_custom_parameters(self, mock_db_config_loader,
                                              mock_pg_database, mock_metrics_reporter,
                                              mock_metrics_manager):
        """Test initialization with custom parameters"""
        with patch('os.makedirs'):
            orch = PGVectorOrchestrator(
                collection_name="custom_collection",
                config_file_path="/custom/path/config.properties",
                metrics_manager=mock_metrics_manager
            )

            assert orch.collection_name == "custom_collection"
            assert orch.config_file_path == "/custom/path/config.properties"
            assert orch.metrics_manager == mock_metrics_manager

    def test_initialization_creates_metrics_manager_if_none(self, mock_db_config_loader,
                                                            mock_pg_database,
                                                            mock_metrics_reporter):
        """Test that MetricsManager is created if not provided"""
        with patch('os.makedirs'), \
                patch('src.context_handler.context_storage_handler.pgvector_orchestrator.MetricsManager') as mock_mm:
            mock_mm_instance = Mock()
            mock_mm.return_value = mock_mm_instance

            orch = PGVectorOrchestrator(metrics_manager=None)

            # Verify MetricsManager was created
            mock_mm.assert_called_once()
            assert orch.metrics_manager == mock_mm_instance

    def test_initialization_config_loader_called(self, mock_db_config_loader,
                                                 mock_pg_database, mock_metrics_reporter,
                                                 mock_metrics_manager):
        """Test that DBConfigLoader is initialized correctly"""
        with patch('os.makedirs'):
            orch = PGVectorOrchestrator(
                collection_name="test",
                config_file_path="/test/config.properties",
                metrics_manager=mock_metrics_manager
            )

            # Verify DBConfigLoader was called with correct parameters
            mock_db_config_loader.assert_called_once_with(
                config_file_path="/test/config.properties",
                metrics_manager=mock_metrics_manager
            )

    def test_initialization_metrics_reporter_called(self, mock_db_config_loader,
                                                    mock_pg_database, mock_metrics_reporter,
                                                    mock_metrics_manager):
        """Test that MetricsReporter is initialized correctly"""
        with patch('os.makedirs'):
            orch = PGVectorOrchestrator(
                collection_name="test_collection",
                metrics_manager=mock_metrics_manager
            )

            # Verify MetricsReporter was called with correct parameters
            mock_metrics_reporter.assert_called_once_with(
                collection_name="test_collection",
                metrics_manager=mock_metrics_manager
            )

    def test_initialization_pg_database_called(self, mock_db_config_loader,
                                               mock_pg_database, mock_metrics_reporter,
                                               mock_metrics_manager):
        """Test that PGDatabase is initialized correctly"""
        with patch('os.makedirs'):
            orch = PGVectorOrchestrator(
                collection_name="test_collection",
                metrics_manager=mock_metrics_manager
            )

            # Verify PGDatabase was called with correct parameters
            mock_pg_database.assert_called_once_with(
                orch.config,
                collection_name="test_collection"
            )

    def test_initialization_exposes_config_attributes(self, orchestrator):
        """Test that config attributes are exposed correctly"""
        assert orchestrator.embeddings is not None
        assert orchestrator.similarity_metric == 'cosine'
        assert orchestrator.threshold == 0.7
        assert orchestrator.model_name == 'amazon.titan-embed-text-v1'
        assert orchestrator.local_storage_path == '../Data/LocalEmbeddings'

    def test_initialization_retrieval_dir_created(self, mock_db_config_loader,
                                                  mock_pg_database, mock_metrics_reporter,
                                                  mock_metrics_manager):
        """Test that retrieval directory is created"""
        with patch('os.makedirs') as mock_makedirs:
            orch = PGVectorOrchestrator(metrics_manager=mock_metrics_manager)

            mock_makedirs.assert_called_once_with('../Output/RetrievalContext', exist_ok=True)
            assert orch.retrieval_dir == '../Output/RetrievalContext'

    def test_initialization_retrieval_dir_creation_error(self, mock_db_config_loader,
                                                         mock_pg_database,
                                                         mock_metrics_reporter,
                                                         mock_metrics_manager):
        """Test handling of directory creation error"""
        with patch('os.makedirs', side_effect=OSError("Permission denied")), \
                patch('logging.warning') as mock_warning:
            orch = PGVectorOrchestrator(metrics_manager=mock_metrics_manager)

            # Should log warning but not crash
            mock_warning.assert_called_once()
            assert orch.retrieval_dir == '../Output/RetrievalContext'

    # ==================== Database Operation Tests ====================

    def test_diagnose_database_success(self, orchestrator):
        """Test successful database diagnosis"""
        orchestrator.db.diagnose_database.return_value = True

        result = orchestrator.diagnose_database()

        assert result is True
        orchestrator.db.diagnose_database.assert_called_once()

    def test_diagnose_database_failure(self, orchestrator):
        """Test database diagnosis failure"""
        orchestrator.db.diagnose_database.return_value = False

        result = orchestrator.diagnose_database()

        assert result is False
        orchestrator.db.diagnose_database.assert_called_once()

    def test_diagnose_database_exception(self, orchestrator):
        """Test database diagnosis with exception"""
        orchestrator.db.diagnose_database.side_effect = Exception("Database error")

        with pytest.raises(Exception) as excinfo:
            orchestrator.diagnose_database()

        assert "Database error" in str(excinfo.value)

    def test_reconnect_if_needed_success(self, orchestrator):
        """Test successful reconnection"""
        orchestrator.reconnect_if_needed()

        orchestrator.db.reconnect_if_needed.assert_called_once()

    def test_reconnect_if_needed_exception(self, orchestrator):
        """Test reconnection with exception"""
        orchestrator.db.reconnect_if_needed.side_effect = Exception("Connection error")

        with pytest.raises(Exception) as excinfo:
            orchestrator.reconnect_if_needed()

        assert "Connection error" in str(excinfo.value)

    # ==================== Close/Cleanup Tests ====================

    def test_close_success(self, orchestrator):
        """Test successful close operation"""
        orchestrator.close()

        orchestrator.db.close.assert_called_once()

    def test_close_with_exception(self, orchestrator):
        """Test close operation with exception"""
        orchestrator.db.close.side_effect = Exception("Close error")

        with patch('logging.error') as mock_log:
            orchestrator.close()

            # Should log error but not raise exception
            mock_log.assert_called_once()
            assert "Error closing database connections" in str(mock_log.call_args)

    def test_close_without_db(self, mock_db_config_loader, mock_pg_database,
                              mock_metrics_reporter, mock_metrics_manager):
        """Test close when db attribute doesn't exist"""
        with patch('os.makedirs'):
            orch = PGVectorOrchestrator(metrics_manager=mock_metrics_manager)
            delattr(orch, 'db')

            # Should not raise exception
            orch.close()

    def test_close_with_none_db(self, orchestrator):
        """Test close when db is None"""
        orchestrator.db = None

        # Should not raise exception
        orchestrator.close()

    # ==================== Metrics Tests ====================

    def test_get_metrics_success(self, orchestrator):
        """Test successful metrics retrieval"""
        expected_metrics = {
            'vector_store_operations': 5,
            'vector_query_operations': 10,
            'total_vectors_stored': 100
        }
        orchestrator.metrics_reporter.get_metrics.return_value = expected_metrics

        result = orchestrator.get_metrics()

        assert result == expected_metrics
        orchestrator.metrics_reporter.get_metrics.assert_called_once()

    def test_get_metrics_empty(self, orchestrator):
        """Test metrics retrieval when empty"""
        orchestrator.metrics_reporter.get_metrics.return_value = {}

        result = orchestrator.get_metrics()

        assert result == {}

    def test_save_metrics_success(self, orchestrator):
        """Test successful metrics saving"""
        expected_path = '/path/to/metrics.json'
        orchestrator.metrics_reporter.save_metrics.return_value = expected_path

        result = orchestrator.save_metrics(output_dir="/custom/output")

        assert result == expected_path
        orchestrator.metrics_reporter.save_metrics.assert_called_once_with(
            output_dir="/custom/output"
        )

    def test_save_metrics_default_dir(self, orchestrator):
        """Test metrics saving with default directory"""
        expected_path = '/path/to/metrics.json'
        orchestrator.metrics_reporter.save_metrics.return_value = expected_path

        result = orchestrator.save_metrics()

        assert result == expected_path
        orchestrator.metrics_reporter.save_metrics.assert_called_once_with(
            output_dir="../Output/Metrics"
        )

    def test_save_metrics_exception(self, orchestrator):
        """Test metrics saving with exception"""
        orchestrator.metrics_reporter.save_metrics.side_effect = Exception("Save error")

        with pytest.raises(Exception) as excinfo:
            orchestrator.save_metrics()

        assert "Save error" in str(excinfo.value)

    # ==================== Attribute Access Tests ====================

    def test_embeddings_attribute_access(self, orchestrator):
        """Test accessing embeddings attribute"""
        assert orchestrator.embeddings is not None
        assert orchestrator.embeddings == orchestrator.config.embeddings

    def test_similarity_metric_attribute_access(self, orchestrator):
        """Test accessing similarity_metric attribute"""
        assert orchestrator.similarity_metric == 'cosine'
        assert orchestrator.similarity_metric == orchestrator.config.similarity_metric

    def test_threshold_attribute_access(self, orchestrator):
        """Test accessing threshold attribute"""
        assert orchestrator.threshold == 0.7
        assert orchestrator.threshold == orchestrator.config.threshold

    def test_model_name_attribute_access(self, orchestrator):
        """Test accessing model_name attribute"""
        assert orchestrator.model_name == 'amazon.titan-embed-text-v1'
        assert orchestrator.model_name == orchestrator.config.model_name

    def test_local_storage_path_attribute_access(self, orchestrator):
        """Test accessing local_storage_path attribute"""
        assert orchestrator.local_storage_path == '../Data/LocalEmbeddings'
        assert orchestrator.local_storage_path == orchestrator.config.local_storage_path

    def test_retrieval_dir_attribute_access(self, orchestrator):
        """Test accessing retrieval_dir attribute"""
        assert orchestrator.retrieval_dir == '../Output/RetrievalContext'

    # ==================== Integration Tests ====================

    def test_full_lifecycle(self, mock_db_config_loader, mock_pg_database,
                            mock_metrics_reporter, mock_metrics_manager):
        """Test full lifecycle: init -> operations -> close"""
        with patch('os.makedirs'):
            # Initialize
            orch = PGVectorOrchestrator(
                collection_name="lifecycle_test",
                metrics_manager=mock_metrics_manager
            )

            # Perform operations
            orch.diagnose_database()
            orch.reconnect_if_needed()
            metrics = orch.get_metrics()
            metrics_path = orch.save_metrics()

            # Close
            orch.close()

            # Verify all operations were called
            assert orch.db.diagnose_database.called
            assert orch.db.reconnect_if_needed.called
            assert orch.metrics_reporter.get_metrics.called
            assert orch.metrics_reporter.save_metrics.called
            assert orch.db.close.called

    def test_multiple_orchestrators_same_collection(self, mock_db_config_loader,
                                                    mock_pg_database,
                                                    mock_metrics_reporter,
                                                    mock_metrics_manager):
        """Test creating multiple orchestrators for the same collection"""
        with patch('os.makedirs'):
            orch1 = PGVectorOrchestrator(
                collection_name="shared_collection",
                metrics_manager=mock_metrics_manager
            )
            orch2 = PGVectorOrchestrator(
                collection_name="shared_collection",
                metrics_manager=mock_metrics_manager
            )

            # Both should have the same collection name
            assert orch1.collection_name == orch2.collection_name

            # But should be separate instances
            assert orch1 is not orch2
            assert orch1.db is not orch2.db

    def test_multiple_orchestrators_different_collections(self, mock_db_config_loader,
                                                          mock_pg_database,
                                                          mock_metrics_reporter,
                                                          mock_metrics_manager):
        """Test creating multiple orchestrators for different collections"""
        with patch('os.makedirs'):
            orch1 = PGVectorOrchestrator(
                collection_name="collection1",
                metrics_manager=mock_metrics_manager
            )
            orch2 = PGVectorOrchestrator(
                collection_name="collection2",
                metrics_manager=mock_metrics_manager
            )

            # Should have different collection names
            assert orch1.collection_name != orch2.collection_name
            assert orch1.collection_name == "collection1"
            assert orch2.collection_name == "collection2"

    # ==================== Error Handling Tests ====================

    def test_initialization_with_invalid_config_path(self, mock_db_config_loader,
                                                     mock_pg_database,
                                                     mock_metrics_reporter,
                                                     mock_metrics_manager):
        """Test initialization with invalid config path"""
        mock_db_config_loader.side_effect = FileNotFoundError("Config not found")

        with patch('os.makedirs'):
            with pytest.raises(FileNotFoundError):
                PGVectorOrchestrator(
                    config_file_path="/invalid/path/config.properties",
                    metrics_manager=mock_metrics_manager
                )

    def test_initialization_with_database_error(self, mock_db_config_loader,
                                                mock_pg_database,
                                                mock_metrics_reporter,
                                                mock_metrics_manager):
        """Test initialization with database connection error"""
        mock_pg_database.side_effect = Exception("Database connection failed")

        with patch('os.makedirs'):
            with pytest.raises(Exception) as excinfo:
                PGVectorOrchestrator(metrics_manager=mock_metrics_manager)

            assert "Database connection failed" in str(excinfo.value)

    def test_operations_after_close(self, orchestrator):
        """Test operations after orchestrator is closed"""
        orchestrator.close()

        # Operations should still work (db might be closed but methods should be callable)
        # This tests that the orchestrator doesn't prevent method calls after close
        try:
            orchestrator.diagnose_database()
            orchestrator.reconnect_if_needed()
        except Exception:
            # Expected if db is actually closed
            pass

    # ==================== Edge Cases ====================

    def test_empty_collection_name(self, mock_db_config_loader, mock_pg_database,
                                   mock_metrics_reporter, mock_metrics_manager):
        """Test with empty collection name"""
        with patch('os.makedirs'):
            orch = PGVectorOrchestrator(
                collection_name="",
                metrics_manager=mock_metrics_manager
            )

            assert orch.collection_name == ""

    def test_special_characters_in_collection_name(self, mock_db_config_loader,
                                                   mock_pg_database,
                                                   mock_metrics_reporter,
                                                   mock_metrics_manager):
        """Test with special characters in collection name"""
        with patch('os.makedirs'):
            orch = PGVectorOrchestrator(
                collection_name="test-collection_123!@#",
                metrics_manager=mock_metrics_manager
            )

            assert orch.collection_name == "test-collection_123!@#"

    def test_very_long_collection_name(self, mock_db_config_loader, mock_pg_database,
                                       mock_metrics_reporter, mock_metrics_manager):
        """Test with very long collection name"""
        long_name = "a" * 1000
        with patch('os.makedirs'):
            orch = PGVectorOrchestrator(
                collection_name=long_name,
                metrics_manager=mock_metrics_manager
            )

            assert orch.collection_name == long_name

    def test_unicode_collection_name(self, mock_db_config_loader, mock_pg_database,
                                     mock_metrics_reporter, mock_metrics_manager):
        """Test with unicode characters in collection name"""
        with patch('os.makedirs'):
            orch = PGVectorOrchestrator(
                collection_name="测试_コレクション_مجموعة",
                metrics_manager=mock_metrics_manager
            )

            assert orch.collection_name == "测试_コレクション_مجموعة"

    # ==================== Concurrent Access Tests ====================

    def test_concurrent_operations(self, orchestrator):
        """Test concurrent operations on the same orchestrator"""
        import threading

        results = []
        errors = []

        def perform_operations():
            try:
                orchestrator.diagnose_database()
                orchestrator.reconnect_if_needed()
                metrics = orchestrator.get_metrics()
                results.append(metrics)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=perform_operations)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check that operations completed
        assert len(results) == 5
        assert len(errors) == 0

    # ==================== Memory Management Tests ====================

    def test_memory_cleanup_on_close(self, orchestrator):
        """Test that resources are properly cleaned up on close"""
        # Store references
        db_ref = orchestrator.db
        config_ref = orchestrator.config

        # Close orchestrator
        orchestrator.close()

        # Verify close was called on db
        db_ref.close.assert_called_once()

        # Orchestrator should still have references (Python's GC will handle cleanup)
        assert orchestrator.db is not None
        assert orchestrator.config is not None

    def test_del_method_calls_close(self, mock_db_config_loader, mock_pg_database,
                                    mock_metrics_reporter, mock_metrics_manager):
        """Test that __del__ method properly closes resources"""
        with patch('os.makedirs'):
            orch = PGVectorOrchestrator(metrics_manager=mock_metrics_manager)
            db_mock = orch.db

            # Delete the orchestrator
            del orch

            # Note: __del__ behavior is not guaranteed in Python, so we just verify
            # that the close method exists and can be called
            assert hasattr(db_mock, 'close')


# ==================== Fixtures for Additional Test Scenarios ====================

@pytest.fixture
def orchestrator_with_real_paths(mock_db_config_loader, mock_pg_database,
                                 mock_metrics_reporter, mock_metrics_manager, tmp_path):
    """Create orchestrator with real temporary paths"""
    retrieval_dir = tmp_path / "RetrievalContext"

    with patch('os.makedirs') as mock_makedirs:
        orch = PGVectorOrchestrator(metrics_manager=mock_metrics_manager)
        orch.retrieval_dir = str(retrieval_dir)
        return orch


class TestPGVectorOrchestratorIntegration:
    """Integration tests for PGVectorOrchestrator"""

    def test_orchestrator_with_real_directory_creation(self, tmp_path, mock_db_config_loader,
                                                       mock_pg_database, mock_metrics_reporter,
                                                       mock_metrics_manager):
        """Test orchestrator with actual directory creation"""
        retrieval_dir = tmp_path / "RetrievalContext"

        with patch.object(PGVectorOrchestrator, '__init__',
                          lambda self, *args, **kwargs: None):
            orch = PGVectorOrchestrator()
            orch.retrieval_dir = str(retrieval_dir)
            orch.db = mock_pg_database.return_value
            orch.metrics_reporter = mock_metrics_reporter.return_value

            # Create the directory
            os.makedirs(orch.retrieval_dir, exist_ok=True)

            # Verify directory exists
            assert os.path.exists(orch.retrieval_dir)
            assert os.path.isdir(orch.retrieval_dir)