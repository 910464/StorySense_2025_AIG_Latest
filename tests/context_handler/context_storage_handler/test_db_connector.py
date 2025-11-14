import pytest
import psycopg2
from unittest.mock import Mock, patch, MagicMock, call, PropertyMock
import time
import sqlalchemy
from psycopg2 import OperationalError, InterfaceError

from src.context_handler.context_storage_handler.db_connector import PGDatabase


class TestPGDatabase:
    @pytest.fixture
    def config_loader_mock(self):
        """Mock config loader for testing"""
        mock = Mock()
        mock.db_host = "test-host"
        mock.db_port = "5432"
        mock.db_name = "testdb"
        mock.db_user = "testuser"
        mock.db_password = "testpass"
        mock.ssl_mode = "prefer"
        mock.similarity_metric = "cosine"
        mock.threshold = 0.8
        return mock

    @pytest.fixture
    def config_loader_mock_inner_product(self):
        """Mock config loader for inner product similarity"""
        mock = Mock()
        mock.db_host = "test-host"
        mock.db_port = "5432"
        mock.db_name = "testdb"
        mock.db_user = "testuser"
        mock.db_password = "testpass"
        mock.ssl_mode = "prefer"
        mock.similarity_metric = "inner_product"
        mock.threshold = 0.8
        return mock

    @pytest.fixture
    def config_loader_mock_euclidean(self):
        """Mock config loader for euclidean similarity"""
        mock = Mock()
        mock.db_host = "test-host"
        mock.db_port = "5432"
        mock.db_name = "testdb"
        mock.db_user = "testuser"
        mock.db_password = "testpass"
        mock.ssl_mode = "prefer"
        mock.similarity_metric = "euclidean"
        mock.threshold = 0.8
        return mock

    @patch('src.context_handler.context_storage_handler.db_connector.PGDatabase.setup_database')
    @patch('src.context_handler.context_storage_handler.db_connector.create_engine')
    def test_initialization(self, mock_create_engine, mock_setup, config_loader_mock):
        """Test PGDatabase initialization"""
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        db = PGDatabase(config_loader_mock)
        
        # Verify configuration is stored
        assert db.config == config_loader_mock
        assert db.conn is None
        assert db.cursor is None
        
        # Verify setup_database was called during initialization
        mock_setup.assert_called_once()

    @patch('src.context_handler.context_storage_handler.db_connector.PGDatabase.setup_database')
    def test_build_connection_string(self, mock_setup, config_loader_mock):
        """Test connection string building"""
        db = PGDatabase(config_loader_mock)
        conn_str = db._build_connection_string()
        
        expected = "host=test-host port=5432 dbname=testdb user=testuser password=testpass sslmode=prefer"
        assert conn_str == expected

    @patch('src.context_handler.context_storage_handler.db_connector.PGDatabase.connect_with_retry')
    @patch('builtins.print')
    def test_setup_database(self, mock_print, mock_connect, config_loader_mock):
        """Test database setup with successful connection"""
        # Create db instance without calling setup_database
        with patch('src.context_handler.context_storage_handler.db_connector.PGDatabase.setup_database'):
            db = PGDatabase(config_loader_mock)
        
        # Mock successful connection
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        
        db.conn = mock_conn
        db.cursor = mock_cursor
        
        # Mock connect_with_retry to not actually connect
        mock_connect.return_value = None
        
        # Call setup_database manually
        db.setup_database()
        
        # Verify connect_with_retry was called
        mock_connect.assert_called_once()
        
        # Verify SQL commands were executed
        expected_calls_contain = [
            "CREATE EXTENSION IF NOT EXISTS vector;",
            "CREATE EXTENSION IF NOT EXISTS pg_trgm;",
            "CREATE TABLE IF NOT EXISTS document_embeddings",
            "CREATE INDEX IF NOT EXISTS idx_collection_name",
            "CREATE INDEX IF NOT EXISTS idx_created_at",
        ]
        
        # Check that key SQL statements were executed
        execute_calls = [call.args[0] for call in mock_cursor.execute.call_args_list]
        for expected_sql in expected_calls_contain:
            assert any(expected_sql in actual_sql for actual_sql in execute_calls)
        
        mock_conn.commit.assert_called_once()

    @patch('src.context_handler.context_storage_handler.db_connector.PGDatabase.connect_with_retry')
    @patch('builtins.print')
    def test_setup_database_with_inner_product_similarity(self, mock_print, mock_connect, config_loader_mock_inner_product):
        """Test database setup with inner product similarity metric"""
        # Create db instance without calling setup_database
        with patch('src.context_handler.context_storage_handler.db_connector.PGDatabase.setup_database'):
            db = PGDatabase(config_loader_mock_inner_product)
        
        # Mock successful connection
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        
        db.conn = mock_conn
        db.cursor = mock_cursor
        
        # Mock connect_with_retry to not actually connect
        mock_connect.return_value = None
        
        # Call setup_database manually
        db.setup_database()
        
        # Verify the correct index was created for inner product
        execute_calls = [call.args[0] for call in mock_cursor.execute.call_args_list]
        assert any("embedding_ip_idx" in call_sql for call_sql in execute_calls)    @patch('src.context_handler.context_storage_handler.db_connector.PGDatabase.connect_with_retry')
    @patch('builtins.print')
    def test_setup_database_with_euclidean_similarity(self, mock_print, mock_connect, config_loader_mock_euclidean):
        """Test database setup with euclidean similarity metric"""
        # Create db instance without calling setup_database
        with patch('src.context_handler.context_storage_handler.db_connector.PGDatabase.setup_database'):
            db = PGDatabase(config_loader_mock_euclidean)
        
        # Mock successful connection
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        
        db.conn = mock_conn
        db.cursor = mock_cursor
        
        # Mock connect_with_retry to not actually connect
        mock_connect.return_value = None
        
        # Call setup_database manually
        db.setup_database()
        
        # Verify the correct index was created for euclidean distance
        execute_calls = [call.args[0] for call in mock_cursor.execute.call_args_list]
        assert any("embedding_l2_idx" in call_sql for call_sql in execute_calls)

    @patch('src.context_handler.context_storage_handler.db_connector.PGDatabase.setup_database')
    @patch('psycopg2.connect')
    @patch('time.sleep')
    def test_connect_with_retry_success_first_attempt(self, mock_sleep, mock_connect, mock_setup, config_loader_mock):
        """Test successful connection on first attempt"""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        db = PGDatabase(config_loader_mock)
        db.connect_with_retry()
        
        # Verify connection was established
        assert db.conn == mock_conn
        assert db.cursor == mock_cursor
        mock_connect.assert_called_once()
        mock_sleep.assert_not_called()

    @patch('src.context_handler.context_storage_handler.db_connector.PGDatabase.setup_database')
    @patch('psycopg2.connect')
    @patch('time.sleep')
    def test_connect_with_retry_success_after_retries(self, mock_sleep, mock_connect, mock_setup, config_loader_mock):
        """Test successful connection after some retries"""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        
        # First two attempts fail, third succeeds
        mock_connect.side_effect = [
            OperationalError("Connection refused"),
            OperationalError("Connection refused"),
            mock_conn
        ]
        
        db = PGDatabase(config_loader_mock)
        db.connect_with_retry()
        
        # Verify connection was established
        assert db.conn == mock_conn
        assert db.cursor == mock_cursor
        assert mock_connect.call_count == 3
        assert mock_sleep.call_count == 2

    @patch('src.context_handler.context_storage_handler.db_connector.PGDatabase.setup_database')
    @patch('psycopg2.connect')
    @patch('time.sleep')
    def test_connect_with_retry_max_retries_exceeded(self, mock_sleep, mock_connect, mock_setup, config_loader_mock):
        """Test connection failure after max retries"""
        # All attempts fail
        mock_connect.side_effect = OperationalError("Connection refused")
        
        db = PGDatabase(config_loader_mock)
        
        with pytest.raises(OperationalError):
            db.connect_with_retry()
        
        # Verify all retry attempts were made
        assert mock_connect.call_count == 3
        assert mock_sleep.call_count == 3

    @patch('src.context_handler.context_storage_handler.db_connector.PGDatabase.setup_database')
    def test_reconnect_if_needed_connection_ok(self, mock_setup, config_loader_mock):
        """Test reconnection when connection is working"""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.execute.return_value = None  # Successful ping
        
        db = PGDatabase(config_loader_mock)
        db.conn = mock_conn
        db.cursor = mock_cursor
        
        db.reconnect_if_needed()
        
        # Verify ping was executed
        mock_cursor.execute.assert_called_once_with("SELECT 1")

    @patch('src.context_handler.context_storage_handler.db_connector.PGDatabase.setup_database')
    @patch('src.context_handler.context_storage_handler.db_connector.PGDatabase.connect_with_retry')
    def test_reconnect_if_needed_no_cursor(self, mock_connect, mock_setup, config_loader_mock):
        """Test reconnection when cursor is None"""
        mock_conn = Mock()
        
        db = PGDatabase(config_loader_mock)
        db.conn = mock_conn
        db.cursor = None
        
        db.reconnect_if_needed()
        
        # Verify reconnection was attempted
        mock_connect.assert_called_once()

    @patch('src.context_handler.context_storage_handler.db_connector.PGDatabase.setup_database')
    @patch('src.context_handler.context_storage_handler.db_connector.PGDatabase.connect_with_retry')
    def test_reconnect_if_needed_connection_lost(self, mock_connect, mock_setup, config_loader_mock):
        """Test reconnection when connection is lost"""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.execute.side_effect = OperationalError("Connection lost")
        
        db = PGDatabase(config_loader_mock)
        db.conn = mock_conn
        db.cursor = mock_cursor
        
        db.reconnect_if_needed()
        
        # Verify reconnection was attempted
        mock_connect.assert_called_once()

    @patch('src.context_handler.context_storage_handler.db_connector.PGDatabase.setup_database')
    @patch('src.context_handler.context_storage_handler.db_connector.PGDatabase.connect_with_retry')
    def test_reconnect_if_needed_interface_error(self, mock_connect, mock_setup, config_loader_mock):
        """Test reconnection when interface error occurs"""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.execute.side_effect = InterfaceError("Interface error")
        
        db = PGDatabase(config_loader_mock)
        db.conn = mock_conn
        db.cursor = mock_cursor
        
        db.reconnect_if_needed()
        
        # Verify reconnection was attempted
        mock_connect.assert_called_once()

    @patch('src.context_handler.context_storage_handler.db_connector.PGDatabase.setup_database')
    @patch('src.context_handler.context_storage_handler.db_connector.PGDatabase.reconnect_if_needed')
    @patch('builtins.print')
    def test_diagnose_database_success(self, mock_print, mock_reconnect, mock_setup, config_loader_mock):
        """Test successful database diagnosis"""
        mock_conn = Mock()
        mock_cursor = Mock()
        
        # Mock cursor responses for different queries
        mock_cursor.fetchall.side_effect = [
            [('collection1', 10), ('collection2', 5)],  # Collections query
            [('collection1', 'Sample content...', '{"key": "value"}'), 
             ('collection2', 'Another sample...', '{"type": "doc"}')]  # Samples query
        ]
        mock_cursor.fetchone.return_value = ('collection1', 1536)  # Dimensions query
        
        db = PGDatabase(config_loader_mock)
        db.conn = mock_conn
        db.cursor = mock_cursor
        
        result = db.diagnose_database()
        
        assert result is True
        mock_reconnect.assert_called_once()
        
        # Verify all expected queries were executed
        expected_calls = [
            call("SELECT collection_name, COUNT(*) FROM document_embeddings GROUP BY collection_name"),
            call("""
            SELECT collection_name, content, metadata 
            FROM document_embeddings 
            ORDER BY id 
            LIMIT 3
            """),
            call("""
            SELECT 
                collection_name, 
                vector_dims(embedding) as dimensions
            FROM document_embeddings
            LIMIT 1
            """)
        ]
        
        mock_cursor.execute.assert_has_calls(expected_calls)

    @patch('src.context_handler.context_storage_handler.db_connector.PGDatabase.setup_database')
    @patch('src.context_handler.context_storage_handler.db_connector.PGDatabase.reconnect_if_needed')
    @patch('builtins.print')
    def test_diagnose_database_no_vectors(self, mock_print, mock_reconnect, mock_setup, config_loader_mock):
        """Test database diagnosis when no vectors are found"""
        mock_conn = Mock()
        mock_cursor = Mock()
        
        # Mock cursor responses
        mock_cursor.fetchall.side_effect = [
            [('collection1', 10), ('collection2', 5)],  # Collections query
            []  # No samples
        ]
        mock_cursor.fetchone.return_value = None  # No dimensions (no vectors)
        
        db = PGDatabase(config_loader_mock)
        db.conn = mock_conn
        db.cursor = mock_cursor
        
        result = db.diagnose_database()
        
        assert result is True
        mock_reconnect.assert_called_once()

    @patch('src.context_handler.context_storage_handler.db_connector.PGDatabase.setup_database')
    @patch('src.context_handler.context_storage_handler.db_connector.PGDatabase.reconnect_if_needed')
    @patch('builtins.print')
    def test_diagnose_database_error(self, mock_print, mock_reconnect, mock_setup, config_loader_mock):
        """Test database diagnosis when error occurs"""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.execute.side_effect = OperationalError("Database error")
        
        db = PGDatabase(config_loader_mock)
        db.conn = mock_conn
        db.cursor = mock_cursor
        
        result = db.diagnose_database()
        
        assert result is False
        mock_reconnect.assert_called_once()
        # Verify rollback was attempted
        mock_conn.rollback.assert_called_once()

    @patch('src.context_handler.context_storage_handler.db_connector.PGDatabase.setup_database')
    def test_close(self, mock_setup, config_loader_mock):
        """Test database connection closure"""
        mock_conn = Mock()
        mock_cursor = Mock()
        
        db = PGDatabase(config_loader_mock)
        db.conn = mock_conn
        db.cursor = mock_cursor
        
        db.close()
        
        # Verify both cursor and connection were closed
        mock_cursor.close.assert_called_once()
        mock_conn.close.assert_called_once()
        
        # Verify references were cleared
        assert db.cursor is None
        assert db.conn is None

    @patch('src.context_handler.context_storage_handler.db_connector.PGDatabase.setup_database')
    def test_close_with_errors(self, mock_setup, config_loader_mock):
        """Test database closure when errors occur during cleanup"""
        mock_conn = Mock()
        mock_cursor = Mock()
        
        # Make close operations raise exceptions
        mock_cursor.close.side_effect = Exception("Cursor close error")
        mock_conn.close.side_effect = Exception("Connection close error")
        
        db = PGDatabase(config_loader_mock)
        db.conn = mock_conn
        db.cursor = mock_cursor
        
        # Should not raise exceptions
        db.close()
        
        # Verify close attempts were made
        mock_cursor.close.assert_called_once()
        mock_conn.close.assert_called_once()
        
        # Verify references were still cleared
        assert db.cursor is None
        assert db.conn is None

    @patch('src.context_handler.context_storage_handler.db_connector.PGDatabase.setup_database')
    def test_close_with_none_values(self, mock_setup, config_loader_mock):
        """Test database closure when connection/cursor are None"""
        db = PGDatabase(config_loader_mock)
        db.conn = None
        db.cursor = None
        
        # Should not raise exceptions
        db.close()
        
        # Values should remain None
        assert db.cursor is None
        assert db.conn is None

    @patch('src.context_handler.context_storage_handler.db_connector.PGDatabase.connect_with_retry')
    @patch('builtins.print')
    def test_setup_database_error_handling(self, mock_print, mock_connect, config_loader_mock):
        """Test error handling during database setup"""
        # Create db instance without calling setup_database
        with patch('src.context_handler.context_storage_handler.db_connector.PGDatabase.setup_database'):
            db = PGDatabase(config_loader_mock)
        
        # Mock connect_with_retry to raise an exception
        mock_connect.side_effect = OperationalError("Connection failed")
        
        with pytest.raises(OperationalError):
            db.setup_database()
        
        mock_connect.assert_called_once()

    @patch('src.context_handler.context_storage_handler.db_connector.PGDatabase.setup_database')
    @patch('psycopg2.connect')
    @patch('time.sleep')
    def test_connect_with_retry_different_delays(self, mock_sleep, mock_connect, mock_setup, config_loader_mock):
        """Test that retry delays increase properly"""
        # All attempts fail
        mock_connect.side_effect = OperationalError("Connection refused")
        
        db = PGDatabase(config_loader_mock)
        
        with pytest.raises(OperationalError):
            db.connect_with_retry()
        
        # Verify sleep delays: 2s, 4s, 6s (retry_backoff=2)
        expected_calls = [call(2), call(4), call(6)]
        mock_sleep.assert_has_calls(expected_calls)

    @patch('src.context_handler.context_storage_handler.db_connector.PGDatabase.setup_database')
    @patch('src.context_handler.context_storage_handler.db_connector.PGDatabase.reconnect_if_needed')
    @patch('builtins.print')
    def test_diagnose_database_with_long_content(self, mock_print, mock_reconnect, mock_setup, config_loader_mock):
        """Test database diagnosis with content that needs truncation"""
        mock_conn = Mock()
        mock_cursor = Mock()
        
        long_content = "A" * 200  # Content longer than 100 chars
        
        # Mock cursor responses
        mock_cursor.fetchall.side_effect = [
            [('collection1', 10)],  # Collections query
            [('collection1', long_content, '{"key": "value"}')]  # Samples query with long content
        ]
        mock_cursor.fetchone.return_value = ('collection1', 1536)  # Dimensions query
        
        db = PGDatabase(config_loader_mock)
        db.conn = mock_conn
        db.cursor = mock_cursor
        
        result = db.diagnose_database()
        
        assert result is True
        
    @patch('src.context_handler.context_storage_handler.db_connector.PGDatabase.setup_database')
    def test_build_connection_string_with_different_ssl_mode(self, mock_setup, config_loader_mock):
        """Test connection string with different SSL mode"""
        config_loader_mock.ssl_mode = "require"
        
        db = PGDatabase(config_loader_mock)
        conn_str = db._build_connection_string()
        
        expected = "host=test-host port=5432 dbname=testdb user=testuser password=testpass sslmode=require"
        assert conn_str == expected

    @patch('src.context_handler.context_storage_handler.db_connector.PGDatabase.setup_database')
    @patch('src.context_handler.context_storage_handler.db_connector.create_engine')
    def test_initialization_without_engine_creation_error(self, mock_create_engine, mock_setup, config_loader_mock):
        """Test initialization stores config correctly"""
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        db = PGDatabase(config_loader_mock)
        
        assert db.config == config_loader_mock
        assert db.conn is None
        assert db.cursor is None

    @patch('src.context_handler.context_storage_handler.db_connector.PGDatabase.setup_database')
    @patch('src.context_handler.context_storage_handler.db_connector.PGDatabase.reconnect_if_needed')
    @patch('builtins.print')
    def test_diagnose_database_exception_without_connection(self, mock_print, mock_reconnect, mock_setup, config_loader_mock):
        """Test database diagnosis when connection is None"""
        mock_reconnect.side_effect = Exception("Reconnect failed")
        
        db = PGDatabase(config_loader_mock)
        db.conn = None
        
        result = db.diagnose_database()
        
        assert result is False
        mock_reconnect.assert_called_once()

    # Additional tests to improve coverage
    @patch('src.context_handler.context_storage_handler.db_connector.PGDatabase.setup_database')
    def test_build_connection_string_psycopg2_format(self, mock_setup, config_loader_mock):
        """Test psycopg2 connection string format"""
        db = PGDatabase(config_loader_mock)
        
        # Test the _build_connection_string method returns psycopg2 format
        conn_str = db._build_connection_string()
        expected = "host=test-host port=5432 dbname=testdb user=testuser password=testpass sslmode=prefer"
        assert conn_str == expected

    @patch('src.context_handler.context_storage_handler.db_connector.PGDatabase.setup_database')
    def test_setup_database_calls_engine_creation(self, mock_setup, config_loader_mock):
        """Test that setup_database creates SQLAlchemy engine"""
        # We can't test the actual engine creation since it's done in __init__
        # But we can verify the connection string building
        db = PGDatabase(config_loader_mock)
        db._build_connection_string()
        
        assert db.connection_string is not None
        assert "postgresql://" in db.connection_string or db.connection_string.startswith("host=")

    @patch('src.context_handler.context_storage_handler.db_connector.PGDatabase.setup_database')
    @patch('src.context_handler.context_storage_handler.db_connector.PGDatabase.reconnect_if_needed')
    def test_diagnose_database_empty_result_handling(self, mock_reconnect, mock_setup, config_loader_mock):
        """Test diagnose_database handles empty results gracefully"""
        mock_conn = Mock()
        mock_cursor = Mock()
        
        # Mock empty responses
        mock_cursor.fetchall.side_effect = [[], []]  # Empty collections and samples
        mock_cursor.fetchone.return_value = None  # No dimensions
        
        db = PGDatabase(config_loader_mock)
        db.conn = mock_conn
        db.cursor = mock_cursor
        
        result = db.diagnose_database()
        
        # Should still return True for successful execution
        assert result is True

    @patch('src.context_handler.context_storage_handler.db_connector.PGDatabase.setup_database')
    def test_reconnect_if_needed_no_connection(self, mock_setup, config_loader_mock):
        """Test reconnect when both conn and cursor are None"""
        db = PGDatabase(config_loader_mock)
        db.conn = None
        db.cursor = None
        
        with patch.object(db, 'connect_with_retry') as mock_connect:
            db.reconnect_if_needed()
            mock_connect.assert_called_once()

    @patch('src.context_handler.context_storage_handler.db_connector.PGDatabase.setup_database')
    def test_close_partial_cleanup(self, mock_setup, config_loader_mock):
        """Test close method with only cursor set"""
        mock_cursor = Mock()
        
        db = PGDatabase(config_loader_mock)
        db.conn = None  # No connection
        db.cursor = mock_cursor
        
        db.close()
        
        # Verify cursor was closed
        mock_cursor.close.assert_called_once()
        assert db.cursor is None
        assert db.conn is None

    @pytest.fixture
    def config_loader_mock_l2(self):
        """Mock config loader for l2 similarity"""
        mock = Mock()
        mock.db_host = "test-host"
        mock.db_port = "5432"
        mock.db_name = "testdb"
        mock.db_user = "testuser"
        mock.db_password = "testpass"
        mock.ssl_mode = "prefer"
        mock.similarity_metric = "l2"
        mock.threshold = 0.8
        return mock

    @pytest.fixture
    def config_loader_mock_inner(self):
        """Mock config loader for inner similarity"""
        mock = Mock()
        mock.db_host = "test-host"
        mock.db_port = "5432"
        mock.db_name = "testdb"
        mock.db_user = "testuser"
        mock.db_password = "testpass"
        mock.ssl_mode = "prefer"
        mock.similarity_metric = "inner"
        mock.threshold = 0.8
        return mock

    @patch('src.context_handler.context_storage_handler.db_connector.PGDatabase.connect_with_retry')
    @patch('builtins.print')
    def test_setup_database_with_l2_similarity(self, mock_print, mock_connect, config_loader_mock_l2):
        """Test database setup with l2 similarity metric"""
        # Create db instance without calling setup_database
        with patch('src.context_handler.context_storage_handler.db_connector.PGDatabase.setup_database'):
            db = PGDatabase(config_loader_mock_l2)
        
        # Mock successful connection
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        
        db.conn = mock_conn
        db.cursor = mock_cursor
        
        # Mock connect_with_retry to not actually connect
        mock_connect.return_value = None
        
        # Call setup_database manually
        db.setup_database()
        
        # Verify the correct index was created for l2
        execute_calls = [call.args[0] for call in mock_cursor.execute.call_args_list]
        assert any("embedding_l2_idx" in call_sql for call_sql in execute_calls)

    @patch('src.context_handler.context_storage_handler.db_connector.PGDatabase.connect_with_retry')
    @patch('builtins.print')
    def test_setup_database_with_inner_similarity(self, mock_print, mock_connect, config_loader_mock_inner):
        """Test database setup with inner similarity metric"""
        # Create db instance without calling setup_database
        with patch('src.context_handler.context_storage_handler.db_connector.PGDatabase.setup_database'):
            db = PGDatabase(config_loader_mock_inner)
        
        # Mock successful connection
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        
        db.conn = mock_conn
        db.cursor = mock_cursor
        
        # Mock connect_with_retry to not actually connect
        mock_connect.return_value = None
        
        # Call setup_database manually
        db.setup_database()
        
        # Verify the correct index was created for inner
        execute_calls = [call.args[0] for call in mock_cursor.execute.call_args_list]
        assert any("embedding_ip_idx" in call_sql for call_sql in execute_calls)

    @patch('src.context_handler.context_storage_handler.db_connector.PGDatabase.connect_with_retry')
    @patch('builtins.print')
    def test_setup_database_with_rollback_on_error(self, mock_print, mock_connect, config_loader_mock):
        """Test database setup with rollback on error"""
        # Create db instance without calling setup_database
        with patch('src.context_handler.context_storage_handler.db_connector.PGDatabase.setup_database'):
            db = PGDatabase(config_loader_mock)
        
        # Mock connection and cursor
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        
        db.conn = mock_conn
        db.cursor = mock_cursor
        
        # Mock connect_with_retry to not actually connect
        mock_connect.return_value = None
        
        # Make cursor.execute raise an exception after the first call
        mock_cursor.execute.side_effect = [None, Exception("Database error")]
        
        with pytest.raises(Exception):
            db.setup_database()
        
        # Verify rollback was called
        mock_conn.rollback.assert_called_once()

    @patch('src.context_handler.context_storage_handler.db_connector.PGDatabase.connect_with_retry')
    @patch('builtins.print')
    def test_setup_database_rollback_also_fails(self, mock_print, mock_connect, config_loader_mock):
        """Test database setup when rollback also fails"""
        # Create db instance without calling setup_database
        with patch('src.context_handler.context_storage_handler.db_connector.PGDatabase.setup_database'):
            db = PGDatabase(config_loader_mock)
        
        # Mock connection and cursor
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        
        db.conn = mock_conn
        db.cursor = mock_cursor
        
        # Mock connect_with_retry to not actually connect
        mock_connect.return_value = None
        
        # Make cursor.execute raise an exception and rollback also fail
        mock_cursor.execute.side_effect = [None, Exception("Database error")]
        mock_conn.rollback.side_effect = Exception("Rollback failed")
        
        with pytest.raises(Exception):
            db.setup_database()
        
        # Verify rollback was attempted
        mock_conn.rollback.assert_called_once()

    @patch('src.context_handler.context_storage_handler.db_connector.PGDatabase.setup_database') 
    def test_build_connection_string_returns_value(self, mock_setup, config_loader_mock):
        """Test that _build_connection_string actually returns the connection string"""
        db = PGDatabase(config_loader_mock)
        
        # Call the method and verify it updates the instance variable
        result = db._build_connection_string()
        
        expected = "host=test-host port=5432 dbname=testdb user=testuser password=testpass sslmode=prefer"
        assert result == expected
        assert db.connection_string == expected

    @patch('src.context_handler.context_storage_handler.db_connector.PGDatabase.setup_database')  
    def test_close_method_calls_real_close(self, mock_setup, config_loader_mock):
        """Test close method actually calls close on connections"""
        db = PGDatabase(config_loader_mock)
        
        # Set up real mock objects that will be closed
        mock_cursor = Mock()
        mock_conn = Mock()
        
        db.cursor = mock_cursor
        db.conn = mock_conn
        
        # Call close
        db.close()
        
        # Verify methods were called and attributes cleared
        mock_cursor.close.assert_called_once()
        mock_conn.close.assert_called_once()
        assert db.cursor is None
        assert db.conn is None

    @patch('src.context_handler.context_storage_handler.db_connector.PGDatabase.setup_database')
    @patch('src.context_handler.context_storage_handler.db_connector.PGDatabase.reconnect_if_needed')
    @patch('builtins.print')
    def test_diagnose_database_rollback_exception_handling(self, mock_print, mock_reconnect, mock_setup, config_loader_mock):
        """Test database diagnosis when rollback also raises exception"""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.execute.side_effect = OperationalError("Database error")
        mock_conn.rollback.side_effect = Exception("Rollback failed")
        
        db = PGDatabase(config_loader_mock)
        db.conn = mock_conn
        db.cursor = mock_cursor
        
        result = db.diagnose_database()
        
        assert result is False
        mock_reconnect.assert_called_once()
        # Verify rollback was attempted
        mock_conn.rollback.assert_called_once()
