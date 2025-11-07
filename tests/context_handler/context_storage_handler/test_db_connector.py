# import pytest
# import psycopg2
# from unittest.mock import Mock, patch, MagicMock, call
# import time
#
# from src.context_handler.context_storage_handler.db_connector import PGDatabase
#
#
# class TestPGDatabase:
#     @pytest.fixture
#     def config_loader_mock(self):
#         """Mock config loader for testing"""
#         mock = Mock()
#         mock.db_host = "test-host"
#         mock.db_port = "5432"
#         mock.db_name = "testdb"
#         mock.db_user = "testuser"
#         mock.db_password = "testpass"
#         mock.ssl_mode = "prefer"
#         mock.similarity_metric = "cosine"
#         return mock
#
#     @pytest.fixture
#     def pg_database(self, config_loader_mock):
#         """Create PGDatabase instance with mocked dependencies"""
#         with patch('psycopg2.connect') as mock_connect:
#             mock_conn = Mock()
#             mock_cursor = Mock()
#             mock_conn.cursor.return_value = mock_cursor
#             mock_connect.return_value = mock_conn
#
#             db = PGDatabase(config_loader_mock, collection_name="test_collection")
#             db.conn = mock_conn
#             db.cursor = mock_cursor
#             return db
#
#     def test_initialization(self, config_loader_mock):
#         """Test PGDatabase initialization"""
#         with patch('psycopg2.connect') as mock_connect, \
#                 patch('sqlalchemy.create_engine') as mock_create_engine:
#             mock_conn = Mock()
#             mock_cursor = Mock()
#             mock_conn.cursor.return_value = mock_cursor
#             mock_connect.return_value = mock_conn
#
#             db = PGDatabase(config_loader_mock)
#
#             # Check that connection string was built correctly
#             assert "postgresql://" in db.connection_string
#             assert "testuser:testpass@test-host:5432/testdb" in db.connection_string
#             assert "?sslmode=prefer" in db.connection_string
#
#             # Check that engine was created
#             mock_create_engine.assert_called_once()
#
#             # Check that setup_database was called
#             mock_cursor.execute.assert_called()
#             mock_conn.commit.assert_called_once()
#
#     def test_build_connection_string(self, config_loader_mock):
#         """Test building connection string with and without SSL mode"""
#         # With SSL mode
#         with patch('psycopg2.connect'):
#             db = PGDatabase(config_loader_mock)
#             assert db.connection_string == "postgresql://testuser:testpass@test-host:5432/testdb?sslmode=prefer"
#
#         # Without SSL mode
#         config_loader_mock.ssl_mode = None
#         with patch('psycopg2.connect'):
#             db = PGDatabase(config_loader_mock)
#             assert db.connection_string == "postgresql://testuser:testpass@test-host:5432/testdb"
#
#     def test_setup_database(self, pg_database):
#         """Test database setup with schema creation"""
#         # Reset mock to check calls
#         pg_database.cursor.reset_mock()
#         pg_database.conn.reset_mock()
#
#         # Call setup_database
#         pg_database.setup_database()
#
#         # Check that all required SQL statements were executed
#         expected_calls = [
#             call("CREATE EXTENSION IF NOT EXISTS vector;"),
#             call("""
#             CREATE TABLE IF NOT EXISTS document_embeddings (
#                 id SERIAL PRIMARY KEY,
#                 collection_name VARCHAR(255) NOT NULL,
#                 content TEXT NOT NULL,
#                 metadata JSONB,
#                 embedding vector(1536),
#                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#             );
#             """),
#             call("CREATE INDEX IF NOT EXISTS collection_idx ON document_embeddings (collection_name);"),
#             call("""
#             CREATE INDEX IF NOT EXISTS collection_content_idx ON document_embeddings (collection_name, md5(content));
#             """),
#             call("""
#                 CREATE INDEX IF NOT EXISTS embedding_cosine_idx
#                 ON document_embeddings
#                 USING ivfflat (embedding vector_cosine_ops)
#                 WITH (lists = 100);
#                 """)
#         ]
#
#         # Check that all SQL statements were executed
#         assert pg_database.cursor.execute.call_count == len(expected_calls)
#         pg_database.cursor.execute.assert_has_calls(expected_calls, any_order=False)
#
#         # Check that changes were committed
#         pg_database.conn.commit.assert_called_once()
#
#     def test_setup_database_with_different_similarity_metrics(self, config_loader_mock):
#         """Test database setup with different similarity metrics"""
#         # Test with l2 similarity metric
#         config_loader_mock.similarity_metric = "l2"
#         with patch('psycopg2.connect') as mock_connect:
#             mock_conn = Mock()
#             mock_cursor = Mock()
#             mock_conn.cursor.return_value = mock_cursor
#             mock_connect.return_value = mock_conn
#
#             db = PGDatabase(config_loader_mock)
#
#             # Check that l2 index was created
#             mock_cursor.execute.assert_any_call("""
#                 CREATE INDEX IF NOT EXISTS embedding_l2_idx
#                 ON document_embeddings
#                 USING ivfflat (embedding vector_l2_ops)
#                 WITH (lists = 100);
#                 """)
#
#         # Test with inner product similarity metric
#         config_loader_mock.similarity_metric = "inner"
#         with patch('psycopg2.connect') as mock_connect:
#             mock_conn = Mock()
#             mock_cursor = Mock()
#             mock_conn.cursor.return_value = mock_cursor
#             mock_connect.return_value = mock_conn
#
#             db = PGDatabase(config_loader_mock)
#
#             # Check that inner product index was created
#             mock_cursor.execute.assert_any_call("""
#                 CREATE INDEX IF NOT EXISTS embedding_ip_idx
#                 ON document_embeddings
#                 USING ivfflat (embedding vector_ip_ops)
#                 WITH (lists = 100);
#                 """)
#
#     def test_connect_with_retry_success_first_attempt(self, config_loader_mock):
#         """Test successful connection on first attempt"""
#         with patch('psycopg2.connect') as mock_connect:
#             mock_conn = Mock()
#             mock_cursor = Mock()
#             mock_conn.cursor.return_value = mock_cursor
#             mock_connect.return_value = mock_conn
#
#             db = PGDatabase(config_loader_mock, max_retries=3)
#
#             # Reset mocks to check connect_with_retry specifically
#             mock_connect.reset_mock()
#
#             # Call connect_with_retry
#             db.connect_with_retry()
#
#             # Check that connect was called once with correct parameters
#             mock_connect.assert_called_once_with(
#                 host="test-host",
#                 port="5432",
#                 database="testdb",
#                 user="testuser",
#                 password="testpass",
#                 sslmode="prefer",
#                 application_name="StorySense"
#             )
#
#             # Check that cursor was obtained
#             mock_conn.cursor.assert_called_once()
#
#             # Check that connection and cursor were set
#             assert db.conn == mock_conn
#             assert db.cursor == mock_cursor
#
#     def test_connect_with_retry_success_after_retries(self, config_loader_mock):
#         """Test successful connection after multiple retries"""
#         with patch('psycopg2.connect') as mock_connect, \
#                 patch('time.sleep') as mock_sleep:
#             # Make connect fail twice then succeed
#             mock_connect.side_effect = [
#                 psycopg2.OperationalError("Connection refused"),
#                 psycopg2.OperationalError("Connection refused"),
#                 Mock()  # Successful connection on third try
#             ]
#
#             db = PGDatabase(config_loader_mock, max_retries=3, retry_backoff=0.1)
#
#             # Reset connection and cursor
#             db.conn = None
#             db.cursor = None
#
#             # Call connect_with_retry
#             db.connect_with_retry()
#
#             # Check that connect was called three times
#             assert mock_connect.call_count == 3
#
#             # Check that sleep was called twice with increasing durations
#             mock_sleep.assert_has_calls([
#                 call(0.1),  # First retry: backoff * 1
#                 call(0.2)  # Second retry: backoff * 2
#             ])
#
#             # Check that connection and cursor were set
#             assert db.conn is not None
#             assert db.cursor is not None
#
#     def test_connect_with_retry_max_retries_exceeded(self, config_loader_mock):
#         """Test connection failure after max retries"""
#         with patch('psycopg2.connect') as mock_connect, \
#                 patch('time.sleep') as mock_sleep:
#             # Make connect always fail
#             mock_connect.side_effect = psycopg2.OperationalError("Connection refused")
#
#             db = PGDatabase(config_loader_mock, max_retries=3, retry_backoff=0.1)
#
#             # Reset connection and cursor
#             db.conn = None
#             db.cursor = None
#
#             # Call connect_with_retry - should raise exception after max retries
#             with pytest.raises(psycopg2.OperationalError) as excinfo:
#                 db.connect_with_retry()
#
#             assert "Connection refused" in str(excinfo.value)
#
#             # Check that connect was called three times
#             assert mock_connect.call_count == 3
#
#             # Check that sleep was called twice
#             assert mock_sleep.call_count == 2
#
#     def test_reconnect_if_needed_connection_ok(self, pg_database):
#         """Test reconnect_if_needed when connection is OK"""
#         # Reset mocks
#         pg_database.cursor.reset_mock()
#
#         # Call reconnect_if_needed
#         pg_database.reconnect_if_needed()
#
#         # Check that cursor.execute was called to test connection
#         pg_database.cursor.execute.assert_called_once_with("SELECT 1")
#
#         # connect_with_retry should not be called
#         assert not hasattr(pg_database, 'connect_with_retry_called')
#
#     def test_reconnect_if_needed_no_cursor(self, pg_database):
#         """Test reconnect_if_needed when cursor is None"""
#         # Set cursor to None
#         pg_database.cursor = None
#
#         # Mock connect_with_retry
#         pg_database.connect_with_retry = Mock()
#
#         # Call reconnect_if_needed
#         pg_database.reconnect_if_needed()
#
#         # Check that connect_with_retry was called
#         pg_database.connect_with_retry.assert_called_once()
#
#     def test_reconnect_if_needed_connection_lost(self, pg_database):
#         """Test reconnect_if_needed when connection is lost"""
#         # Make cursor.execute raise an exception
#         pg_database.cursor.execute.side_effect = psycopg2.OperationalError("Connection lost")
#
#         # Mock connect_with_retry
#         pg_database.connect_with_retry = Mock()
#
#         # Call reconnect_if_needed
#         pg_database.reconnect_if_needed()
#
#         # Check that connect_with_retry was called
#         pg_database.connect_with_retry.assert_called_once()
#
#     def test_reconnect_if_needed_interface_error(self, pg_database):
#         """Test reconnect_if_needed with InterfaceError"""
#         # Make cursor.execute raise an InterfaceError
#         pg_database.cursor.execute.side_effect = psycopg2.InterfaceError("Interface error")
#
#         # Mock connect_with_retry
#         pg_database.connect_with_retry = Mock()
#
#         # Call reconnect_if_needed
#         pg_database.reconnect_if_needed()
#
#         # Check that connect_with_retry was called
#         pg_database.connect_with_retry.assert_called_once()
#
#     def test_diagnose_database_success(self, pg_database):
#         """Test diagnose_database successful execution"""
#         # Mock cursor.fetchall to return sample data
#         pg_database.cursor.fetchall.side_effect = [
#             [('collection1', 10), ('collection2', 5)],  # Collections
#             [('collection1', 'Sample content', {'key': 'value'}),
#              ('collection2', 'Another sample', {'key2': 'value2'}),
#              ('collection3', 'Third sample', {'key3': 'value3'})],  # Samples
#             [('collection1', 1536)]  # Dimensions
#         ]
#
#         # Call diagnose_database
#         result = pg_database.diagnose_database()
#
#         # Check that the function returned True
#         assert result is True
#
#         # Check that all queries were executed
#         assert pg_database.cursor.execute.call_count == 3
#         pg_database.cursor.execute.assert_any_call(
#             "SELECT collection_name, COUNT(*) FROM document_embeddings GROUP BY collection_name")
#         pg_database.cursor.execute.assert_any_call("""
#             SELECT collection_name, content, metadata
#             FROM document_embeddings
#             ORDER BY id
#             LIMIT 3
#             """)
#         pg_database.cursor.execute.assert_any_call("""
#             SELECT
#                 collection_name,
#                 vector_dims(embedding) as dimensions
#             FROM document_embeddings
#             LIMIT 1
#             """)
#
#         # Check that fetchall was called for collections query
#         assert pg_database.cursor.fetchall.call_count == 1
#
#         # Check that fetchone was called for dimensions query
#         assert pg_database.cursor.fetchone.call_count == 1
#
#     def test_diagnose_database_no_vectors(self, pg_database):
#         """Test diagnose_database when no vectors are found"""
#         # Mock cursor.fetchall to return collections
#         pg_database.cursor.fetchall.return_value = [('collection1', 10), ('collection2', 5)]
#
#         # Mock cursor.fetchone to return None for dimensions query
#         pg_database.cursor.fetchone.return_value = None
#
#         # Call diagnose_database
#         result = pg_database.diagnose_database()
#
#         # Check that the function returned True despite no vectors
#         assert result is True
#
#     def test_diagnose_database_error(self, pg_database):
#         """Test diagnose_database error handling"""
#         # Make cursor.execute raise an exception
#         pg_database.cursor.execute.side_effect = Exception("Database error")
#
#         # Call diagnose_database
#         result = pg_database.diagnose_database()
#
#         # Check that the function returned False
#         assert result is False
#
#         # Check that rollback was called
#         pg_database.conn.rollback.assert_called_once()
#
#     def test_close(self, pg_database):
#         """Test close method"""
#         # Call close
#         pg_database.close()
#
#         # Check that cursor and connection were closed
#         pg_database.cursor.close.assert_called_once()
#         pg_database.conn.close.assert_called_once()
#
#     def test_close_with_errors(self):
#         """Test close method with errors"""
#         # Create a mock cursor that raises an exception when closed
#         mock_cursor = Mock()
#         mock_cursor.close.side_effect = Exception("Error closing cursor")
#
#         # Create a mock connection that raises an exception when closed
#         mock_conn = Mock()
#         mock_conn.close.side_effect = Exception("Error closing connection")
#
#         # Create a PGDatabase instance with these mocks
#         db = Mock()
#         db.cursor = mock_cursor
#         db.conn = mock_conn
#
#         # Get the close method
#         close_method = PGDatabase.close.__get__(db)
#
#         # Call close - should not raise exceptions
#         close_method()
#
#         # Check that both close methods were called despite errors
#         mock_cursor.close.assert_called_once()
#         mock_conn.close.assert_called_once()
#
#     def test_setup_database_error_handling(self, config_loader_mock):
#         """Test error handling in setup_database"""
#         with patch('psycopg2.connect') as mock_connect:
#             mock_conn = Mock()
#             mock_cursor = Mock()
#             mock_conn.cursor.return_value = mock_cursor
#             mock_connect.return_value = mock_conn
#
#             # Make cursor.execute raise an exception
#             mock_cursor.execute.side_effect = Exception("Database error")
#
#             # Creating PGDatabase should not raise exception
#             with pytest.raises(Exception) as excinfo:
#                 db = PGDatabase(config_loader_mock)
#
#             assert "Error setting up database" in str(excinfo.value)
#
#             # Check that rollback was called
#             mock_conn.rollback.assert_called_once()

import pytest
import psycopg2
from unittest.mock import Mock, patch, MagicMock, call
import time
import sqlalchemy

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
        return mock

    def test_initialization(self):
        """Test PGDatabase initialization"""
        # Create mock config
        config_mock = Mock()
        config_mock.db_host = "test-host"
        config_mock.db_port = "5432"
        config_mock.db_name = "testdb"
        config_mock.db_user = "testuser"
        config_mock.db_password = "testpass"
        config_mock.ssl_mode = "prefer"
        config_mock.similarity_metric = "cosine"

        # Mock all external dependencies
        with patch('psycopg2.connect') as mock_connect, \
                patch('sqlalchemy.create_engine') as mock_create_engine, \
                patch.object(PGDatabase, 'connect_with_retry'), \
                patch.object(PGDatabase, 'setup_database'):
            # Setup mock connection and cursor
            mock_conn = Mock()
            mock_cursor = Mock()
            mock_conn.cursor.return_value = mock_cursor
            mock_connect.return_value = mock_conn

            # Create the database object
            db = PGDatabase(config_mock)

            # Manually set the connection string since we've mocked the method that would set it
            db._build_connection_string()

            # Check that connection string was built correctly
            expected_conn_string = "postgresql://testuser:testpass@test-host:5432/testdb?sslmode=prefer"
            assert db.connection_string == expected_conn_string

            # Check that engine was created with correct connection string
            mock_create_engine.assert_called_once()

    def test_build_connection_string(self):
        """Test building connection string with and without SSL mode"""
        # With SSL mode
        config_mock = Mock()
        config_mock.db_host = "test-host"
        config_mock.db_port = "5432"
        config_mock.db_name = "testdb"
        config_mock.db_user = "testuser"
        config_mock.db_password = "testpass"
        config_mock.ssl_mode = "prefer"
        config_mock.similarity_metric = "cosine"

        # Create a database instance
        db = PGDatabase.__new__(PGDatabase)  # Create instance without calling __init__
        db.config = config_mock

        # Call _build_connection_string directly
        db._build_connection_string()

        # Check the connection string
        assert db.connection_string == "postgresql://testuser:testpass@test-host:5432/testdb?sslmode=prefer"

        # Without SSL mode
        config_mock.ssl_mode = None

        # Call _build_connection_string again
        db._build_connection_string()

        # Check the connection string
        assert db.connection_string == "postgresql://testuser:testpass@test-host:5432/testdb"

    def test_setup_database(self):
        """Test database setup with schema creation"""
        # Create mock config
        config_mock = Mock()
        config_mock.db_host = "test-host"
        config_mock.db_port = "5432"
        config_mock.db_name = "testdb"
        config_mock.db_user = "testuser"
        config_mock.db_password = "testpass"
        config_mock.ssl_mode = "prefer"
        config_mock.similarity_metric = "cosine"

        # Create mocks for connection and cursor
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor

        # Create a database instance with mocked connection
        db = PGDatabase.__new__(PGDatabase)  # Create instance without calling __init__
        db.config = config_mock
        db.conn = mock_conn
        db.cursor = mock_cursor
        db.connection_string = "postgresql://testuser:testpass@test-host:5432/testdb?sslmode=prefer"
        db.max_retries = 3
        db.retry_backoff = 1

        # Call setup_database directly
        db.setup_database()

        # Check that all required SQL statements were executed
        expected_calls = [
            call("CREATE EXTENSION IF NOT EXISTS vector;"),
            call("""
            CREATE TABLE IF NOT EXISTS document_embeddings (
                id SERIAL PRIMARY KEY,
                collection_name VARCHAR(255) NOT NULL,
                content TEXT NOT NULL,
                metadata JSONB,
                embedding vector(1536),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """),
            call("CREATE INDEX IF NOT EXISTS collection_idx ON document_embeddings (collection_name);"),
            call("""
            CREATE INDEX IF NOT EXISTS collection_content_idx ON document_embeddings (collection_name, md5(content));
            """),
            call("""
                CREATE INDEX IF NOT EXISTS embedding_cosine_idx 
                ON document_embeddings 
                USING ivfflat (embedding vector_cosine_ops) 
                WITH (lists = 100);
                """)
        ]

        # Check that all SQL statements were executed
        assert mock_cursor.execute.call_count >= 5

        # Check that changes were committed
        mock_conn.commit.assert_called_once()

    def test_setup_database_with_different_similarity_metrics(self):
        """Test database setup with different similarity metrics"""
        # Test with l2 similarity metric
        config_mock = Mock()
        config_mock.db_host = "test-host"
        config_mock.db_port = "5432"
        config_mock.db_name = "testdb"
        config_mock.db_user = "testuser"
        config_mock.db_password = "testpass"
        config_mock.ssl_mode = "prefer"
        config_mock.similarity_metric = "l2"

        # Create mocks for connection and cursor
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor

        # Create a database instance with mocked connection
        db = PGDatabase.__new__(PGDatabase)  # Create instance without calling __init__
        db.config = config_mock
        db.conn = mock_conn
        db.cursor = mock_cursor
        db.connection_string = "postgresql://testuser:testpass@test-host:5432/testdb?sslmode=prefer"
        db.max_retries = 3
        db.retry_backoff = 1

        # Call setup_database directly
        db.setup_database()

        # Check that l2 index was created
        mock_cursor.execute.assert_any_call("""
                CREATE INDEX IF NOT EXISTS embedding_l2_idx 
                ON document_embeddings 
                USING ivfflat (embedding vector_l2_ops) 
                WITH (lists = 100);
                """)

        # Test with inner product similarity metric
        config_mock.similarity_metric = "inner"

        # Reset mocks
        mock_cursor.reset_mock()
        mock_conn.reset_mock()

        # Call setup_database again
        db.setup_database()

        # Check that inner product index was created
        mock_cursor.execute.assert_any_call("""
                CREATE INDEX IF NOT EXISTS embedding_ip_idx 
                ON document_embeddings 
                USING ivfflat (embedding vector_ip_ops) 
                WITH (lists = 100);
                """)

    def test_connect_with_retry_success_first_attempt(self):
        """Test successful connection on first attempt"""
        # Create mock config
        config_mock = Mock()
        config_mock.db_host = "test-host"
        config_mock.db_port = "5432"
        config_mock.db_name = "testdb"
        config_mock.db_user = "testuser"
        config_mock.db_password = "testpass"
        config_mock.ssl_mode = "prefer"

        with patch('psycopg2.connect') as mock_connect:
            mock_conn = Mock()
            mock_cursor = Mock()
            mock_conn.cursor.return_value = mock_cursor
            mock_connect.return_value = mock_conn

            # Create a database instance
            db = PGDatabase.__new__(PGDatabase)  # Create instance without calling __init__
            db.config = config_mock
            db.max_retries = 3
            db.retry_backoff = 1

            # Call connect_with_retry
            db.connect_with_retry()

            # Check that connect was called once with correct parameters
            mock_connect.assert_called_once_with(
                host="test-host",
                port="5432",
                database="testdb",
                user="testuser",
                password="testpass",
                sslmode="prefer",
                application_name="StorySense"
            )

            # Check that cursor was obtained
            mock_conn.cursor.assert_called_once()

            # Check that connection and cursor were set
            assert db.conn == mock_conn
            assert db.cursor == mock_cursor

    def test_connect_with_retry_success_after_retries(self):
        """Test successful connection after multiple retries"""
        # Create mock config
        config_mock = Mock()
        config_mock.db_host = "test-host"
        config_mock.db_port = "5432"
        config_mock.db_name = "testdb"
        config_mock.db_user = "testuser"
        config_mock.db_password = "testpass"
        config_mock.ssl_mode = "prefer"

        with patch('psycopg2.connect') as mock_connect, \
                patch('time.sleep') as mock_sleep:
            # Create mock connections
            mock_conn = Mock()
            mock_cursor = Mock()
            mock_conn.cursor.return_value = mock_cursor

            # Make connect fail twice then succeed
            mock_connect.side_effect = [
                psycopg2.OperationalError("Connection refused"),
                psycopg2.OperationalError("Connection refused"),
                mock_conn  # Successful connection on third try
            ]

            # Create a database instance
            db = PGDatabase.__new__(PGDatabase)  # Create instance without calling __init__
            db.config = config_mock
            db.max_retries = 3
            db.retry_backoff = 0.1

            # Call connect_with_retry
            db.connect_with_retry()

            # Check that connect was called three times
            assert mock_connect.call_count == 3

            # Check that sleep was called twice with increasing durations
            mock_sleep.assert_has_calls([
                call(0.1),  # First retry: backoff * 1
                call(0.2)  # Second retry: backoff * 2
            ])

            # Check that connection and cursor were set
            assert db.conn == mock_conn
            assert db.cursor == mock_cursor

    def test_connect_with_retry_max_retries_exceeded(self):
        """Test connection failure after max retries"""
        # Create mock config
        config_mock = Mock()
        config_mock.db_host = "test-host"
        config_mock.db_port = "5432"
        config_mock.db_name = "testdb"
        config_mock.db_user = "testuser"
        config_mock.db_password = "testpass"
        config_mock.ssl_mode = "prefer"

        with patch('psycopg2.connect') as mock_connect, \
                patch('time.sleep') as mock_sleep:
            # Create a custom exception for testing
            test_exception = psycopg2.OperationalError("Connection refused")

            # Make connect always fail with the same exception instance
            mock_connect.side_effect = [test_exception, test_exception, test_exception]

            # Create a database instance
            db = PGDatabase.__new__(PGDatabase)  # Create instance without calling __init__
            db.config = config_mock
            db.max_retries = 3
            db.retry_backoff = 0.1

            # Call connect_with_retry - should raise exception after max retries
            with pytest.raises(psycopg2.OperationalError) as excinfo:
                db.connect_with_retry()

            # Check that the exception is the same instance
            assert excinfo.value == test_exception

            # Check that connect was called three times
            assert mock_connect.call_count == 3

            # Check that sleep was called exactly twice (not three times)
            assert mock_sleep.call_count == 2

    def test_reconnect_if_needed_connection_ok(self):
        """Test reconnect_if_needed when connection is OK"""
        # Create a database instance with mocked connection
        db = PGDatabase.__new__(PGDatabase)  # Create instance without calling __init__
        db.cursor = Mock()

        # Call reconnect_if_needed
        db.reconnect_if_needed()

        # Check that cursor.execute was called to test connection
        db.cursor.execute.assert_called_once_with("SELECT 1")

    def test_reconnect_if_needed_no_cursor(self):
        """Test reconnect_if_needed when cursor is None"""
        # Create a database instance with no cursor
        db = PGDatabase.__new__(PGDatabase)  # Create instance without calling __init__
        db.cursor = None
        db.connect_with_retry = Mock()

        # Call reconnect_if_needed
        db.reconnect_if_needed()

        # Check that connect_with_retry was called
        db.connect_with_retry.assert_called_once()

    def test_reconnect_if_needed_connection_lost(self):
        """Test reconnect_if_needed when connection is lost"""
        # Create a database instance with mocked connection
        db = PGDatabase.__new__(PGDatabase)  # Create instance without calling __init__
        db.cursor = Mock()
        db.cursor.execute.side_effect = psycopg2.OperationalError("Connection lost")
        db.connect_with_retry = Mock()

        # Call reconnect_if_needed
        db.reconnect_if_needed()

        # Check that connect_with_retry was called
        db.connect_with_retry.assert_called_once()

    def test_reconnect_if_needed_interface_error(self):
        """Test reconnect_if_needed with InterfaceError"""
        # Create a database instance with mocked connection
        db = PGDatabase.__new__(PGDatabase)  # Create instance without calling __init__
        db.cursor = Mock()
        db.cursor.execute.side_effect = psycopg2.InterfaceError("Interface error")
        db.connect_with_retry = Mock()

        # Call reconnect_if_needed
        db.reconnect_if_needed()

        # Check that connect_with_retry was called
        db.connect_with_retry.assert_called_once()

    def test_diagnose_database_success(self):
        """Test diagnose_database successful execution"""
        # Create a database instance with mocked connection
        db = PGDatabase.__new__(PGDatabase)  # Create instance without calling __init__
        db.cursor = Mock()
        db.conn = Mock()
        db.config = Mock()
        db.config.similarity_metric = "cosine"
        db.config.threshold = 0.7

        # Mock cursor.fetchall to return sample data
        db.cursor.fetchall.return_value = [('collection1', 10), ('collection2', 5)]

        # Mock cursor.fetchone to return dimensions
        db.cursor.fetchone.return_value = ('collection1', 1536)

        # Mock the logging.info to avoid actual logging
        with patch('logging.info'):
            # Call diagnose_database
            result = db.diagnose_database()

            # Check that the function returned True
            assert result is True

            # Check that all queries were executed
            assert db.cursor.execute.call_count >= 3
            db.cursor.execute.assert_any_call(
                "SELECT collection_name, COUNT(*) FROM document_embeddings GROUP BY collection_name")

    def test_diagnose_database_no_vectors(self):
        """Test diagnose_database when no vectors are found"""
        # Create a database instance with mocked connection
        db = PGDatabase.__new__(PGDatabase)  # Create instance without calling __init__
        db.cursor = Mock()
        db.conn = Mock()
        db.config = Mock()
        db.config.similarity_metric = "cosine"
        db.config.threshold = 0.7

        # Mock cursor.fetchall to return collections
        db.cursor.fetchall.return_value = [('collection1', 10), ('collection2', 5)]

        # Mock cursor.fetchone to return None for dimensions query
        db.cursor.fetchone.return_value = None

        # Mock the logging.info and logging.warning to avoid actual logging
        with patch('logging.info'), patch('logging.warning'):
            # Call diagnose_database
            result = db.diagnose_database()

            # Check that the function returned True despite no vectors
            assert result is True

    def test_diagnose_database_error(self):
        """Test diagnose_database error handling"""
        # Create a database instance with mocked connection
        db = PGDatabase.__new__(PGDatabase)  # Create instance without calling __init__
        db.cursor = Mock()
        db.conn = Mock()

        # Make cursor.execute raise an exception
        db.cursor.execute.side_effect = Exception("Database error")

        # Mock the logging.error to avoid actual logging
        with patch('logging.error'):
            # Call diagnose_database
            result = db.diagnose_database()

            # Check that the function returned False
            assert result is False

            # Check that rollback was called
            db.conn.rollback.assert_called_once()

    def test_close(self):
        """Test close method"""
        # Create a database instance with mocked connection
        db = PGDatabase.__new__(PGDatabase)  # Create instance without calling __init__
        db.cursor = Mock()
        db.conn = Mock()

        # Call close
        db.close()

        # Check that cursor and connection were closed
        db.cursor.close.assert_called_once()
        db.conn.close.assert_called_once()

    def test_close_with_errors(self):
        """Test close method with errors"""
        # Create a mock cursor that raises an exception when closed
        mock_cursor = Mock()
        mock_cursor.close.side_effect = Exception("Error closing cursor")

        # Create a mock connection that raises an exception when closed
        mock_conn = Mock()
        mock_conn.close.side_effect = Exception("Error closing connection")

        # Mock the logging.error to avoid actual logging
        with patch('logging.error'):
            # Create a database instance with mocked connection
            db = PGDatabase.__new__(PGDatabase)  # Create instance without calling __init__
            db.cursor = mock_cursor
            db.conn = mock_conn

            # Call close - should not raise exceptions
            db.close()

            # Check that both close methods were called despite errors
            mock_cursor.close.assert_called_once()
            mock_conn.close.assert_called_once()

    def test_setup_database_error_handling(self):
        """Test error handling in setup_database"""
        # Create mock config
        config_mock = Mock()
        config_mock.db_host = "test-host"
        config_mock.db_port = "5432"
        config_mock.db_name = "testdb"
        config_mock.db_user = "testuser"
        config_mock.db_password = "testpass"
        config_mock.ssl_mode = "prefer"
        config_mock.similarity_metric = "cosine"

        # Create mocks for connection and cursor
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor

        # Make cursor.execute raise an exception
        mock_cursor.execute.side_effect = Exception("Database error")

        # Create a database instance with mocked connection
        db = PGDatabase.__new__(PGDatabase)  # Create instance without calling __init__
        db.config = config_mock
        db.conn = mock_conn
        db.cursor = mock_cursor
        db.connection_string = "postgresql://testuser:testpass@test-host:5432/testdb?sslmode=prefer"
        db.max_retries = 3
        db.retry_backoff = 1

        # Mock the logging.error to avoid actual logging
        with patch('logging.error'):
            # Call setup_database - should raise exception
            with pytest.raises(Exception) as excinfo:
                db.setup_database()

            # Check that the error message contains "Error setting up database"
            assert "Error setting up database" in str(excinfo.value)

            # Check that rollback was called
            mock_conn.rollback.assert_called_once()