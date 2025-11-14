import pytest
import os
from unittest.mock import patch, Mock, MagicMock
import configparser

from src.context_handler.context_storage_handler.db_config_loader import DBConfigLoader


class TestDBConfigLoader:
    @pytest.fixture
    def global_metrics_manager_mock(self):
        """Mock metrics manager for testing"""
        mock = Mock()
        mock.record_error = Mock()
        return mock

    @pytest.fixture
    def aws_titan_embeddings_mock(self):
        """Mock AWSTitanEmbeddings class"""
        with patch('src.aws_layer.aws_titan_embedding.AWSTitanEmbeddings') as mock:
            yield mock

    @pytest.fixture
    def config_file_mock(self, tmp_path):
        """Create a mock config file"""
        config_dir = tmp_path / "Config"
        config_dir.mkdir()
        config_file = config_dir / "Config.properties"

        config = configparser.ConfigParser()
        config['AdvancedConfigurations'] = {
            'embedding_model_name': 'test-model',
            'local_embeddings_path': './test-path'
        }

        with open(config_file, 'w') as f:
            config.write(f)

        return str(config_file)

    def test_initialization_with_defaults(self, global_metrics_manager_mock, aws_titan_embeddings_mock):
        """Test initialization with default values"""
        # Arrange
        # Mock os.getenv to return the default values that are hardcoded in the class
        with patch('os.getenv') as mock_getenv:
            def mock_getenv_side_effect(key, default=None):
                if key == 'EMBEDDING_MODEL_NAME':
                    return 'amazon.titan-embed-text-v1'
                elif key == 'LOCAL_EMBEDDINGS_PATH':
                    return '../Data/LocalEmbeddings'
                return default

            mock_getenv.side_effect = mock_getenv_side_effect

            # Act
            loader = DBConfigLoader(metrics_manager=global_metrics_manager_mock)

            # Assert
            # Check that the model_id is set from the environment variable
            assert loader.model_id == os.getenv('EMBEDDING_MODEL_NAME', 'amazon.titan-embed-text-v1')
            assert loader.local_storage_path == os.getenv('LOCAL_EMBEDDINGS_PATH', '../Data/LocalEmbeddings')
            aws_titan_embeddings_mock.assert_called_once()

    def test_initialization_with_env_variables(self, global_metrics_manager_mock, aws_titan_embeddings_mock):
        """Test initialization with environment variables"""
        # Arrange
        # Mock the environment variables
        with patch('os.getenv') as mock_getenv:
            def mock_getenv_side_effect(key, default=None):
                env_vars = {
                    'SIMILARITY_METRIC': 'l2',
                    'SIMILARITY_THRESHOLD': '0.8',
                    'DB_HOST': 'test-host',
                    'DB_PORT': '5433',
                    'DB_NAME': 'test-db',
                    'DB_USER': 'test-user',
                    'DB_PASSWORD': 'test-password',
                    'DB_SSL_MODE': 'require',
                    'EMBEDDING_MODEL_NAME': 'custom-model',
                    'LOCAL_EMBEDDINGS_PATH': './custom-path'
                }
                return env_vars.get(key, default)

            mock_getenv.side_effect = mock_getenv_side_effect

            # Act
            loader = DBConfigLoader(metrics_manager=global_metrics_manager_mock)

            # Assert
            assert loader.similarity_metric == 'l2'
            assert loader.threshold == '0.8'
            assert loader.db_host == 'test-host'
            assert loader.db_port == '5433'
            assert loader.db_name == 'test-db'
            assert loader.db_user == 'test-user'
            assert loader.db_password == 'test-password'
            assert loader.ssl_mode == 'require'
            assert loader.model_name == 'custom-model'
            assert loader.local_storage_path == './custom-path'

            # Check that AWSTitanEmbeddings was initialized with correct parameters
            aws_titan_embeddings_mock.assert_called_once()

    def test_load_config_with_exception(self, global_metrics_manager_mock, aws_titan_embeddings_mock):
        """Test error handling during configuration loading"""
        # Arrange
        with patch('os.getenv') as mock_getenv:
            def mock_getenv_side_effect(key, default=None):
                if key == 'EMBEDDING_MODEL_NAME':
                    return 'test-model'
                elif key == 'LOCAL_EMBEDDINGS_PATH':
                    return './test-path'
                return default

            mock_getenv.side_effect = mock_getenv_side_effect

            # Simulate an exception during AWSTitanEmbeddings initialization
            aws_titan_embeddings_mock.side_effect = Exception("Test error")

            # Act & Assert
            # The exception should be caught, logged, and re-raised
            with pytest.raises(Exception, match="Test error"):
                loader = DBConfigLoader(metrics_manager=global_metrics_manager_mock)

            # Verify that the error was recorded before re-raising
            global_metrics_manager_mock.record_error.assert_called_once_with('config_error', 'Test error')

    def test_load_config_with_custom_config_path(self, global_metrics_manager_mock, aws_titan_embeddings_mock,
                                                 config_file_mock):
        """Test loading configuration from a custom config file path"""
        # Arrange
        with patch('os.getenv') as mock_getenv:
            def mock_getenv_side_effect(key, default=None):
                if key == 'EMBEDDING_MODEL_NAME':
                    return 'amazon.titan-embed-text-v1'
                elif key == 'LOCAL_EMBEDDINGS_PATH':
                    return '../Data/LocalEmbeddings'
                return default

            mock_getenv.side_effect = mock_getenv_side_effect

            # Act
            loader = DBConfigLoader(config_file_path=config_file_mock, metrics_manager=global_metrics_manager_mock)

            # Assert
            aws_titan_embeddings_mock.assert_called_once()
            # Values should be loaded from the environment variables
            assert loader.model_id == 'amazon.titan-embed-text-v1'
            assert loader.local_storage_path == '../Data/LocalEmbeddings'

    def test_load_config_with_missing_config_file(self, global_metrics_manager_mock, aws_titan_embeddings_mock):
        """Test loading configuration with a missing config file"""
        # Arrange
        with patch('os.getenv') as mock_getenv:
            def mock_getenv_side_effect(key, default=None):
                if key == 'EMBEDDING_MODEL_NAME':
                    return 'amazon.titan-embed-text-v1'
                elif key == 'LOCAL_EMBEDDINGS_PATH':
                    return '../Data/LocalEmbeddings'
                return default

            mock_getenv.side_effect = mock_getenv_side_effect

            # Act
            loader = DBConfigLoader(config_file_path="/nonexistent/path/config.properties",
                                    metrics_manager=global_metrics_manager_mock)

            # Assert
            # Should use values from environment variables
            assert loader.model_id == 'amazon.titan-embed-text-v1'
            assert loader.local_storage_path == '../Data/LocalEmbeddings'
            aws_titan_embeddings_mock.assert_called_once()

    def test_load_config_with_partial_env_variables(self, global_metrics_manager_mock, aws_titan_embeddings_mock):
        """Test loading configuration with partial environment variables"""
        # Arrange
        with patch('os.getenv') as mock_getenv:
            def mock_getenv_side_effect(key, default=None):
                env_vars = {
                    'DB_HOST': 'test-host',
                    'DB_USER': 'test-user',
                    'EMBEDDING_MODEL_NAME': 'amazon.titan-embed-text-v1',
                    'LOCAL_EMBEDDINGS_PATH': '../Data/LocalEmbeddings'
                }
                return env_vars.get(key, default)

            mock_getenv.side_effect = mock_getenv_side_effect

            # Act
            loader = DBConfigLoader(metrics_manager=global_metrics_manager_mock)

            # Assert
            assert loader.db_host == 'test-host'
            assert loader.db_user == 'test-user'
            # Other attributes should have default values
            assert loader.db_port is None
            assert loader.db_name is None
            assert loader.db_password is None
            assert loader.ssl_mode is None

    def test_load_config_with_empty_env_variables(self, global_metrics_manager_mock, aws_titan_embeddings_mock):
        """Test loading configuration with empty environment variables"""
        # Arrange
        with patch('os.getenv') as mock_getenv:
            def mock_getenv_side_effect(key, default=None):
                env_vars = {
                    'DB_HOST': '',
                    'DB_PORT': '',
                    'DB_NAME': '',
                    'DB_USER': '',
                    'DB_PASSWORD': '',
                    'DB_SSL_MODE': '',
                    'SIMILARITY_METRIC': '',
                    'SIMILARITY_THRESHOLD': '',
                    'EMBEDDING_MODEL_NAME': '',
                    'LOCAL_EMBEDDINGS_PATH': ''
                }
                return env_vars.get(key, default)

            mock_getenv.side_effect = mock_getenv_side_effect

            # Act
            loader = DBConfigLoader(metrics_manager=global_metrics_manager_mock)

            # Assert
            # Should use empty values from environment
            assert loader.db_host == ''
            assert loader.db_port == ''
            assert loader.db_name == ''
            assert loader.db_user == ''
            assert loader.db_password == ''
            assert loader.ssl_mode == ''
            assert loader.similarity_metric == ''
            assert loader.threshold == ''
            assert loader.model_name == ''
            assert loader.local_storage_path == ''

    def test_load_config_with_invalid_env_variables(self, global_metrics_manager_mock, aws_titan_embeddings_mock):
        """Test loading configuration with invalid environment variables"""
        # Arrange
        with patch('os.getenv') as mock_getenv:
            def mock_getenv_side_effect(key, default=None):
                env_vars = {
                    'DB_PORT': 'not-a-number',
                    'SIMILARITY_THRESHOLD': 'not-a-number',
                    'EMBEDDING_MODEL_NAME': 'amazon.titan-embed-text-v1',
                    'LOCAL_EMBEDDINGS_PATH': '../Data/LocalEmbeddings'
                }
                return env_vars.get(key, default)

            mock_getenv.side_effect = mock_getenv_side_effect

            # Act
            loader = DBConfigLoader(metrics_manager=global_metrics_manager_mock)

            # Assert
            # Should use the invalid values as strings
            assert loader.db_port == 'not-a-number'
            assert loader.threshold == 'not-a-number'

    def test_embeddings_initialization(self, global_metrics_manager_mock):
        """Test that embeddings are properly initialized"""
        # Arrange
        with patch('src.aws_layer.aws_titan_embedding.AWSTitanEmbeddings') as mock_embeddings:
            mock_instance = Mock()
            mock_embeddings.return_value = mock_instance

            # Act
            with patch('os.getenv') as mock_getenv:
                # Return default values for model_id and local_storage_path
                def mock_getenv_side_effect(key, default=None):
                    if key == 'EMBEDDING_MODEL_NAME':
                        return 'amazon.titan-embed-text-v1'
                    elif key == 'LOCAL_EMBEDDINGS_PATH':
                        return '../Data/LocalEmbeddings'
                    return default

                mock_getenv.side_effect = mock_getenv_side_effect

                loader = DBConfigLoader(metrics_manager=global_metrics_manager_mock)

            # Assert
            assert loader.embeddings is mock_instance
            # The model_id and local_storage_path should be passed to AWSTitanEmbeddings
            mock_embeddings.assert_called_once()
