import pytest
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from pathlib import Path

# Import the module to test
from src.configuration_handler.config_loader import load_configuration


class TestConfigLoader:
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def valid_config_file(self, temp_dir):
        """Create a valid config file for testing"""
        config_path = os.path.join(temp_dir, 'valid_config.env')
        with open(config_path, 'w') as f:
            f.write("""
            # LLM Configuration
            LLM_FAMILY=AWS
            LLM_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0
            LLM_TEMPERATURE=0.05
            LLM_MAX_TOKENS=100000

            # AWS Credentials
            AWS_ACCESS_KEY_ID=test_key
            AWS_SECRET_ACCESS_KEY=test_secret
            AWS_REGION=us-east-1

            # Database Configuration
            DB_HOST=test-host
            DB_PORT=5432
            DB_NAME=testdb
            DB_USER=testuser
            DB_PASSWORD=testpass

            # Vector Database Configuration
            SIMILARITY_METRIC=cosine
            SIMILARITY_THRESHOLD=0.7
            """)
        return config_path

    @pytest.fixture
    def invalid_config_file(self, temp_dir):
        """Create an invalid config file for testing"""
        config_path = os.path.join(temp_dir, 'invalid_config.env')
        with open(config_path, 'w') as f:
            f.write("""
            # Invalid format
            LLM_FAMILY AWS
            DB_HOST: test-host
            INVALID LINE
            """)
        return config_path

    @pytest.fixture
    def empty_config_file(self, temp_dir):
        """Create an empty config file for testing"""
        config_path = os.path.join(temp_dir, 'empty_config.env')
        with open(config_path, 'w') as f:
            f.write("")
        return config_path

    def test_load_valid_configuration(self, valid_config_file):
        """Test loading a valid configuration file"""
        # Clear any existing environment variables that might interfere
        with patch.dict(os.environ, {}, clear=True):
            # Load the configuration
            result = load_configuration(valid_config_file)

            # Check that loading was successful
            assert result is True

            # Check that environment variables were set correctly
            assert os.environ.get('LLM_FAMILY') == 'AWS'
            assert os.environ.get('LLM_MODEL_ID') == 'anthropic.claude-3-sonnet-20240229-v1:0'
            assert os.environ.get('LLM_TEMPERATURE') == '0.05'
            assert os.environ.get('LLM_MAX_TOKENS') == '100000'
            assert os.environ.get('AWS_ACCESS_KEY_ID') == 'test_key'
            assert os.environ.get('AWS_SECRET_ACCESS_KEY') == 'test_secret'
            assert os.environ.get('DB_HOST') == 'test-host'
            assert os.environ.get('SIMILARITY_METRIC') == 'cosine'
            assert os.environ.get('SIMILARITY_THRESHOLD') == '0.7'

    def test_load_nonexistent_file(self):
        """Test loading a configuration file that doesn't exist"""
        # Try to load a non-existent file
        result = load_configuration('/path/to/nonexistent/file.env')

        # Should return False indicating failure
        assert result is False

    def test_load_invalid_configuration(self, invalid_config_file):
        """Test loading an invalid configuration file"""
        # Load the invalid configuration
        result = load_configuration(invalid_config_file)

        # The function returns False when it encounters invalid lines
        assert result is False

        # Check that no environment variables were set from the invalid file
        assert 'LLM_FAMILY' not in os.environ
        assert 'DB_HOST' not in os.environ

    def test_load_empty_configuration(self, empty_config_file):
        """Test loading an empty configuration file"""
        # Load the empty configuration
        result = load_configuration(empty_config_file)

        # Should return True as the file exists, even if empty
        assert result is True

    def test_environment_variable_precedence(self, valid_config_file):
        """Test that existing environment variables take precedence"""
        # Set environment variables before loading config
        with patch.dict(os.environ, {
            'LLM_FAMILY': 'EXISTING_VALUE',
            'DB_HOST': 'existing-host'
        }):
            # Load the configuration
            result = load_configuration(valid_config_file)

            # Check that existing values were preserved
            assert os.environ.get('LLM_FAMILY') == 'EXISTING_VALUE'
            assert os.environ.get('DB_HOST') == 'existing-host'

            # But new values should be loaded
            assert os.environ.get('LLM_TEMPERATURE') == '0.05'

    @patch('dotenv.load_dotenv')
    def test_dotenv_called_correctly(self, mock_load_dotenv, valid_config_file):
        """Test that dotenv.load_dotenv is called with the correct parameters"""
        # Make the mock return True
        mock_load_dotenv.return_value = True

        # Load the configuration
        result = load_configuration(valid_config_file)

        # Check that dotenv.load_dotenv was called correctly
        mock_load_dotenv.assert_called_once_with(dotenv_path=valid_config_file)

    def test_default_path(self):
        """Test loading from the default path when no path is provided"""
        # Mock the default path
        default_path = r"C:\Users\910464\OneDrive - Cognizant\Documents\GitHub\DevOrganization\StorySense_2025_AIG_Latest\.env"

        # Mock Path.exists to return True for the default path
        with patch('pathlib.Path.exists', return_value=True), \
                patch('dotenv.load_dotenv', return_value=True) as mock_load_dotenv:
            # Call load_configuration without a path
            result = load_configuration()

            # Check that dotenv.load_dotenv was called with the default path
            mock_load_dotenv.assert_called_once_with(dotenv_path=default_path)
            assert result is True

    def test_with_permission_error(self):
        """Test handling of permission errors"""
        # Mock Path.exists to return True
        # Mock load_dotenv to raise a PermissionError
        with patch('pathlib.Path.exists', return_value=True), \
                patch('dotenv.load_dotenv', side_effect=PermissionError("Permission denied")):
            # Load the configuration
            result = load_configuration("some_path")

            # Should return False indicating failure
            assert result is False

    def test_with_io_error(self):
        """Test handling of IO errors"""
        # Mock Path.exists to return True
        # Mock load_dotenv to raise an IOError
        with patch('pathlib.Path.exists', return_value=True), \
                patch('dotenv.load_dotenv', side_effect=IOError("IO Error")):
            # Load the configuration
            result = load_configuration("some_path")

            # Should return False indicating failure
            assert result is False

    def test_with_unicode_decode_error(self):
        """Test handling of UnicodeDecodeError"""
        # Mock Path.exists to return True
        # Mock load_dotenv to raise a UnicodeDecodeError
        with patch('pathlib.Path.exists', return_value=True), \
                patch('dotenv.load_dotenv', side_effect=UnicodeDecodeError("utf-8", b"test", 0, 1, "invalid")):
            # Load the configuration
            result = load_configuration("some_path")

            # Should return False indicating failure
            assert result is False
