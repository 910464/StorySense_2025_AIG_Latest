import pytest
import os
import configparser
from unittest.mock import patch, mock_open
import tempfile
import shutil

# Import the relevant module that uses config.properties
from src.configuration_handler.config_loader import load_configuration
from src.html_report.storysense_processor import StorySenseProcessor


class TestConfigProperties:
    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary directory for config files"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def config_file_path(self, temp_config_dir):
        """Create a temporary config.properties file"""
        config_path = os.path.join(temp_config_dir, 'Config.properties')
        with open(config_path, 'w') as f:
            f.write("""[LLM]
LLM_FAMILY=AWS
TEMPERATURE=0.05

[AdvancedConfigurations]
embedding_model_name=amazon.titan-embed-text-v1
embedding_model_path=../Data/ExternalEmbeddingModel
external_model_threshold=0.7
default_model_threshold=0.50
local_embeddings_path=../Data/LocalEmbeddings

[Guardrails]
guardrail_id=3xr1mcliy9u6
region=us-east-1
description=Default guardrail for general use

[AWS]
MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0
MAX_TOKENS=150000
""")
        return config_path

    def test_load_configuration(self, config_file_path):
        """Test that configuration can be loaded from file"""
        # Test loading the configuration
        result = load_configuration(config_file_path)
        assert result is True

    def test_config_values(self, config_file_path):
        """Test that configuration values are correctly loaded"""
        # Load the configuration
        load_configuration(config_file_path)

        # Check environment variables are set correctly
        assert os.getenv('LLM_FAMILY') == 'AWS'
        assert os.getenv('LLM_TEMPERATURE') == '0.05'
        assert os.getenv('EMBEDDING_MODEL_NAME') == 'amazon.titan-embed-text-v1'
        assert os.getenv('EXTERNAL_MODEL_THRESHOLD') == '0.7'
        assert os.getenv('DEFAULT_MODEL_THRESHOLD') == '0.50'
        assert os.getenv('LOCAL_EMBEDDINGS_PATH') == '../Data/LocalEmbeddings'
        assert os.getenv('LLM_MODEL_ID') == 'anthropic.claude-3-sonnet-20240229-v1:0'
        assert os.getenv('LLM_MAX_TOKENS') == '150000'

    def test_create_default_config(self, temp_config_dir):
        """Test that default config is created if it doesn't exist"""
        # Set up a path where config doesn't exist
        config_path = os.path.join(temp_config_dir, 'Config')

        # Patch the config path in StorySenseProcessor
        with patch.object(StorySenseProcessor, 'config_path', config_path):
            # Initialize processor which should create default config
            processor = StorySenseProcessor()

            # Check that default config was created
            default_config_path = os.path.join(config_path, 'Config.properties')
            assert os.path.exists(default_config_path)

            # Parse the created config
            config = configparser.ConfigParser()
            config.read(default_config_path)

            # Check default values
            assert config.has_section('LLM')
            assert config.get('LLM', 'LLM_FAMILY') == 'AWS'
            assert config.has_section('AdvancedConfigurations')
            assert config.get('AdvancedConfigurations', 'embedding_model_name') == 'amazon.titan-embed-text-v1'

    def test_config_override_with_env_vars(self, config_file_path):
        """Test that environment variables override config file values"""
        # Set environment variables
        with patch.dict(os.environ, {
            'LLM_FAMILY': 'CUSTOM',
            'LLM_TEMPERATURE': '0.8',
            'EMBEDDING_MODEL_NAME': 'custom-model'
        }):
            # Load configuration
            load_configuration(config_file_path)

            # Check that environment variables take precedence
            assert os.getenv('LLM_FAMILY') == 'CUSTOM'
            assert os.getenv('LLM_TEMPERATURE') == '0.8'
            assert os.getenv('EMBEDDING_MODEL_NAME') == 'custom-model'

            # But other values should still be loaded from file
            assert os.getenv('EXTERNAL_MODEL_THRESHOLD') == '0.7'

    def test_missing_config_file(self):
        """Test behavior when config file is missing"""
        # Try to load a non-existent config file
        result = load_configuration('/nonexistent/path/config.properties')

        # Should return False indicating failure
        assert result is False

    def test_invalid_config_file(self, temp_config_dir):
        """Test behavior with invalid config file"""
        # Create an invalid config file
        invalid_config_path = os.path.join(temp_config_dir, 'invalid.properties')
        with open(invalid_config_path, 'w') as f:
            f.write("This is not a valid config file")

        # Try to load the invalid config
        result = load_configuration(invalid_config_path)

        # Should return False or handle the error gracefully
        assert result is False