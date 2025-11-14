import pytest
import os
import configparser
from unittest.mock import patch, mock_open
import tempfile
import shutil

# Import the relevant module that uses configIO.properties
from src.html_report.storysense_processor import StorySenseProcessor


class TestConfigIOProperties:
    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary directory for config files"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def configio_file_path(self, temp_config_dir):
        """Create a temporary configIO.properties file"""
        config_path = os.path.join(temp_config_dir, 'ConfigIO.properties')
        with open(config_path, 'w') as f:
            f.write("""[Input]
input_file_path=../Input/UserStories.xlsx
additional_context_path=../Input/AdditionalContext.xlsx

[Output]
output_file_path=../Output/StorySense
retrieval_context=../Output/RetrievalContext
num_context_retrieve=8
manual_test_type=Functional

[Processing]
batch_size=5
parallel_processing=false

[Context]
context_library_path=../Input/ContextLibrary
context_types=business_rules,requirements,documentation,policies,examples,glossary
""")
        return config_path

    def test_load_configio(self, configio_file_path, temp_config_dir):
        """Test that ConfigIO.properties can be loaded"""
        # Create a StorySenseProcessor with the test config path
        with patch.object(StorySenseProcessor, 'config_path', temp_config_dir):
            # Copy the test configIO file to the expected location
            shutil.copy(configio_file_path, os.path.join(temp_config_dir, 'ConfigIO.properties'))

            # Initialize processor which should load the config
            processor = StorySenseProcessor()

            # Check that config was loaded
            assert processor.config_parser_io is not None
            assert processor.config_parser_io.has_section('Input')
            assert processor.config_parser_io.has_section('Output')
            assert processor.config_parser_io.has_section('Processing')
            assert processor.config_parser_io.has_section('Context')

    def test_configio_values(self, configio_file_path, temp_config_dir):
        """Test that ConfigIO.properties values are correctly loaded"""
        # Create a StorySenseProcessor with the test config path
        with patch.object(StorySenseProcessor, 'config_path', temp_config_dir):
            # Copy the test configIO file to the expected location
            shutil.copy(configio_file_path, os.path.join(temp_config_dir, 'ConfigIO.properties'))

            # Initialize processor which should load the config
            processor = StorySenseProcessor()

            # Check specific values
            assert processor.config_parser_io.get('Input', 'input_file_path') == '../Input/UserStories.xlsx'
            assert processor.config_parser_io.get('Output', 'num_context_retrieve') == '8'
            assert processor.config_parser_io.get('Processing', 'batch_size') == '5'
            assert processor.config_parser_io.get('Processing', 'parallel_processing') == 'false'
            assert processor.config_parser_io.get('Context',
                                                  'context_types') == 'business_rules,requirements,documentation,policies,examples,glossary'

    def test_create_default_configio(self, temp_config_dir):
        """Test that default ConfigIO.properties is created if it doesn't exist"""
        # Create a StorySenseProcessor with the test config path
        with patch.object(StorySenseProcessor, 'config_path', temp_config_dir):
            # Initialize processor which should create default config
            processor = StorySenseProcessor()

            # Check that default config was created
            default_config_path = os.path.join(temp_config_dir, 'ConfigIO.properties')
            assert os.path.exists(default_config_path)

            # Parse the created config
            config = configparser.ConfigParser()
            config.read(default_config_path)

            # Check default values
            assert config.has_section('Input')
            assert config.get('Input', 'input_file_path') == '../Input/UserStories.xlsx'
            assert config.has_section('Output')
            assert config.get('Output', 'num_context_retrieve') == '8'
            assert config.has_section('Processing')
            assert config.get('Processing', 'batch_size') == '5'

    def test_configio_path_resolution(self, configio_file_path, temp_config_dir):
        """Test that relative paths in ConfigIO.properties are resolved correctly"""
        # Create a StorySenseProcessor with the test config path
        with patch.object(StorySenseProcessor, 'config_path', temp_config_dir):
            # Copy the test configIO file to the expected location
            shutil.copy(configio_file_path, os.path.join(temp_config_dir, 'ConfigIO.properties'))

            # Initialize processor with specific input paths
            processor = StorySenseProcessor(user_stories_path="custom_path.xlsx")

            # Check that paths are resolved correctly
            assert processor.input_file_path == "custom_path.xlsx"  # Should use provided path
            assert os.path.isabs(processor.input_file_path)  # Should be absolute path

            # Default path from config should be used when not provided
            assert processor.additional_context_path is not None
            assert os.path.isabs(processor.additional_context_path)

    def test_batch_size_from_config(self, configio_file_path, temp_config_dir):
        """Test that batch size is correctly read from config"""
        # Create a modified config with different batch size
        modified_config_path = os.path.join(temp_config_dir, 'ConfigIO.properties')
        with open(modified_config_path, 'w') as f:
            f.write("""[Input]
input_file_path=../Input/UserStories.xlsx
additional_context_path=../Input/AdditionalContext.xlsx

[Output]
output_file_path=../Output/StorySense
retrieval_context=../Output/RetrievalContext
num_context_retrieve=8
manual_test_type=Functional

[Processing]
batch_size=10
parallel_processing=true

[Context]
context_library_path=../Input/ContextLibrary
context_types=business_rules,requirements,documentation,policies,examples,glossary
""")

        # Create a StorySenseProcessor with the test config path
        with patch.object(StorySenseProcessor, 'config_path', temp_config_dir):
            # Initialize processor which should load the config
            processor = StorySenseProcessor()

            # Mock the analyze_stories_in_batches method to capture arguments
            with patch.object(processor, 'analyze_stories_in_batches') as mock_analyze:
                # Call analyze_stories which should use config values
                processor.analyze_stories(None, None)

                # Check that batch_size=10 was passed to analyze_stories_in_batches
                mock_analyze.assert_called_once()
                args, kwargs = mock_analyze.call_args
                assert kwargs.get('batch_size') == 10
                assert kwargs.get('parallel') is True

    def test_missing_configio_section(self, temp_config_dir):
        """Test behavior when a section is missing from ConfigIO.properties"""
        # Create a config file with missing sections
        incomplete_config_path = os.path.join(temp_config_dir, 'ConfigIO.properties')
        with open(incomplete_config_path, 'w') as f:
            f.write("""[Input]
input_file_path=../Input/UserStories.xlsx
additional_context_path=../Input/AdditionalContext.xlsx

# Missing Output section

[Processing]
batch_size=5
parallel_processing=false
""")

        # Create a StorySenseProcessor with the test config path
        with patch.object(StorySenseProcessor, 'config_path', temp_config_dir):
            # Initialize processor which should handle missing sections
            processor = StorySenseProcessor()

            # Should use default values for missing sections
            assert processor.num_context_retrieve == 8  # Default value
