import pytest
import sys
import os
import configparser
import pandas as pd
from unittest.mock import Mock, patch, MagicMock, mock_open, call
from pathlib import Path
import tempfile
import shutil


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment and mock all dependencies before each test"""
    # Mock all problematic modules that cause import errors
    mock_modules = {
        'main_service_router': Mock(story_sense_router=Mock()),
        'src.aws_layer.aws_titan_embedding': Mock(AWSTitanEmbeddings=Mock()),
        'src.metrics.metrics_manager': Mock(MetricsManager=Mock()),
        'src.html_report.storysense_processor': Mock(StorySenseProcessor=Mock()),
        'src.context_handler.context_file_handler.enhanced_context_processor': Mock(EnhancedContextProcessor=Mock()),
        'src.prompt_layer.storysense_analyzer': Mock(StorySenseAnalyzer=Mock()),
        'src.llm_layer.model_manual_test_llm': Mock(LLM=Mock()),
        'langchain.prompts.prompt': Mock(PromptTemplate=Mock()),
        'langchain_core.prompts.prompt': Mock(PromptTemplate=Mock()),
        'transformers': Mock(),
        'torch': Mock(),
        'botocore': Mock(args=Mock()),
        'colorama': Mock(Fore=Mock(GREEN='', CYAN='', RESET=''), Style=Mock(RESET_ALL='')),
    }
    
    # Add all mock modules to sys.modules
    for module_name, mock_module in mock_modules.items():
        sys.modules[module_name] = mock_module

    yield

    # Cleanup - remove mocked modules
    for module_name in mock_modules.keys():
        if module_name in sys.modules:
            del sys.modules[module_name]


@pytest.fixture
def mock_config_files():
    """Mock configuration file operations"""
    config_content = """
[Input]
input_file_path = ../Input/UserStories.xlsx
additional_context_path = ../Input/AdditionalContext.xlsx

[Output]
output_file_path = ../Output/StorySense
retrieval_context = ../Output/RetrievalContext
num_context_retrieve = 8

[Processing]
batch_size = 5
parallel_processing = false

[Context]
context_library_path = ../Input/ContextLibrary
"""

    with patch('os.path.exists', return_value=True), \
            patch('builtins.open', mock_open(read_data=config_content)):
        yield


@pytest.fixture
def mock_all_dependencies():
    """Mock all external dependencies"""
    with patch('os.makedirs'), \
            patch('os.path.exists', return_value=True), \
            patch('builtins.open', mock_open(read_data="[Input]\ninput_file_path=../Input/UserStories.xlsx\n")):
        yield


class TestStorySenseGeneratorInitialization:
    """Test StorySenseGenerator initialization"""

    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('os.path.isabs')
    @patch('os.path.abspath')
    @patch('os.path.basename')
    @patch('builtins.open', new_callable=mock_open,
           read_data="[Input]\ninput_file_path=../Input/UserStories.xlsx\nadditional_context_path=../Input/AdditionalContext.xlsx\n[Output]\noutput_file_path=../Output/StorySense\n[LLM]\nLLM_FAMILY=AWS\n")
    def test_initialization_with_defaults(self, mock_file, mock_basename, mock_abspath, mock_isabs, mock_exists, mock_makedirs):
        """Test initialization with default parameters"""
        mock_exists.return_value = True
        mock_isabs.return_value = False  # Paths are relative
        mock_abspath.side_effect = lambda x: f"/absolute{x}"
        mock_basename.side_effect = lambda x: os.path.basename(x)

        # Clear any existing modules from sys.modules that might cause conflicts
        modules_to_clear = [name for name in sys.modules.keys() if 'StorySenseGenerator' in name]
        for module in modules_to_clear:
            del sys.modules[module]

        from src.interface_layer.StorySenseGenerator import StorySenseGenerator

        ssg = StorySenseGenerator()

        assert ssg is not None
        assert ssg.config_path == '../Config'
        assert mock_makedirs.called

    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open,
           read_data="[Input]\ninput_file_path=../Input/UserStories.xlsx\nadditional_context_path=../Input/AdditionalContext.xlsx\n[Output]\noutput_file_path=../Output/StorySense\n[LLM]\nLLM_FAMILY=AWS\n")
    def test_initialization_with_user_stories_path(self, mock_file, mock_exists, mock_makedirs):
        """Test initialization with user stories path"""
        mock_exists.return_value = True

        from src.interface_layer.StorySenseGenerator import StorySenseGenerator

        ssg = StorySenseGenerator(user_stories_path='../Input/test_stories.xlsx')

        assert '../Input/test_stories.xlsx' in ssg.input_file_path

    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open,
           read_data="[Input]\ninput_file_path=../Input/UserStories.xlsx\nadditional_context_path=../Input/AdditionalContext.xlsx\n[Output]\noutput_file_path=../Output/StorySense\n[LLM]\nLLM_FAMILY=AWS\n")
    def test_initialization_with_context_path(self, mock_file, mock_exists, mock_makedirs):
        """Test initialization with context path"""
        mock_exists.return_value = True

        from src.interface_layer.StorySenseGenerator import StorySenseGenerator

        ssg = StorySenseGenerator(additional_context_path='../Input/test_context.xlsx')

        assert '../Input/test_context.xlsx' in ssg.additional_context_path

    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open,
           read_data="[Input]\ninput_file_path=../Input/UserStories.xlsx\nadditional_context_path=../Input/AdditionalContext.xlsx\n[Output]\noutput_file_path=../Output/StorySense\n[LLM]\nLLM_FAMILY=AWS\n")
    def test_initialization_with_both_paths(self, mock_file, mock_exists, mock_makedirs):
        """Test initialization with both paths"""
        mock_exists.return_value = True

        from src.interface_layer.StorySenseGenerator import StorySenseGenerator

        ssg = StorySenseGenerator(
            user_stories_path='../Input/test_stories.xlsx',
            additional_context_path='../Input/test_context.xlsx'
        )

        assert '../Input/test_stories.xlsx' in ssg.input_file_path
        assert '../Input/test_context.xlsx' in ssg.additional_context_path

    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open,
           read_data="[Input]\ninput_file_path=../Input/UserStories.xlsx\n[Output]\noutput_file_path=../Output/StorySense\n[LLM]\nLLM_FAMILY=AWS\n")
    def test_directory_creation(self, mock_file, mock_exists, mock_makedirs):
        """Test that necessary directories are created"""
        mock_exists.return_value = True

        from src.interface_layer.StorySenseGenerator import StorySenseGenerator

        ssg = StorySenseGenerator()

        # Verify directories were created
        assert mock_makedirs.call_count >= 5

    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open,
           read_data="[Input]\ninput_file_path=../Input/UserStories.xlsx\n[Output]\noutput_file_path=../Output/StorySense\n[LLM]\nLLM_FAMILY=AWS\n")
    def test_absolute_path_conversion(self, mock_file, mock_exists, mock_makedirs):
        """Test that relative paths are converted to absolute"""
        mock_exists.return_value = True

        from src.interface_layer.StorySenseGenerator import StorySenseGenerator

        ssg = StorySenseGenerator(user_stories_path='../Input/stories.xlsx')

        assert os.path.isabs(ssg.input_file_path)

    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open,
           read_data="[Input]\ninput_file_path=../Input/UserStories.xlsx\n[Output]\noutput_file_path=../Output/StorySense\n[LLM]\nLLM_FAMILY=AWS\n")
    def test_filename_extraction(self, mock_file, mock_exists, mock_makedirs):
        """Test that filenames are extracted correctly"""
        mock_exists.return_value = True

        from src.interface_layer.StorySenseGenerator import StorySenseGenerator

        ssg = StorySenseGenerator(user_stories_path='../Input/test_stories.xlsx')

        assert ssg.input_filename == 'test_stories.xlsx'


class TestCreateDefaultConfig:
    """Test create_default_config method"""

    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open,
           read_data="[Input]\ninput_file_path=../Input/UserStories.xlsx\n[Output]\noutput_file_path=../Output/StorySense\n[LLM]\nLLM_FAMILY=AWS\n")
    def test_create_default_config(self, mock_file, mock_exists, mock_makedirs):
        """Test creating default config file"""
        mock_exists.return_value = True

        from src.interface_layer.StorySenseGenerator import StorySenseGenerator

        ssg = StorySenseGenerator()

        with patch('builtins.open', mock_open()) as mock_config_file:
            ssg.create_default_config()

            # Verify file was opened for writing
            assert mock_config_file.called

    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open,
           read_data="[Input]\ninput_file_path=../Input/UserStories.xlsx\n[Output]\noutput_file_path=../Output/StorySense\n[LLM]\nLLM_FAMILY=AWS\n")
    def test_config_directory_creation(self, mock_file, mock_exists, mock_makedirs):
        """Test that config directory is created"""
        mock_exists.return_value = True

        from src.interface_layer.StorySenseGenerator import StorySenseGenerator

        ssg = StorySenseGenerator()
        ssg.create_default_config()

        # Verify directory creation was called
        assert mock_makedirs.called


class TestCreateDefaultConfigIO:
    """Test create_default_config_io method"""

    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open,
           read_data="[Input]\ninput_file_path=../Input/UserStories.xlsx\n[Output]\noutput_file_path=../Output/StorySense\n[LLM]\nLLM_FAMILY=AWS\n")
    def test_create_default_config_io(self, mock_file, mock_exists, mock_makedirs):
        """Test creating default ConfigIO file"""
        mock_exists.return_value = True

        from src.interface_layer.StorySenseGenerator import StorySenseGenerator

        ssg = StorySenseGenerator()

        with patch('builtins.open', mock_open()) as mock_config_file:
            ssg.create_default_config_io()

            # Verify file was opened for writing
            assert mock_config_file.called


class TestProcessContextLibrary:
    """Test process_context_library method"""

    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open,
           read_data="[Input]\ninput_file_path=../Input/UserStories.xlsx\n[Output]\noutput_file_path=../Output/StorySense\n[Context]\ncontext_library_path=../Input/ContextLibrary\n[LLM]\nLLM_FAMILY=AWS\n")
    @patch('src.context_handler.context_file_handler.enhanced_context_processor.EnhancedContextProcessor')
    @patch('src.metrics.metrics_manager.MetricsManager')
    def test_process_context_library_default_folder(self, mock_metrics, mock_processor,
                                                    mock_file, mock_exists, mock_makedirs):
        """Test processing context library with default folder"""
        mock_exists.return_value = True

        mock_processor_instance = Mock()
        mock_processor_instance.process_all_context_files.return_value = {
            'processed_files': 5,
            'skipped_files': 2,
            'failed_files': 0
        }
        mock_processor.return_value = mock_processor_instance

        from src.interface_layer.StorySenseGenerator import StorySenseGenerator
        ssg = StorySenseGenerator()

        result = ssg.process_context_library()

        assert result['processed_files'] == 5
        mock_processor.assert_called_once()

    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open,
           read_data="[Input]\ninput_file_path=../Input/UserStories.xlsx\n[Output]\noutput_file_path=../Output/StorySense\n[Context]\ncontext_library_path=../Input/ContextLibrary\n[LLM]\nLLM_FAMILY=AWS\n")
    @patch('src.context_handler.context_file_handler.enhanced_context_processor.EnhancedContextProcessor')
    @patch('src.metrics.metrics_manager.MetricsManager')
    def test_process_context_library_custom_folder(self, mock_metrics, mock_processor,
                                                   mock_file, mock_exists, mock_makedirs):
        """Test processing context library with custom folder"""
        mock_exists.return_value = True

        mock_processor_instance = Mock()
        mock_processor_instance.process_all_context_files.return_value = {
            'processed_files': 3
        }
        mock_processor.return_value = mock_processor_instance

        from src.interface_layer.StorySenseGenerator import StorySenseGenerator
        ssg = StorySenseGenerator()

        result = ssg.process_context_library(context_folder='../Input/CustomContext')

        assert result['processed_files'] == 3

    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open,
           read_data="[Input]\ninput_file_path=../Input/UserStories.xlsx\n[Output]\noutput_file_path=../Output/StorySense\n[Context]\ncontext_library_path=../Input/ContextLibrary\n[LLM]\nLLM_FAMILY=AWS\n")
    @patch('src.context_handler.context_file_handler.enhanced_context_processor.EnhancedContextProcessor')
    @patch('src.metrics.metrics_manager.MetricsManager')
    def test_process_context_library_force_reprocess(self, mock_metrics, mock_processor,
                                                     mock_file, mock_exists, mock_makedirs):
        """Test force reprocessing of context library"""
        mock_exists.return_value = True

        mock_processor_instance = Mock()
        mock_processor_instance.process_all_context_files.return_value = {
            'processed_files': 10
        }
        mock_processor.return_value = mock_processor_instance

        from src.interface_layer.StorySenseGenerator import StorySenseGenerator
        ssg = StorySenseGenerator()

        result = ssg.process_context_library(force_reprocess=True)

        mock_processor_instance.process_all_context_files.assert_called_once_with(force_reprocess=True)


class TestProcessUserStories:
    """Test process_user_stories method"""

    @pytest.fixture
    def sample_excel_file(self, sample_user_stories):
        """Create a sample Excel file"""
        temp_file = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)
        sample_user_stories.to_excel(temp_file.name, index=False)
        yield temp_file.name
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open,
           read_data="[Input]\ninput_file_path=../Input/UserStories.xlsx\n[Output]\noutput_file_path=../Output/StorySense\n[Processing]\nbatch_size=5\n[LLM]\nLLM_FAMILY=AWS\n")
    @patch('src.html_report.storysense_processor.StorySenseProcessor')
    @patch('pandas.read_excel')
    def test_process_user_stories_excel_file(self, mock_read_excel, mock_processor,
                                             mock_file, mock_exists, mock_makedirs,
                                             sample_user_stories, sample_excel_file):
        """Test processing Excel file"""
        mock_exists.return_value = True
        mock_read_excel.return_value = sample_user_stories

        mock_processor_instance = Mock()
        mock_processor_instance.analyze_stories_in_batches = Mock(return_value=[])
        mock_processor.return_value = mock_processor_instance

        from src.interface_layer.StorySenseGenerator import StorySenseGenerator
        ssg = StorySenseGenerator(user_stories_path=sample_excel_file)

        ssg.process_user_stories()

        mock_processor_instance.analyze_stories_in_batches.assert_called_once()

    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open,
           read_data="[Input]\ninput_file_path=../Input/UserStories.xlsx\n[Output]\noutput_file_path=../Output/StorySense\n[Processing]\nbatch_size=5\n[LLM]\nLLM_FAMILY=AWS\n")
    @patch('src.html_report.storysense_processor.StorySenseProcessor')
    @patch('pandas.read_csv')
    def test_process_user_stories_csv_file(self, mock_read_csv, mock_processor,
                                           mock_file, mock_exists, mock_makedirs,
                                           sample_user_stories):
        """Test processing CSV file"""
        mock_exists.return_value = True
        mock_read_csv.return_value = sample_user_stories

        temp_csv = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
        sample_user_stories.to_csv(temp_csv.name, index=False)

        mock_processor_instance = Mock()
        mock_processor_instance.analyze_stories_in_batches = Mock(return_value=[])
        mock_processor.return_value = mock_processor_instance

        from src.interface_layer.StorySenseGenerator import StorySenseGenerator
        ssg = StorySenseGenerator(user_stories_path=temp_csv.name)

        ssg.process_user_stories()

        mock_processor_instance.analyze_stories_in_batches.assert_called_once()

        os.unlink(temp_csv.name)

    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open,
           read_data="[Input]\ninput_file_path=../Input/UserStories.xlsx\n[Output]\noutput_file_path=../Output/StorySense\n[LLM]\nLLM_FAMILY=AWS\n")
    def test_process_user_stories_file_not_found(self, mock_file, mock_exists, mock_makedirs):
        """Test handling of missing file"""
        mock_exists.side_effect = lambda path: False if 'nonexistent' in path else True

        from src.interface_layer.StorySenseGenerator import StorySenseGenerator
        ssg = StorySenseGenerator(user_stories_path='/nonexistent/file.xlsx')

        with pytest.raises(SystemExit):
            ssg.process_user_stories()

    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open,
           read_data="[Input]\ninput_file_path=../Input/UserStories.xlsx\n[Output]\noutput_file_path=../Output/StorySense\n[LLM]\nLLM_FAMILY=AWS\n")
    def test_process_user_stories_unsupported_format(self, mock_file, mock_exists, mock_makedirs):
        """Test handling of unsupported file format"""
        mock_exists.return_value = True

        from src.interface_layer.StorySenseGenerator import StorySenseGenerator
        ssg = StorySenseGenerator(user_stories_path='../Input/test.txt')

        with pytest.raises(SystemExit):
            ssg.process_user_stories()

    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open,
           read_data="[Input]\ninput_file_path=../Input/UserStories.xlsx\n[Output]\noutput_file_path=../Output/StorySense\n[Processing]\nbatch_size=5\n[LLM]\nLLM_FAMILY=AWS\n")
    @patch('src.html_report.storysense_processor.StorySenseProcessor')
    @patch('pandas.read_excel')
    def test_process_user_stories_missing_columns(self, mock_read_excel, mock_processor,
                                                  mock_file, mock_exists, mock_makedirs):
        """Test handling of missing required columns"""
        mock_exists.return_value = True

        # DataFrame missing AcceptanceCriteria
        mock_read_excel.return_value = pd.DataFrame({
            'ID': ['US001'],
            'Description': ['Test']
        })

        temp_file = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)

        from src.interface_layer.StorySenseGenerator import StorySenseGenerator
        ssg = StorySenseGenerator(user_stories_path=temp_file.name)

        with pytest.raises(SystemExit):
            ssg.process_user_stories()

        os.unlink(temp_file.name)

    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open,
           read_data="[Input]\ninput_file_path=../Input/UserStories.xlsx\n[Output]\noutput_file_path=../Output/StorySense\n[Processing]\nbatch_size=5\n[LLM]\nLLM_FAMILY=AWS\n")
    @patch('src.html_report.storysense_processor.StorySenseProcessor')
    @patch('pandas.read_excel')
    def test_process_user_stories_with_batch_size(self, mock_read_excel, mock_processor,
                                                  mock_file, mock_exists, mock_makedirs,
                                                  sample_user_stories):
        """Test processing with custom batch size"""
        mock_exists.return_value = True
        mock_read_excel.return_value = sample_user_stories

        mock_processor_instance = Mock()
        mock_processor_instance.analyze_stories_in_batches = Mock(return_value=[])
        mock_processor.return_value = mock_processor_instance

        temp_file = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)

        from src.interface_layer.StorySenseGenerator import StorySenseGenerator
        ssg = StorySenseGenerator(user_stories_path=temp_file.name)

        ssg.process_user_stories(batch_size=10)

        # Verify batch_size was passed
        call_args = mock_processor_instance.analyze_stories_in_batches.call_args
        assert call_args[0][2] == 10

        os.unlink(temp_file.name)

    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open,
           read_data="[Input]\ninput_file_path=../Input/UserStories.xlsx\n[Output]\noutput_file_path=../Output/StorySense\n[Processing]\nbatch_size=5\n[LLM]\nLLM_FAMILY=AWS\n")
    @patch('src.html_report.storysense_processor.StorySenseProcessor')
    @patch('pandas.read_excel')
    def test_process_user_stories_with_parallel(self, mock_read_excel, mock_processor,
                                                mock_file, mock_exists, mock_makedirs,
                                                sample_user_stories):
        """Test processing with parallel execution"""
        mock_exists.return_value = True
        mock_read_excel.return_value = sample_user_stories

        mock_processor_instance = Mock()
        mock_processor_instance.analyze_stories_in_batches = Mock(return_value=[])
        mock_processor.return_value = mock_processor_instance

        temp_file = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)

        from src.interface_layer.StorySenseGenerator import StorySenseGenerator
        ssg = StorySenseGenerator(user_stories_path=temp_file.name)

        ssg.process_user_stories(parallel=True)

        # Verify parallel was passed
        call_args = mock_processor_instance.analyze_stories_in_batches.call_args
        assert call_args[0][3] is True

        os.unlink(temp_file.name)


class TestParseArguments:
    """Test parse_arguments function"""

    def test_parse_arguments_no_args(self):
        """Test parsing with no arguments"""
        from src.interface_layer.StorySenseGenerator import parse_arguments

        with patch('sys.argv', ['script.py']):
            result = parse_arguments()

            assert result[0] is None  # user_stories_path
            assert result[1] is None  # additional_context_path
            assert result[2] == 5  # batch_size
            assert result[3] is False  # parallel

    def test_parse_arguments_with_user_stories(self):
        """Test parsing with user stories argument"""
        from src.interface_layer.StorySenseGenerator import parse_arguments

        with patch('sys.argv', ['script.py', '--user-stories', '../Input/stories.xlsx']):
            result = parse_arguments()

            assert result[0] == '../Input/stories.xlsx'

    def test_parse_arguments_with_context(self):
        """Test parsing with context argument"""
        from src.interface_layer.StorySenseGenerator import parse_arguments

        with patch('sys.argv', ['script.py', '--context', '../Input/context.xlsx']):
            result = parse_arguments()

            assert result[1] == '../Input/context.xlsx'

    def test_parse_arguments_with_batch_size(self):
        """Test parsing with batch size argument"""
        from src.interface_layer.StorySenseGenerator import parse_arguments

        with patch('sys.argv', ['script.py', '--batch-size', '10']):
            result = parse_arguments()

            assert result[2] == 10

    def test_parse_arguments_with_parallel(self):
        """Test parsing with parallel flag"""
        from src.interface_layer.StorySenseGenerator import parse_arguments

        with patch('sys.argv', ['script.py', '--parallel']):
            result = parse_arguments()

            assert result[3] is True

    def test_parse_arguments_manage_context(self):
        """Test parsing with manage context flag"""
        from src.interface_layer.StorySenseGenerator import parse_arguments

        with patch('sys.argv', ['script.py', '--manage-context']):
            result = parse_arguments()

            assert result[4].manage_context is True

    def test_parse_arguments_process_context(self):
        """Test parsing with process context flag"""
        from src.interface_layer.StorySenseGenerator import parse_arguments

        with patch('sys.argv', ['script.py', '--process-context']):
            result = parse_arguments()

            assert result[4].process_context is True

    def test_parse_arguments_context_folder(self):
        """Test parsing with context folder argument"""
        from src.interface_layer.StorySenseGenerator import parse_arguments

        with patch('sys.argv', ['script.py', '--context-folder', '../Input/CustomContext']):
            result = parse_arguments()

            assert result[5] == '../Input/CustomContext'

    def test_parse_arguments_force_reprocess(self):
        """Test parsing with force reprocess flag"""
        from src.interface_layer.StorySenseGenerator import parse_arguments

        with patch('sys.argv', ['script.py', '--force-reprocess']):
            result = parse_arguments()

            assert result[6] is True

    def test_parse_arguments_positional_user_stories(self):
        """Test parsing with positional user stories argument"""
        from src.interface_layer.StorySenseGenerator import parse_arguments

        with patch('sys.argv', ['script.py', '../Input/stories.xlsx']):
            result = parse_arguments()

            assert result[0] == '../Input/stories.xlsx'

    def test_parse_arguments_short_flags(self):
        """Test short flag arguments"""
        from src.interface_layer.StorySenseGenerator import parse_arguments

        with patch('sys.argv', ['script.py', '-u', 'stories.xlsx', '-c', 'context.xlsx', '-b', '10', '-p']):
            result = parse_arguments()

            assert result[0] == 'stories.xlsx'
            assert result[1] == 'context.xlsx'
            assert result[2] == 10
            assert result[3] is True


class TestStorySenseGeneratorEdgeCases:
    """Test edge cases and error conditions"""

    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open,
           read_data="[Input]\ninput_file_path=../Input/UserStories.xlsx\n[Output]\noutput_file_path=../Output/StorySense\n[Processing]\nbatch_size=5\n[LLM]\nLLM_FAMILY=AWS\n")
    @patch('src.html_report.storysense_processor.StorySenseProcessor')
    @patch('pandas.read_excel')
    def test_empty_user_stories_file(self, mock_read_excel, mock_processor,
                                     mock_file, mock_exists, mock_makedirs):
        """Test processing empty user stories file"""
        mock_exists.return_value = True
        mock_read_excel.return_value = pd.DataFrame()

        temp_file = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)

        from src.interface_layer.StorySenseGenerator import StorySenseGenerator
        ssg = StorySenseGenerator(user_stories_path=temp_file.name)

        with pytest.raises(SystemExit):
            ssg.process_user_stories()

        os.unlink(temp_file.name)

    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open,
           read_data="[Input]\ninput_file_path=../Input/UserStories.xlsx\n[Output]\noutput_file_path=../Output/StorySense\n[LLM]\nLLM_FAMILY=AWS\n")
    def test_corrupted_excel_file(self, mock_file, mock_exists, mock_makedirs):
        """Test processing corrupted Excel file"""
        mock_exists.return_value = True

        temp_file = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)
        temp_file.write(b'corrupted data')
        temp_file.close()

        from src.interface_layer.StorySenseGenerator import StorySenseGenerator
        ssg = StorySenseGenerator(user_stories_path=temp_file.name)

        with pytest.raises(SystemExit):
            ssg.process_user_stories()

        os.unlink(temp_file.name)

    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open,
           read_data="[Input]\ninput_file_path=../Input/UserStories.xlsx\n[Output]\noutput_file_path=../Output/StorySense\n[Processing]\nbatch_size=5\n[LLM]\nLLM_FAMILY=AWS\n")
    @patch('src.html_report.storysense_processor.StorySenseProcessor')
    @patch('pandas.read_excel')
    def test_unicode_in_user_stories(self, mock_read_excel, mock_processor,
                                     mock_file, mock_exists, mock_makedirs):
        """Test processing user stories with Unicode characters"""
        mock_exists.return_value = True

        df = pd.DataFrame({
            'ID': ['US001'],
            'Description': ['Test with émojis 🎉 and ñoñ-ASCII'],
            'AcceptanceCriteria': ['Criteria with 中文 characters']
        })
        mock_read_excel.return_value = df

        mock_processor_instance = Mock()
        mock_processor_instance.analyze_stories_in_batches = Mock(return_value=[])
        mock_processor.return_value = mock_processor_instance

        temp_file = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)

        from src.interface_layer.StorySenseGenerator import StorySenseGenerator
        ssg = StorySenseGenerator(user_stories_path=temp_file.name)

        ssg.process_user_stories()

        assert mock_processor_instance.analyze_stories_in_batches.called

        os.unlink(temp_file.name)


class TestStorySenseGeneratorConfiguration:
    """Test configuration handling"""

    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open,
           read_data="[Input]\ninput_file_path=../Input/UserStories.xlsx\n[Output]\noutput_file_path=../Output/StorySense\n[LLM]\nLLM_FAMILY=AWS\n")
    def test_config_parser_initialization(self, mock_file, mock_exists, mock_makedirs):
        """Test ConfigParser initialization"""
        mock_exists.return_value = True

        from src.interface_layer.StorySenseGenerator import StorySenseGenerator
        ssg = StorySenseGenerator()

        assert hasattr(ssg, 'config_parser_io')
        assert hasattr(ssg, 'config_parser')

    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open,
           read_data="[Input]\ninput_file_path=../Input/UserStories.xlsx\n[Output]\noutput_file_path=../Output/StorySense\n[LLM]\nLLM_FAMILY=AWS\n")
    def test_config_file_paths(self, mock_file, mock_exists, mock_makedirs):
        """Test configuration file paths"""
        mock_exists.return_value = True

        from src.interface_layer.StorySenseGenerator import StorySenseGenerator
        ssg = StorySenseGenerator()

        assert ssg.config_path == '../Config'


class TestStorySenseGeneratorBatchProcessing:
    """Test batch processing functionality"""

    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open,
           read_data="[Input]\ninput_file_path=../Input/UserStories.xlsx\n[Output]\noutput_file_path=../Output/StorySense\n[Processing]\nbatch_size=5\n[LLM]\nLLM_FAMILY=AWS\n")
    @patch('src.html_report.storysense_processor.StorySenseProcessor')
    @patch('pandas.read_excel')
    def test_batch_processing_default_size(self, mock_read_excel, mock_processor,
                                           mock_file, mock_exists, mock_makedirs,
                                           sample_user_stories):
        """Test batch processing with default size"""
        mock_exists.return_value = True
        mock_read_excel.return_value = sample_user_stories

        mock_processor_instance = Mock()
        mock_processor_instance.analyze_stories_in_batches = Mock(return_value=[])
        mock_processor.return_value = mock_processor_instance

        temp_file = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)

        from src.interface_layer.StorySenseGenerator import StorySenseGenerator
        ssg = StorySenseGenerator(user_stories_path=temp_file.name)

        ssg.process_user_stories()

        # Default batch size should be 5
        call_args = mock_processor_instance.analyze_stories_in_batches.call_args
        assert call_args[0][2] == 5

        os.unlink(temp_file.name)


class TestStorySenseGeneratorErrorHandling:
    """Test error handling"""

    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open,
           read_data="[Input]\ninput_file_path=../Input/UserStories.xlsx\n[Output]\noutput_file_path=../Output/StorySense\n[LLM]\nLLM_FAMILY=AWS\n")
    @patch('pandas.read_excel')
    def test_read_excel_error(self, mock_read_excel, mock_file, mock_exists, mock_makedirs):
        """Test handling of Excel read errors"""
        mock_exists.return_value = True
        mock_read_excel.side_effect = Exception("Read error")

        temp_file = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)

        from src.interface_layer.StorySenseGenerator import StorySenseGenerator
        ssg = StorySenseGenerator(user_stories_path=temp_file.name)

        with pytest.raises(SystemExit):
            ssg.process_user_stories()

        os.unlink(temp_file.name)

    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open,
           read_data="[Input]\ninput_file_path=../Input/UserStories.xlsx\n[Output]\noutput_file_path=../Output/StorySense\n[LLM]\nLLM_FAMILY=AWS\n")
    @patch('pandas.read_csv')
    def test_read_csv_error(self, mock_read_csv, mock_file, mock_exists, mock_makedirs):
        """Test handling of CSV read errors"""
        mock_exists.return_value = True
        mock_read_csv.side_effect = Exception("Read error")

        temp_file = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)

        from src.interface_layer.StorySenseGenerator import StorySenseGenerator
        ssg = StorySenseGenerator(user_stories_path=temp_file.name)

        with pytest.raises(SystemExit):
            ssg.process_user_stories()

        os.unlink(temp_file.name)


class TestStorySenseGeneratorPathHandling:
    """Test path handling functionality"""

    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open,
           read_data="[Input]\ninput_file_path=../Input/UserStories.xlsx\n[Output]\noutput_file_path=../Output/StorySense\n[LLM]\nLLM_FAMILY=AWS\n")
    def test_relative_path_handling(self, mock_file, mock_exists, mock_makedirs):
        """Test handling of relative paths"""
        mock_exists.return_value = True

        from src.interface_layer.StorySenseGenerator import StorySenseGenerator
        ssg = StorySenseGenerator(user_stories_path='../Input/stories.xlsx')

        # Should be converted to absolute
        assert os.path.isabs(ssg.input_file_path)

    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open,
           read_data="[Input]\ninput_file_path=../Input/UserStories.xlsx\n[Output]\noutput_file_path=../Output/StorySense\n[LLM]\nLLM_FAMILY=AWS\n")
    def test_absolute_path_handling(self, mock_file, mock_exists, mock_makedirs):
        """Test handling of absolute paths"""
        mock_exists.return_value = True

        abs_path = os.path.abspath('../Input/stories.xlsx')

        from src.interface_layer.StorySenseGenerator import StorySenseGenerator
        ssg = StorySenseGenerator(user_stories_path=abs_path)

        assert os.path.isabs(ssg.input_file_path)


class TestStorySenseGeneratorPrintOutput:
    """Test print output and user feedback"""

    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open,
           read_data="[Input]\ninput_file_path=../Input/UserStories.xlsx\n[Output]\noutput_file_path=../Output/StorySense\n[Processing]\nbatch_size=5\n[LLM]\nLLM_FAMILY=AWS\n")
    @patch('src.html_report.storysense_processor.StorySenseProcessor')
    @patch('pandas.read_excel')
    def test_processing_header_printed(self, mock_read_excel, mock_processor,
                                       mock_file, mock_exists, mock_makedirs,
                                       sample_user_stories, capsys):
        """Test that processing header is printed"""
        mock_exists.return_value = True
        mock_read_excel.return_value = sample_user_stories

        mock_processor_instance = Mock()
        mock_processor_instance.analyze_stories_in_batches = Mock(return_value=[])
        mock_processor.return_value = mock_processor_instance

        temp_file = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)

        from src.interface_layer.StorySenseGenerator import StorySenseGenerator
        ssg = StorySenseGenerator(user_stories_path=temp_file.name)

        ssg.process_user_stories()

        captured = capsys.readouterr()
        assert "STORY SENSE ANALYZER" in captured.out

        os.unlink(temp_file.name)


class TestStorySenseGeneratorSystemExit:
    """Test system exit handling"""

    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open,
           read_data="[Input]\ninput_file_path=../Input/UserStories.xlsx\n[Output]\noutput_file_path=../Output/StorySense\n[LLM]\nLLM_FAMILY=AWS\n")
    def test_sys_exit_on_missing_file(self, mock_file, mock_exists, mock_makedirs):
        """Test sys.exit on missing file"""
        mock_exists.side_effect = lambda path: False if 'nonexistent' in path else True

        from src.interface_layer.StorySenseGenerator import StorySenseGenerator
        ssg = StorySenseGenerator(user_stories_path='/nonexistent/file.xlsx')

        with pytest.raises(SystemExit) as exc_info:
            ssg.process_user_stories()

        assert exc_info.value.code == 1

    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open,
           read_data="[Input]\ninput_file_path=../Input/UserStories.xlsx\n[Output]\noutput_file_path=../Output/StorySense\n[LLM]\nLLM_FAMILY=AWS\n")
    def test_sys_exit_on_unsupported_format(self, mock_file, mock_exists, mock_makedirs):
        """Test sys.exit on unsupported format"""
        mock_exists.return_value = True

        from src.interface_layer.StorySenseGenerator import StorySenseGenerator
        ssg = StorySenseGenerator(user_stories_path='../Input/test.txt')

        with pytest.raises(SystemExit) as exc_info:
            ssg.process_user_stories()

        assert exc_info.value.code == 1


class TestStorySenseGeneratorIntegration:
    """Integration tests"""

    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open,
           read_data="[Input]\ninput_file_path=../Input/UserStories.xlsx\nadditional_context_path=../Input/AdditionalContext.xlsx\n[Output]\noutput_file_path=../Output/StorySense\n[Processing]\nbatch_size=5\nparallel_processing=false\n[LLM]\nLLM_FAMILY=AWS\n")
    @patch('src.html_report.storysense_processor.StorySenseProcessor')
    @patch('pandas.read_excel')
    def test_full_processing_workflow(self, mock_read_excel, mock_processor,
                                      mock_file, mock_exists, mock_makedirs,
                                      sample_user_stories):
        """Test complete processing workflow"""
        mock_exists.return_value = True
        mock_read_excel.return_value = sample_user_stories

        mock_processor_instance = Mock()
        mock_processor_instance.analyze_stories_in_batches = Mock(return_value=[])
        mock_processor.return_value = mock_processor_instance

        temp_file = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)
        sample_user_stories.to_excel(temp_file.name, index=False)

        from src.interface_layer.StorySenseGenerator import StorySenseGenerator
        ssg = StorySenseGenerator(user_stories_path=temp_file.name)

        ssg.process_user_stories(batch_size=2, parallel=False)

        # Verify processing was called
        assert mock_processor_instance.analyze_stories_in_batches.called

        os.unlink(temp_file.name)


class TestStorySenseGeneratorCommandLine:
    """Test command-line interface"""

    def test_parse_arguments_all_flags(self):
        """Test parsing with all flags"""
        from src.interface_layer.StorySenseGenerator import parse_arguments

        with patch('sys.argv', [
            'script.py',
            '--user-stories', '../Input/stories.xlsx',
            '--context', '../Input/context.xlsx',
            '--batch-size', '15',
            '--parallel',
            '--process-context',
            '--manage-context',
            '--force-reprocess',
            '--context-folder', '../Input/CustomContext'
        ]):
            result = parse_arguments()

            assert result[0] == '../Input/stories.xlsx'
            assert result[1] == '../Input/context.xlsx'
            assert result[2] == 15
            assert result[3] is True
            assert result[4].process_context is True
            assert result[4].manage_context is True
            assert result[5] == '../Input/CustomContext'
            assert result[6] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src.interface_layer.StorySenseGenerator",
                 "--cov-report=term-missing"])
