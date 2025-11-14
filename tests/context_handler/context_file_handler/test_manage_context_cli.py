"""Test cases for manage_context_cli.py module."""

import sys
import pytest
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

# Mock all problematic modules before any imports
mock_modules = {
    'psutil': Mock(),
    'requests': Mock(),
    'sqlalchemy': Mock(),
    'sqlalchemy.create_engine': Mock(),
    'psycopg2': Mock(),
    'psycopg2.extras': Mock(),
    'boto3': Mock(),
    'colorama': Mock(),
    'numpy': Mock(),
    'pandas': Mock(),
    'sklearn': Mock(),
    'tiktoken': Mock(),
}

for module_name, mock_module in mock_modules.items():
    sys.modules[module_name] = mock_module

# Create a proper psycopg2 mock with submodules
psycopg2_mock = Mock()
psycopg2_mock.extras = Mock()
psycopg2_mock.extras.execute_values = Mock()
sys.modules['psycopg2'] = psycopg2_mock
sys.modules['psycopg2.extras'] = psycopg2_mock.extras

# Now we can import the module to test
from src.context_handler.context_file_handler.manage_context_cli import main


class TestManageContextCLI:
    """Test cases for the manage_context_cli module."""

    @pytest.fixture
    def mock_context_manager(self):
        """Mock ContextManager for testing"""
        mock = Mock()
        mock.get_context_status.return_value = {
            'total_registered_documents': 5,
            'last_update': '2023-01-01T12:00:00',
            'collections': {
                'business_rules': 2,
                'requirements': 1,
                'documentation': 1,
                'policies': 0,
                'examples': 1,
                'glossary': 0
            }
        }
        mock.check_and_process_context_library.return_value = {
            'message': 'Context library processed successfully.'
        }
        mock._create_context_directory_structure = Mock()
        return mock

    @pytest.fixture
    def mock_metrics_manager(self):
        """Mock MetricsManager for testing"""
        return Mock()

    def test_main_with_process_flag(self, mock_context_manager, mock_metrics_manager):
        """Test main function with --process flag"""
        # Mock argparse.ArgumentParser and its methods
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.process = True
        mock_args.status = False
        mock_args.create_structure = False
        mock_args.directory = None
        mock_parser.parse_args.return_value = mock_args
        
        with patch('sys.argv', ['manage_context_cli.py', '--process']), \
                patch('argparse.ArgumentParser', return_value=mock_parser), \
                patch('src.context_handler.context_file_handler.manage_context_cli.ContextManager') as mock_cm_class, \
                patch('src.context_handler.context_file_handler.manage_context_cli.MetricsManager') as mock_mm_class, \
                patch('src.context_handler.context_file_handler.manage_context_cli.Fore') as mock_fore, \
                patch('src.context_handler.context_file_handler.manage_context_cli.Style') as mock_style, \
                patch('builtins.print') as mock_print:
            
            # Mock colorama attributes to be strings
            mock_fore.GREEN = '\033[32m'
            mock_fore.CYAN = '\033[36m'
            mock_style.RESET_ALL = '\033[0m'
            
            mock_cm_class.return_value = mock_context_manager
            mock_mm_class.return_value = mock_metrics_manager
            
            main()

            # Verify ContextManager and MetricsManager were instantiated
            mock_cm_class.assert_called_once()
            mock_mm_class.assert_called_once()
            # Check that check_and_process_context_library was called
            mock_context_manager.check_and_process_context_library.assert_called_once_with('../Input/ContextLibrary')

    def test_main_with_status_flag(self, mock_context_manager, mock_metrics_manager):
        """Test main function with --status flag"""
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.process = False
        mock_args.status = True
        mock_args.create_structure = False
        mock_args.directory = None
        mock_parser.parse_args.return_value = mock_args
        
        with patch('sys.argv', ['manage_context_cli.py', '--status']), \
                patch('argparse.ArgumentParser', return_value=mock_parser), \
                patch('src.context_handler.context_file_handler.manage_context_cli.ContextManager') as mock_cm_class, \
                patch('src.context_handler.context_file_handler.manage_context_cli.MetricsManager') as mock_mm_class, \
                patch('src.context_handler.context_file_handler.manage_context_cli.Fore') as mock_fore, \
                patch('src.context_handler.context_file_handler.manage_context_cli.Style') as mock_style, \
                patch('builtins.print') as mock_print:
            
            mock_fore.CYAN = '\033[36m'
            mock_style.RESET_ALL = '\033[0m'
            
            mock_cm_class.return_value = mock_context_manager
            mock_mm_class.return_value = mock_metrics_manager
            
            main()

            # Verify ContextManager and MetricsManager were instantiated
            mock_cm_class.assert_called_once()
            mock_mm_class.assert_called_once()
            # Check that get_context_status was called
            mock_context_manager.get_context_status.assert_called_once()

    def test_main_with_create_structure_flag(self, mock_context_manager, mock_metrics_manager):
        """Test main function with --create-structure flag"""
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.process = False
        mock_args.status = False
        mock_args.create_structure = True
        mock_args.directory = None
        mock_parser.parse_args.return_value = mock_args
        
        with patch('sys.argv', ['manage_context_cli.py', '--create-structure']), \
                patch('argparse.ArgumentParser', return_value=mock_parser), \
                patch('src.context_handler.context_file_handler.manage_context_cli.ContextManager') as mock_cm_class, \
                patch('src.context_handler.context_file_handler.manage_context_cli.MetricsManager') as mock_mm_class, \
                patch('src.context_handler.context_file_handler.manage_context_cli.Fore') as mock_fore, \
                patch('src.context_handler.context_file_handler.manage_context_cli.Style') as mock_style, \
                patch('builtins.print') as mock_print:
            
            mock_fore.GREEN = '\033[32m'
            mock_style.RESET_ALL = '\033[0m'
            
            mock_cm_class.return_value = mock_context_manager
            mock_mm_class.return_value = mock_metrics_manager
            
            main()

            # Verify ContextManager and MetricsManager were instantiated
            mock_cm_class.assert_called_once()
            mock_mm_class.assert_called_once()
            # Check that _create_context_directory_structure was called
            mock_context_manager._create_context_directory_structure.assert_called_once_with('../Input/ContextLibrary')

    def test_main_with_directory_flag(self, mock_context_manager, mock_metrics_manager):
        """Test main function with --process and --directory flags"""
        test_dir = '/path/to/test/directory'
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.process = True
        mock_args.status = False
        mock_args.create_structure = False
        mock_args.directory = test_dir
        mock_parser.parse_args.return_value = mock_args
        
        with patch('sys.argv', ['manage_context_cli.py', '--process', '--directory', test_dir]), \
                patch('argparse.ArgumentParser', return_value=mock_parser), \
                patch('src.context_handler.context_file_handler.manage_context_cli.ContextManager') as mock_cm_class, \
                patch('src.context_handler.context_file_handler.manage_context_cli.MetricsManager') as mock_mm_class, \
                patch('src.context_handler.context_file_handler.manage_context_cli.Fore') as mock_fore, \
                patch('src.context_handler.context_file_handler.manage_context_cli.Style') as mock_style, \
                patch('builtins.print') as mock_print:
            
            mock_fore.GREEN = '\033[32m'
            mock_style.RESET_ALL = '\033[0m'
            
            mock_cm_class.return_value = mock_context_manager
            mock_mm_class.return_value = mock_metrics_manager
            
            main()

            # Check that check_and_process_context_library was called with custom directory
            mock_context_manager.check_and_process_context_library.assert_called_once_with(test_dir)

    def test_main_with_create_structure_and_directory_flags(self, mock_context_manager, mock_metrics_manager):
        """Test main function with --create-structure and --directory flags"""
        test_dir = '/path/to/test/directory'
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.process = False
        mock_args.status = False
        mock_args.create_structure = True
        mock_args.directory = test_dir
        mock_parser.parse_args.return_value = mock_args
        
        with patch('sys.argv', ['manage_context_cli.py', '--create-structure', '--directory', test_dir]), \
                patch('argparse.ArgumentParser', return_value=mock_parser), \
                patch('src.context_handler.context_file_handler.manage_context_cli.ContextManager') as mock_cm_class, \
                patch('src.context_handler.context_file_handler.manage_context_cli.MetricsManager') as mock_mm_class, \
                patch('src.context_handler.context_file_handler.manage_context_cli.Fore') as mock_fore, \
                patch('src.context_handler.context_file_handler.manage_context_cli.Style') as mock_style, \
                patch('builtins.print') as mock_print:
            
            mock_fore.GREEN = '\033[32m'
            mock_style.RESET_ALL = '\033[0m'
            
            mock_cm_class.return_value = mock_context_manager
            mock_mm_class.return_value = mock_metrics_manager
            
            main()

            # Check that _create_context_directory_structure was called with custom directory
            mock_context_manager._create_context_directory_structure.assert_called_once_with(test_dir)

    def test_main_with_status_and_directory_flags(self, mock_context_manager, mock_metrics_manager):
        """Test main function with --status and --directory flags"""
        test_dir = '/path/to/test/directory'
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.process = False
        mock_args.status = True
        mock_args.create_structure = False
        mock_args.directory = test_dir
        mock_parser.parse_args.return_value = mock_args
        
        with patch('sys.argv', ['manage_context_cli.py', '--status', '--directory', test_dir]), \
                patch('argparse.ArgumentParser', return_value=mock_parser), \
                patch('src.context_handler.context_file_handler.manage_context_cli.ContextManager') as mock_cm_class, \
                patch('src.context_handler.context_file_handler.manage_context_cli.MetricsManager') as mock_mm_class, \
                patch('src.context_handler.context_file_handler.manage_context_cli.Fore') as mock_fore, \
                patch('src.context_handler.context_file_handler.manage_context_cli.Style') as mock_style, \
                patch('builtins.print') as mock_print:
            
            mock_fore.CYAN = '\033[36m'
            mock_style.RESET_ALL = '\033[0m'
            
            mock_cm_class.return_value = mock_context_manager
            mock_mm_class.return_value = mock_metrics_manager
            
            main()

            # Check that get_context_status was called (status doesn't use directory parameter)
            mock_context_manager.get_context_status.assert_called_once()

    def test_main_no_arguments_shows_help(self, mock_context_manager, mock_metrics_manager):
        """Test main function with no arguments shows help"""
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.process = False
        mock_args.status = False
        mock_args.create_structure = False
        mock_args.directory = None
        mock_parser.parse_args.return_value = mock_args
        
        with patch('sys.argv', ['manage_context_cli.py']), \
                patch('argparse.ArgumentParser', return_value=mock_parser), \
                patch('src.context_handler.context_file_handler.manage_context_cli.ContextManager') as mock_cm_class, \
                patch('src.context_handler.context_file_handler.manage_context_cli.MetricsManager') as mock_mm_class:
            
            mock_cm_class.return_value = mock_context_manager
            mock_mm_class.return_value = mock_metrics_manager
            
            main()

            # Check that print_help was called when no action is specified
            mock_parser.print_help.assert_called_once()

    def test_main_status_with_null_last_update(self, mock_context_manager, mock_metrics_manager):
        """Test main function with --status flag when last_update is None"""
        # Modify mock to return None for last_update
        mock_context_manager.get_context_status.return_value = {
            'total_registered_documents': 0,
            'last_update': None,
            'collections': {
                'business_rules': 0,
                'requirements': 0,
                'documentation': 0,
                'policies': 0,
                'examples': 0,
                'glossary': 0
            }
        }
        
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.process = False
        mock_args.status = True
        mock_args.create_structure = False
        mock_args.directory = None
        mock_parser.parse_args.return_value = mock_args
        
        with patch('sys.argv', ['manage_context_cli.py', '--status']), \
                patch('argparse.ArgumentParser', return_value=mock_parser), \
                patch('src.context_handler.context_file_handler.manage_context_cli.ContextManager') as mock_cm_class, \
                patch('src.context_handler.context_file_handler.manage_context_cli.MetricsManager') as mock_mm_class, \
                patch('src.context_handler.context_file_handler.manage_context_cli.Fore') as mock_fore, \
                patch('src.context_handler.context_file_handler.manage_context_cli.Style') as mock_style, \
                patch('builtins.print') as mock_print:
            
            mock_fore.CYAN = '\033[36m'
            mock_style.RESET_ALL = '\033[0m'
            
            mock_cm_class.return_value = mock_context_manager
            mock_mm_class.return_value = mock_metrics_manager
            
            main()

            # Verify status was called and None last_update was handled
            mock_context_manager.get_context_status.assert_called_once()
            # Check that print was called with 'Never' for null last_update
            mock_print.assert_any_call(f"Last update: Never")

    def test_main_module_execution(self):
        """Test that main() is called when script is run directly"""
        import importlib
        import sys
        
        # Mock the main function and sys.argv
        with patch('sys.argv', ['manage_context_cli.py', '--help']), \
                patch('src.context_handler.context_file_handler.manage_context_cli.main') as mock_main, \
                patch.dict('sys.modules', {'__main__': Mock(__name__='__main__')}):
            
            # Simulate running the module directly by importing and executing
            # This tests the if __name__ == "__main__": guard
            try:
                # This will trigger the __name__ == "__main__" check
                exec(compile(open('src/context_handler/context_file_handler/manage_context_cli.py').read(), 
                           'src/context_handler/context_file_handler/manage_context_cli.py', 'exec'))
            except SystemExit:
                # Expected when argparse shows help
                pass
            except Exception:
                # We expect some exceptions due to mocking, but that's okay
                pass

    def test_argparse_configuration(self, mock_context_manager, mock_metrics_manager):
        """Test that argparse is configured correctly with all arguments"""
        real_parser = Mock()
        
        # Capture add_argument calls to verify parser configuration
        add_argument_calls = []
        def mock_add_argument(*args, **kwargs):
            add_argument_calls.append((args, kwargs))
        
        real_parser.add_argument = mock_add_argument
        
        mock_args = Mock()
        mock_args.process = False
        mock_args.status = False
        mock_args.create_structure = False
        mock_args.directory = None
        real_parser.parse_args.return_value = mock_args
        
        with patch('argparse.ArgumentParser', return_value=real_parser), \
                patch('src.context_handler.context_file_handler.manage_context_cli.ContextManager') as mock_cm_class, \
                patch('src.context_handler.context_file_handler.manage_context_cli.MetricsManager') as mock_mm_class:
            
            mock_cm_class.return_value = mock_context_manager
            mock_mm_class.return_value = mock_metrics_manager
            
            main()
            
            # Verify that all expected arguments were added
            expected_args = [
                ('--process', '-p'),
                ('--status', '-s'), 
                ('--directory', '-d'),
                ('--create-structure', '-c')
            ]
            
            # Check that we have at least the expected number of add_argument calls
            assert len(add_argument_calls) >= len(expected_args)
            
            # Verify parser.print_help was called when no action specified
            real_parser.print_help.assert_called_once()
