import pytest
import sys
import os
from unittest.mock import patch, Mock, MagicMock
from io import StringIO
from pathlib import Path

# Import the module to test
from src.context_handler.context_file_handler.manage_context_cli import main


class TestManageContextCLI:
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
            'status': 'context_processed',
            'message': 'Context library processed. 3 new files, 2 existing files.',
            'has_context': True,
            'processed_files': 3,
            'skipped_files': 2,
            'new_files': 3,
            'total_documents': 5,
            'collections_updated': ['business_rules', 'requirements']
        }
        mock._create_context_directory_structure = Mock()
        return mock

    @pytest.fixture
    def mock_metrics_manager(self):
        """Mock MetricsManager for testing"""
        return Mock()

    @pytest.fixture
    def mock_configparser(self):
        """Mock ConfigParser for testing"""
        mock = Mock()
        mock_instance = Mock()
        mock_instance.get.return_value = '8'  # Return '8' for any config value
        mock.return_value = mock_instance
        return mock

    def test_main_with_process_flag(self, mock_context_manager, mock_metrics_manager, mock_configparser):
        """Test main function with --process flag"""
        with patch('sys.argv', ['manage_context_cli.py', '--process']), \
                patch('src.context_handler.context_file_handler.context_manager.ContextManager',
                      return_value=mock_context_manager), \
                patch('src.metrics.metrics_manager.MetricsManager',
                      return_value=mock_metrics_manager), \
                patch('configparser.ConfigParser', mock_configparser), \
                patch('os.path.exists', return_value=True), \
                patch('builtins.open', MagicMock()), \
                patch('sys.stdout', new=StringIO()) as fake_out:
            main()

            # Check that check_and_process_context_library was called
            mock_context_manager.check_and_process_context_library.assert_called_once()

            # Check output
            output = fake_out.getvalue()
            assert "Processing complete" in output

    def test_main_with_status_flag(self, mock_context_manager, mock_metrics_manager, mock_configparser):
        """Test main function with --status flag"""
        with patch('sys.argv', ['manage_context_cli.py', '--status']), \
                patch('src.context_handler.context_file_handler.context_manager.ContextManager',
                      return_value=mock_context_manager), \
                patch('src.metrics.metrics_manager.MetricsManager',
                      return_value=mock_metrics_manager), \
                patch('configparser.ConfigParser', mock_configparser), \
                patch('os.path.exists', return_value=True), \
                patch('builtins.open', MagicMock()), \
                patch('sys.stdout', new=StringIO()) as fake_out:
            main()

            # Check that get_context_status was called
            mock_context_manager.get_context_status.assert_called_once()

            # Check output
            output = fake_out.getvalue()
            assert "CONTEXT LIBRARY STATUS" in output
            assert "Total registered documents: 5" in output
            assert "Last update: 2023-01-01T12:00:00" in output
            assert "Business Rules: 2" in output

    def test_main_with_create_structure_flag(self, mock_context_manager, mock_metrics_manager, mock_configparser):
        """Test main function with --create-structure flag"""
        with patch('sys.argv', ['manage_context_cli.py', '--create-structure']), \
                patch('src.context_handler.context_file_handler.context_manager.ContextManager',
                      return_value=mock_context_manager), \
                patch('src.metrics.metrics_manager.MetricsManager',
                      return_value=mock_metrics_manager), \
                patch('configparser.ConfigParser', mock_configparser), \
                patch('os.path.exists', return_value=True), \
                patch('builtins.open', MagicMock()), \
                patch('sys.stdout', new=StringIO()) as fake_out:
            main()

            # Check that _create_context_directory_structure was called
            mock_context_manager._create_context_directory_structure.assert_called_once()

            # Check output
            output = fake_out.getvalue()
            assert "Created context library structure" in output

    def test_main_with_directory_flag(self, mock_context_manager, mock_metrics_manager, mock_configparser):
        """Test main function with --directory flag"""
        custom_dir = "/custom/directory"
        with patch('sys.argv', ['manage_context_cli.py', '--process', '--directory', custom_dir]), \
                patch('src.context_handler.context_file_handler.context_manager.ContextManager',
                      return_value=mock_context_manager), \
                patch('src.metrics.metrics_manager.MetricsManager',
                      return_value=mock_metrics_manager), \
                patch('configparser.ConfigParser', mock_configparser), \
                patch('os.path.exists', return_value=True), \
                patch('builtins.open', MagicMock()), \
                patch('sys.stdout', new=StringIO()) as fake_out:
            main()

            # Check that check_and_process_context_library was called with custom directory
            mock_context_manager.check_and_process_context_library.assert_called_once_with(custom_dir)

    def test_main_with_create_structure_and_directory_flag(self, mock_context_manager, mock_metrics_manager,
                                                           mock_configparser):
        """Test main function with --create-structure and --directory flags"""
        custom_dir = "/custom/directory"
        with patch('sys.argv', ['manage_context_cli.py', '--create-structure', '--directory', custom_dir]), \
                patch('src.context_handler.context_file_handler.context_manager.ContextManager',
                      return_value=mock_context_manager), \
                patch('src.metrics.metrics_manager.MetricsManager',
                      return_value=mock_metrics_manager), \
                patch('configparser.ConfigParser', mock_configparser), \
                patch('os.path.exists', return_value=True), \
                patch('builtins.open', MagicMock()), \
                patch('sys.stdout', new=StringIO()) as fake_out:
            main()

            # Check that _create_context_directory_structure was called with custom directory
            mock_context_manager._create_context_directory_structure.assert_called_once_with(custom_dir)

    def test_main_with_no_args(self, mock_context_manager, mock_metrics_manager, mock_configparser):
        """Test main function with no arguments (should show help)"""
        with patch('sys.argv', ['manage_context_cli.py']), \
                patch('src.context_handler.context_file_handler.context_manager.ContextManager',
                      return_value=mock_context_manager), \
                patch('src.metrics.metrics_manager.MetricsManager',
                      return_value=mock_metrics_manager), \
                patch('configparser.ConfigParser', mock_configparser), \
                patch('os.path.exists', return_value=True), \
                patch('builtins.open', MagicMock()), \
                patch('argparse.ArgumentParser.print_help') as mock_print_help, \
                patch('sys.stdout', new=StringIO()):
            main()

            # Check that print_help was called
            mock_print_help.assert_called_once()