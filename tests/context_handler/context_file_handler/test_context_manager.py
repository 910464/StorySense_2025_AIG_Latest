import pytest
import os
import json
import shutil
import tempfile
from unittest.mock import Mock, patch, MagicMock, ANY, mock_open
from pathlib import Path

# Import the module to test
from src.context_handler.context_file_handler.context_manager import ContextManager


class TestContextManager:
    @pytest.fixture
    def metrics_manager_mock(self):
        """Mock metrics manager for testing"""
        mock = Mock()
        mock.record_llm_call = Mock()
        mock.record_error = Mock()
        mock.record_vector_operation = Mock()
        return mock

    @pytest.fixture
    def pgvector_connector_mock(self):
        """Mock PGVectorConnector for testing"""
        mock = Mock()
        mock.vector_store_documents.return_value = True
        mock.retrieval_context.return_value = ("Sample context", {"0.8": "Document 1"},
                                               {"0.8": {"source_file": "test.pdf", "file_type": "pdf"}}, 0.7)
        return mock

    @pytest.fixture
    def context_manager(self, metrics_manager_mock, pgvector_connector_mock):
        """Create a ContextManager instance with mocked dependencies"""
        with patch('src.context_handler.context_file_handler.context_manager.PGVectorConnector',
                   return_value=pgvector_connector_mock):
            manager = ContextManager(metrics_manager=metrics_manager_mock)
            # Mock the context collections
            manager.context_collections = {
                'business_rules': pgvector_connector_mock,
                'requirements': pgvector_connector_mock,
                'documentation': pgvector_connector_mock,
                'policies': pgvector_connector_mock,
                'examples': pgvector_connector_mock,
                'glossary': pgvector_connector_mock
            }
            # Mock document processor and detector
            manager.document_processor = Mock()
            # Add file_processors dictionary to the document_processor mock
            manager.document_processor.file_processors = {
                '.pdf': Mock(),
                '.docx': Mock(),
                '.txt': Mock(),
                '.xlsx': Mock(),
                '.pptx': Mock()
            }
            manager.context_detector = Mock()
            manager.document_registry = Mock()
            return manager

    @pytest.fixture
    def temp_context_dir(self):
        """Create a temporary directory for context files"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_document(self):
        """Create a sample document for testing"""
        return {
            'text': 'This is a sample document for testing',
            'metadata': {
                'source_file': 'test.pdf',
                'file_type': 'pdf'
            }
        }

    def test_initialization(self, metrics_manager_mock):
        """Test initialization of ContextManager"""
        with patch('src.context_handler.context_file_handler.context_manager.PGVectorConnector') as mock_pgvector:
            # Configure the mock to return a new mock for each call
            mock_pgvector.side_effect = lambda **kwargs: Mock()

            manager = ContextManager(metrics_manager=metrics_manager_mock)

            # Check that context collections were initialized
            assert 'business_rules' in manager.context_collections
            assert 'requirements' in manager.context_collections
            assert 'documentation' in manager.context_collections
            assert 'policies' in manager.context_collections
            assert 'examples' in manager.context_collections
            assert 'glossary' in manager.context_collections

            # Check that PGVectorConnector was called for each collection
            assert mock_pgvector.call_count == 6

    def test_check_and_process_context_library_no_directory(self, context_manager):
        """Test check_and_process_context_library when directory doesn't exist"""
        with patch('os.path.exists', return_value=False), \
                patch.object(context_manager, '_create_context_directory_structure') as mock_create_dir:
            result = context_manager.check_and_process_context_library('/nonexistent/path')

            # Should create directory structure
            mock_create_dir.assert_called_once_with('/nonexistent/path')

            # Should return appropriate status
            assert result['status'] == 'no_context_directory'
            assert not result['has_context']

    def test_check_and_process_context_library_no_files(self, context_manager):
        """Test check_and_process_context_library when directory exists but has no files"""
        with patch('os.path.exists', return_value=True), \
                patch.object(context_manager, '_find_supported_files', return_value=[]):
            result = context_manager.check_and_process_context_library('/empty/directory')

            # Should return appropriate status
            assert result['status'] == 'no_context_files'
            assert not result['has_context']

    def test_check_and_process_context_library_with_files(self, context_manager):
        """Test check_and_process_context_library with supported files"""
        supported_files = ['/path/to/file1.pdf', '/path/to/file2.docx']

        with patch('os.path.exists', return_value=True), \
                patch.object(context_manager, '_find_supported_files', return_value=supported_files), \
                patch.object(context_manager, '_process_files_with_duplicate_check') as mock_process:
            # Configure mock_process to return a result
            mock_process.return_value = {
                'new_files': 2,
                'skipped_files': 0,
                'failed_files': 0,
                'total_documents': 5,
                'collections_updated': set(['business_rules', 'documentation']),
                'processing_details': [],
                'skipped_details': []
            }

            result = context_manager.check_and_process_context_library('/context/directory')

            # Should process files
            mock_process.assert_called_once_with(supported_files)

            # Should return appropriate status
            assert result['status'] == 'context_processed'
            assert result['has_context']
            assert result['processed_files'] == 2
            assert result['total_documents'] == 5
            assert 'business_rules' in result['collections_updated']
            assert 'documentation' in result['collections_updated']

    def test_find_supported_files(self, context_manager, temp_context_dir):
        """Test _find_supported_files method"""
        # Create some test files
        pdf_file = os.path.join(temp_context_dir, 'test.pdf')
        docx_file = os.path.join(temp_context_dir, 'test.docx')
        txt_file = os.path.join(temp_context_dir, 'test.txt')
        unsupported_file = os.path.join(temp_context_dir, 'test.xyz')

        # Create empty files
        for file_path in [pdf_file, docx_file, txt_file, unsupported_file]:
            with open(file_path, 'w') as f:
                f.write('')

        # Mock the supported file types
        context_manager.document_processor.file_processors = {
            '.pdf': Mock(),
            '.docx': Mock(),
            '.txt': Mock()
        }

        # Call the method
        result = context_manager._find_supported_files(temp_context_dir)

        # Check that only supported files were found
        assert len(result) == 3
        assert pdf_file in result
        assert docx_file in result
        assert txt_file in result
        assert unsupported_file not in result

    def test_process_files_with_duplicate_check(self, context_manager):
        """Test _process_files_with_duplicate_check method"""
        # Mock document registry
        context_manager.document_registry.is_document_processed.side_effect = [
            {'exists': True, 'reason': 'already processed'},  # First file is a duplicate
            {'exists': False}  # Second file is new
        ]
        context_manager.document_registry.register_document.return_value = 'doc123'

        # Mock _process_single_file
        with patch.object(context_manager, '_process_single_file') as mock_process:
            mock_process.return_value = {
                'success': True,
                'document_count': 3,
                'collections': ['business_rules'],
                'context_type': 'business_rules',
                'processing_time': 1.5
            }

            # Call the method
            result = context_manager._process_files_with_duplicate_check(['/path/to/file1.pdf', '/path/to/file2.docx'])

            # Check results
            assert result['skipped_files'] == 1
            assert result['new_files'] == 1
            assert result['failed_files'] == 0
            assert result['total_documents'] == 3
            assert 'business_rules' in result['collections_updated']

            # Check that only the second file was processed
            mock_process.assert_called_once_with('/path/to/file2.docx')

            # Check that document was registered
            context_manager.document_registry.register_document.assert_called_once_with(
                '/path/to/file2.docx', mock_process.return_value)

    def test_process_single_file_success(self, context_manager, sample_document):
        """Test _process_single_file with successful processing"""
        # Mock document processor
        context_manager.document_processor.process.return_value = [sample_document]

        # Mock context type detector
        context_manager.context_detector.determine.return_value = 'business_rules'

        # Call the method
        result = context_manager._process_single_file('/path/to/file.pdf')

        # Check results
        assert result['success'] is True
        assert result['document_count'] == 1
        assert result['collections'] == ['business_rules']
        assert result['context_type'] == 'business_rules'
        assert 'processing_time' in result

        # Check that document was processed and stored
        context_manager.document_processor.process.assert_called_once_with('/path/to/file.pdf')
        context_manager.context_detector.determine.assert_called_once_with('/path/to/file.pdf', [sample_document])
        context_manager.context_collections['business_rules'].vector_store_documents.assert_called_once_with(
            [sample_document])

    def test_process_single_file_no_content(self, context_manager):
        """Test _process_single_file when no content is extracted"""
        # Mock document processor to return empty list
        context_manager.document_processor.process.return_value = []

        # Call the method
        result = context_manager._process_single_file('/path/to/file.pdf')

        # Check results
        assert result['success'] is False
        assert result['error'] == "No content extracted from file"
        assert 'processing_time' in result

    def test_process_single_file_unknown_context_type(self, context_manager, sample_document):
        """Test _process_single_file with unknown context type"""
        # Mock document processor
        context_manager.document_processor.process.return_value = [sample_document]

        # Mock context type detector to return unknown type
        context_manager.context_detector.determine.return_value = 'unknown_type'

        # Call the method
        result = context_manager._process_single_file('/path/to/file.pdf')

        # Check results
        assert result['success'] is False
        assert result['error'] == "Unknown context type: unknown_type"
        assert 'processing_time' in result

    def test_process_single_file_storage_failure(self, context_manager, sample_document):
        """Test _process_single_file when storage fails"""
        # Mock document processor
        context_manager.document_processor.process.return_value = [sample_document]

        # Mock context type detector
        context_manager.context_detector.determine.return_value = 'business_rules'

        # Mock vector store to fail
        context_manager.context_collections['business_rules'].vector_store_documents.return_value = False

        # Call the method
        result = context_manager._process_single_file('/path/to/file.pdf')

        # Check results
        assert result['success'] is False
        assert result['error'] == "Failed to store documents in vector database"
        assert 'processing_time' in result

    def test_process_single_file_exception(self, context_manager):
        """Test _process_single_file when an exception occurs"""
        # Mock document processor to raise exception
        context_manager.document_processor.process.side_effect = Exception("Test error")

        # Call the method
        result = context_manager._process_single_file('/path/to/file.pdf')

        # Check results
        assert result['success'] is False
        assert result['error'] == "Test error"
        assert 'processing_time' in result

    def test_create_context_directory_structure(self, context_manager, temp_context_dir):
        """Test _create_context_directory_structure method"""
        # Mock the file writing operations to avoid I/O errors
        with patch('builtins.open', mock_open()) as mock_file:
            # Call the method
            context_manager._create_context_directory_structure(temp_context_dir)

            # Check that directories were created
            assert os.path.exists(os.path.join(temp_context_dir, 'business_rules'))
            assert os.path.exists(os.path.join(temp_context_dir, 'requirements'))
            assert os.path.exists(os.path.join(temp_context_dir, 'documentation'))
            assert os.path.exists(os.path.join(temp_context_dir, 'policies'))
            assert os.path.exists(os.path.join(temp_context_dir, 'examples'))
            assert os.path.exists(os.path.join(temp_context_dir, 'glossary'))

    def test_get_context_status(self, context_manager):
        """Test get_context_status method"""
        # Mock document registry
        context_manager.document_registry.get_status_summary.return_value = {
            'registry_exists': True,
            'total_registered_documents': 10,
            'last_update': '2023-01-01T12:00:00'
        }
        context_manager.document_registry.get_all_documents.return_value = {
            'doc1': {'collections': ['business_rules', 'documentation']},
            'doc2': {'collections': ['business_rules']},
            'doc3': {'collections': ['requirements']},
            'doc4': {'collections': ['glossary']}
        }

        # Call the method
        result = context_manager.get_context_status()

        # Check results
        assert result['registry_exists'] is True
        assert result['total_registered_documents'] == 10
        assert result['last_update'] == '2023-01-01T12:00:00'
        assert result['collections']['business_rules'] == 2
        assert result['collections']['documentation'] == 1
        assert result['collections']['requirements'] == 1
        assert result['collections']['glossary'] == 1
        assert result['collections']['policies'] == 0
        assert result['collections']['examples'] == 0

    def test_search_context(self, context_manager):
        """Test search_context method"""
        # Create separate mock objects for each collection
        business_rules_mock = Mock()
        business_rules_mock.retrieval_context.return_value = (
            "Business rules context", {"0.8": "BR Doc"}, {"0.8": {"source_file": "br.pdf", "file_type": "pdf"}}, 0.7
        )

        requirements_mock = Mock()
        requirements_mock.retrieval_context.return_value = (
            "Requirements context", {"0.9": "Req Doc"}, {"0.9": {"source_file": "req.docx", "file_type": "docx"}}, 0.7
        )

        # Replace the collections with our specific mocks
        context_manager.context_collections = {
            'business_rules': business_rules_mock,
            'requirements': requirements_mock,
            'documentation': Mock(),
            'policies': Mock(),
            'examples': Mock(),
            'glossary': Mock()
        }

        # Call the method with specific context types
        result = context_manager.search_context("test query", context_types=['business_rules', 'requirements'])

        # Check results
        assert 'business_rules' in result
        assert 'requirements' in result
        assert result['business_rules']['context'] == "Business rules context"
        assert result['requirements']['context'] == "Requirements context"
        assert result['business_rules']['documents'] == {"0.8": "BR Doc"}
        assert result['requirements']['documents'] == {"0.9": "Req Doc"}
        assert 'pdf' in result['file_type_stats']
        assert 'docx' in result['file_type_stats']
        assert result['file_type_stats']['pdf'] == 1
        assert result['file_type_stats']['docx'] == 1

    def test_search_context_with_error(self, context_manager):
        """Test search_context when an error occurs"""
        # Create separate mock objects for each collection
        business_rules_mock = Mock()
        business_rules_mock.retrieval_context.return_value = (
            "Business rules context", {"0.8": "BR Doc"}, {"0.8": {"source_file": "br.pdf", "file_type": "pdf"}}, 0.7
        )

        requirements_mock = Mock()
        requirements_mock.retrieval_context.side_effect = Exception("Test error")

        # Replace the collections with our specific mocks
        context_manager.context_collections = {
            'business_rules': business_rules_mock,
            'requirements': requirements_mock,
            'documentation': Mock(),
            'policies': Mock(),
            'examples': Mock(),
            'glossary': Mock()
        }

        # Call the method
        result = context_manager.search_context("test query", context_types=['business_rules', 'requirements'])

        # Check results
        assert 'business_rules' in result
        assert 'requirements' in result
        assert result['business_rules']['context'] == "Business rules context"
        assert 'error' in result['requirements']
        assert result['requirements']['error'] == "Test error"
        assert 'pdf' in result['file_type_stats']
        assert result['file_type_stats']['pdf'] == 1

    def test_search_context_default_types(self, context_manager):
        """Test search_context with default context types"""
        # Create separate mock objects for each collection
        business_rules_mock = Mock()
        business_rules_mock.retrieval_context.return_value = (
            "Business rules context", {"0.8": "BR Doc"}, {"0.8": {"source_file": "br.pdf", "file_type": "pdf"}}, 0.7
        )

        requirements_mock = Mock()
        requirements_mock.retrieval_context.return_value = (
            "Requirements context", {"0.9": "Req Doc"}, {"0.9": {"source_file": "req.docx", "file_type": "docx"}}, 0.7
        )

        # Set up all the collections
        context_manager.context_collections = {
            'business_rules': business_rules_mock,
            'requirements': requirements_mock,
            'documentation': Mock(),
            'policies': Mock(),
            'examples': Mock(),
            'glossary': Mock()
        }

        # Call the method without specifying context types
        result = context_manager.search_context("test query")

        # Should search all collections
        assert 'business_rules' in result
        assert 'requirements' in result
        assert result['business_rules']['context'] == "Business rules context"
        assert result['requirements']['context'] == "Requirements context"

    def test_get_file_type_distribution(self, context_manager):
        """Test get_file_type_distribution method"""
        # Create mock search results
        search_results = {
            'business_rules': {
                'file_types': {
                    0.8: {'type': 'pdf', 'source': 'br.pdf'},
                    0.7: {'type': 'docx', 'source': 'br.docx'}
                }
            },
            'requirements': {
                'file_types': {
                    0.9: {'type': 'pdf', 'source': 'req.pdf'},
                    0.6: {'type': 'xlsx', 'source': 'req.xlsx'}
                }
            }
        }

        # Call the method
        result = context_manager.get_file_type_distribution(search_results)

        # Check results
        assert result['pdf'] == 2
        assert result['docx'] == 1
        assert result['xlsx'] == 1

    def test_get_file_type_distribution_with_stats(self, context_manager):
        """Test get_file_type_distribution when file_type_stats is already present"""
        # Create mock search results with file_type_stats
        search_results = {
            'file_type_stats': {
                'pdf': 3,
                'docx': 2,
                'txt': 1
            }
        }

        # Call the method
        result = context_manager.get_file_type_distribution(search_results)

        # Should return the existing stats
        assert result == search_results['file_type_stats']
        assert result['pdf'] == 3
        assert result['docx'] == 2
        assert result['txt'] == 1