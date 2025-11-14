import pytest
import os
import pandas as pd
import json
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock, ANY, mock_open
from pathlib import Path
from datetime import datetime

from src.context_handler.context_file_handler.enhanced_context_processor import EnhancedContextProcessor


class TestEnhancedContextProcessor:
    @pytest.fixture
    def global_metrics_manager_mock(self):
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
        mock.diagnose_database.return_value = True
        return mock

    @pytest.fixture
    def processor(self, global_metrics_manager_mock, pgvector_connector_mock):
        """Create an EnhancedContextProcessor instance with mocked dependencies"""
        with patch('src.context_handler.context_file_handler.enhanced_context_processor.PGVectorConnector',
                   return_value=pgvector_connector_mock), \
             patch('src.context_handler.context_file_handler.enhanced_context_processor.AWSTitanEmbeddings') as mock_embeddings, \
             patch('src.context_handler.context_file_handler.enhanced_context_processor.MetricsManager', 
                   return_value=global_metrics_manager_mock), \
             patch('os.makedirs'), \
             patch('pathlib.Path.exists', return_value=False):  # Mock registry file doesn't exist
            
            # Mock the embeddings instance
            mock_embeddings_instance = Mock()
            mock_embeddings.return_value = mock_embeddings_instance
            
            processor = EnhancedContextProcessor('../Input/ContextLibrary', metrics_manager=global_metrics_manager_mock)
            
            # Replace context collections with mocks
            processor.context_collections = {
                'business_rules': pgvector_connector_mock,
                'requirements': pgvector_connector_mock,
                'documentation': pgvector_connector_mock,
                'policies': pgvector_connector_mock,
                'examples': pgvector_connector_mock,
                'glossary': pgvector_connector_mock,
                'wireframes': pgvector_connector_mock
            }
            
            # Mock additional attributes and methods
            processor.document_registry = {}
            processor.embeddings = mock_embeddings_instance
            processor._save_document_registry = Mock()
            processor._calculate_file_hash = Mock(return_value="test_hash")
            processor.image_parser = Mock()
            processor.stats = {
                'total_files': 0,
                'processed_files': 0,
                'skipped_files': 0,
                'failed_files': 0,
                'total_chunks': 0,
                'processing_time': 0,
                'by_category': {}
            }
            
            return processor

    @pytest.fixture
    def temp_context_dir(self):
        """Create a temporary directory for context files"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_files(self, temp_context_dir):
        """Create sample files for testing"""
        # Create subdirectories
        os.makedirs(os.path.join(temp_context_dir, 'business_rules'), exist_ok=True)
        os.makedirs(os.path.join(temp_context_dir, 'documentation'), exist_ok=True)

        # Create sample files
        files = {
            'pdf': os.path.join(temp_context_dir, 'business_rules', 'test.pdf'),
            'docx': os.path.join(temp_context_dir, 'documentation', 'test.docx'),
            'xlsx': os.path.join(temp_context_dir, 'business_rules', 'test.xlsx'),
            'txt': os.path.join(temp_context_dir, 'documentation', 'test.txt'),
            'pptx': os.path.join(temp_context_dir, 'documentation', 'test.pptx'),
            'jpg': os.path.join(temp_context_dir, 'documentation', 'test.jpg'),
            'hidden': os.path.join(temp_context_dir, '.hidden_file'),
            'large': os.path.join(temp_context_dir, 'large_file.txt')
        }

        # Create empty files
        for file_path in files.values():
            with open(file_path, 'w') as f:
                f.write('test content')

        # Make the large file appear large
        with patch('os.path.getsize', lambda path: 600 * 1024 * 1024 if path == files['large'] else 1024):
            yield files

    def test_initialization(self, global_metrics_manager_mock):
        """Test initialization of EnhancedContextProcessor"""
        with patch(
                'src.context_handler.context_file_handler.enhanced_context_processor.PGVectorConnector') as mock_pgvector, \
                patch(
                    'src.context_handler.context_file_handler.enhanced_context_processor.AWSTitanEmbeddings') as mock_embeddings, \
                patch('os.makedirs') as mock_makedirs, \
                patch(
                    'src.context_handler.context_file_handler.enhanced_context_processor.EnhancedContextProcessor._load_document_registry',
                    return_value={}):
            processor = EnhancedContextProcessor('../Input/ContextLibrary', metrics_manager=global_metrics_manager_mock)

            # Check that context collections were initialized
            assert 'business_rules' in processor.context_collections
            assert 'requirements' in processor.context_collections
            assert 'documentation' in processor.context_collections
            assert 'policies' in processor.context_collections
            assert 'examples' in processor.context_collections
            assert 'glossary' in processor.context_collections
            assert 'wireframes' in processor.context_collections

            # Check that PGVectorConnector was called for each collection
            assert mock_pgvector.call_count == 7

            # Check that AWSTitanEmbeddings was initialized
            mock_embeddings.assert_called_once()

            # Check that directories were created
            mock_makedirs.assert_any_call('../Data/DocumentRegistry', exist_ok=True)
            mock_makedirs.assert_any_call('../Data/LocalEmbeddings', exist_ok=True)

    def test_load_document_registry_success(self):
        """Test loading document registry when file exists"""
        mock_registry = {'doc1': {'file_name': 'test.pdf'}}

        with patch('src.context_handler.context_file_handler.enhanced_context_processor.PGVectorConnector'), \
             patch('src.context_handler.context_file_handler.enhanced_context_processor.AWSTitanEmbeddings'), \
             patch('src.context_handler.context_file_handler.enhanced_context_processor.MetricsManager'), \
             patch('os.makedirs'), \
             patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=json.dumps(mock_registry))):
            
            # Create processor and load registry normally during init
            processor = EnhancedContextProcessor('../Input/ContextLibrary')
            
            # Now test the method by calling it again with our mock setup still active
            registry = processor._load_document_registry()

            assert registry == mock_registry

    def test_load_document_registry_not_exists(self):
        """Test loading document registry when file doesn't exist"""
        with patch('src.context_handler.context_file_handler.enhanced_context_processor.PGVectorConnector'), \
             patch('src.context_handler.context_file_handler.enhanced_context_processor.AWSTitanEmbeddings'), \
             patch('src.context_handler.context_file_handler.enhanced_context_processor.MetricsManager'), \
             patch('os.makedirs'), \
             patch('pathlib.Path.exists', return_value=False):
            
            # Create processor and load registry normally during init
            processor = EnhancedContextProcessor('../Input/ContextLibrary')
            
            # Now test the method by calling it again with our mock setup still active
            registry = processor._load_document_registry()

            assert registry == {}

    def test_load_document_registry_error(self):
        """Test loading document registry with JSON error"""
        with patch('src.context_handler.context_file_handler.enhanced_context_processor.PGVectorConnector'), \
             patch('src.context_handler.context_file_handler.enhanced_context_processor.AWSTitanEmbeddings'), \
             patch('src.context_handler.context_file_handler.enhanced_context_processor.MetricsManager'), \
             patch('os.makedirs'), \
             patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data='invalid json')):
            
            # Create processor and load registry normally during init  
            processor = EnhancedContextProcessor('../Input/ContextLibrary')
            
            # Now test the method by calling it again with our mock setup still active
            registry = processor._load_document_registry()

            assert registry == {}

    def test_save_document_registry(self, processor):
        """Test saving document registry"""
        mock_registry = {'doc1': {'file_name': 'test.pdf'}}
        processor.document_registry = mock_registry

        with patch('builtins.open', mock_open()) as mock_file:
            processor._save_document_registry()

            mock_file.assert_called_once_with(processor.registry_file, 'w', encoding='utf-8')
            mock_file().write.assert_called_once()
            # Check that JSON was written
            args, _ = mock_file().write.call_args
            assert 'doc1' in args[0]

    def test_save_document_registry_error(self, processor):
        """Test saving document registry with error"""
        processor.document_registry = {'doc1': {'file_name': 'test.pdf'}}

        with patch('builtins.open', side_effect=Exception("Test error")), \
                patch('logging.error') as mock_log:
            processor._save_document_registry()

            mock_log.assert_called_once_with("Could not save document registry: Test error")

    def test_calculate_file_hash(self):
        """Test calculating file hash"""
        with patch('src.context_handler.context_file_handler.enhanced_context_processor.PGVectorConnector'), \
             patch('src.context_handler.context_file_handler.enhanced_context_processor.AWSTitanEmbeddings'), \
             patch('src.context_handler.context_file_handler.enhanced_context_processor.MetricsManager'), \
             patch('os.makedirs'), \
             patch('pathlib.Path.exists', return_value=False), \
             patch('builtins.open', mock_open(read_data=b'test content')):
            processor = EnhancedContextProcessor('../Input/ContextLibrary')
            file_hash = processor._calculate_file_hash('/path/to/file.pdf')

            # Check that a hash was returned
            assert isinstance(file_hash, str)
            assert len(file_hash) > 0

    def test_calculate_file_hash_error(self):
        """Test calculating file hash with error"""
        with patch('src.context_handler.context_file_handler.enhanced_context_processor.PGVectorConnector'), \
             patch('src.context_handler.context_file_handler.enhanced_context_processor.AWSTitanEmbeddings'), \
             patch('src.context_handler.context_file_handler.enhanced_context_processor.MetricsManager'), \
             patch('os.makedirs'), \
             patch('pathlib.Path.exists', return_value=False):
            processor = EnhancedContextProcessor('../Input/ContextLibrary')
            
            # Now test the hash calculation with an error  
            with patch('builtins.open', side_effect=Exception("Test error")), \
                 patch('src.context_handler.context_file_handler.enhanced_context_processor.logger.error') as mock_log:
                file_hash = processor._calculate_file_hash('/path/to/file.pdf')

                mock_log.assert_called_once_with("Could not calculate hash for /path/to/file.pdf: Test error")
                assert file_hash == ""

    def test_is_document_processed_by_hash(self, processor):
        """Test checking if document is processed by hash"""
        # Set up mock registry
        processor.document_registry = {
            'doc1': {
                'file_hash': 'test_hash',
                'file_name': 'test.pdf',
                'file_size': 1024,
                'file_mtime': 1234567890
            }
        }

        # Mock file properties
        with patch('os.path.getsize', return_value=2048), \
                patch('os.path.getmtime', return_value=9876543210):
            # Should match by hash
            result = processor._is_document_processed('/path/to/different_name.pdf')

            assert result['exists'] is True
            assert result['reason'] == 'identical_content'
            assert result['existing_doc_id'] == 'doc1'

    def test_is_document_processed_by_attributes(self, processor):
        """Test checking if document is processed by attributes"""
        # Set up mock registry
        processor.document_registry = {
            'doc1': {
                'file_hash': 'different_hash',
                'file_name': 'test.pdf',
                'file_size': 1024,
                'file_mtime': 1234567890
            }
        }

        # Mock file properties and hash calculation
        processor._calculate_file_hash.return_value = 'new_hash'

        with patch('os.path.getsize', return_value=1024), \
                patch('os.path.getmtime', return_value=1234567890):
            # Should match by name, size, and mtime
            result = processor._is_document_processed('/path/to/test.pdf')

            assert result['exists'] is True
            assert result['reason'] == 'same_file_attributes'
            assert result['existing_doc_id'] == 'doc1'

    def test_is_document_processed_new_file(self, processor):
        """Test checking if document is processed when it's new"""
        # Set up mock registry
        processor.document_registry = {
            'doc1': {
                'file_hash': 'different_hash',
                'file_name': 'different.pdf',
                'file_size': 2048,
                'file_mtime': 9876543210
            }
        }

        # Mock file properties and hash calculation
        processor._calculate_file_hash.return_value = 'new_hash'

        with patch('os.path.getsize', return_value=1024), \
                patch('os.path.getmtime', return_value=1234567890):
            # Should not match
            result = processor._is_document_processed('/path/to/test.pdf')

            assert result['exists'] is False

    def test_register_document(self, processor):
        """Test registering a document"""
        file_path = '/path/to/test.pdf'
        processing_result = {
            'success': True,
            'document_count': 3,
            'collections': ['business_rules'],
            'context_type': 'business_rules',
            'processing_time': 1.5
        }

        # Mock file properties and hash calculation
        with patch('os.path.getsize', return_value=1024), \
                patch('os.path.getmtime', return_value=1234567890), \
                patch('datetime.datetime.now', return_value=datetime(2023, 1, 1, 12, 0, 0)):
            doc_id = processor._register_document(file_path, processing_result)

            # Check that document was registered
            assert doc_id in processor.document_registry
            assert processor.document_registry[doc_id]['file_name'] == 'test.pdf'
            assert processor.document_registry[doc_id]['file_path'] == file_path
            assert processor.document_registry[doc_id]['file_hash'] == 'test_hash'
            assert processor.document_registry[doc_id]['file_size'] == 1024
            assert processor.document_registry[doc_id]['file_mtime'] == 1234567890
            assert processor.document_registry[doc_id]['file_type'] == '.pdf'
            assert processor.document_registry[doc_id]['processed_date'] == '2023-01-01T12:00:00'
            assert processor.document_registry[doc_id]['document_count'] == 3
            assert processor.document_registry[doc_id]['collections'] == ['business_rules']
            assert processor.document_registry[doc_id]['context_type'] == 'business_rules'
            assert processor.document_registry[doc_id]['processing_time'] == 1.5
            assert processor.document_registry[doc_id]['success'] is True

            # Check that registry was saved
            processor._save_document_registry.assert_called_once()

    def test_process_all_context_files(self, processor, temp_context_dir, sample_files):
        """Test processing all context files"""
        # Mock _discover_files and _process_category_files
        with patch.object(processor, '_discover_files') as mock_discover, \
                patch.object(processor, '_process_category_files') as mock_process, \
                patch.object(processor, '_display_final_stats') as mock_display:
            # Configure mock_discover to return files by category
            mock_discover.return_value = {
                'business_rules': [sample_files['pdf'], sample_files['xlsx']],
                'documentation': [sample_files['docx'], sample_files['txt'], sample_files['pptx']],
                'requirements': [],
                'policies': [],
                'examples': [],
                'glossary': [],
                'wireframes': [sample_files['jpg']]
            }

            # Call the method
            result = processor.process_all_context_files()

            # Check that _discover_files was called
            mock_discover.assert_called_once()

            # Check that _process_category_files was called for each category with files
            assert mock_process.call_count == 3
            mock_process.assert_any_call('business_rules', [sample_files['pdf'], sample_files['xlsx']], False)
            mock_process.assert_any_call('documentation',
                                         [sample_files['docx'], sample_files['txt'], sample_files['pptx']], False)
            mock_process.assert_any_call('wireframes', [sample_files['jpg']], False)

            # Check that _display_final_stats was called
            mock_display.assert_called_once()

            # Check the stats
            assert result['total_files'] == 6
            assert 'processing_time' in result

    def test_process_all_context_files_directory_not_found(self, processor):
        """Test processing all context files when directory doesn't exist"""
        with patch('os.path.exists', return_value=False):
            with pytest.raises(FileNotFoundError):
                processor.process_all_context_files()

    def test_discover_files(self, processor, temp_context_dir, sample_files):
        """Test discovering files"""
        processor.context_folder_path = Path(temp_context_dir)

        # Mock os.walk to return our sample files
        with patch('os.walk') as mock_walk, \
                patch('os.path.getsize', lambda path: 600 * 1024 * 1024 if path == sample_files['large'] else 1024):
            mock_walk.return_value = [
                (temp_context_dir, [], ['test.txt', '.hidden_file', 'large_file.txt']),
                (os.path.join(temp_context_dir, 'business_rules'), [], ['test.pdf', 'test.xlsx']),
                (os.path.join(temp_context_dir, 'documentation'), [],
                 ['test.docx', 'test.txt', 'test.pptx', 'test.jpg'])
            ]

            result = processor._discover_files()

            # Check that files were categorized correctly
            assert sample_files['pdf'] in result['business_rules']
            assert sample_files['xlsx'] in result['business_rules']
            assert sample_files['docx'] in result['documentation']
            assert sample_files['txt'] in result['documentation']
            assert sample_files['pptx'] in result['documentation']
            assert sample_files['jpg'] in result['wireframes']  # Should be categorized as wireframes

            # Check that hidden and large files were skipped
            assert sample_files['hidden'] not in sum(result.values(), [])
            assert sample_files['large'] not in sum(result.values(), [])

    def test_categorize_file(self, processor):
        """Test categorizing files"""
        # Test categorization by folder name
        assert processor._categorize_file(Path('/path/to/business_rules/test.pdf'),
                                          'business_rules') == 'business_rules'
        assert processor._categorize_file(Path('/path/to/requirements/test.pdf'), 'requirements') == 'requirements'
        assert processor._categorize_file(Path('/path/to/documentation/test.pdf'), 'documentation') == 'documentation'
        assert processor._categorize_file(Path('/path/to/policies/test.pdf'), 'policies') == 'policies'
        assert processor._categorize_file(Path('/path/to/examples/test.pdf'), 'examples') == 'examples'
        assert processor._categorize_file(Path('/path/to/glossary/test.pdf'), 'glossary') == 'glossary'
        assert processor._categorize_file(Path('/path/to/wireframes/test.pdf'), 'wireframes') == 'wireframes'

        # Test categorization by file name
        assert processor._categorize_file(Path('/path/to/generic/business_rules.pdf'), 'generic') == 'business_rules'
        assert processor._categorize_file(Path('/path/to/generic/requirements.pdf'), 'generic') == 'requirements'
        assert processor._categorize_file(Path('/path/to/generic/documentation.pdf'), 'generic') == 'documentation'
        assert processor._categorize_file(Path('/path/to/generic/policy.pdf'), 'generic') == 'policies'
        assert processor._categorize_file(Path('/path/to/generic/example.pdf'), 'generic') == 'examples'
        assert processor._categorize_file(Path('/path/to/generic/glossary.pdf'), 'generic') == 'glossary'
        assert processor._categorize_file(Path('/path/to/generic/wireframe.pdf'), 'generic') == 'wireframes'

        # Test categorization by file extension
        assert processor._categorize_file(Path('/path/to/generic/image.png'), 'generic') == 'documentation'
        assert processor._categorize_file(Path('/path/to/generic/ui_screen.png'), 'generic') == 'wireframes'
        assert processor._categorize_file(Path('/path/to/generic/data.xlsx'), 'generic') == 'business_rules'
        assert processor._categorize_file(Path('/path/to/generic/glossary.xlsx'), 'generic') == 'glossary'
        assert processor._categorize_file(Path('/path/to/generic/document.pdf'), 'generic') == 'documentation'
        assert processor._categorize_file(Path('/path/to/generic/presentation.pptx'), 'generic') == 'requirements'
        assert processor._categorize_file(Path('/path/to/generic/unknown.xyz'), 'generic') == 'documentation'

    def test_process_category_files(self, processor):
        """Test processing files in a category"""
        category = 'business_rules'
        files = ['/path/to/file1.pdf', '/path/to/file2.xlsx']

        # Mock methods
        with patch.object(processor, '_is_document_processed') as mock_is_processed, \
                patch.object(processor, '_process_unstructured_document') as mock_process_pdf, \
                patch.object(processor, '_process_structured_document') as mock_process_excel, \
                patch.object(processor, '_handle_processing_result') as mock_handle_result:
            # Configure mocks
            mock_is_processed.side_effect = [
                {'exists': True, 'reason': 'already processed'},  # First file is a duplicate
                {'exists': False}  # Second file is new
            ]

            mock_process_excel.return_value = {
                'success': True,
                'document_count': 3,
                'collections': [category],
                'context_type': category,
                'processing_time': 1.5,
                'error': None
            }

            # Call the method
            processor._process_category_files(category, files, False)

            # Check that duplicate check was performed for both files
            assert mock_is_processed.call_count == 2

            # Check that only the second file was processed
            mock_process_pdf.assert_not_called()
            mock_process_excel.assert_called_once_with('/path/to/file2.xlsx', category)

            # Check that result was handled
            mock_handle_result.assert_called_once_with(mock_process_excel.return_value, '/path/to/file2.xlsx', category)

            # Check stats
            assert processor.stats['by_category'][category]['total'] == 2
            assert processor.stats['by_category'][category]['skipped'] == 1
            assert processor.stats['by_category'][category]['processed'] == 0  # Handled by mock_handle_result
            assert processor.stats['by_category'][category]['failed'] == 0

    def test_process_category_files_with_error(self, processor):
        """Test processing files in a category with error"""
        category = 'business_rules'
        files = ['/path/to/file.pdf']

        # Mock methods
        with patch.object(processor, '_is_document_processed', return_value={'exists': False}), \
                patch.object(processor, '_process_unstructured_document', side_effect=Exception("Test error")), \
                patch('logging.error') as mock_log:
            # Call the method
            processor._process_category_files(category, files, False)

            # Check that error was logged
            mock_log.assert_called_once_with(ANY)

            # Check stats
            assert processor.stats['by_category'][category]['total'] == 1
            assert processor.stats['by_category'][category]['skipped'] == 0
            assert processor.stats['by_category'][category]['processed'] == 0
            assert processor.stats['by_category'][category]['failed'] == 1

    def test_process_category_files_with_image_batch(self, processor):
        """Test processing files in a category with image batch"""
        category = 'wireframes'
        files = ['/path/to/image1.jpg', '/path/to/image2.png']

        # Mock methods
        with patch.object(processor, '_is_document_processed', return_value={'exists': False}), \
                patch.object(processor, '_process_image_batch') as mock_process_batch, \
                patch('os.path.getsize', return_value=1024 * 1024):  # 1MB per image

            # Call the method
            processor._process_category_files(category, files, False)

            # Check that image batch was processed
            mock_process_batch.assert_called_once_with(files, category)

    def test_process_image_batch(self, processor):
        """Test processing image batch"""
        category = 'wireframes'
        image_paths = ['/path/to/image1.jpg', '/path/to/image2.png']

        # Mock image parser
        processor.image_parser = Mock()
        processor.image_parser.parse_image_batch.return_value = [
            "Image 1 description",
            "Error: Failed to process image 2"
        ]

        # Mock methods
        with patch.object(processor, '_handle_processing_result') as mock_handle_result:
            # Call the method
            processor._process_image_batch(image_paths, category)

            # Check that image parser was called
            processor.image_parser.parse_image_batch.assert_called_once_with(image_paths)

            # Check that results were handled
            assert mock_handle_result.call_count == 2

            # Check first result (success)
            args1, _ = mock_handle_result.call_args_list[0]
            assert args1[0]['success'] is True
            assert args1[0]['document_count'] == 1
            assert args1[1] == '/path/to/image1.jpg'
            assert args1[2] == category

            # Check second result (error)
            args2, _ = mock_handle_result.call_args_list[1]
            assert args2[0]['success'] is False
            assert args2[0]['error'] == "Error: Failed to process image 2"
            assert args2[1] == '/path/to/image2.png'
            assert args2[2] == category

    def test_process_image_batch_with_error(self, processor):
        """Test processing image batch with error"""
        category = 'wireframes'
        image_paths = ['/path/to/image1.jpg', '/path/to/image2.png']

        # Mock image parser to raise exception
        processor.image_parser = Mock()
        processor.image_parser.parse_image_batch.side_effect = Exception("Batch processing error")

        # Mock logging
        with patch('logging.error') as mock_log:
            # Call the method
            processor._process_image_batch(image_paths, category)

            # Check that error was logged
            mock_log.assert_called_once_with(ANY)

            # Check stats
            assert processor.stats['failed_files'] == 2
            assert processor.stats['by_category'][category]['failed'] == 2

    def test_handle_processing_result_success(self, processor):
        """Test handling processing result with success"""
        file_path = '/path/to/test.pdf'
        category = 'business_rules'
        result = {
            'success': True,
            'document_count': 3,
            'collections': [category],
            'context_type': category,
            'processing_time': 1.5,
            'error': None
        }

        # Mock methods
        with patch.object(processor, '_register_document') as mock_register:
            # Call the method
            processor._handle_processing_result(result, file_path, category)

            # Check that document was registered
            mock_register.assert_called_once_with(file_path, result)

            # Check stats
            assert processor.stats['processed_files'] == 1
            assert processor.stats['by_category'][category]['processed'] == 1
            assert processor.stats['total_chunks'] == 3

    def test_handle_processing_result_failure(self, processor):
        """Test handling processing result with failure"""
        file_path = '/path/to/test.pdf'
        category = 'business_rules'
        result = {
            'success': False,
            'document_count': 0,
            'collections': [category],
            'context_type': category,
            'processing_time': 1.5,
            'error': "Test error"
        }

        # Mock methods
        with patch.object(processor, '_register_document') as mock_register:
            # Call the method
            processor._handle_processing_result(result, file_path, category)

            # Check that document was not registered
            mock_register.assert_not_called()

            # Check stats
            assert processor.stats['failed_files'] == 1
            assert processor.stats['by_category'][category]['failed'] == 1

    def test_process_text_file_success(self, processor):
        """Test processing text file with success"""
        file_path = '/path/to/test.txt'
        category = 'documentation'

        # Mock file reading
        with patch('builtins.open', mock_open(read_data='Test content')), \
                patch('os.path.getsize', return_value=1024):
            # Call the method
            result = processor._process_unstructured_document(file_path, category)

            # Check result
            assert result['success'] is True
            assert result['document_count'] == 1
            assert result['collections'] == [category]
            assert result['context_type'] == category
            assert 'processing_time' in result
            assert result['error'] is None

            # Check that vector store was called
            processor.context_collections[category].vector_store_documents.assert_called_once()
            args, _ = processor.context_collections[category].vector_store_documents.call_args
            assert len(args[0]) == 1
            assert args[0][0]['text'] == 'Test content'
            assert args[0][0]['metadata']['source_file'] == 'test.txt'
            assert args[0][0]['metadata']['file_type'] == 'text'
            assert args[0][0]['metadata']['category'] == category
            assert args[0][0]['metadata']['file_size'] == 1024
            assert args[0][0]['metadata']['character_count'] == 12
            assert args[0][0]['metadata']['line_count'] == 1

    def test_process_text_file_unicode_error(self, processor):
        """Test processing text file with Unicode error"""
        file_path = '/path/to/test.txt'
        category = 'documentation'

        # Mock file reading with Unicode error on first attempt
        with patch('builtins.open') as mock_open:
            mock_open.side_effect = [
                UnicodeDecodeError('utf-8', b'test', 0, 1, 'invalid'),
                mock_open(read_data='Test content').return_value
            ]

            with patch('os.path.getsize', return_value=1024):
                # Call the method
                result = processor._process_unstructured_document(file_path, category)

                # Check result
                assert result['success'] is True
                assert result['document_count'] == 1

    def test_process_text_file_storage_failure(self, processor):
        """Test processing text file with storage failure"""
        file_path = '/path/to/test.txt'
        category = 'documentation'

        # Mock file reading and vector store failure
        with patch('builtins.open', mock_open(read_data='Test content')), \
                patch('os.path.getsize', return_value=1024):
            # Configure vector store to fail
            processor.context_collections[category].vector_store_documents.return_value = False

            # Call the method
            result = processor._process_unstructured_document(file_path, category)

            # Check result
            assert result['success'] is False
            assert result['error'] == "Failed to store documents in vector database"

    def test_process_text_file_exception(self, processor):
        """Test processing text file with exception"""
        file_path = '/path/to/test.txt'
        category = 'documentation'

        # Mock file reading to raise exception
        with patch('builtins.open', side_effect=Exception("Test error")), \
                patch('logging.error') as mock_log:
            # Call the method
            result = processor._process_unstructured_document(file_path, category)

            # Check result
            assert result['success'] is False
            assert result['error'] == "Test error"

            # Check that error was logged
            mock_log.assert_called_once_with(ANY)

    def test_process_unstructured_document_success(self, processor):
        """Test processing PDF file with success"""
        file_path = '/path/to/test.pdf'
        category = 'documentation'

        # Mock PyPDF2
        with patch('PyPDF2.PdfReader') as mock_pdf_reader, \
                patch('builtins.open', mock_open()):
            # Configure mock PDF reader
            mock_page1 = Mock()
            mock_page1.extract_text.return_value = "Page 1 content"
            mock_page2 = Mock()
            mock_page2.extract_text.return_value = "Page 2 content"

            mock_pdf_reader.return_value.pages = [mock_page1, mock_page2]

            # Call the method
            result = processor._process_unstructured_document(file_path, category)

            # Check result
            assert result['success'] is True
            assert result['document_count'] == 2
            assert result['collections'] == [category]
            assert result['context_type'] == category
            assert 'processing_time' in result
            assert result['error'] is None

            # Check that vector store was called
            processor.context_collections[category].vector_store_documents.assert_called_once()
            args, _ = processor.context_collections[category].vector_store_documents.call_args
            assert len(args[0]) == 2
            assert args[0][0]['text'] == "Page 1 content"
            assert args[0][0]['metadata']['source_file'] == 'test.pdf'
            assert args[0][0]['metadata']['file_type'] == 'pdf'
            assert args[0][0]['metadata']['category'] == category
            assert args[0][0]['metadata']['page_number'] == 1
            assert args[0][0]['metadata']['total_pages'] == 2

            assert args[0][1]['text'] == "Page 2 content"
            assert args[0][1]['metadata']['page_number'] == 2

    def test_process_unstructured_document_no_text(self, processor):
        """Test processing PDF file with no text"""
        file_path = '/path/to/test.pdf'
        category = 'documentation'

        # Mock PyPDF2
        with patch('PyPDF2.PdfReader') as mock_pdf_reader, \
                patch('builtins.open', mock_open()):
            # Configure mock PDF reader with empty pages
            mock_page = Mock()
            mock_page.extract_text.return_value = ""

            mock_pdf_reader.return_value.pages = [mock_page]

            # Call the method
            result = processor._process_unstructured_document(file_path, category)

            # Check result
            assert result['success'] is True
            assert result['document_count'] == 1  # Should create a placeholder document

            # Check that vector store was called with placeholder
            processor.context_collections[category].vector_store_documents.assert_called_once()
            args, _ = processor.context_collections[category].vector_store_documents.call_args
            assert len(args[0]) == 1
            assert "PDF file: test.pdf (no text extracted)" in args[0][0]['text']
            assert args[0][0]['metadata']['extraction_failed'] is True

    def test_process_unstructured_document_import_error(self, processor):
        """Test processing PDF file with import error"""
        file_path = '/path/to/test.pdf'
        category = 'documentation'

        # Mock import error
        with patch('PyPDF2.PdfReader', side_effect=ImportError("No module named 'PyPDF2'")), \
                patch('logging.warning') as mock_log:
            # Call the method
            result = processor._process_unstructured_document(file_path, category)

            # Check result
            assert result['success'] is True  # Still succeeds with placeholder
            assert result['document_count'] == 1

            # Check that warning was logged
            mock_log.assert_called_once_with(ANY)

            # Check that vector store was called with placeholder
            processor.context_collections[category].vector_store_documents.assert_called_once()
            args, _ = processor.context_collections[category].vector_store_documents.call_args
            assert len(args[0]) == 1
            assert "PDF file: test.pdf (text extraction failed:" in args[0][0]['text']

    def test_process_unstructured_document_extraction_error(self, processor):
        """Test processing PDF file with extraction error"""
        file_path = '/path/to/test.pdf'
        category = 'documentation'

        # Mock PyPDF2 to raise exception
        with patch('PyPDF2.PdfReader', side_effect=Exception("Extraction error")), \
                patch('logging.warning') as mock_log:
            # Call the method
            result = processor._process_unstructured_document(file_path, category)

            # Check result
            assert result['success'] is True  # Still succeeds with placeholder
            assert result['document_count'] == 1

            # Check that warning was logged
            mock_log.assert_called_once_with(ANY)

            # Check that vector store was called with placeholder
            processor.context_collections[category].vector_store_documents.assert_called_once()
            args, _ = processor.context_collections[category].vector_store_documents.call_args
            assert len(args[0]) == 1
            assert "PDF file: test.pdf (text extraction failed:" in args[0][0]['text']

    def test_process_unstructured_document_storage_failure(self, processor):
        """Test processing PDF file with storage failure"""
        file_path = '/path/to/test.pdf'
        category = 'documentation'

        # Mock PyPDF2
        with patch('PyPDF2.PdfReader') as mock_pdf_reader, \
                patch('builtins.open', mock_open()):
            # Configure mock PDF reader
            mock_page = Mock()
            mock_page.extract_text.return_value = "Page content"

            mock_pdf_reader.return_value.pages = [mock_page]

            # Configure vector store to fail
            processor.context_collections[category].vector_store_documents.return_value = False

            # Call the method
            result = processor._process_unstructured_document(file_path, category)

            # Check result
            assert result['success'] is False
            assert result['error'] == "Failed to store documents in vector database"

    def test_process_docx_success(self, processor):
        """Test processing DOCX file with success"""
        file_path = '/path/to/test.docx'
        category = 'documentation'

        # Mock docx
        with patch('docx.Document') as mock_document:
            # Configure mock document
            mock_doc = Mock()

            # Mock paragraphs
            mock_paragraph1 = Mock()
            mock_paragraph1.text = "Paragraph 1"
            mock_paragraph2 = Mock()
            mock_paragraph2.text = "Paragraph 2"
            mock_doc.paragraphs = [mock_paragraph1, mock_paragraph2]

            # Mock tables
            mock_cell1 = Mock()
            mock_cell1.text = "Cell 1"
            mock_cell2 = Mock()
            mock_cell2.text = "Cell 2"
            mock_row = Mock()
            mock_row.cells = [mock_cell1, mock_cell2]
            mock_table = Mock()
            mock_table.rows = [mock_row]
            mock_doc.tables = [mock_table]

            mock_document.return_value = mock_doc

            # Call the method
            result = processor._process_docx(file_path, category)

            # Check result
            assert result['success'] is True
            assert result['document_count'] == 1
            assert result['collections'] == [category]
            assert result['context_type'] == category
            assert 'processing_time' in result
            assert result['error'] is None

            # Check that vector store was called
            processor.context_collections[category].vector_store_documents.assert_called_once()
            args, _ = processor.context_collections[category].vector_store_documents.call_args
            assert len(args[0]) == 1
            assert "Paragraph 1\nParagraph 2" in args[0][0]['text']
            assert "Tables:\nCell 1 | Cell 2" in args[0][0]['text']
            assert args[0][0]['metadata']['source_file'] == 'test.docx'
            assert args[0][0]['metadata']['file_type'] == 'docx'
            assert args[0][0]['metadata']['category'] == category
            assert args[0][0]['metadata']['paragraph_count'] == 2
            assert args[0][0]['metadata']['table_count'] == 1

    def test_process_docx_no_content(self, processor):
        """Test processing DOCX file with no content"""
        file_path = '/path/to/test.docx'
        category = 'documentation'

        # Mock docx
        with patch('docx.Document') as mock_document:
            # Configure mock document with empty content
            mock_doc = Mock()
            mock_doc.paragraphs = []
            mock_doc.tables = []

            mock_document.return_value = mock_doc

            # Call the method
            result = processor._process_docx(file_path, category)

            # Check result
            assert result['success'] is True
            assert result['document_count'] == 1  # Should create a placeholder document

            # Check that vector store was called with placeholder
            processor.context_collections[category].vector_store_documents.assert_called_once()
            args, _ = processor.context_collections[category].vector_store_documents.call_args
            assert len(args[0]) == 1
            assert "Word document: test.docx (no text content)" in args[0][0]['text']
            assert args[0][0]['metadata']['empty_document'] is True

    def test_process_docx_import_error(self, processor):
        """Test processing DOCX file with import error"""
        file_path = '/path/to/test.docx'
        category = 'documentation'

        # Mock import error
        with patch('docx.Document', side_effect=ImportError("No module named 'docx'")), \
                patch('logging.warning') as mock_log:
            # Call the method
            result = processor._process_docx(file_path, category)

            # Check result
            assert result['success'] is True  # Still succeeds with placeholder
            assert result['document_count'] == 1

            # Check that warning was logged
            mock_log.assert_called_once_with(ANY)

            # Check that vector store was called with placeholder
            processor.context_collections[category].vector_store_documents.assert_called_once()
            args, _ = processor.context_collections[category].vector_store_documents.call_args
            assert len(args[0]) == 1
            assert "Word document: test.docx (text extraction failed:" in args[0][0]['text']

    def test_process_excel_file_success(self, processor):
        """Test processing Excel file with success"""
        file_path = Path('/path/to/test.xlsx')
        category = 'business_rules'

        # Mock pandas
        with patch('pandas.read_excel') as mock_read_excel:
            # Configure mock DataFrame
            mock_df = pd.DataFrame({
                'Column1': ['Value1', 'Value2'],
                'Column2': ['Value3', 'Value4']
            })
            mock_read_excel.return_value = mock_df

            # Call the method
            result = processor._process_structured_document(file_path, category)

            # Check result
            assert result['success'] is True
            assert result['document_count'] == 2  # 2 rows
            assert result['collections'] == [category]
            assert result['context_type'] == category
            assert 'processing_time' in result
            assert result['error'] is None

            # Check that vector store was called
            processor.context_collections[category].vector_store_documents.assert_called_once()
            args, _ = processor.context_collections[category].vector_store_documents.call_args
            assert len(args[0]) == 3

            # Check row documents
            assert args[0][0]['metadata']['source_file'] == 'test.xlsx'
            assert args[0][0]['metadata']['file_type'] == 'excel'
            assert args[0][0]['metadata']['category'] == category
            assert args[0][0]['metadata']['row_index'] == 0

            assert args[0][1]['metadata']['row_index'] == 1

            # Check full sheet document
            assert args[0][2]['metadata']['full_sheet'] is True
            assert args[0][2]['metadata']['row_count'] == 2
            assert args[0][2]['metadata']['column_count'] == 2

    def test_process_excel_file_empty(self, processor):
        """Test processing Excel file with no data"""
        file_path = Path('/path/to/test.xlsx')
        category = 'business_rules'

        # Mock pandas
        with patch('pandas.read_excel') as mock_read_excel:
            # Configure mock DataFrame with no data
            mock_df = pd.DataFrame()
            mock_read_excel.return_value = mock_df

            # Call the method
            result = processor._process_structured_document(file_path, category)

            # Check result
            assert result['success'] is True
            assert result['document_count'] == 1  # Should create a placeholder document

            # Check that vector store was called with placeholder
            processor.context_collections[category].vector_store_documents.assert_called_once()
            args, _ = processor.context_collections[category].vector_store_documents.call_args
            assert len(args[0]) == 1
            assert "Excel file: test.xlsx (no data extracted)" in args[0][0]['text']
            assert args[0][0]['metadata']['empty_file'] is True

    def test_process_excel_file_error(self, processor):
        """Test processing Excel file with error"""
        file_path = Path('/path/to/test.xlsx')
        category = 'business_rules'

        # Mock pandas to raise exception
        with patch('pandas.read_excel', side_effect=Exception("Excel error")), \
                patch('logging.warning') as mock_log:
            # Call the method
            result = processor._process_structured_document(file_path, category)

            # Check result
            assert result['success'] is True  # Still succeeds with placeholder
            assert result['document_count'] == 1

            # Check that warning was logged
            mock_log.assert_called_once_with(ANY)

            # Check that vector store was called with placeholder
            processor.context_collections[category].vector_store_documents.assert_called_once()
            args, _ = processor.context_collections[category].vector_store_documents.call_args
            assert len(args[0]) == 1
            assert "Excel file: test.xlsx (data extraction failed:" in args[0][0]['text']

    def test_row_to_text(self, processor):
        """Test converting DataFrame row to text"""
        row = pd.Series({
            'Column1': 'Value1',
            'Column2': 'Value2',
            'Column3': None,  # Should be skipped
            'Column4': ''  # Should be skipped
        })

        result = processor._row_to_text(row)

        assert "Column1: Value1" in result
        assert "Column2: Value2" in result
        assert "Column3" not in result
        assert "Column4" not in result

    def test_process_presentation_file_success(self, processor):
        """Test processing PowerPoint file with success"""
        file_path = '/path/to/test.pptx'
        category = 'requirements'

        # Mock python-pptx
        with patch('pptx.Presentation') as mock_presentation:
            # Configure mock presentation
            mock_prs = Mock()

            # Mock slides
            mock_shape1 = Mock()
            mock_shape1.text = "Slide 1 Shape 1"
            mock_shape2 = Mock()
            mock_shape2.text = "Slide 1 Shape 2"
            mock_slide1 = Mock()
            mock_slide1.shapes = [mock_shape1, mock_shape2]

            mock_shape3 = Mock()
            mock_shape3.text = "Slide 2 Shape 1"
            mock_slide2 = Mock()
            mock_slide2.shapes = [mock_shape3]

            mock_prs.slides = [mock_slide1, mock_slide2]

            mock_presentation.return_value = mock_prs

            # Call the method
            result = processor._process_presentation(file_path, category)

            # Check result
            assert result['success'] is True
            assert result['document_count'] == 2
            assert result['collections'] == [category]
            assert result['context_type'] == category
            assert 'processing_time' in result
            assert result['error'] is None

            # Check that vector store was called
            processor.context_collections[category].vector_store_documents.assert_called_once()
            args, _ = processor.context_collections[category].vector_store_documents.call_args
            assert len(args[0]) == 2

            # Check slide documents
            assert args[0][0]['text'] == "Slide 1 Shape 1\nSlide 1 Shape 2"
            assert args[0][0]['metadata']['source_file'] == 'test.pptx'
            assert args[0][0]['metadata']['file_type'] == 'pptx'
            assert args[0][0]['metadata']['slide_number'] == 1
            assert args[0][0]['metadata']['total_slides'] == 2

            assert args[0][1]['text'] == "Slide 2 Shape 1"
            assert args[0][1]['metadata']['slide_number'] == 2

    def test_process_presentation_file_no_text(self, processor):
        """Test processing PowerPoint file with no text"""
        file_path = '/path/to/test.pptx'
        category = 'requirements'

        # Mock python-pptx
        with patch('pptx.Presentation') as mock_presentation:
            # Configure mock presentation with empty slides
            mock_prs = Mock()

            # Mock empty slide
            mock_shape = Mock()
            mock_shape.text = ""
            mock_slide = Mock()
            mock_slide.shapes = [mock_shape]

            mock_prs.slides = [mock_slide]

            mock_presentation.return_value = mock_prs

            # Call the method
            result = processor._process_presentation(file_path, category)

            # Check result
            assert result['success'] is True
            assert result['document_count'] == 1  # Should create a placeholder document

            # Check that vector store was called with placeholder
            processor.context_collections[category].vector_store_documents.assert_called_once()
            args, _ = processor.context_collections[category].vector_store_documents.call_args
            assert len(args[0]) == 1
            assert "PowerPoint file: test.pptx (no text extracted)" in args[0][0]['text']
            assert args[0][0]['metadata']['empty_presentation'] is True

    def test_process_presentation_file_import_error(self, processor):
        """Test processing PowerPoint file with import error"""
        file_path = '/path/to/test.pptx'
        category = 'requirements'

        # Mock import error
        with patch('pptx.Presentation', side_effect=ImportError("No module named 'pptx'")), \
                patch('logging.warning') as mock_log:
            # Call the method
            result = processor._process_presentation(file_path, category)

            # Check result
            assert result['success'] is True  # Still succeeds with placeholder
            assert result['document_count'] == 1

            # Check that warning was logged
            mock_log.assert_called_once_with(ANY)

            # Check that vector store was called with placeholder
            processor.context_collections[category].vector_store_documents.assert_called_once()
            args, _ = processor.context_collections[category].vector_store_documents.call_args
            assert len(args[0]) == 1
            assert "PowerPoint file: test.pptx (text extraction failed:" in args[0][0]['text']

    def test_process_image_file_success(self, processor):
        """Test processing image file with success"""
        file_path = Path('/path/to/test.jpg')
        category = 'wireframes'

        # Mock image metadata extraction
        with patch.object(processor, '_extract_image_metadata') as mock_extract_metadata, \
                patch.object(processor, '_extract_text_from_image') as mock_extract_text:
            # Configure mocks
            mock_extract_metadata.return_value = {
                'format': 'JPEG',
                'mode': 'RGB',
                'size': '800x600',
                'file_size': 102400
            }
            mock_extract_text.return_value = "Extracted text from image"

            # Set OCR_ENABLED to true
            with patch.dict('os.environ', {'OCR_ENABLED': 'true'}):
                # Call the method
                result = processor._process_image_file(file_path, category)

                # Check result
                assert result['success'] is True
                assert result['document_count'] == 1
                assert result['collections'] == [category]
                assert result['context_type'] == category
                assert 'processing_time' in result
                assert result['error'] is None

                # Check that vector store was called
                processor.context_collections[category].vector_store_documents.assert_called_once()
                args, _ = processor.context_collections[category].vector_store_documents.call_args
                assert len(args[0]) == 1
                assert "Image: test.jpg" in args[0][0]['text']
                assert "Type: JPEG" in args[0][0]['text']
                assert "Size: 800x600" in args[0][0]['text']
                assert "Extracted Text: Extracted text from image" in args[0][0]['text']
                assert args[0][0]['metadata']['source_file'] == 'test.jpg'
                assert args[0][0]['metadata']['file_type'] == 'image'
                assert args[0][0]['metadata']['image_metadata']['format'] == 'JPEG'

    def test_process_image_file_no_ocr(self, processor):
        """Test processing image file without OCR"""
        file_path = Path('/path/to/test.jpg')
        category = 'wireframes'

        # Mock image metadata extraction
        with patch.object(processor, '_extract_image_metadata') as mock_extract_metadata, \
                patch.object(processor, '_extract_text_from_image') as mock_extract_text:
            # Configure mocks
            mock_extract_metadata.return_value = {
                'format': 'JPEG',
                'mode': 'RGB',
                'size': '800x600',
                'file_size': 102400
            }

            # Set OCR_ENABLED to false
            with patch.dict('os.environ', {'OCR_ENABLED': 'false'}):
                # Call the method
                result = processor._process_image_file(file_path, category)

                # Check result
                assert result['success'] is True
                assert result['document_count'] == 1

                # Check that OCR was not called
                mock_extract_text.assert_not_called()

                # Check that vector store was called
                processor.context_collections[category].vector_store_documents.assert_called_once()
                args, _ = processor.context_collections[category].vector_store_documents.call_args
                assert len(args[0]) == 1
                assert "Image: test.jpg" in args[0][0]['text']
                assert "Extracted Text:" not in args[0][0]['text']

    def test_process_image_file_storage_failure(self, processor):
        """Test processing image file with storage failure"""
        file_path = Path('/path/to/test.jpg')
        category = 'wireframes'

        # Mock image metadata extraction
        with patch.object(processor, '_extract_image_metadata') as mock_extract_metadata:
            # Configure mocks
            mock_extract_metadata.return_value = {
                'format': 'JPEG',
                'size': '800x600'
            }

            # Configure vector store to fail
            processor.context_collections[category].vector_store_documents.return_value = False

            # Call the method
            result = processor._process_image_file(file_path, category)

            # Check result
            assert result['success'] is False
            assert result['error'] == "Failed to store documents in vector database"

    def test_extract_image_metadata(self, processor):
        """Test extracting image metadata"""
        file_path = '/path/to/test.jpg'

        # Mock PIL.Image
        with patch('PIL.Image.open') as mock_open:
            # Configure mock image
            mock_img = Mock()
            mock_img.format = 'JPEG'
            mock_img.mode = 'RGB'
            mock_img.width = 800
            mock_img.height = 600

            mock_open.return_value.__enter__.return_value = mock_img

            with patch('os.path.getsize', return_value=102400):
                # Call the method
                result = processor._extract_image_metadata(file_path)

                # Check result
                assert result['format'] == 'JPEG'
                assert result['mode'] == 'RGB'
                assert result['size'] == '800x600'
                assert result['file_size'] == 102400

    def test_extract_image_metadata_error(self, processor):
        """Test extracting image metadata with error"""
        file_path = '/path/to/test.jpg'

        # Mock PIL.Image to raise exception
        with patch('PIL.Image.open', side_effect=Exception("Image error")):
            # Call the method
            result = processor._extract_image_metadata(file_path)

            # Check result
            assert result['format'] == 'Unknown'
            assert result['size'] == 'Unknown'

    def test_extract_text_from_image(self, processor):
        """Test extracting text from image"""
        file_path = '/path/to/test.jpg'

        # Mock pytesseract
        with patch('pytesseract.image_to_string', return_value="Extracted text"), \
                patch('PIL.Image.open') as mock_open:
            # Configure mock image
            mock_img = Mock()
            mock_open.return_value.__enter__.return_value = mock_img

            # Call the method
            result = processor._extract_text_from_image(file_path)

            # Check result
            assert result == "Extracted text"

    def test_extract_text_from_image_error(self, processor):
        """Test extracting text from image with error"""
        file_path = '/path/to/test.jpg'

        # Mock pytesseract to raise exception
        with patch('pytesseract.image_to_string', side_effect=Exception("OCR error")), \
                patch('PIL.Image.open') as mock_open, \
                patch('logging.warning') as mock_log:
            # Configure mock image
            mock_img = Mock()
            mock_open.return_value.__enter__.return_value = mock_img

            # Call the method
            result = processor._extract_text_from_image(file_path)

            # Check result
            assert result == ""

            # Check that warning was logged
            mock_log.assert_called_once_with(ANY)

    def test_display_final_stats(self, processor, capsys):
        """Test displaying final stats"""
        # Set up stats
        processor.stats = {
            'total_files': 10,
            'processed_files': 7,
            'skipped_files': 2,
            'failed_files': 1,
            'total_chunks': 20,
            'processing_time': 5.5,
            'by_category': {
                'business_rules': {'total': 5, 'processed': 4, 'skipped': 1, 'failed': 0},
                'documentation': {'total': 3, 'processed': 2, 'skipped': 0, 'failed': 1},
                'requirements': {'total': 2, 'processed': 1, 'skipped': 1, 'failed': 0},
                'empty_category': {'total': 0, 'processed': 0, 'skipped': 0, 'failed': 0}
            }
        }

        # Call the method
        processor._display_final_stats()

        # Capture output
        captured = capsys.readouterr()

        # Check that stats were displayed
        assert "Context Processing Complete" in captured.out
        assert "Total files found: 10" in captured.out
        assert "Successfully processed: 7" in captured.out
        assert "Skipped files: 2" in captured.out
        assert "Failed files: 1" in captured.out
        assert "Total chunks created: 20" in captured.out
        assert "Processing time: 5.50s" in captured.out

        # Check category stats
        assert "business_rules: 4 processed, 1 skipped, 0 failed" in captured.out
        assert "documentation: 2 processed, 0 skipped, 1 failed" in captured.out
        assert "requirements: 1 processed, 1 skipped, 0 failed" in captured.out
        assert "empty_category" not in captured.out  # Empty categories should be skipped
