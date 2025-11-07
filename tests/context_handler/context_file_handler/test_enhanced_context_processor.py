# import pytest
# import os
# import json
# import tempfile
# import shutil
# import pandas as pd
# import numpy as np
# from unittest.mock import patch, Mock, MagicMock, ANY, mock_open
# from datetime import datetime
# from pathlib import Path
#
# from src.context_handler.context_file_handler.enhanced_context_processor import EnhancedContextProcessor
#
#
# class TestEnhancedContextProcessor:
#     @pytest.fixture
#     def metrics_manager_mock(self):
#         """Mock metrics manager for testing"""
#         mock = Mock()
#         mock.record_llm_call = Mock()
#         mock.record_error = Mock()
#         mock.record_vector_operation = Mock()
#         return mock
#
#     @pytest.fixture
#     def pgvector_connector_mock(self):
#         """Mock PGVectorConnector for testing"""
#         mock = Mock()
#         mock.vector_store_documents.return_value = True
#         mock.diagnose_database.return_value = True
#         return mock
#
#     @pytest.fixture
#     def temp_context_dir(self):
#         """Create a temporary directory for context files"""
#         temp_dir = tempfile.mkdtemp()
#         yield temp_dir
#         shutil.rmtree(temp_dir)
#
#     @pytest.fixture
#     def processor(self, metrics_manager_mock):
#         """Create an EnhancedContextProcessor instance with mocked dependencies"""
#         with patch('src.context_handler.context_file_handler.enhanced_context_processor.PGVectorConnector',
#                    return_value=Mock()) as mock_pgvector, \
#                 patch('src.context_handler.context_file_handler.enhanced_context_processor.AWSTitanEmbeddings',
#                       return_value=Mock()) as mock_embeddings:
#             processor = EnhancedContextProcessor('/fake/context/path', metrics_manager=metrics_manager_mock)
#
#             # Replace context collections with mocks
#             for collection_name in processor.context_collections:
#                 processor.context_collections[collection_name] = Mock()
#                 processor.context_collections[collection_name].vector_store_documents.return_value = True
#
#             yield processor
#
#     @pytest.fixture
#     def sample_files(self, temp_context_dir):
#         """Create sample files of different types for testing"""
#         # Create directories
#         os.makedirs(os.path.join(temp_context_dir, 'business_rules'), exist_ok=True)
#         os.makedirs(os.path.join(temp_context_dir, 'requirements'), exist_ok=True)
#
#         # Create sample files
#         files = {
#             'pdf': os.path.join(temp_context_dir, 'business_rules', 'test.pdf'),
#             'docx': os.path.join(temp_context_dir, 'requirements', 'test.docx'),
#             'xlsx': os.path.join(temp_context_dir, 'test.xlsx'),
#             'txt': os.path.join(temp_context_dir, 'test.txt'),
#             'jpg': os.path.join(temp_context_dir, 'test.jpg'),
#             'pptx': os.path.join(temp_context_dir, 'test.pptx'),
#             'hidden': os.path.join(temp_context_dir, '.hidden.txt'),
#             'large': os.path.join(temp_context_dir, 'large.pdf')
#         }
#
#         # Create empty files
#         for file_path in files.values():
#             with open(file_path, 'w') as f:
#                 f.write('test content')
#
#         # Make the large file appear large
#         with patch('os.path.getsize', lambda path: 600 * 1024 * 1024 if path == files['large'] else 1024):
#             yield files
#
#     def test_initialization(self, metrics_manager_mock):
#         """Test initialization of EnhancedContextProcessor"""
#         with patch(
#                 'src.context_handler.context_file_handler.enhanced_context_processor.PGVectorConnector') as mock_pgvector, \
#                 patch(
#                     'src.context_handler.context_file_handler.enhanced_context_processor.AWSTitanEmbeddings') as mock_embeddings, \
#                 patch('os.makedirs') as mock_makedirs:
#             processor = EnhancedContextProcessor('/test/path', metrics_manager=metrics_manager_mock)
#
#             # Check that directories were created
#             mock_makedirs.assert_any_call('../Data/DocumentRegistry', exist_ok=True)
#             mock_makedirs.assert_any_call('../Data/LocalEmbeddings', exist_ok=True)
#
#             # Check that context collections were initialized
#             assert 'business_rules' in processor.context_collections
#             assert 'requirements' in processor.context_collections
#             assert 'documentation' in processor.context_collections
#             assert 'policies' in processor.context_collections
#             assert 'examples' in processor.context_collections
#             assert 'glossary' in processor.context_collections
#             assert 'wireframes' in processor.context_collections
#
#             # Check that PGVectorConnector was called for each collection
#             assert mock_pgvector.call_count == 7
#
#             # Check that AWSTitanEmbeddings was initialized
#             mock_embeddings.assert_called_once()
#
#     def test_load_document_registry_success(self, processor, temp_context_dir):
#         """Test loading document registry successfully"""
#         # Create a registry file
#         registry_file = os.path.join(temp_context_dir, 'document_registry.json')
#         test_data = {'doc1': {'file_name': 'test.pdf'}}
#
#         with open(registry_file, 'w') as f:
#             json.dump(test_data, f)
#
#         # Set the registry file path
#         processor.registry_file = Path(registry_file)
#
#         # Load the registry
#         result = processor._load_document_registry()
#
#         # Check that the registry was loaded
#         assert result == test_data
#
#     def test_load_document_registry_not_exists(self, processor):
#         """Test loading document registry when file doesn't exist"""
#         # Set the registry file to a non-existent path
#         processor.registry_file = Path('/nonexistent/path/registry.json')
#
#         # Load the registry
#         result = processor._load_document_registry()
#
#         # Should return empty dict
#         assert result == {}
#
#     def test_load_document_registry_json_error(self, processor, temp_context_dir):
#         """Test loading document registry with invalid JSON"""
#         # Create an invalid registry file
#         registry_file = os.path.join(temp_context_dir, 'invalid_registry.json')
#         with open(registry_file, 'w') as f:
#             f.write('{"invalid": json}')
#
#         # Set the registry file path
#         processor.registry_file = Path(registry_file)
#
#         # Load the registry with warning logged
#         with patch('logging.warning') as mock_warning:
#             result = processor._load_document_registry()
#             mock_warning.assert_called_once()
#             assert result == {}
#
#     def test_save_document_registry_success(self, processor, temp_context_dir):
#         """Test saving document registry successfully"""
#         # Set up registry file and data
#         registry_file = os.path.join(temp_context_dir, 'save_registry.json')
#         processor.registry_file = Path(registry_file)
#         processor.document_registry = {'doc1': {'file_name': 'test.pdf'}}
#
#         # Save the registry
#         processor._save_document_registry()
#
#         # Check that the file was created with correct content
#         assert os.path.exists(registry_file)
#         with open(registry_file, 'r') as f:
#             saved_data = json.load(f)
#             assert saved_data == processor.document_registry
#
#     def test_save_document_registry_error(self, processor):
#         """Test error handling when saving registry fails"""
#         # Set up registry file to a path that will cause an error
#         processor.registry_file = Path('/nonexistent/directory/registry.json')
#         processor.document_registry = {'doc1': {'file_name': 'test.pdf'}}
#
#         # Try to save with error logged
#         with patch('logging.error') as mock_error:
#             processor._save_document_registry()
#             mock_error.assert_called_once()
#
#     def test_calculate_file_hash(self, processor, temp_context_dir):
#         """Test calculating file hash"""
#         # Create a test file
#         test_file = os.path.join(temp_context_dir, 'hash_test.txt')
#         with open(test_file, 'w') as f:
#             f.write('test content')
#
#         # Calculate hash
#         file_hash = processor._calculate_file_hash(test_file)
#
#         # Should be a non-empty string
#         assert isinstance(file_hash, str)
#         assert len(file_hash) > 0
#
#     def test_calculate_file_hash_error(self, processor):
#         """Test error handling when calculating hash fails"""
#         # Try to calculate hash for non-existent file
#         with patch('logging.error') as mock_error:
#             file_hash = processor._calculate_file_hash('/nonexistent/file.txt')
#             mock_error.assert_called_once()
#             assert file_hash == ""
#
#     def test_is_document_processed_by_hash(self, processor):
#         """Test checking if a document is processed by hash"""
#         # Set up registry with a known document
#         processor.document_registry = {
#             'doc1': {
#                 'file_hash': 'abc123',
#                 'file_name': 'test.pdf',
#                 'file_size': 1024,
#                 'file_mtime': 1625097600.0
#             }
#         }
#
#         # Mock hash calculation to return the known hash
#         with patch.object(processor, '_calculate_file_hash', return_value='abc123'):
#             result = processor._is_document_processed('/path/to/test.pdf')
#
#             assert result['exists'] is True
#             assert result['reason'] == 'identical_content'
#             assert result['existing_doc_id'] == 'doc1'
#
#     def test_is_document_processed_by_attributes(self, processor):
#         """Test checking if a document is processed by attributes"""
#         # Set up registry with a known document
#         processor.document_registry = {
#             'doc1': {
#                 'file_hash': 'abc123',
#                 'file_name': 'test.pdf',
#                 'file_size': 1024,
#                 'file_mtime': 1625097600.0
#             }
#         }
#
#         # Mock hash calculation to return a different hash but same attributes
#         with patch.object(processor, '_calculate_file_hash', return_value='different_hash'), \
#                 patch('os.path.getsize', return_value=1024), \
#                 patch('os.path.getmtime', return_value=1625097600.0):
#             result = processor._is_document_processed('/path/to/test.pdf')
#
#             assert result['exists'] is True
#             assert result['reason'] == 'same_file_attributes'
#             assert result['existing_doc_id'] == 'doc1'
#
#     def test_is_document_processed_new_document(self, processor):
#         """Test checking a new document that hasn't been processed"""
#         # Set up registry with a known document
#         processor.document_registry = {
#             'doc1': {
#                 'file_hash': 'abc123',
#                 'file_name': 'test.pdf',
#                 'file_size': 1024,
#                 'file_mtime': 1625097600.0
#             }
#         }
#
#         # Mock hash calculation and file attributes for a new document
#         with patch.object(processor, '_calculate_file_hash', return_value='new_hash'), \
#                 patch('os.path.getsize', return_value=2048), \
#                 patch('os.path.getmtime', return_value=1625184000.0):
#             result = processor._is_document_processed('/path/to/new.pdf')
#
#             assert result['exists'] is False
#
#     def test_register_document(self, processor):
#         """Test registering a new document"""
#         # Mock file operations
#         with patch.object(processor, '_calculate_file_hash', return_value='test_hash'), \
#                 patch('os.path.getsize', return_value=1024), \
#                 patch('os.path.getmtime', return_value=1625097600.0), \
#                 patch.object(processor, '_save_document_registry') as mock_save:
#             # Processing result
#             processing_result = {
#                 'success': True,
#                 'document_count': 3,
#                 'collections': ['business_rules'],
#                 'context_type': 'business_rules',
#                 'processing_time': 1.2
#             }
#
#             # Register the document
#             doc_id = processor._register_document('/path/to/test.pdf', processing_result)
#
#             # Check that the document was registered
#             assert doc_id in processor.document_registry
#             assert processor.document_registry[doc_id]['file_name'] == 'test.pdf'
#             assert processor.document_registry[doc_id]['file_hash'] == 'test_hash'
#             assert processor.document_registry[doc_id]['document_count'] == 3
#             assert processor.document_registry[doc_id]['collections'] == ['business_rules']
#             assert processor.document_registry[doc_id]['context_type'] == 'business_rules'
#             assert processor.document_registry[doc_id]['processing_time'] == 1.2
#             assert processor.document_registry[doc_id]['success'] is True
#
#             # Check that save was called
#             mock_save.assert_called_once()
#
#     def test_process_all_context_files(self, processor):
#         """Test processing all context files"""
#         # Mock _discover_files and _process_category_files
#         with patch.object(processor, '_discover_files') as mock_discover, \
#                 patch.object(processor, '_process_category_files') as mock_process, \
#                 patch.object(processor, '_display_final_stats') as mock_display:
#             # Set up mock discover to return some files
#             mock_discover.return_value = {
#                 'business_rules': ['/path/to/rule1.pdf', '/path/to/rule2.docx'],
#                 'requirements': ['/path/to/req1.xlsx'],
#                 'documentation': [],
#                 'policies': [],
#                 'examples': [],
#                 'glossary': [],
#                 'wireframes': []
#             }
#
#             # Process all files
#             result = processor.process_all_context_files()
#
#             # Check that discover and process were called
#             mock_discover.assert_called_once()
#             assert mock_process.call_count == 2  # Only for categories with files
#             mock_display.assert_called_once()
#
#             # Check stats
#             assert result['total_files'] == 3
#             assert 'by_category' in result
#
#     def test_process_all_context_files_nonexistent_dir(self, processor):
#         """Test processing with non-existent directory"""
#         # Set context folder to non-existent path
#         processor.context_folder_path = Path('/nonexistent/path')
#
#         # Should raise FileNotFoundError
#         with pytest.raises(FileNotFoundError):
#             processor.process_all_context_files()
#
#     def test_discover_files(self, processor, sample_files, temp_context_dir):
#         """Test discovering files in context folder"""
#         # Set context folder to temp directory
#         processor.context_folder_path = Path(temp_context_dir)
#
#         # Discover files
#         with patch('os.path.getsize', lambda path: 600 * 1024 * 1024 if 'large' in path else 1024):
#             result = processor._discover_files()
#
#         # Check that files were discovered and categorized
#         assert len(result['business_rules']) == 1  # test.pdf in business_rules dir
#         assert len(result['requirements']) == 1  # test.docx in requirements dir
#         assert any(str(path).endswith('test.xlsx') for path in result['business_rules'])  # xlsx in root
#         assert any(str(path).endswith('test.txt') for path in result['documentation'])  # txt in root
#         assert any(str(path).endswith('test.jpg') for path in result['documentation'])  # jpg in root
#         assert any(str(path).endswith('test.pptx') for path in result['requirements'])  # pptx in root
#
#         # Hidden and large files should be skipped
#         for category in result.values():
#             assert not any('hidden' in str(path) for path in category)
#             assert not any('large' in str(path) for path in category)
#
#     def test_categorize_file(self, processor):
#         """Test categorizing files based on path and name"""
#         # Test directory-based categorization
#         assert processor._categorize_file(Path('/path/to/business_rules/doc.pdf'), 'business_rules') == 'business_rules'
#         assert processor._categorize_file(Path('/path/to/requirements/doc.pdf'), 'requirements') == 'requirements'
#         assert processor._categorize_file(Path('/path/to/documentation/doc.pdf'), 'documentation') == 'documentation'
#         assert processor._categorize_file(Path('/path/to/policies/doc.pdf'), 'policies') == 'policies'
#         assert processor._categorize_file(Path('/path/to/examples/doc.pdf'), 'examples') == 'examples'
#         assert processor._categorize_file(Path('/path/to/glossary/doc.pdf'), 'glossary') == 'glossary'
#         assert processor._categorize_file(Path('/path/to/wireframes/doc.pdf'), 'wireframes') == 'wireframes'
#
#         # Test filename-based categorization
#         assert processor._categorize_file(Path('/path/to/business_rule.pdf'), 'generic') == 'business_rules'
#         assert processor._categorize_file(Path('/path/to/requirement_spec.pdf'), 'generic') == 'requirements'
#         assert processor._categorize_file(Path('/path/to/user_guide.pdf'), 'generic') == 'documentation'
#         assert processor._categorize_file(Path('/path/to/policy_doc.pdf'), 'generic') == 'policies'
#         assert processor._categorize_file(Path('/path/to/example_template.pdf'), 'generic') == 'examples'
#         assert processor._categorize_file(Path('/path/to/glossary_terms.pdf'), 'generic') == 'glossary'
#         assert processor._categorize_file(Path('/path/to/ui_wireframe.pdf'), 'generic') == 'wireframes'
#
#         # Test extension-based categorization
#         assert processor._categorize_file(Path('/path/to/generic.pdf'), 'generic') == 'documentation'
#         assert processor._categorize_file(Path('/path/to/generic.xlsx'), 'generic') == 'business_rules'
#         assert processor._categorize_file(Path('/path/to/glossary.xlsx'), 'generic') == 'glossary'
#         assert processor._categorize_file(Path('/path/to/generic.pptx'), 'generic') == 'requirements'
#         assert processor._categorize_file(Path('/path/to/ui_screen.jpg'), 'generic') == 'wireframes'
#         assert processor._categorize_file(Path('/path/to/generic.jpg'), 'generic') == 'documentation'
#
#     def test_process_category_files(self, processor):
#         """Test processing files in a category"""
#         # Mock methods
#         with patch.object(processor, '_is_document_processed') as mock_is_processed, \
#                 patch.object(processor, '_process_single_file') as mock_process_file, \
#                 patch.object(processor, '_register_document') as mock_register:
#             # Set up mock to indicate new files
#             mock_is_processed.return_value = {'exists': False}
#
#             # Set up mock to return success
#             mock_process_file.return_value = {
#                 'success': True,
#                 'document_count': 3,
#                 'collections': ['business_rules'],
#                 'error': None
#             }
#
#             # Process category files
#             files = [Path('/path/to/file1.pdf'), Path('/path/to/file2.docx')]
#             processor._process_category_files('business_rules', files, False)
#
#             # Check that methods were called correctly
#             assert mock_is_processed.call_count == 2
#             assert mock_process_file.call_count == 2
#             assert mock_register.call_count == 2
#
#             # Check stats
#             assert processor.stats['processed_files'] == 2
#             assert processor.stats['by_category']['business_rules']['processed'] == 2
#             assert processor.stats['total_chunks'] == 6
#
#     def test_process_category_files_with_duplicates(self, processor):
#         """Test processing files with duplicates"""
#         # Mock methods
#         with patch.object(processor, '_is_document_processed') as mock_is_processed, \
#                 patch.object(processor, '_process_single_file') as mock_process_file:
#             # Set up mock to indicate duplicates
#             mock_is_processed.return_value = {'exists': True, 'reason': 'already processed'}
#
#             # Process category files
#             files = [Path('/path/to/file1.pdf'), Path('/path/to/file2.docx')]
#             processor._process_category_files('business_rules', files, False)
#
#             # Check that process_file was not called
#             mock_process_file.assert_not_called()
#
#             # Check stats
#             assert processor.stats['skipped_files'] == 2
#             assert processor.stats['by_category']['business_rules']['skipped'] == 2
#
#     def test_process_category_files_with_errors(self, processor):
#         """Test processing files with errors"""
#         # Mock methods
#         with patch.object(processor, '_is_document_processed') as mock_is_processed, \
#                 patch.object(processor, '_process_single_file') as mock_process_file:
#             # Set up mock to indicate new files
#             mock_is_processed.return_value = {'exists': False}
#
#             # Set up mock to return failure
#             mock_process_file.return_value = {
#                 'success': False,
#                 'document_count': 0,
#                 'collections': [],
#                 'error': 'Test error'
#             }
#
#             # Process category files
#             files = [Path('/path/to/file1.pdf')]
#             processor._process_category_files('business_rules', files, False)
#
#             # Check stats
#             assert processor.stats['failed_files'] == 1
#             assert processor.stats['by_category']['business_rules']['failed'] == 1
#
#     def test_process_category_files_with_exception(self, processor):
#         """Test processing files with exception"""
#         # Mock methods
#         with patch.object(processor, '_is_document_processed') as mock_is_processed, \
#                 patch.object(processor, '_process_single_file') as mock_process_file, \
#                 patch('logging.error') as mock_error:
#             # Set up mock to indicate new files
#             mock_is_processed.return_value = {'exists': False}
#
#             # Set up mock to raise exception
#             mock_process_file.side_effect = Exception("Test exception")
#
#             # Process category files
#             files = [Path('/path/to/file1.pdf')]
#             processor._process_category_files('business_rules', files, False)
#
#             # Check that error was logged
#             mock_error.assert_called_once()
#
#             # Check stats
#             assert processor.stats['failed_files'] == 1
#             assert processor.stats['by_category']['business_rules']['failed'] == 1
#
#     def test_process_category_files_force_reprocess(self, processor):
#         """Test processing files with force_reprocess=True"""
#         # Mock methods
#         with patch.object(processor, '_is_document_processed') as mock_is_processed, \
#                 patch.object(processor, '_process_single_file') as mock_process_file, \
#                 patch.object(processor, '_register_document') as mock_register:
#             # Set up mock to return success
#             mock_process_file.return_value = {
#                 'success': True,
#                 'document_count': 3,
#                 'collections': ['business_rules'],
#                 'error': None
#             }
#
#             # Process category files with force_reprocess=True
#             files = [Path('/path/to/file1.pdf')]
#             processor._process_category_files('business_rules', files, True)
#
#             # Check that is_processed was not called
#             mock_is_processed.assert_not_called()
#
#             # Check that process_file was called
#             mock_process_file.assert_called_once()
#
#             # Check stats
#             assert processor.stats['processed_files'] == 1
#             assert processor.stats['by_category']['business_rules']['processed'] == 1
#
#     def test_process_structured_document_excel(self, processor):
#         """Test processing Excel document"""
#         # Mock pandas read_excel
#         mock_df = pd.DataFrame({
#             'col1': ['value1', 'value2'],
#             'col2': ['value3', 'value4']
#         })
#
#         with patch('pandas.read_excel', return_value=mock_df), \
#                 patch('time.time', side_effect=[100, 101]):  # Start and end times
#
#             result = processor._process_structured_document(Path('/path/to/test.xlsx'), 'business_rules')
#
#             # Check result
#             assert result['success'] is True
#             assert result['document_count'] == 2  # One per row
#             assert result['collections'] == ['business_rules']
#             assert result['context_type'] == 'business_rules'
#             assert result['processing_time'] == 1  # 101 - 100
#
#             # Check that vector_store_documents was called
#             processor.context_collections['business_rules'].vector_store_documents.assert_called_once()
#
#     def test_process_structured_document_csv(self, processor):
#         """Test processing CSV document"""
#         # Mock pandas read_csv
#         mock_df = pd.DataFrame({
#             'col1': ['value1', 'value2'],
#             'col2': ['value3', 'value4']
#         })
#
#         with patch('pandas.read_csv', return_value=mock_df), \
#                 patch('time.time', side_effect=[100, 101]):  # Start and end times
#
#             result = processor._process_structured_document(Path('/path/to/test.csv'), 'business_rules')
#
#             # Check result
#             assert result['success'] is True
#             assert result['document_count'] == 2  # One per row
#             assert result['collections'] == ['business_rules']
#             assert result['context_type'] == 'business_rules'
#             assert result['processing_time'] == 1  # 101 - 100
#
#             # Check that vector_store_documents was called
#             processor.context_collections['business_rules'].vector_store_documents.assert_called_once()
#
#     def test_process_structured_document_empty(self, processor):
#         """Test processing empty structured document"""
#         # Mock pandas read_excel to return empty DataFrame
#         mock_df = pd.DataFrame()
#
#         with patch('pandas.read_excel', return_value=mock_df), \
#                 patch('time.time', side_effect=[100, 101]):  # Start and end times
#
#             result = processor._process_structured_document(Path('/path/to/test.xlsx'), 'business_rules')
#
#             # Check result
#             assert result['success'] is False
#             assert result['document_count'] == 0
#             assert result['error'] == "No content extracted from file"
#
#             # Check that vector_store_documents was not called
#             processor.context_collections['business_rules'].vector_store_documents.assert_not_called()
#
#     def test_process_structured_document_error(self, processor):
#         """Test error handling in structured document processing"""
#         # Mock pandas read_excel to raise exception
#         with patch('pandas.read_excel', side_effect=Exception("Test error")), \
#                 patch('time.time', side_effect=[100, 101]), \
#                 patch('logging.error') as mock_error:
#             result = processor._process_structured_document(Path('/path/to/test.xlsx'), 'business_rules')
#
#             # Check result
#             assert result['success'] is False
#             assert result['document_count'] == 0
#             assert result['error'] == "Test error"
#
#             # Check that error was logged
#             mock_error.assert_called_once()
#
#     def test_process_structured_document_storage_failure(self, processor):
#         """Test handling storage failure in structured document processing"""
#         # Mock pandas read_excel
#         mock_df = pd.DataFrame({
#             'col1': ['value1', 'value2'],
#             'col2': ['value3', 'value4']
#         })
#
#         # Mock vector_store_documents to return False (failure)
#         processor.context_collections['business_rules'].vector_store_documents.return_value = False
#
#         with patch('pandas.read_excel', return_value=mock_df), \
#                 patch('time.time', side_effect=[100, 101]):  # Start and end times
#
#             result = processor._process_structured_document(Path('/path/to/test.xlsx'), 'business_rules')
#
#             # Check result
#             assert result['success'] is False
#             assert result['document_count'] == 0
#             assert result['error'] == "Failed to store documents in vector database"
#
#     def test_process_unstructured_document_pdf(self, processor):
#         """Test processing PDF document"""
#         # Mock _process_pdf
#         with patch.object(processor, '_process_pdf') as mock_process_pdf, \
#                 patch('time.time', side_effect=[100, 101]):  # Start and end times
#
#             # Set up mock to return some documents
#             mock_process_pdf.return_value = [
#                 {'text': 'Document 1', 'metadata': {'source_file': 'test.pdf'}},
#                 {'text': 'Document 2', 'metadata': {'source_file': 'test.pdf'}}
#             ]
#
#             result = processor._process_unstructured_document(Path('/path/to/test.pdf'), 'documentation')
#
#             # Check result
#             assert result['success'] is True
#             assert result['document_count'] == 2
#             assert result['collections'] == ['documentation']
#             assert result['context_type'] == 'documentation'
#             assert result['processing_time'] == 1  # 101 - 100
#
#             # Check that vector_store_documents was called
#             processor.context_collections['documentation'].vector_store_documents.assert_called_once()
#
#     def test_process_unstructured_document_docx(self, processor):
#         """Test processing DOCX document"""
#         # Mock _process_docx
#         with patch.object(processor, '_process_docx') as mock_process_docx, \
#                 patch('time.time', side_effect=[100, 101]):  # Start and end times
#
#             # Set up mock to return some documents
#             mock_process_docx.return_value = [
#                 {'text': 'Document 1', 'metadata': {'source_file': 'test.docx'}}
#             ]
#
#             result = processor._process_unstructured_document(Path('/path/to/test.docx'), 'documentation')
#
#             # Check result
#             assert result['success'] is True
#             assert result['document_count'] == 1
#             assert result['collections'] == ['documentation']
#             assert result['context_type'] == 'documentation'
#             assert result['processing_time'] == 1  # 101 - 100
#
#             # Check that vector_store_documents was called
#             processor.context_collections['documentation'].vector_store_documents.assert_called_once()
#
#     def test_process_unstructured_document_text(self, processor):
#         """Test processing text document"""
#         # Mock _process_text
#         with patch.object(processor, '_process_text') as mock_process_text, \
#                 patch('time.time', side_effect=[100, 101]):  # Start and end times
#
#             # Set up mock to return some documents
#             mock_process_text.return_value = [
#                 {'text': 'Document 1', 'metadata': {'source_file': 'test.txt'}}
#             ]
#
#             result = processor._process_unstructured_document(Path('/path/to/test.txt'), 'documentation')
#
#             # Check result
#             assert result['success'] is True
#             assert result['document_count'] == 1
#             assert result['collections'] == ['documentation']
#             assert result['context_type'] == 'documentation'
#             assert result['processing_time'] == 1  # 101 - 100
#
#             # Check that vector_store_documents was called
#             processor.context_collections['documentation'].vector_store_documents.assert_called_once()
#
#     def test_process_unstructured_document_unsupported(self, processor):
#         """Test processing unsupported document type"""
#         with patch('time.time', side_effect=[100, 101]):  # Start and end times
#
#             with pytest.raises(ValueError) as excinfo:
#                 processor._process_unstructured_document(Path('/path/to/test.xyz'), 'documentation')
#
#             assert "Unsupported unstructured document format" in str(excinfo.value)
#
#     def test_process_unstructured_document_empty(self, processor):
#         """Test processing empty unstructured document"""
#         # Mock _process_pdf to return empty list
#         with patch.object(processor, '_process_pdf', return_value=[]), \
#                 patch('time.time', side_effect=[100, 101]):  # Start and end times
#
#             result = processor._process_unstructured_document(Path('/path/to/test.pdf'), 'documentation')
#
#             # Check result
#             assert result['success'] is False
#             assert result['document_count'] == 0
#             assert result['error'] == "No content extracted from file"
#
#             # Check that vector_store_documents was not called
#             processor.context_collections['documentation'].vector_store_documents.assert_not_called()
#
#     def test_process_unstructured_document_storage_failure(self, processor):
#         """Test handling storage failure in unstructured document processing"""
#         # Mock _process_pdf
#         with patch.object(processor, '_process_pdf') as mock_process_pdf, \
#                 patch('time.time', side_effect=[100, 101]):  # Start and end times
#
#             # Set up mock to return some documents
#             mock_process_pdf.return_value = [
#                 {'text': 'Document 1', 'metadata': {'source_file': 'test.pdf'}}
#             ]
#
#             # Mock vector_store_documents to return False (failure)
#             processor.context_collections['documentation'].vector_store_documents.return_value = False
#
#             result = processor._process_unstructured_document(Path('/path/to/test.pdf'), 'documentation')
#
#             # Check result
#             assert result['success'] is False
#             assert result['document_count'] == 0
#             assert result['error'] == "Failed to store documents in vector database"
#
#     def test_process_presentation(self, processor):
#         """Test processing presentation document"""
#         # Mock _process_pptx
#         with patch.object(processor, '_process_pptx') as mock_process_pptx, \
#                 patch('time.time', side_effect=[100, 101]):  # Start and end times
#
#             # Set up mock to return some documents
#             mock_process_pptx.return_value = [
#                 {'text': 'Slide 1', 'metadata': {'source_file': 'test.pptx'}},
#                 {'text': 'Slide 2', 'metadata': {'source_file': 'test.pptx'}}
#             ]
#
#             result = processor._process_presentation(Path('/path/to/test.pptx'), 'requirements')
#
#             # Check result
#             assert result['success'] is True
#             assert result['document_count'] == 2
#             assert result['collections'] == ['requirements']
#             assert result['context_type'] == 'requirements'
#             assert result['processing_time'] == 1  # 101 - 100
#
#             # Check that vector_store_documents was called
#             processor.context_collections['requirements'].vector_store_documents.assert_called_once()
#
#     def test_process_presentation_empty(self, processor):
#         """Test processing empty presentation"""
#         # Mock _process_pptx to return empty list
#         with patch.object(processor, '_process_pptx', return_value=[]), \
#                 patch('time.time', side_effect=[100, 101]):  # Start and end times
#
#             result = processor._process_presentation(Path('/path/to/test.pptx'), 'requirements')
#
#             # Check result
#             assert result['success'] is False
#             assert result['document_count'] == 0
#             assert result['error'] == "No content extracted from file"
#
#             # Check that vector_store_documents was not called
#             processor.context_collections['requirements'].vector_store_documents.assert_not_called()
#
#     def test_process_presentation_error(self, processor):
#         """Test error handling in presentation processing"""
#         # Mock _process_pptx to raise exception
#         with patch.object(processor, '_process_pptx', side_effect=Exception("Test error")), \
#                 patch('time.time', side_effect=[100, 101]), \
#                 patch('logging.error') as mock_error:
#             result = processor._process_presentation(Path('/path/to/test.pptx'), 'requirements')
#
#             # Check result
#             assert result['success'] is False
#             assert result['document_count'] == 0
#             assert result['error'] == "Test error"
#
#             # Check that error was logged
#             mock_error.assert_called_once()
#
#     def test_process_image_file(self, processor):
#         """Test processing image file"""
#         # Mock _extract_image_metadata and _extract_text_from_image
#         with patch.object(processor, '_extract_image_metadata') as mock_extract_metadata, \
#                 patch.object(processor, '_extract_text_from_image') as mock_extract_text, \
#                 patch('time.time', side_effect=[100, 101]), \
#                 patch.dict('os.environ', {'OCR_ENABLED': 'true'}):
#             # Set up mocks
#             mock_extract_metadata.return_value = {'format': 'JPEG', 'size': '800x600'}
#             mock_extract_text.return_value = 'Extracted text from image'
#
#             result = processor._process_image_file(Path('/path/to/test.jpg'), 'documentation')
#
#             # Check result
#             assert result['success'] is True
#             assert result['document_count'] == 1
#             assert result['collections'] == ['documentation']
#             assert result['context_type'] == 'documentation'
#             assert result['processing_time'] == 1  # 101 - 100
#
#             # Check that vector_store_documents was called
#             processor.context_collections['documentation'].vector_store_documents.assert_called_once()
#
#             # Check that the document contains the extracted text
#             args, kwargs = processor.context_collections['documentation'].vector_store_documents.call_args
#             documents = args[0]
#             assert len(documents) == 1
#             assert 'Extracted text from image' in documents[0]['text']
#
#     def test_process_image_file_no_ocr(self, processor):
#         """Test processing image file without OCR"""
#         # Mock _extract_image_metadata
#         with patch.object(processor, '_extract_image_metadata') as mock_extract_metadata, \
#                 patch('time.time', side_effect=[100, 101]), \
#                 patch.dict('os.environ', {'OCR_ENABLED': 'false'}):
#             # Set up mock
#             mock_extract_metadata.return_value = {'format': 'JPEG', 'size': '800x600'}
#
#             result = processor._process_image_file(Path('/path/to/test.jpg'), 'documentation')
#
#             # Check result
#             assert result['success'] is True
#             assert result['document_count'] == 1
#             assert result['collections'] == ['documentation']
#             assert result['context_type'] == 'documentation'
#             assert result['processing_time'] == 1  # 101 - 100
#
#             # Check that vector_store_documents was called
#             processor.context_collections['documentation'].vector_store_documents.assert_called_once()
#
#             # Check that the document doesn't contain extracted text
#             args, kwargs = processor.context_collections['documentation'].vector_store_documents.call_args
#             documents = args[0]
#             assert len(documents) == 1
#             assert 'Extracted Text:' not in documents[0]['text']
#
#     def test_process_image_file_error(self, processor):
#         """Test error handling in image processing"""
#         # Mock _extract_image_metadata to raise exception
#         with patch.object(processor, '_extract_image_metadata', side_effect=Exception("Test error")), \
#                 patch('time.time', side_effect=[100, 101]), \
#                 patch('logging.error') as mock_error:
#             result = processor._process_image_file(Path('/path/to/test.jpg'), 'documentation')
#
#             # Check result
#             assert result['success'] is False
#             assert result['document_count'] == 0
#             assert result['error'] == "Test error"
#
#             # Check that error was logged
#             mock_error.assert_called_once()
#
#     def test_row_to_text(self, processor):
#         """Test converting DataFrame row to text"""
#         # Create a test row
#         row = pd.Series({
#             'col1': 'value1',
#             'col2': 'value2',
#             'col3': np.nan,  # Should be skipped
#             'col4': ''  # Should be skipped
#         })
#
#         result = processor._row_to_text(row)
#
#         # Check result
#         assert 'col1: value1' in result
#         assert 'col2: value2' in result
#         assert 'col3' not in result  # NaN should be skipped
#         assert 'col4' not in result  # Empty string should be skipped
#
#     def test_extract_image_metadata(self, processor):
#         """Test extracting metadata from image"""
#         # Mock PIL.Image
#         mock_img = Mock()
#         mock_img.format = 'JPEG'
#         mock_img.mode = 'RGB'
#         mock_img.width = 800
#         mock_img.height = 600
#
#         with patch('PIL.Image.open') as mock_open, \
#                 patch('os.path.getsize', return_value=1024):
#             # Set up mock to return our mock image
#             mock_open.return_value.__enter__.return_value = mock_img
#
#             result = processor._extract_image_metadata('/path/to/test.jpg')
#
#             # Check result
#             assert result['format'] == 'JPEG'
#             assert result['mode'] == 'RGB'
#             assert result['size'] == '800x600'
#             assert result['file_size'] == 1024
#
#     def test_extract_image_metadata_error(self, processor):
#         """Test error handling in image metadata extraction"""
#         # Mock PIL.Image.open to raise exception
#         with patch('PIL.Image.open', side_effect=Exception("Test error")):
#             result = processor._extract_image_metadata('/path/to/test.jpg')
#
#             # Should return default values
#             assert result['format'] == 'Unknown'
#             assert result['size'] == 'Unknown'
#
#     def test_extract_text_from_image(self, processor):
#         """Test extracting text from image using OCR"""
#         # Mock pytesseract
#         with patch('pytesseract.image_to_string', return_value='Extracted text'), \
#                 patch('PIL.Image.open') as mock_open:
#             # Set up mock image
#             mock_img = Mock()
#             mock_open.return_value.__enter__.return_value = mock_img
#
#             result = processor._extract_text_from_image('/path/to/test.jpg')
#
#             # Check result
#             assert result == 'Extracted text'
#
#     def test_extract_text_from_image_error(self, processor):
#         """Test error handling in OCR text extraction"""
#         # Mock pytesseract to raise exception
#         with patch('pytesseract.image_to_string', side_effect=Exception("OCR error")), \
#                 patch('PIL.Image.open') as mock_open, \
#                 patch('logging.warning') as mock_warning:
#             # Set up mock image
#             mock_img = Mock()
#             mock_open.return_value.__enter__.return_value = mock_img
#
#             result = processor._extract_text_from_image('/path/to/test.jpg')
#
#             # Should return empty string and log warning
#             assert result == ""
#             mock_warning.assert_called_once()
#
#     def test_process_pdf(self, processor):
#         """Test processing PDF file"""
#         # Mock PyPDF2
#         mock_page1 = Mock()
#         mock_page1.extract_text.return_value = 'Page 1 content'
#
#         mock_page2 = Mock()
#         mock_page2.extract_text.return_value = 'Page 2 content'
#
#         mock_reader = Mock()
#         mock_reader.pages = [mock_page1, mock_page2]
#
#         with patch('PyPDF2.PdfReader', return_value=mock_reader):
#             result = processor._process_pdf('/path/to/test.pdf')
#
#             # Check result
#             assert len(result) == 2
#             assert result[0]['text'] == 'Page 1 content'
#             assert result[0]['metadata']['page_number'] == 1
#             assert result[0]['metadata']['total_pages'] == 2
#             assert result[1]['text'] == 'Page 2 content'
#             assert result[1]['metadata']['page_number'] == 2
#
#     def test_process_pdf_import_error(self, processor):
#         """Test handling import error in PDF processing"""
#         # Mock import error
#         with patch('PyPDF2.PdfReader', side_effect=ImportError("No module named 'PyPDF2'")), \
#                 patch('logging.warning') as mock_warning:
#             result = processor._process_pdf('/path/to/test.pdf')
#
#             # Should return empty list and log warning
#             assert result == []
#             mock_warning.assert_called_once()
#
#     def test_process_pdf_processing_error(self, processor):
#         """Test handling processing error in PDF processing"""
#         # Mock processing error
#         with patch('PyPDF2.PdfReader', side_effect=Exception("PDF processing error")), \
#                 patch('logging.error') as mock_error:
#             result = processor._process_pdf('/path/to/test.pdf')
#
#             # Should return empty list and log error
#             assert result == []
#             mock_error.assert_called_once()
#
#     def test_process_docx(self, processor):
#         """Test processing DOCX file"""
#         # Mock docx
#         mock_paragraph1 = Mock()
#         mock_paragraph1.text = 'Paragraph 1'
#
#         mock_paragraph2 = Mock()
#         mock_paragraph2.text = 'Paragraph 2'
#
#         mock_cell1 = Mock()
#         mock_cell1.text = 'Cell 1'
#
#         mock_cell2 = Mock()
#         mock_cell2.text = 'Cell 2'
#
#         mock_row = Mock()
#         mock_row.cells = [mock_cell1, mock_cell2]
#
#         mock_table = Mock()
#         mock_table.rows = [mock_row]
#
#         mock_doc = Mock()
#         mock_doc.paragraphs = [mock_paragraph1, mock_paragraph2]
#         mock_doc.tables = [mock_table]
#
#         with patch('docx.Document', return_value=mock_doc):
#             result = processor._process_docx('/path/to/test.docx')
#
#             # Check result
#             assert len(result) == 1
#             assert 'Paragraph 1' in result[0]['text']
#             assert 'Paragraph 2' in result[0]['text']
#             assert 'Cell 1 | Cell 2' in result[0]['text']
#             assert result[0]['metadata']['paragraph_count'] == 2
#             assert result[0]['metadata']['table_count'] == 1
#
#     def test_process_docx_import_error(self, processor):
#         """Test handling import error in DOCX processing"""
#         # Mock import error
#         with patch('docx.Document', side_effect=ImportError("No module named 'docx'")), \
#                 patch('logging.warning') as mock_warning:
#             result = processor._process_docx('/path/to/test.docx')
#
#             # Should return empty list and log warning
#             assert result == []
#             mock_warning.assert_called_once()
#
#     def test_process_docx_processing_error(self, processor):
#         """Test handling processing error in DOCX processing"""
#         # Mock processing error
#         with patch('docx.Document', side_effect=Exception("DOCX processing error")), \
#                 patch('logging.error') as mock_error:
#             result = processor._process_docx('/path/to/test.docx')
#
#             # Should return empty list and log error
#             assert result == []
#             mock_error.assert_called_once()
#
#     def test_process_pptx(self, processor):
#         """Test processing PPTX file"""
#         # Mock pptx
#         mock_shape1 = Mock()
#         mock_shape1.text = 'Shape 1 text'
#
#         mock_shape2 = Mock()
#         mock_shape2.text = 'Shape 2 text'
#
#         mock_shape3 = Mock()  # Shape without text
#
#         mock_slide1 = Mock()
#         mock_slide1.shapes = [mock_shape1, mock_shape2]
#
#         mock_slide2 = Mock()
#         mock_slide2.shapes = [mock_shape3]  # No text, should be skipped
#
#         mock_presentation = Mock()
#         mock_presentation.slides = [mock_slide1, mock_slide2]
#
#         with patch('pptx.Presentation', return_value=mock_presentation):
#             result = processor._process_pptx('/path/to/test.pptx')
#
#             # Check result
#             assert len(result) == 1  # Only one slide has text
#             assert 'Shape 1 text' in result[0]['text']
#             assert 'Shape 2 text' in result[0]['text']
#             assert result[0]['metadata']['slide_number'] == 1
#             assert result[0]['metadata']['total_slides'] == 2
#
#     def test_process_pptx_import_error(self, processor):
#         """Test handling import error in PPTX processing"""
#         # Mock import error
#         with patch('pptx.Presentation', side_effect=ImportError("No module named 'pptx'")), \
#                 patch('logging.warning') as mock_warning:
#             result = processor._process_pptx('/path/to/test.pptx')
#
#             # Should return empty list and log warning
#             assert result == []
#             mock_warning.assert_called_once()
#
#     def test_process_pptx_processing_error(self, processor):
#         """Test handling processing error in PPTX processing"""
#         # Mock processing error
#         with patch('pptx.Presentation', side_effect=Exception("PPTX processing error")), \
#                 patch('logging.error') as mock_error:
#             result = processor._process_pptx('/path/to/test.pptx')
#
#             # Should return empty list and log error
#             assert result == []
#             mock_error.assert_called_once()
#
#     def test_process_text(self, processor):
#         """Test processing text file"""
#         # Mock open
#         with patch('builtins.open', mock_open(read_data='Test content\nLine 2')):
#             result = processor._process_text('/path/to/test.txt')
#
#             # Check result
#             assert len(result) == 1
#             assert result[0]['text'] == 'Test content\nLine 2'
#             assert result[0]['metadata']['file_type'] == '.txt'
#             assert result[0]['metadata']['character_count'] == 18
#             assert result[0]['metadata']['line_count'] == 2
#
#     def test_process_text_unicode_error(self, processor):
#         """Test handling unicode error in text processing"""
#         # Mock open to raise UnicodeDecodeError first, then succeed with latin-1
#         with patch('builtins.open') as mock_open_func:
#             # First call raises UnicodeDecodeError
#             mock_file_utf8 = mock_open(read_data='').return_value
#             mock_file_utf8.__enter__.side_effect = UnicodeDecodeError('utf-8', b'test', 0, 1, 'invalid')
#
#             # Second call succeeds with latin-1
#             mock_file_latin1 = mock_open(read_data='Test content with latin-1 encoding').return_value
#
#             # Set up the mock to return different values on successive calls
#             mock_open_func.side_effect = [mock_file_utf8, mock_file_latin1]
#
#             result = processor._process_text('/path/to/test.txt')
#
#             # Check result
#             assert len(result) == 1
#             assert result[0]['text'] == 'Test content with latin-1 encoding'
#             assert result[0]['metadata']['encoding'] == 'latin-1'
#
#     def test_display_final_stats(self, processor):
#         """Test displaying final stats"""
#         # Set up some stats
#         processor.stats = {
#             'total_files': 10,
#             'processed_files': 7,
#             'skipped_files': 2,
#             'failed_files': 1,
#             'total_chunks': 20,
#             'processing_time': 5.5,
#             'by_category': {
#                 'business_rules': {'total': 5, 'processed': 4, 'skipped': 1, 'failed': 0},
#                 'documentation': {'total': 3, 'processed': 2, 'skipped': 0, 'failed': 1},
#                 'empty_category': {'total': 0, 'processed': 0, 'skipped': 0, 'failed': 0}
#             }
#         }
#
#         # Call the method (just to ensure it doesn't raise exceptions)
#         processor._display_final_stats()
#
#         # No assertions needed, just checking that it runs without errors


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
        mock.diagnose_database.return_value = True
        return mock

    @pytest.fixture
    def processor(self, metrics_manager_mock, pgvector_connector_mock):
        """Create an EnhancedContextProcessor instance with mocked dependencies"""
        with patch('src.context_handler.context_file_handler.enhanced_context_processor.PGVectorConnector',
                   return_value=pgvector_connector_mock):
            with patch('src.context_handler.context_file_handler.enhanced_context_processor.AWSTitanEmbeddings'):
                processor = EnhancedContextProcessor('../Input/ContextLibrary', metrics_manager=metrics_manager_mock)
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
                # Mock document registry
                processor.document_registry = {}
                processor._load_document_registry = Mock(return_value={})
                processor._save_document_registry = Mock()
                processor._calculate_file_hash = Mock(return_value="test_hash")
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

    def test_initialization(self, metrics_manager_mock):
        """Test initialization of EnhancedContextProcessor"""
        with patch(
                'src.context_handler.context_file_handler.enhanced_context_processor.PGVectorConnector') as mock_pgvector, \
                patch(
                    'src.context_handler.context_file_handler.enhanced_context_processor.AWSTitanEmbeddings') as mock_embeddings, \
                patch('os.makedirs') as mock_makedirs, \
                patch(
                    'src.context_handler.context_file_handler.enhanced_context_processor.EnhancedContextProcessor._load_document_registry',
                    return_value={}):
            processor = EnhancedContextProcessor('../Input/ContextLibrary', metrics_manager=metrics_manager_mock)

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

        with patch('os.path.exists', return_value=True), \
                patch('builtins.open', mock_open(read_data=json.dumps(mock_registry))):
            processor = EnhancedContextProcessor('../Input/ContextLibrary')
            registry = processor._load_document_registry()

            assert registry == mock_registry

    def test_load_document_registry_not_exists(self):
        """Test loading document registry when file doesn't exist"""
        with patch('os.path.exists', return_value=False):
            processor = EnhancedContextProcessor('../Input/ContextLibrary')
            registry = processor._load_document_registry()

            assert registry == {}

    def test_load_document_registry_error(self):
        """Test loading document registry with JSON error"""
        with patch('os.path.exists', return_value=True), \
                patch('builtins.open', mock_open(read_data='invalid json')):
            processor = EnhancedContextProcessor('../Input/ContextLibrary')
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
        with patch('builtins.open', mock_open(read_data=b'test content')):
            processor = EnhancedContextProcessor('../Input/ContextLibrary')
            file_hash = processor._calculate_file_hash('/path/to/file.pdf')

            # Check that a hash was returned
            assert isinstance(file_hash, str)
            assert len(file_hash) > 0

    def test_calculate_file_hash_error(self):
        """Test calculating file hash with error"""
        with patch('builtins.open', side_effect=Exception("Test error")), \
                patch('logging.error') as mock_log:
            processor = EnhancedContextProcessor('../Input/ContextLibrary')
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
                patch('datetime.now', return_value=datetime(2023, 1, 1, 12, 0, 0)):
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
                patch.object(processor, '_process_pdf_file') as mock_process_pdf, \
                patch.object(processor, '_process_excel_file') as mock_process_excel, \
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
                patch.object(processor, '_process_pdf_file', side_effect=Exception("Test error")), \
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
            result = processor._process_text_file(file_path, category)

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
                result = processor._process_text_file(file_path, category)

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
            result = processor._process_text_file(file_path, category)

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
            result = processor._process_text_file(file_path, category)

            # Check result
            assert result['success'] is False
            assert result['error'] == "Test error"

            # Check that error was logged
            mock_log.assert_called_once_with(ANY)

    def test_process_pdf_file_success(self, processor):
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
            result = processor._process_pdf_file(file_path, category)

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

    def test_process_pdf_file_no_text(self, processor):
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
            result = processor._process_pdf_file(file_path, category)

            # Check result
            assert result['success'] is True
            assert result['document_count'] == 1  # Should create a placeholder document

            # Check that vector store was called with placeholder
            processor.context_collections[category].vector_store_documents.assert_called_once()
            args, _ = processor.context_collections[category].vector_store_documents.call_args
            assert len(args[0]) == 1
            assert "PDF file: test.pdf (no text extracted)" in args[0][0]['text']
            assert args[0][0]['metadata']['extraction_failed'] is True

    def test_process_pdf_file_import_error(self, processor):
        """Test processing PDF file with import error"""
        file_path = '/path/to/test.pdf'
        category = 'documentation'

        # Mock import error
        with patch('PyPDF2.PdfReader', side_effect=ImportError("No module named 'PyPDF2'")), \
                patch('logging.warning') as mock_log:
            # Call the method
            result = processor._process_pdf_file(file_path, category)

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

    def test_process_pdf_file_extraction_error(self, processor):
        """Test processing PDF file with extraction error"""
        file_path = '/path/to/test.pdf'
        category = 'documentation'

        # Mock PyPDF2 to raise exception
        with patch('PyPDF2.PdfReader', side_effect=Exception("Extraction error")), \
                patch('logging.warning') as mock_log:
            # Call the method
            result = processor._process_pdf_file(file_path, category)

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

    def test_process_pdf_file_storage_failure(self, processor):
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
            result = processor._process_pdf_file(file_path, category)

            # Check result
            assert result['success'] is False
            assert result['error'] == "Failed to store documents in vector database"

    def test_process_docx_file_success(self, processor):
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
            result = processor._process_docx_file(file_path, category)

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

    def test_process_docx_file_no_content(self, processor):
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
            result = processor._process_docx_file(file_path, category)

            # Check result
            assert result['success'] is True
            assert result['document_count'] == 1  # Should create a placeholder document

            # Check that vector store was called with placeholder
            processor.context_collections[category].vector_store_documents.assert_called_once()
            args, _ = processor.context_collections[category].vector_store_documents.call_args
            assert len(args[0]) == 1
            assert "Word document: test.docx (no text content)" in args[0][0]['text']
            assert args[0][0]['metadata']['empty_document'] is True

    def test_process_docx_file_import_error(self, processor):
        """Test processing DOCX file with import error"""
        file_path = '/path/to/test.docx'
        category = 'documentation'

        # Mock import error
        with patch('docx.Document', side_effect=ImportError("No module named 'docx'")), \
                patch('logging.warning') as mock_log:
            # Call the method
            result = processor._process_docx_file(file_path, category)

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
        file_path = '/path/to/test.xlsx'
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
            result = processor._process_excel_file(file_path, category)

            # Check result
            assert result['success'] is True
            assert result['document_count'] == 3  # 2 rows + 1 full sheet
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
        file_path = '/path/to/test.xlsx'
        category = 'business_rules'

        # Mock pandas
        with patch('pandas.read_excel') as mock_read_excel:
            # Configure mock DataFrame with no data
            mock_df = pd.DataFrame()
            mock_read_excel.return_value = mock_df

            # Call the method
            result = processor._process_excel_file(file_path, category)

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
        file_path = '/path/to/test.xlsx'
        category = 'business_rules'

        # Mock pandas to raise exception
        with patch('pandas.read_excel', side_effect=Exception("Excel error")), \
                patch('logging.warning') as mock_log:
            # Call the method
            result = processor._process_excel_file(file_path, category)

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
            result = processor._process_presentation_file(file_path, category)

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
            assert args[0][0]['metadata']['category'] == category
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
            result = processor._process_presentation_file(file_path, category)

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
            result = processor._process_presentation_file(file_path, category)

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
        file_path = '/path/to/test.jpg'
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
                assert args[0][0]['metadata']['category'] == category
                assert args[0][0]['metadata']['image_metadata']['format'] == 'JPEG'

    def test_process_image_file_no_ocr(self, processor):
        """Test processing image file without OCR"""
        file_path = '/path/to/test.jpg'
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
        file_path = '/path/to/test.jpg'
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