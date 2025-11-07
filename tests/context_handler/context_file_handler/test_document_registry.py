import pytest
import os
import json
import tempfile
import shutil
from unittest.mock import patch, mock_open, MagicMock
from datetime import datetime
from pathlib import Path

from src.context_handler.context_file_handler.document_registry import DocumentRegistry


class TestDocumentRegistry:
    @pytest.fixture
    def temp_registry_dir(self):
        """Create a temporary directory for the registry"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def empty_registry(self, temp_registry_dir):
        """Create an empty registry"""
        return DocumentRegistry(temp_registry_dir)

    @pytest.fixture
    def populated_registry(self, temp_registry_dir):
        """Create a registry with some test data"""
        # Create a registry file with test data
        registry_file = os.path.join(temp_registry_dir, 'document_registry.json')
        test_data = {
            'doc_123': {
                'file_name': 'test.pdf',
                'file_path': '/path/to/test.pdf',
                'file_hash': 'abc123',
                'file_size': 1024,
                'file_mtime': 1625097600.0,
                'file_type': '.pdf',
                'processed_date': '2023-01-01T12:00:00',
                'document_count': 5,
                'collections': ['business_rules'],
                'context_type': 'business_rules',
                'processing_time': 1.5,
                'success': True
            },
            'doc_456': {
                'file_name': 'example.docx',
                'file_path': '/path/to/example.docx',
                'file_hash': 'def456',
                'file_size': 2048,
                'file_mtime': 1625184000.0,
                'file_type': '.docx',
                'processed_date': '2023-01-02T12:00:00',
                'document_count': 3,
                'collections': ['documentation'],
                'context_type': 'documentation',
                'processing_time': 0.8,
                'success': True
            }
        }

        with open(registry_file, 'w') as f:
            json.dump(test_data, f)

        return DocumentRegistry(temp_registry_dir)

    def test_initialization_with_empty_directory(self, temp_registry_dir):
        """Test initialization with an empty directory"""
        registry = DocumentRegistry(temp_registry_dir)
        assert registry.registry_file == os.path.join(temp_registry_dir, 'document_registry.json')
        assert registry.document_registry == {}

    def test_initialization_with_existing_registry(self, populated_registry):
        """Test initialization with an existing registry file"""
        assert len(populated_registry.document_registry) == 2
        assert 'doc_123' in populated_registry.document_registry
        assert 'doc_456' in populated_registry.document_registry
        assert populated_registry.document_registry['doc_123']['file_name'] == 'test.pdf'
        assert populated_registry.document_registry['doc_456']['file_name'] == 'example.docx'

    def test_initialization_with_corrupted_registry(self, temp_registry_dir):
        """Test initialization with a corrupted registry file"""
        # Create a corrupted registry file
        registry_file = os.path.join(temp_registry_dir, 'document_registry.json')
        with open(registry_file, 'w') as f:
            f.write('{"invalid": "json')  # Corrupted JSON

        # Should handle the error and create a new empty registry
        with patch('logging.warning') as mock_warning:
            registry = DocumentRegistry(temp_registry_dir)
            mock_warning.assert_called_once()
            assert registry.document_registry == {}

    def test_save_document_registry(self, empty_registry):
        """Test saving the document registry"""
        # Add some test data
        empty_registry.document_registry = {
            'doc_test': {
                'file_name': 'test.pdf',
                'file_path': '/path/to/test.pdf'
            }
        }

        # Save the registry
        empty_registry._save_document_registry()

        # Check that the file was created with the correct content
        registry_file = empty_registry.registry_file
        assert os.path.exists(registry_file)

        with open(registry_file, 'r') as f:
            saved_data = json.load(f)
            assert 'doc_test' in saved_data
            assert saved_data['doc_test']['file_name'] == 'test.pdf'

    def test_save_document_registry_error(self, empty_registry):
        """Test error handling when saving the registry fails"""
        # Mock open to raise an exception
        with patch('builtins.open', side_effect=Exception("Test error")), \
                patch('logging.error') as mock_error:
            empty_registry._save_document_registry()
            mock_error.assert_called_once()

    def test_calculate_file_hash(self, empty_registry, temp_registry_dir):
        """Test calculating file hash"""
        # Create a test file
        test_file = os.path.join(temp_registry_dir, 'test_file.txt')
        with open(test_file, 'w') as f:
            f.write('test content')

        # Calculate hash
        file_hash = empty_registry._calculate_file_hash(test_file)

        # Should be a non-empty string
        assert isinstance(file_hash, str)
        assert len(file_hash) > 0

    def test_calculate_file_hash_error(self, empty_registry):
        """Test error handling when calculating file hash fails"""
        # Try to calculate hash for a non-existent file
        with patch('logging.error') as mock_error:
            file_hash = empty_registry._calculate_file_hash('/nonexistent/file.txt')
            mock_error.assert_called_once()
            assert file_hash == ""

    def test_is_document_processed_by_hash(self, populated_registry):
        """Test checking if a document is processed by hash"""
        # Mock _calculate_file_hash to return a known hash
        with patch.object(populated_registry, '_calculate_file_hash', return_value='abc123'):
            result = populated_registry.is_document_processed('/path/to/some/file.pdf')

            assert result['exists'] is True
            assert result['reason'] == 'identical_content'
            assert result['existing_doc_id'] == 'doc_123'
            assert 'existing_info' in result

    def test_is_document_processed_by_attributes(self, populated_registry):
        """Test checking if a document is processed by attributes"""
        # Mock file attributes but with a different hash
        with patch.object(populated_registry, '_calculate_file_hash', return_value='different_hash'), \
                patch('os.path.getsize', return_value=1024), \
                patch('os.path.getmtime', return_value=1625097600.0):
            result = populated_registry.is_document_processed('/path/to/test.pdf')

            assert result['exists'] is True
            assert result['reason'] == 'same_file_attributes'
            assert result['existing_doc_id'] == 'doc_123'
            assert 'existing_info' in result

    def test_is_document_processed_new_document(self, populated_registry):
        """Test checking a new document that hasn't been processed"""
        # Mock file attributes for a new document
        with patch.object(populated_registry, '_calculate_file_hash', return_value='new_hash'), \
                patch('os.path.getsize', return_value=3072), \
                patch('os.path.getmtime', return_value=1625270400.0):
            result = populated_registry.is_document_processed('/path/to/new.pdf')

            assert result['exists'] is False

    def test_register_document(self, empty_registry, temp_registry_dir):
        """Test registering a new document"""
        # Create a test file
        test_file = os.path.join(temp_registry_dir, 'register_test.txt')
        with open(test_file, 'w') as f:
            f.write('test content')

        # Mock file hash calculation
        with patch.object(empty_registry, '_calculate_file_hash', return_value='test_hash'):
            # Processing result
            processing_result = {
                'success': True,
                'document_count': 3,
                'collections': ['business_rules'],
                'context_type': 'business_rules',
                'processing_time': 1.2
            }

            # Register the document
            doc_id = empty_registry.register_document(test_file, processing_result)

            # Check that the document was registered
            assert doc_id in empty_registry.document_registry
            assert empty_registry.document_registry[doc_id]['file_name'] == 'register_test.txt'
            assert empty_registry.document_registry[doc_id]['file_hash'] == 'test_hash'
            assert empty_registry.document_registry[doc_id]['document_count'] == 3
            assert empty_registry.document_registry[doc_id]['collections'] == ['business_rules']
            assert empty_registry.document_registry[doc_id]['context_type'] == 'business_rules'
            assert empty_registry.document_registry[doc_id]['processing_time'] == 1.2
            assert empty_registry.document_registry[doc_id]['success'] is True

    def test_register_document_with_file_errors(self, empty_registry):
        """Test registering a document with file attribute errors"""
        # Mock file operations to raise exceptions
        with patch.object(empty_registry, '_calculate_file_hash', return_value='test_hash'), \
                patch('os.path.getsize', side_effect=Exception("Size error")), \
                patch('os.path.getmtime', side_effect=Exception("Time error")), \
                patch.object(empty_registry, '_save_document_registry') as mock_save:
            # Processing result
            processing_result = {
                'success': True,
                'document_count': 3,
                'collections': ['business_rules'],
                'context_type': 'business_rules',
                'processing_time': 1.2
            }

            # Register the document
            doc_id = empty_registry.register_document('/path/to/error_file.txt', processing_result)

            # Check that the document was registered despite errors
            assert doc_id in empty_registry.document_registry
            assert empty_registry.document_registry[doc_id]['file_name'] == 'error_file.txt'
            assert empty_registry.document_registry[doc_id]['file_hash'] == 'test_hash'
            assert empty_registry.document_registry[doc_id]['file_size'] is None
            assert empty_registry.document_registry[doc_id]['file_mtime'] is None

            # Check that save was called
            mock_save.assert_called_once()

    def test_get_status_summary_empty(self, empty_registry):
        """Test getting status summary for an empty registry"""
        status = empty_registry.get_status_summary()

        assert status['registry_exists'] is False
        assert status['total_registered_documents'] == 0
        assert status['last_update'] is None

    def test_get_status_summary_populated(self, populated_registry):
        """Test getting status summary for a populated registry"""
        status = populated_registry.get_status_summary()

        assert status['registry_exists'] is True
        assert status['total_registered_documents'] == 2
        assert status['last_update'] == '2023-01-02T12:00:00'  # The most recent date

    def test_get_all_documents(self, populated_registry):
        """Test getting all documents"""
        documents = populated_registry.get_all_documents()

        assert len(documents) == 2
        assert 'doc_123' in documents
        assert 'doc_456' in documents
        assert documents['doc_123']['file_name'] == 'test.pdf'
        assert documents['doc_456']['file_name'] == 'example.docx'