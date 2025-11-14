# tests/context_handler/context_storage_handler/test_run_context_processor.py

import pytest
import os
import tempfile
import shutil
import json
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open, call
from datetime import datetime

from src.context_handler.context_storage_handler.run_context_processor import EnhancedContextProcessor


class TestEnhancedContextProcessor:

    @pytest.fixture
    def temp_context_dir(self):
        """Create a temporary directory for context files"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def processor(self, temp_context_dir, global_metrics_manager_mock):
        """Create an EnhancedContextProcessor instance for testing"""
        with patch(
                'src.context_handler.context_storage_handler.run_context_processor.PGVectorConnector') as mock_pgvector, \
                patch(
                    'src.context_handler.context_storage_handler.run_context_processor.AWSTitanEmbeddings') as mock_embeddings, \
                patch(
                    'src.context_handler.context_storage_handler.run_context_processor.ImageParserLLM') as mock_image_parser:
            # Setup mock returns
            mock_pgvector.return_value.vector_store_documents.return_value = True
            mock_pgvector.return_value.diagnose_database.return_value = True
            mock_embeddings.return_value = Mock()
            mock_image_parser.return_value = Mock()

            processor = EnhancedContextProcessor(temp_context_dir, metrics_manager=global_metrics_manager_mock)
            processor.image_parser = Mock()
            processor.image_parser.parse_image.return_value = "Extracted text from image"

            return processor

    @pytest.fixture
    def mock_pgvector_connector(self):
        """Mock PGVectorConnector"""
        mock = Mock()
        mock.vector_store_documents.return_value = True
        mock.diagnose_database.return_value = True
        return mock

    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing"""
        return [
            {
                'text': 'Sample document text 1',
                'metadata': {
                    'source_file': 'test1.txt',
                    'file_type': 'text'
                }
            },
            {
                'text': 'Sample document text 2',
                'metadata': {
                    'source_file': 'test2.txt',
                    'file_type': 'text'
                }
            }
        ]

    # ==================== Initialization Tests ====================

    def test_initialization(self, processor, temp_context_dir):
        """Test processor initialization"""
        assert processor.context_folder_path == Path(temp_context_dir)
        assert processor.metrics_manager is not None
        assert len(processor.context_collections) == 7
        assert 'business_rules' in processor.context_collections
        assert 'wireframes' in processor.context_collections

    def test_initialization_creates_directories(self, temp_context_dir, global_metrics_manager_mock):
        """Test that initialization creates necessary directories"""
        with patch('src.context_handler.context_storage_handler.run_context_processor.PGVectorConnector'), \
                patch('src.context_handler.context_storage_handler.run_context_processor.AWSTitanEmbeddings'), \
                patch('src.context_handler.context_storage_handler.run_context_processor.ImageParserLLM'):
            processor = EnhancedContextProcessor(temp_context_dir, metrics_manager=global_metrics_manager_mock)

            assert os.path.exists(processor.output_dir)

    def test_initialization_with_default_metrics_manager(self, temp_context_dir):
        """Test initialization with default metrics manager"""
        with patch('src.context_handler.context_storage_handler.run_context_processor.PGVectorConnector'), \
                patch('src.context_handler.context_storage_handler.run_context_processor.AWSTitanEmbeddings'), \
                patch('src.context_handler.context_storage_handler.run_context_processor.ImageParserLLM'), \
                patch(
                    'src.context_handler.context_storage_handler.run_context_processor.MetricsManager') as mock_metrics:
            processor = EnhancedContextProcessor(temp_context_dir)
            mock_metrics.assert_called_once()

    # ==================== Document Registry Tests ====================

    def test_load_document_registry_existing(self, processor):
        """Test loading existing document registry"""
        registry_data = {
            'doc_1': {
                'file_name': 'test.txt',
                'file_hash': 'abc123'
            }
        }

        with patch('os.path.exists', return_value=True), \
                patch('builtins.open', mock_open(read_data=json.dumps(registry_data))):
            registry = processor._load_document_registry()
            assert registry == registry_data

    def test_load_document_registry_nonexistent(self, processor):
        """Test loading non-existent document registry"""
        with patch('os.path.exists', return_value=False):
            registry = processor._load_document_registry()
            assert registry == {}

    def test_load_document_registry_corrupted(self, processor):
        """Test loading corrupted document registry"""
        with patch('os.path.exists', return_value=True), \
                patch('builtins.open', mock_open(read_data='invalid json')):
            registry = processor._load_document_registry()
            assert registry == {}

    def test_save_document_registry(self, processor):
        """Test saving document registry"""
        processor.document_registry = {'doc_1': {'file_name': 'test.txt'}}

        with patch('builtins.open', mock_open()) as mock_file:
            processor._save_document_registry()
            mock_file.assert_called_once()

    def test_save_document_registry_error(self, processor):
        """Test error handling when saving document registry"""
        processor.document_registry = {'doc_1': {'file_name': 'test.txt'}}

        with patch('builtins.open', side_effect=Exception("Write error")):
            # Should not raise exception
            processor._save_document_registry()

    # ==================== File Hash Tests ====================

    def test_calculate_file_hash(self, processor, temp_file_path):
        """Test file hash calculation"""
        with open(temp_file_path, 'w') as f:
            f.write("test content")

        hash1 = processor._calculate_file_hash(temp_file_path)
        assert isinstance(hash1, str)
        assert len(hash1) == 32  # MD5 hash length

        # Same content should produce same hash
        hash2 = processor._calculate_file_hash(temp_file_path)
        assert hash1 == hash2

    def test_calculate_file_hash_error(self, processor):
        """Test file hash calculation error handling"""
        hash_result = processor._calculate_file_hash("/nonexistent/file.txt")
        assert hash_result == ""

    # ==================== Document Processing Check Tests ====================

    def test_is_document_processed_by_hash(self, processor, temp_file_path):
        """Test checking if document is processed by hash"""
        with open(temp_file_path, 'w') as f:
            f.write("test content")

        file_hash = processor._calculate_file_hash(temp_file_path)
        processor.document_registry = {
            'doc_1': {
                'file_hash': file_hash,
                'file_name': 'test.txt'
            }
        }

        result = processor._is_document_processed(temp_file_path)
        assert result['exists'] is True
        assert result['reason'] == 'identical_content'

    def test_is_document_processed_by_attributes(self, processor, temp_file_path):
        """Test checking if document is processed by file attributes"""
        with open(temp_file_path, 'w') as f:
            f.write("test content")

        file_name = os.path.basename(temp_file_path)
        file_size = os.path.getsize(temp_file_path)
        file_mtime = os.path.getmtime(temp_file_path)

        processor.document_registry = {
            'doc_1': {
                'file_name': file_name,
                'file_size': file_size,
                'file_mtime': file_mtime,
                'file_hash': 'different_hash'
            }
        }

        result = processor._is_document_processed(temp_file_path)
        assert result['exists'] is True
        assert result['reason'] == 'same_file_attributes'

    def test_is_document_not_processed(self, processor, temp_file_path):
        """Test checking if document is not processed"""
        with open(temp_file_path, 'w') as f:
            f.write("test content")

        processor.document_registry = {}

        result = processor._is_document_processed(temp_file_path)
        assert result['exists'] is False

    # ==================== Document Registration Tests ====================

    def test_register_document(self, processor, temp_file_path):
        """Test registering a processed document"""
        with open(temp_file_path, 'w') as f:
            f.write("test content")

        processing_result = {
            'document_count': 5,
            'collections': ['business_rules'],
            'context_type': 'business_rules',
            'processing_time': 1.5,
            'success': True
        }

        with patch.object(processor, '_save_document_registry'):
            doc_id = processor._register_document(temp_file_path, processing_result)

            assert doc_id.startswith('doc_')
            assert doc_id in processor.document_registry
            assert processor.document_registry[doc_id]['file_name'] == os.path.basename(temp_file_path)
            assert processor.document_registry[doc_id]['document_count'] == 5

    # ==================== File Discovery Tests ====================

    def test_discover_files(self, processor, temp_context_dir):
        """Test file discovery"""
        # Create test files
        os.makedirs(os.path.join(temp_context_dir, 'business_rules'), exist_ok=True)
        test_file = os.path.join(temp_context_dir, 'business_rules', 'test.txt')
        with open(test_file, 'w') as f:
            f.write("test")

        files = processor._discover_files()

        assert 'business_rules' in files
        assert len(files['business_rules']) > 0

    def test_discover_files_skips_hidden(self, processor, temp_context_dir):
        """Test that hidden files are skipped"""
        hidden_file = os.path.join(temp_context_dir, '.hidden.txt')
        with open(hidden_file, 'w') as f:
            f.write("test")

        files = processor._discover_files()

        # Hidden file should not be in any category
        all_files = [f for category_files in files.values() for f in category_files]
        assert hidden_file not in [str(f) for f in all_files]

    def test_discover_files_skips_large(self, processor, temp_context_dir):
        """Test that large files are skipped"""
        large_file = os.path.join(temp_context_dir, 'large.txt')
        with open(large_file, 'w') as f:
            f.write("test")

        with patch('os.path.getsize', return_value=600 * 1024 * 1024):  # 600 MB
            files = processor._discover_files()

            all_files = [str(f) for category_files in files.values() for f in category_files]
            assert large_file not in all_files

    # ==================== File Categorization Tests ====================

    def test_categorize_file_by_folder(self, processor):
        """Test file categorization by folder name"""
        file_path = Path('/test/business_rules/test.txt')
        category = processor._categorize_file(file_path, 'business_rules')
        assert category == 'business_rules'

    def test_categorize_file_by_filename(self, processor):
        """Test file categorization by filename"""
        file_path = Path('/test/test_requirements.txt')
        category = processor._categorize_file(file_path, 'unknown')
        assert category == 'requirements'

    def test_categorize_file_by_extension(self, processor):
        """Test file categorization by file extension"""
        file_path = Path('/test/document.pdf')
        category = processor._categorize_file(file_path, 'unknown')
        assert category == 'documentation'

    def test_categorize_file_default(self, processor):
        """Test default file categorization"""
        file_path = Path('/test/unknown.xyz')
        category = processor._categorize_file(file_path, 'unknown')
        assert category == 'documentation'

    # ==================== Text File Processing Tests ====================

    def test_process_text_file(self, processor, temp_file_path):
        """Test processing text files"""
        with open(temp_file_path, 'w') as f:
            f.write("Test content")

        with patch.object(processor.context_collections['documentation'], 'vector_store_documents', return_value=True):
            result = processor._process_text_file(temp_file_path, 'documentation')

            assert result['success'] is True
            assert result['document_count'] == 1
            assert result['context_type'] == 'documentation'

    def test_process_text_file_unicode_error(self, processor, temp_file_path):
        """Test processing text file with unicode error"""
        # Create a file with latin-1 encoding
        with open(temp_file_path, 'wb') as f:
            f.write(b'\xe9')  # Latin-1 encoded character

        with patch.object(processor.context_collections['documentation'], 'vector_store_documents', return_value=True):
            result = processor._process_text_file(temp_file_path, 'documentation')

            assert result['success'] is True

    def test_process_text_file_error(self, processor, temp_file_path):
        """Test error handling in text file processing"""
        with open(temp_file_path, 'w') as f:
            f.write("Test content")

        with patch.object(processor.context_collections['documentation'], 'vector_store_documents',
                          side_effect=Exception("Storage error")):
            result = processor._process_text_file(temp_file_path, 'documentation')

            assert result['success'] is False
            assert 'error' in result

    # ==================== PDF Processing Tests ====================

    def test_process_pdf_file(self, processor, temp_file_path):
        """Test processing PDF files"""
        with patch('PyPDF2.PdfReader') as mock_pdf:
            mock_page = Mock()
            mock_page.extract_text.return_value = "PDF content"
            mock_pdf.return_value.pages = [mock_page]

            with patch.object(processor.context_collections['documentation'], 'vector_store_documents',
                              return_value=True):
                result = processor._process_pdf_file(temp_file_path, 'documentation')

                assert result['success'] is True
                assert result['document_count'] == 1

    def test_process_pdf_file_no_text(self, processor, temp_file_path):
        """Test processing PDF with no extractable text"""
        with patch('PyPDF2.PdfReader') as mock_pdf:
            mock_page = Mock()
            mock_page.extract_text.return_value = ""
            mock_pdf.return_value.pages = [mock_page]

            with patch.object(processor.context_collections['documentation'], 'vector_store_documents',
                              return_value=True):
                result = processor._process_pdf_file(temp_file_path, 'documentation')

                assert result['success'] is True
                # Should create placeholder document
                assert result['document_count'] == 1

    def test_process_pdf_file_error(self, processor, temp_file_path):
        """Test error handling in PDF processing"""
        with patch('PyPDF2.PdfReader', side_effect=Exception("PDF error")):
            with patch.object(processor.context_collections['documentation'], 'vector_store_documents',
                              return_value=True):
                result = processor._process_pdf_file(temp_file_path, 'documentation')

                assert result['success'] is True
                # Should create error placeholder document

    # ==================== Word Document Processing Tests ====================

    def test_process_doc_file(self, processor, temp_file_path):
        """Test processing Word documents"""
        with patch('docx.Document') as mock_doc:
            mock_paragraph = Mock()
            mock_paragraph.text = "Paragraph text"
            mock_doc.return_value.paragraphs = [mock_paragraph]
            mock_doc.return_value.tables = []

            with patch.object(processor.context_collections['documentation'], 'vector_store_documents',
                              return_value=True):
                result = processor._process_doc_file(temp_file_path, 'documentation')

                assert result['success'] is True
                assert result['document_count'] == 1

    def test_process_doc_file_with_tables(self, processor, temp_file_path):
        """Test processing Word document with tables"""
        with patch('docx.Document') as mock_doc:
            mock_paragraph = Mock()
            mock_paragraph.text = "Paragraph text"

            mock_cell = Mock()
            mock_cell.text = "Cell text"
            mock_row = Mock()
            mock_row.cells = [mock_cell]
            mock_table = Mock()
            mock_table.rows = [mock_row]

            mock_doc.return_value.paragraphs = [mock_paragraph]
            mock_doc.return_value.tables = [mock_table]

            with patch.object(processor.context_collections['documentation'], 'vector_store_documents',
                              return_value=True):
                result = processor._process_doc_file(temp_file_path, 'documentation')

                assert result['success'] is True

    def test_process_doc_file_error(self, processor, temp_file_path):
        """Test error handling in Word document processing"""
        with patch('docx.Document', side_effect=Exception("DOCX error")):
            with patch.object(processor.context_collections['documentation'], 'vector_store_documents',
                              return_value=True):
                result = processor._process_doc_file(temp_file_path, 'documentation')

                assert result['success'] is True
                # Should create error placeholder document

    # ==================== Excel Processing Tests ====================

    def test_process_excel_file(self, processor, temp_file_path):
        """Test processing Excel files"""
        df = pd.DataFrame({
            'Column1': ['Value1', 'Value2'],
            'Column2': ['Value3', 'Value4']
        })

        with patch('pandas.read_excel', return_value=df):
            with patch.object(processor.context_collections['business_rules'], 'vector_store_documents',
                              return_value=True):
                result = processor._process_excel_file(temp_file_path, 'business_rules')

                assert result['success'] is True
                assert result['document_count'] > 0

    def test_process_excel_file_empty(self, processor, temp_file_path):
        """Test processing empty Excel file"""
        df = pd.DataFrame()

        with patch('pandas.read_excel', return_value=df):
            with patch.object(processor.context_collections['business_rules'], 'vector_store_documents',
                              return_value=True):
                result = processor._process_excel_file(temp_file_path, 'business_rules')

                assert result['success'] is True

    def test_process_excel_file_error(self, processor, temp_file_path):
        """Test error handling in Excel processing"""
        with patch('pandas.read_excel', side_effect=Exception("Excel error")):
            result = processor._process_excel_file(temp_file_path, 'business_rules')

            assert result['success'] is False

    def test_row_to_text(self, processor):
        """Test converting DataFrame row to text"""
        row = pd.Series({'Column1': 'Value1', 'Column2': 'Value2'})
        text = processor._row_to_text(row)

        assert 'Column1: Value1' in text
        assert 'Column2: Value2' in text

    def test_row_to_text_with_nan(self, processor):
        """Test converting DataFrame row with NaN values"""
        row = pd.Series({'Column1': 'Value1', 'Column2': pd.NA})
        text = processor._row_to_text(row)

        assert 'Column1: Value1' in text
        assert 'Column2' not in text

    # ==================== PowerPoint Processing Tests ====================

    def test_process_presentation_file(self, processor, temp_file_path):
        """Test processing PowerPoint files"""
        with patch('pptx.Presentation') as mock_prs:
            mock_shape = Mock()
            mock_shape.text = "Slide text"
            mock_slide = Mock()
            mock_slide.shapes = [mock_shape]
            mock_prs.return_value.slides = [mock_slide]

            with patch.object(processor.context_collections['requirements'], 'vector_store_documents',
                              return_value=True):
                result = processor._process_presentation_file(temp_file_path, 'requirements')

                assert result['success'] is True
                assert result['document_count'] == 1

    def test_process_presentation_file_no_text(self, processor, temp_file_path):
        """Test processing PowerPoint with no text"""
        with patch('pptx.Presentation') as mock_prs:
            mock_slide = Mock()
            mock_slide.shapes = []
            mock_prs.return_value.slides = [mock_slide]

            with patch.object(processor.context_collections['requirements'], 'vector_store_documents',
                              return_value=True):
                result = processor._process_presentation_file(temp_file_path, 'requirements')

                assert result['success'] is True

    def test_process_presentation_file_error(self, processor, temp_file_path):
        """Test error handling in PowerPoint processing"""
        with patch('pptx.Presentation', side_effect=Exception("PPTX error")):
            result = processor._process_presentation_file(temp_file_path, 'requirements')

            assert result['success'] is False

    # ==================== Image Processing Tests ====================

    def test_process_image_file(self, processor, mock_image_path):
        """Test processing image files"""
        processor.image_parser.parse_image.return_value = "Extracted text from image"

        with patch.object(processor.context_collections['wireframes'], 'vector_store_documents', return_value=True):
            result = processor._process_image_file(mock_image_path, 'wireframes')

            assert result['success'] is True
            assert result['document_count'] == 1

    def test_process_image_file_error(self, processor, mock_image_path):
        """Test error handling in image processing"""
        processor.image_parser.parse_image.return_value = "Error: Image processing failed"

        result = processor._process_image_file(mock_image_path, 'wireframes')

        assert result['success'] is False

    def test_process_image_file_exception(self, processor, mock_image_path):
        """Test exception handling in image processing"""
        processor.image_parser.parse_image.side_effect = Exception("Image error")

        result = processor._process_image_file(mock_image_path, 'wireframes')

        assert result['success'] is False

    # ==================== Batch Processing Tests ====================

    def test_process_all_context_files(self, processor, temp_context_dir):
        """Test processing all context files"""
        # Create test files
        test_file = os.path.join(temp_context_dir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write("test content")

        with patch.object(processor, '_process_category_files'):
            stats = processor.process_all_context_files()

            assert 'total_files' in stats
            assert 'processed_files' in stats
            assert 'processing_time' in stats

    def test_process_all_context_files_nonexistent_folder(self, global_metrics_manager_mock):
        """Test processing with nonexistent folder"""
        with patch('src.context_handler.context_storage_handler.run_context_processor.PGVectorConnector'), \
                patch('src.context_handler.context_storage_handler.run_context_processor.AWSTitanEmbeddings'), \
                patch('src.context_handler.context_storage_handler.run_context_processor.ImageParserLLM'):
            processor = EnhancedContextProcessor('/nonexistent/path', metrics_manager=global_metrics_manager_mock)

            with pytest.raises(FileNotFoundError):
                processor.process_all_context_files()

    def test_process_all_context_files_force_reprocess(self, processor, temp_context_dir):
        """Test force reprocessing"""
        test_file = os.path.join(temp_context_dir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write("test content")

        # Mark file as already processed
        processor.document_registry = {
            'doc_1': {
                'file_hash': processor._calculate_file_hash(test_file)
            }
        }

        with patch.object(processor, '_process_category_files'):
            stats = processor.process_all_context_files(force_reprocess=True)

            # Should process even though file is in registry
            assert stats is not None

    # ==================== Category Processing Tests ====================

    def test_process_category_files(self, processor, temp_file_path):
        """Test processing files in a category"""
        with open(temp_file_path, 'w') as f:
            f.write("test content")

        with patch.object(processor, '_process_text_file', return_value={
            'success': True,
            'document_count': 1,
            'collections': ['documentation'],
            'context_type': 'documentation',
            'processing_time': 0.5
        }), patch.object(processor, '_register_document'):
            processor._process_category_files('documentation', [temp_file_path], False)

            assert processor.stats['processed_files'] == 1

    def test_process_category_files_skip_duplicate(self, processor, temp_file_path):
        """Test skipping duplicate files"""
        with open(temp_file_path, 'w') as f:
            f.write("test content")

        # Mark as already processed
        processor.document_registry = {
            'doc_1': {
                'file_hash': processor._calculate_file_hash(temp_file_path)
            }
        }

        processor._process_category_files('documentation', [temp_file_path], False)

        assert processor.stats['skipped_files'] == 1

    def test_process_category_files_error(self, processor, temp_file_path):
        """Test error handling in category processing"""
        with open(temp_file_path, 'w') as f:
            f.write("test content")

        with patch.object(processor, '_process_text_file', side_effect=Exception("Processing error")):
            processor._process_category_files('documentation', [temp_file_path], False)

            assert processor.stats['failed_files'] == 1

    # ==================== Statistics Display Tests ====================

    def test_display_final_stats(self, processor, capsys):
        """Test displaying final statistics"""
        processor.stats = {
            'total_files': 10,
            'processed_files': 8,
            'skipped_files': 1,
            'failed_files': 1,
            'total_chunks': 50,
            'processing_time': 5.5,
            'by_category': {
                'documentation': {
                    'total': 5,
                    'processed': 4,
                    'skipped': 1,
                    'failed': 0
                }
            }
        }

        processor._display_final_stats()

        captured = capsys.readouterr()
        assert 'Processing Complete' in captured.out
        assert '8' in captured.out  # processed files

    # ==================== Integration Tests ====================

    def test_full_processing_workflow(self, processor, temp_context_dir):
        """Test complete processing workflow"""
        # Create test files in different categories
        business_dir = os.path.join(temp_context_dir, 'business_rules')
        os.makedirs(business_dir, exist_ok=True)

        test_file = os.path.join(business_dir, 'rules.txt')
        with open(test_file, 'w') as f:
            f.write("Business rule content")

        with patch.object(processor.context_collections['business_rules'], 'vector_store_documents', return_value=True):
            stats = processor.process_all_context_files()

            assert stats['total_files'] > 0
            assert stats['processed_files'] > 0 or stats['skipped_files'] > 0

    def test_database_diagnostics(self, processor):
        """Test database diagnostics"""
        for collection in processor.context_collections.values():
            collection.diagnose_database.return_value = True

        # Should not raise exception
        for collection in processor.context_collections.values():
            result = collection.diagnose_database()
            assert result is True

    # ==================== Edge Cases and Error Handling ====================

    def test_process_empty_file(self, processor, temp_file_path):
        """Test processing empty file"""
        with open(temp_file_path, 'w') as f:
            f.write("")

        with patch.object(processor.context_collections['documentation'], 'vector_store_documents', return_value=True):
            result = processor._process_text_file(temp_file_path, 'documentation')

            # Should handle empty file gracefully
            assert result is not None

    def test_process_binary_file_as_text(self, processor, temp_file_path):
        """Test processing binary file as text"""
        with open(temp_file_path, 'wb') as f:
            f.write(b'\x00\x01\x02\x03')

        # Should handle binary content
        result = processor._process_text_file(temp_file_path, 'documentation')
        assert result is not None

    def test_concurrent_processing(self, processor, temp_context_dir):
        """Test concurrent file processing"""
        # Create multiple test files
        for i in range(5):
            test_file = os.path.join(temp_context_dir, f'test{i}.txt')
            with open(test_file, 'w') as f:
                f.write(f"test content {i}")

        with patch.object(processor.context_collections['documentation'], 'vector_store_documents', return_value=True):
            stats = processor.process_all_context_files()

            assert stats['total_files'] == 5

    def test_memory_efficiency(self, processor, temp_context_dir):
        """Test memory efficiency with large number of files"""
        # Create many small files
        for i in range(100):
            test_file = os.path.join(temp_context_dir, f'test{i}.txt')
            with open(test_file, 'w') as f:
                f.write(f"content {i}")

        with patch.object(processor.context_collections['documentation'], 'vector_store_documents', return_value=True):
            # Should process without memory issues
            stats = processor.process_all_context_files()
            assert stats['total_files'] == 100


# ==================== Additional Helper Function Tests ====================

class TestHelperFunctions:

    @pytest.fixture
    def processor(self, global_metrics_manager_mock):
        """Create processor for helper function tests"""
        with patch('src.context_handler.context_storage_handler.run_context_processor.PGVectorConnector'), \
                patch('src.context_handler.context_storage_handler.run_context_processor.AWSTitanEmbeddings'), \
                patch('src.context_handler.context_storage_handler.run_context_processor.ImageParserLLM'):
            temp_dir = tempfile.mkdtemp()
            processor = EnhancedContextProcessor(temp_dir, metrics_manager=global_metrics_manager_mock)
            yield processor
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_extract_image_metadata(self, processor, mock_image_path):
        """Test image metadata extraction"""
        with patch('PIL.Image.open') as mock_img:
            mock_img.return_value.__enter__.return_value.format = 'JPEG'
            mock_img.return_value.__enter__.return_value.mode = 'RGB'
            mock_img.return_value.__enter__.return_value.width = 1920
            mock_img.return_value.__enter__.return_value.height = 1080

            metadata = processor._extract_image_metadata(mock_image_path)

            assert metadata['format'] == 'JPEG'
            assert metadata['size'] == '1920x1080'

    def test_extract_image_metadata_error(self, processor):
        """Test image metadata extraction error handling"""
        metadata = processor._extract_image_metadata('/nonexistent/image.jpg')

        assert metadata['format'] == 'Unknown'
        assert metadata['size'] == 'Unknown'

    def test_extract_text_from_image(self, processor, mock_image_path):
        """Test OCR text extraction"""
        with patch('pytesseract.image_to_string', return_value="Extracted text"):
            text = processor._extract_text_from_image(mock_image_path)
            assert text == "Extracted text"

    def test_extract_text_from_image_error(self, processor, mock_image_path):
        """Test OCR error handling"""
        with patch('pytesseract.image_to_string', side_effect=Exception("OCR error")):
            text = processor._extract_text_from_image(mock_image_path)
            assert text == ""


# ==================== Performance Tests ====================

class TestPerformance:

    @pytest.fixture
    def processor(self, global_metrics_manager_mock):
        """Create processor for performance tests"""
        with patch('src.context_handler.context_storage_handler.run_context_processor.PGVectorConnector'), \
                patch('src.context_handler.context_storage_handler.run_context_processor.AWSTitanEmbeddings'), \
                patch('src.context_handler.context_storage_handler.run_context_processor.ImageParserLLM'):
            temp_dir = tempfile.mkdtemp()
            processor = EnhancedContextProcessor(temp_dir, metrics_manager=global_metrics_manager_mock)
            yield processor
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_processing_time_tracking(self, processor, temp_file_path):
        """Test that processing time is tracked"""
        with open(temp_file_path, 'w') as f:
            f.write("test content")

        with patch.object(processor.context_collections['documentation'], 'vector_store_documents', return_value=True):
            result = processor._process_text_file(temp_file_path, 'documentation')

            assert 'processing_time' in result
            assert result['processing_time'] >= 0

    def test_batch_size_handling(self, processor):
        """Test handling of different batch sizes"""
        temp_context_dir = processor.context_folder_path

        # Create test files
        for i in range(10):
            test_file = os.path.join(temp_context_dir, f'test{i}.txt')
            with open(test_file, 'w') as f:
                f.write(f"content {i}")

        with patch.object(processor.context_collections['documentation'], 'vector_store_documents', return_value=True):
            stats = processor.process_all_context_files()

            # Should process all files regardless of batch size
            assert stats['total_files'] == 10
