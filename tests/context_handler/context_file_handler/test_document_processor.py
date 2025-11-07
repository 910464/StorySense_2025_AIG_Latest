import pytest
import os
import json
import tempfile
import pandas as pd
from unittest.mock import patch, Mock, mock_open, MagicMock
from pathlib import Path

from src.context_handler.context_file_handler.document_processor import DocumentProcessor


class TestDocumentProcessor:
    @pytest.fixture
    def processor(self):
        """Create a DocumentProcessor instance"""
        return DocumentProcessor()

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_files(self, temp_dir):
        """Create sample files for testing"""
        # Create a text file
        text_path = os.path.join(temp_dir, "sample.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write("This is a sample text file\nWith multiple lines\nFor testing")

        # Create a JSON file
        json_path = os.path.join(temp_dir, "sample.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"key1": "value1", "key2": {"nested": "value2"}}, f)

        # Create a CSV file
        csv_path = os.path.join(temp_dir, "sample.csv")
        pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"]
        }).to_csv(csv_path, index=False)

        return {
            "text": text_path,
            "json": json_path,
            "csv": csv_path
        }

    def test_initialization(self, processor):
        """Test initialization of DocumentProcessor"""
        assert processor is not None
        assert hasattr(processor, 'file_processors')
        assert '.pdf' in processor.file_processors
        assert '.docx' in processor.file_processors
        assert '.txt' in processor.file_processors
        assert '.json' in processor.file_processors
        assert '.csv' in processor.file_processors

    def test_process_unsupported_file(self, processor):
        """Test processing an unsupported file type"""
        with pytest.raises(ValueError) as excinfo:
            processor.process("file.unsupported")
        assert "Unsupported file type" in str(excinfo.value)

    def test_process_text_file(self, processor, sample_files):
        """Test processing a text file"""
        result = processor.process(sample_files["text"])

        assert len(result) == 1
        assert result[0]['text'] == "This is a sample text file\nWith multiple lines\nFor testing"
        assert result[0]['metadata']['source_file'] == "sample.txt"
        assert result[0]['metadata']['file_type'] == ".txt"
        assert result[0]['metadata']['character_count'] > 0
        assert result[0]['metadata']['line_count'] == 3

    def test_process_text_file_with_unicode_error(self, processor):
        """Test processing a text file with unicode error"""
        # Create a mock that properly handles the context manager pattern
        mock_file = MagicMock()
        mock_file.__enter__.return_value.read.return_value = "Content after fallback to latin-1"

        # Create a mock for open that first raises UnicodeDecodeError, then returns our mock file
        with patch('builtins.open', side_effect=[
            UnicodeDecodeError('utf-8', b'test', 0, 1, 'invalid'),
            mock_file
        ]):
            # We need to patch Path.exists to avoid file not found errors
            with patch('pathlib.Path.exists', return_value=True):
                result = processor._process_text("file.txt")

                assert len(result) == 1
                assert result[0]['text'] == "Content after fallback to latin-1"
                assert result[0]['metadata']['encoding'] == 'latin-1'

    def test_process_json_file(self, processor, sample_files):
        """Test processing a JSON file"""
        result = processor.process(sample_files["json"])

        assert len(result) == 1
        assert "key1: value1" in result[0]['text']
        assert "key2:" in result[0]['text']
        assert "nested: value2" in result[0]['text']
        assert result[0]['metadata']['source_file'] == "sample.json"
        assert result[0]['metadata']['file_type'] == "json"
        assert result[0]['metadata']['data_type'] == "dict"

    def test_process_json_file_with_list(self, processor):
        """Test processing a JSON file with a list as root element"""
        json_content = [{"item1": "value1"}, {"item2": "value2"}]

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_file:
            temp_file.write(json.dumps(json_content).encode('utf-8'))
            temp_file_path = temp_file.name

        try:
            result = processor.process(temp_file_path)

            assert len(result) == 1
            assert "item1: value1" in result[0]['text']
            assert "item2: value2" in result[0]['text']
            assert result[0]['metadata']['data_type'] == "list"
        finally:
            os.unlink(temp_file_path)

    def test_process_json_file_with_scalar(self, processor):
        """Test processing a JSON file with a scalar value"""
        json_content = "Just a string"

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_file:
            temp_file.write(json.dumps(json_content).encode('utf-8'))
            temp_file_path = temp_file.name

        try:
            result = processor.process(temp_file_path)

            assert len(result) == 1
            assert result[0]['text'] == "Just a string"
            assert result[0]['metadata']['data_type'] == "str"
        finally:
            os.unlink(temp_file_path)

    def test_process_csv_file(self, processor, sample_files):
        """Test processing a CSV file"""
        result = processor.process(sample_files["csv"])

        assert len(result) == 1
        # Fix the assertion to match the actual format of the CSV output
        assert "col1" in result[0]['text']
        assert "col2" in result[0]['text']
        assert "1" in result[0]['text']
        assert "a" in result[0]['text']
        assert "2" in result[0]['text']
        assert "b" in result[0]['text']
        assert "3" in result[0]['text']
        assert "c" in result[0]['text']
        assert result[0]['metadata']['source_file'] == "sample.csv"
        assert result[0]['metadata']['file_type'] == "csv"
        assert result[0]['metadata']['row_count'] == 3
        assert result[0]['metadata']['column_count'] == 2
        assert "col1" in result[0]['metadata']['columns']
        assert "col2" in result[0]['metadata']['columns']

    @patch('PyPDF2.PdfReader')
    def test_process_pdf_file(self, mock_pdf_reader, processor):
        """Test processing a PDF file"""
        # Mock the PDF reader
        mock_page1 = Mock()
        mock_page1.extract_text.return_value = "Page 1 content"
        mock_page2 = Mock()
        mock_page2.extract_text.return_value = "Page 2 content"
        mock_page3 = Mock()
        mock_page3.extract_text.return_value = ""  # Empty page

        # Create a mock PDF reader with pages attribute
        mock_pdf = Mock()
        mock_pdf.pages = [mock_page1, mock_page2, mock_page3]
        mock_pdf_reader.return_value = mock_pdf

        # Mock the open function
        with patch('builtins.open', mock_open()):
            result = processor._process_pdf("test.pdf")

            assert len(result) == 2  # Should only include non-empty pages
            assert result[0]['text'] == "Page 1 content"
            assert result[0]['metadata']['source_file'] == "test.pdf"
            assert result[0]['metadata']['file_type'] == "pdf"
            assert result[0]['metadata']['page_number'] == 1
            assert result[0]['metadata']['total_pages'] == 3

            assert result[1]['text'] == "Page 2 content"
            assert result[1]['metadata']['page_number'] == 2

    @patch('PyPDF2.PdfReader')
    def test_process_pdf_file_exception(self, mock_pdf_reader, processor):
        """Test processing a PDF file with an exception"""
        # Make the PDF reader raise an exception
        mock_pdf_reader.side_effect = Exception("PDF processing error")

        # Mock the open function
        with patch('builtins.open', mock_open()):
            with pytest.raises(Exception) as excinfo:
                processor._process_pdf("test.pdf")

            assert "PDF processing error" in str(excinfo.value)

    @patch('docx.Document')
    def test_process_docx_file(self, mock_document, processor):
        """Test processing a Word document"""
        # Mock the Document
        mock_doc = Mock()

        # Mock paragraphs
        mock_para1 = Mock()
        mock_para1.text = "Paragraph 1"
        mock_para2 = Mock()
        mock_para2.text = "Paragraph 2"
        mock_para3 = Mock()
        mock_para3.text = ""  # Empty paragraph
        mock_doc.paragraphs = [mock_para1, mock_para2, mock_para3]

        # Mock tables
        mock_cell1 = Mock()
        mock_cell1.text = "Cell 1"
        mock_cell2 = Mock()
        mock_cell2.text = "Cell 2"
        mock_cell3 = Mock()
        mock_cell3.text = ""  # Empty cell
        mock_row = Mock()
        mock_row.cells = [mock_cell1, mock_cell2, mock_cell3]
        mock_empty_row = Mock()
        mock_empty_row.cells = [mock_cell3]  # Row with only empty cells
        mock_table = Mock()
        mock_table.rows = [mock_row, mock_empty_row]
        mock_doc.tables = [mock_table]

        mock_document.return_value = mock_doc

        result = processor._process_docx("test.docx")

        assert len(result) == 1
        assert "Paragraph 1" in result[0]['text']
        assert "Paragraph 2" in result[0]['text']
        assert "Cell 1 | Cell 2" in result[0]['text']
        assert result[0]['metadata']['source_file'] == "test.docx"
        assert result[0]['metadata']['file_type'] == "docx"
        assert result[0]['metadata']['paragraph_count'] == 2  # Should only count non-empty paragraphs
        assert result[0]['metadata']['table_count'] == 1

    @patch('docx.Document')
    def test_process_docx_file_empty(self, mock_document, processor):
        """Test processing an empty Word document"""
        # Mock the Document with no content
        mock_doc = Mock()
        mock_doc.paragraphs = []
        mock_doc.tables = []
        mock_document.return_value = mock_doc

        result = processor._process_docx("empty.docx")

        assert len(result) == 1
        assert result[0]['text'] == ""
        assert result[0]['metadata']['source_file'] == "empty.docx"
        assert result[0]['metadata']['file_type'] == "docx"
        assert result[0]['metadata']['paragraph_count'] == 0
        assert result[0]['metadata']['table_count'] == 0

    @patch('docx.Document')
    def test_process_docx_file_exception(self, mock_document, processor):
        """Test processing a Word document with an exception"""
        # Make the Document constructor raise an exception
        mock_document.side_effect = Exception("DOCX processing error")

        with pytest.raises(Exception) as excinfo:
            processor._process_docx("test.docx")

        assert "DOCX processing error" in str(excinfo.value)

    def test_process_pptx_file(self, processor):
        """Test processing a PowerPoint presentation"""
        # Mock the entire pptx module
        with patch('src.context_handler.context_file_handler.document_processor.Presentation') as mock_presentation:
            # Mock the Presentation
            mock_pres = Mock()

            # Mock shapes with text
            mock_shape1 = Mock()
            mock_shape1.text = "Slide 1 Shape 1"
            mock_shape2 = Mock()
            mock_shape2.text = "Slide 1 Shape 2"
            mock_shape3 = Mock()
            mock_shape3.text = ""  # Empty shape

            # Mock slides
            mock_slide1 = Mock()
            mock_slide1.shapes = [mock_shape1, mock_shape2]
            mock_slide2 = Mock()
            mock_slide2.shapes = [mock_shape3]  # No text
            mock_pres.slides = [mock_slide1, mock_slide2]

            mock_presentation.return_value = mock_pres

            result = processor._process_pptx("test.pptx")

            assert len(result) == 1  # Only one slide has text
            assert "Slide 1 Shape 1" in result[0]['text']
            assert "Slide 1 Shape 2" in result[0]['text']
            assert result[0]['metadata']['source_file'] == "test.pptx"
            assert result[0]['metadata']['file_type'] == "pptx"
            assert result[0]['metadata']['slide_number'] == 1
            assert result[0]['metadata']['total_slides'] == 2

    def test_process_pptx_file_empty(self, processor):
        """Test processing an empty PowerPoint presentation"""
        # Mock the entire pptx module
        with patch('src.context_handler.context_file_handler.document_processor.Presentation') as mock_presentation:
            # Mock the Presentation with no content
            mock_pres = Mock()
            mock_pres.slides = []
            mock_presentation.return_value = mock_pres

            result = processor._process_pptx("empty.pptx")

            assert len(result) == 0  # No documents should be returned

    def test_process_pptx_file_exception(self, processor):
        """Test processing a PowerPoint presentation with an exception"""
        # Mock the entire pptx module to raise an exception
        with patch('src.context_handler.context_file_handler.document_processor.Presentation',
                   side_effect=Exception("PPTX processing error")):
            with pytest.raises(Exception) as excinfo:
                processor._process_pptx("test.pptx")

            assert "PPTX processing error" in str(excinfo.value)

    def test_process_excel_file(self, processor):
        """Test processing an Excel file"""
        # Create a mock DataFrame
        mock_df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"]
        })

        # Mock both pandas.read_excel and pandas.ExcelFile
        with patch('pandas.read_excel', return_value=mock_df), \
                patch('pandas.ExcelFile') as mock_excel_file:
            # Configure the mock ExcelFile
            mock_excel = Mock()
            mock_excel.sheet_names = ["Sheet1"]
            mock_excel_file.return_value = mock_excel

            result = processor._process_excel("test.xlsx")

            assert len(result) == 1
            assert "col1" in result[0]['text']
            assert "col2" in result[0]['text']
            assert "1" in result[0]['text']
            assert "a" in result[0]['text']
            assert result[0]['metadata']['source_file'] == "test.xlsx"
            assert result[0]['metadata']['file_type'] == "xlsx"
            assert result[0]['metadata']['sheet_name'] == "Sheet1"
            assert result[0]['metadata']['row_count'] == 3
            assert result[0]['metadata']['column_count'] == 2

    def test_process_excel_file_multiple_sheets(self, processor):
        """Test processing an Excel file with multiple sheets"""
        # Create mock DataFrames for each sheet
        mock_df1 = pd.DataFrame({
            "col1": [1, 2],
            "col2": ["a", "b"]
        })
        mock_df2 = pd.DataFrame({
            "col3": [3, 4],
            "col4": ["c", "d"]
        })

        # Mock pandas.read_excel to return different DataFrames for different sheets
        def mock_read_excel(file_path, sheet_name=None):
            if sheet_name == "Sheet1":
                return mock_df1
            elif sheet_name == "Sheet2":
                return mock_df2
            return pd.DataFrame()

        # Mock both pandas.read_excel and pandas.ExcelFile
        with patch('pandas.read_excel', side_effect=mock_read_excel), \
                patch('pandas.ExcelFile') as mock_excel_file:
            # Configure the mock ExcelFile
            mock_excel = Mock()
            mock_excel.sheet_names = ["Sheet1", "Sheet2"]
            mock_excel_file.return_value = mock_excel

            result = processor._process_excel("test.xlsx")

            assert len(result) == 2
            # Check Sheet1 content
            assert "col1" in result[0]['text']
            assert "col2" in result[0]['text']
            assert result[0]['metadata']['sheet_name'] == "Sheet1"
            # Check Sheet2 content
            assert "col3" in result[1]['text']
            assert "col4" in result[1]['text']
            assert result[1]['metadata']['sheet_name'] == "Sheet2"

    def test_process_excel_file_empty(self, processor):
        """Test processing an Excel file with empty sheets"""
        # Create an empty DataFrame
        mock_df = pd.DataFrame()

        # Mock both pandas.read_excel and pandas.ExcelFile
        with patch('pandas.read_excel', return_value=mock_df), \
                patch('pandas.ExcelFile') as mock_excel_file:
            # Configure the mock ExcelFile
            mock_excel = Mock()
            mock_excel.sheet_names = ["Sheet1"]
            mock_excel_file.return_value = mock_excel

            result = processor._process_excel("empty.xlsx")

            assert len(result) == 0  # No documents should be returned for empty sheets

    def test_process_excel_file_exception(self, processor):
        """Test processing an Excel file with an exception"""
        # Mock pandas.ExcelFile to raise an exception
        with patch('pandas.ExcelFile', side_effect=Exception("Excel processing error")):
            with pytest.raises(Exception) as excinfo:
                processor._process_excel("test.xlsx")

            assert "Excel processing error" in str(excinfo.value)

    def test_dict_to_text(self, processor):
        """Test _dict_to_text method"""
        test_dict = {
            "key1": "value1",
            "key2": {
                "nested1": "nested_value1",
                "nested2": "nested_value2"
            },
            "key3": ["item1", "item2", "item3"]
        }

        result = processor._dict_to_text(test_dict)

        assert "key1: value1" in result
        assert "key2:" in result
        assert "nested1: nested_value1" in result
        assert "nested2: nested_value2" in result
        assert "key3: item1, item2, item3" in result

    def test_dict_to_text_with_prefix(self, processor):
        """Test _dict_to_text method with prefix"""
        test_dict = {
            "key1": "value1",
            "key2": {
                "nested1": "nested_value1"
            }
        }

        result = processor._dict_to_text(test_dict, prefix="  ")

        assert "  key1: value1" in result
        assert "  key2:" in result
        assert "    nested1: nested_value1" in result  # Should have additional indentation

    def test_process_with_missing_libraries(self):
        """Test processing when document libraries are not available"""
        with patch('src.context_handler.context_file_handler.document_processor.HAS_DOC_PROCESSORS', False):
            processor = DocumentProcessor()

            # Test PDF processing
            with pytest.raises(ImportError) as excinfo:
                processor._process_pdf("test.pdf")
            assert "PyPDF2 not installed" in str(excinfo.value)

            # Test DOCX processing
            with pytest.raises(ImportError) as excinfo:
                processor._process_docx("test.docx")
            assert "python-docx not installed" in str(excinfo.value)

            # Test PPTX processing
            with pytest.raises(ImportError) as excinfo:
                processor._process_pptx("test.pptx")
            assert "python-pptx not installed" in str(excinfo.value)

            # Test Excel processing
            with pytest.raises(ImportError) as excinfo:
                processor._process_excel("test.xlsx")
            assert "openpyxl not installed" in str(excinfo.value)