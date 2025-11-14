import pytest
import base64
import logging
import os
import sys
from unittest.mock import Mock, patch, MagicMock, mock_open
from src.llm_layer.image_parser_llm import ImageParserLLM
from src.llm_layer.model_manual_test_llm import LLM
from src.metrics.metrics_manager import MetricsManager


class TestImageParserLLM:
    """Comprehensive test suite for ImageParserLLM class"""

    @pytest.fixture
    def global_metrics_manager_mock(self):
        """Mock metrics manager fixture"""
        return Mock(spec=MetricsManager)

    @pytest.fixture
    def mock_llm(self):
        """Mock LLM instance"""
        return Mock(spec=LLM)

    @pytest.fixture
    def image_parser(self, global_metrics_manager_mock):
        """ImageParserLLM instance with mocked LLM"""
        with patch('src.llm_layer.image_parser_llm.LLM') as mock_llm_class:
            mock_llm_instance = Mock(spec=LLM)
            mock_llm_class.return_value = mock_llm_instance
            
            parser = ImageParserLLM(metrics_manager=global_metrics_manager_mock)
            parser.llm = mock_llm_instance
            return parser

    @pytest.fixture
    def image_parser_no_metrics(self):
        """ImageParserLLM instance without metrics manager"""
        with patch('src.llm_layer.image_parser_llm.LLM') as mock_llm_class:
            mock_llm_instance = Mock(spec=LLM)
            mock_llm_class.return_value = mock_llm_instance
            
            parser = ImageParserLLM()
            parser.llm = mock_llm_instance
            return parser

    @pytest.fixture
    def mock_image_path(self):
        """Mock image file path - use fixed path to avoid permission issues"""
        return "test_image.jpg"

    # ==================== Initialization Tests ====================

    @patch('src.llm_layer.image_parser_llm.LLM')
    def test_initialization_with_metrics_manager(self, mock_llm_class, global_metrics_manager_mock):
        """Test initialization with metrics manager"""
        mock_llm_instance = Mock(spec=LLM)
        mock_llm_class.return_value = mock_llm_instance
        
        parser = ImageParserLLM(metrics_manager=global_metrics_manager_mock)
        
        # Verify LLM was initialized with metrics manager
        mock_llm_class.assert_called_once_with(metrics_manager=global_metrics_manager_mock)
        assert parser.llm == mock_llm_instance
        assert isinstance(parser.logger, logging.Logger)
        assert parser.logger.name == "src.llm_layer.image_parser_llm"

    @patch('src.llm_layer.image_parser_llm.LLM')
    def test_initialization_without_metrics_manager(self, mock_llm_class):
        """Test initialization without metrics manager"""
        mock_llm_instance = Mock(spec=LLM)
        mock_llm_class.return_value = mock_llm_instance
        
        parser = ImageParserLLM()
        
        # Verify LLM was initialized with None metrics manager
        mock_llm_class.assert_called_once_with(metrics_manager=None)
        assert parser.llm == mock_llm_instance
        assert isinstance(parser.logger, logging.Logger)

    def test_logger_initialization(self, image_parser):
        """Test logger is properly initialized"""
        assert hasattr(image_parser, 'logger')
        assert isinstance(image_parser.logger, logging.Logger)

    # ==================== _encode_image_to_base64 Tests ====================

    def test_encode_image_to_base64_success(self, image_parser, mock_image_path):
        """Test successful image encoding to base64"""
        with patch("builtins.open", mock_open(read_data=b"fake image data")):
            result = image_parser._encode_image_to_base64(mock_image_path)
            
            expected = base64.b64encode(b"fake image data").decode('utf-8')
            assert result == expected

    def test_encode_image_to_base64_file_not_found(self, image_parser):
        """Test handling of file not found error"""
        non_existent_path = "/path/to/nonexistent/image.jpg"
        
        with patch("builtins.open", side_effect=FileNotFoundError("File not found")):
            with pytest.raises(FileNotFoundError):
                image_parser._encode_image_to_base64(non_existent_path)

    def test_encode_image_to_base64_permission_error(self, image_parser, mock_image_path):
        """Test handling of permission error"""
        with patch("builtins.open", side_effect=PermissionError("Permission denied")):
            with pytest.raises(PermissionError):
                image_parser._encode_image_to_base64(mock_image_path)

    def test_encode_image_to_base64_io_error(self, image_parser, mock_image_path):
        """Test handling of IO error"""
        with patch("builtins.open", side_effect=IOError("IO error")):
            with pytest.raises(IOError):
                image_parser._encode_image_to_base64(mock_image_path)

    def test_encode_image_to_base64_generic_exception(self, image_parser, mock_image_path):
        """Test handling of generic exception"""
        with patch("builtins.open", side_effect=Exception("Generic error")):
            with pytest.raises(Exception):
                image_parser._encode_image_to_base64(mock_image_path)

    # ==================== parse_image Tests ====================

    def test_parse_image_success(self, image_parser, mock_image_path):
        """Test successful image parsing"""
        # Mock the LLM response
        expected_response = "This is a detailed analysis of the image content"
        image_parser.llm.send_request_multimodal.return_value = expected_response
        
        result = image_parser.parse_image(mock_image_path)
        
        assert result == expected_response
        
        # Verify the LLM was called with correct parameters
        image_parser.llm.send_request_multimodal.assert_called_once()
        call_args = image_parser.llm.send_request_multimodal.call_args
        
        # Check the template prompt contains expected text
        assert "expert at analyzing images" in call_args[1]['template_prompt']
        assert call_args[1]['input_variables'] == []
        assert call_args[1]['input_variables_dict'] == {}
        assert call_args[1]['call_type'] == "image_analysis"
        assert call_args[1]['image_path'] == mock_image_path

    def test_parse_image_llm_exception(self, image_parser, mock_image_path):
        """Test image parsing when LLM raises an exception"""
        # Mock LLM to raise an exception
        error_message = "AWS connection failed"
        image_parser.llm.send_request_multimodal.side_effect = Exception(error_message)
        
        with patch.object(image_parser.logger, 'error') as mock_logger:
            result = image_parser.parse_image(mock_image_path)
            
            # Should return error message
            assert result.startswith("Error: Unable to get response from LLM")
            assert error_message in result
            
            # Verify error was logged
            mock_logger.assert_called_once()
            assert error_message in str(mock_logger.call_args)

    def test_parse_image_template_prompt_content(self, image_parser, mock_image_path):
        """Test that the template prompt contains expected content"""
        image_parser.llm.send_request_multimodal.return_value = "success"
        
        image_parser.parse_image(mock_image_path)
        
        call_args = image_parser.llm.send_request_multimodal.call_args
        template_prompt = call_args[1]['template_prompt']
        
        # Verify template prompt contains key instructions
        assert "expert at analyzing images" in template_prompt
        assert "detailed description" in template_prompt
        assert "extract it verbatim" in template_prompt
        assert "diagram or wireframe" in template_prompt

    # ==================== parse_image_batch Tests ====================

    def test_parse_image_batch_success(self, image_parser):
        """Test successful batch image parsing"""
        # Use simple string paths to avoid file permission issues
        image_paths = [f"test_image_{i}.jpg" for i in range(3)]
        
        # Mock individual parse_image calls
        expected_results = [f"Analysis result {i}" for i in range(3)]
        with patch.object(image_parser, 'parse_image', side_effect=expected_results):
            results = image_parser.parse_image_batch(image_paths)
            
            assert len(results) == 3
            assert results == expected_results
            
            # Verify parse_image was called for each path
            assert image_parser.parse_image.call_count == 3
            for path in image_paths:
                image_parser.parse_image.assert_any_call(path)

    def test_parse_image_batch_empty_list(self, image_parser):
        """Test batch parsing with empty list"""
        results = image_parser.parse_image_batch([])
        assert results == []

    def test_parse_image_batch_with_errors(self, image_parser):
        """Test batch parsing when some images fail"""
        # Use simple string paths to avoid file permission issues
        image_paths = ["test_image_0.jpg", "test_image_1.jpg"]
        
        # Mock parse_image to succeed for first, fail for second
        def mock_parse_image(path):
            if "0" in path:
                return "Success result"
            else:
                return "Error: Failed to parse"
        
        with patch.object(image_parser, 'parse_image', side_effect=mock_parse_image):
            results = image_parser.parse_image_batch(image_paths)
            
            assert len(results) == 2
            assert results[0] == "Success result"
            assert results[1] == "Error: Failed to parse"

    # ==================== Integration Tests ====================

    def test_full_parsing_workflow_integration(self, mock_image_path):
        """Test complete image parsing workflow with real object initialization"""
        with patch('src.llm_layer.image_parser_llm.LLM') as mock_llm_class:
            mock_llm_instance = Mock(spec=LLM)
            mock_llm_class.return_value = mock_llm_instance
            
            # Mock LLM response
            mock_llm_instance.send_request_multimodal.return_value = "Integrated parsing result"
            
            # Create parser instance
            parser = ImageParserLLM()
            
            # Execute parsing - only pass image_path (single argument)
            result = parser.parse_image(mock_image_path)
            
            # Verify result
            assert result == "Integrated parsing result"
            
            # Verify LLM was initialized and called correctly
            mock_llm_class.assert_called_once_with(metrics_manager=None)
            mock_llm_instance.send_request_multimodal.assert_called_once()

    def test_llm_integration_with_metrics(self, global_metrics_manager_mock, mock_image_path):
        """Test LLM integration with metrics manager"""
        with patch('src.llm_layer.image_parser_llm.LLM') as mock_llm_class:
            mock_llm_instance = Mock(spec=LLM)
            mock_llm_class.return_value = mock_llm_instance
            mock_llm_instance.send_request_multimodal.return_value = "Metrics test result"
            
            # Create parser with metrics manager
            parser = ImageParserLLM(metrics_manager=global_metrics_manager_mock)
            
            # Execute parsing
            result = parser.parse_image(mock_image_path)
            
            # Verify metrics manager was passed to LLM
            mock_llm_class.assert_called_once_with(metrics_manager=global_metrics_manager_mock)
            assert result == "Metrics test result"


# ==================== Additional Fixtures ====================

@pytest.fixture
def sample_image_data():
    """Sample binary image data (PNG header)"""
    return b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"

@pytest.fixture
def base64_encoded_image():
    """Base64 encoded sample image"""
    sample_data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    return base64.b64encode(sample_data).decode('utf-8')

@pytest.fixture
def mock_llm_multimodal_response():
    """Mock multimodal LLM response"""
    return "This image contains a diagram showing user interface wireframes with buttons and text fields."

@pytest.fixture
def multiple_image_paths():
    """Create multiple test image paths"""
    return [f"batch_test_image_{i}.jpg" for i in range(3)]


# ==================== Environment Variable Tests ====================

class TestImageParserLLMEnvironment:
    """Test ImageParserLLM behavior with different environment configurations"""
    
    @patch.dict(os.environ, {'LLM_FAMILY': 'AWS', 'LLM_MODEL_ID': 'test-model'})
    @patch('src.llm_layer.image_parser_llm.LLM')
    def test_environment_variables_passed_to_llm(self, mock_llm_class):
        """Test that environment variables are properly used"""
        mock_llm_instance = Mock(spec=LLM)
        mock_llm_class.return_value = mock_llm_instance
        
        parser = ImageParserLLM()
        
        # Verify LLM was created (environment variables are handled by LLM class)
        mock_llm_class.assert_called_once_with(metrics_manager=None)
        assert parser.llm == mock_llm_instance


# ==================== Error Handling and Logging Tests ====================

class TestImageParserLLMErrorHandling:
    """Test error handling and logging in ImageParserLLM"""
    
    @patch('src.llm_layer.image_parser_llm.LLM')
    def test_logger_error_on_multimodal_failure(self, mock_llm_class):
        """Test that errors are properly logged when multimodal request fails"""
        mock_llm_instance = Mock(spec=LLM)
        mock_llm_class.return_value = mock_llm_instance
        
        # Use simple string path
        image_path = "error_test.jpg"
        
        # Mock LLM to raise exception
        error_msg = "Network timeout error"
        mock_llm_instance.send_request_multimodal.side_effect = Exception(error_msg)
        
        parser = ImageParserLLM()
        
        with patch.object(parser.logger, 'error') as mock_logger:
            result = parser.parse_image(image_path)
            
            # Verify error was logged with correct message
            mock_logger.assert_called_once()
            logged_message = mock_logger.call_args[0][0]
            assert error_msg in logged_message
            assert image_path in logged_message
            
            # Verify error response format
            assert result.startswith("Error: Unable to get response from LLM")
            assert error_msg in result
    
    def test_encode_image_logging_on_various_errors(self):
        """Test that _encode_image_to_base64 logs errors appropriately"""
        with patch('src.llm_layer.image_parser_llm.LLM') as mock_llm_class:
            mock_llm_instance = Mock(spec=LLM)
            mock_llm_class.return_value = mock_llm_instance
            
            parser = ImageParserLLM()
            
            # Test different types of exceptions
            test_cases = [
                (FileNotFoundError("File not found"), "File not found"),
                (PermissionError("Permission denied"), "Permission denied"),
                (IOError("Disk error"), "Disk error"),
                (Exception("Generic error"), "Generic error")
            ]
            
            for exception, error_text in test_cases:
                with patch("builtins.open", side_effect=exception):
                    with patch.object(parser.logger, 'error') as mock_logger:
                        with pytest.raises(type(exception)):
                            parser._encode_image_to_base64("test_path.jpg")
                        
                        # Verify error was logged
                        mock_logger.assert_called_once()
                        logged_message = mock_logger.call_args[0][0]
                        assert "test_path.jpg" in logged_message
                        assert "base64" in logged_message


# ==================== Performance and Resource Tests ====================

class TestImageParserLLMPerformance:
    """Test performance-related aspects of ImageParserLLM"""
    
    def test_batch_processing_efficiency(self):
        """Test that batch processing calls parse_image efficiently"""
        with patch('src.llm_layer.image_parser_llm.LLM') as mock_llm_class:
            mock_llm_instance = Mock(spec=LLM)
            mock_llm_class.return_value = mock_llm_instance
            
            parser = ImageParserLLM()
            
            # Use simple string paths
            image_paths = [f"batch_{i}.jpg" for i in range(5)]
            
            # Mock parse_image method
            with patch.object(parser, 'parse_image', return_value="result") as mock_parse:
                results = parser.parse_image_batch(image_paths)
                
                # Verify efficiency: exactly one call per image
                assert mock_parse.call_count == 5
                assert len(results) == 5
                assert all(result == "result" for result in results)
                
                # Verify each image path was called
                for path in image_paths:
                    mock_parse.assert_any_call(path)
    
    def test_memory_usage_with_large_base64_encoding(self):
        """Test base64 encoding with larger image data"""
        with patch('src.llm_layer.image_parser_llm.LLM') as mock_llm_class:
            mock_llm_instance = Mock(spec=LLM)
            mock_llm_class.return_value = mock_llm_instance
            
            parser = ImageParserLLM()
            
            # Create a larger fake image (simulate a 1KB image)
            large_image_data = b"fake_image_data" * 100  # ~1.5KB
            
            # Mock file operations to avoid permission issues
            with patch("builtins.open", mock_open(read_data=large_image_data)):
                # Test encoding
                result = parser._encode_image_to_base64("large_image.jpg")
                
                # Verify base64 encoding worked
                expected = base64.b64encode(large_image_data).decode('utf-8')
                assert result == expected
                assert len(result) > 1000  # Should be reasonably large
