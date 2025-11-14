import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open
import pytest
import os
import configparser
import logging
import sys
from pathlib import Path

# Add the src directory to the Python path
current_dir = Path(__file__).parent
src_dir = current_dir.parent.parent / "src"
sys.path.insert(0, str(src_dir))

try:
    from src.llm_layer.model_manual_test_llm import LLM
    from src.aws_layer.aws_bedrock_connector import AWSBedrockConnector
    from src.metrics.metrics_manager import MetricsManager
except ImportError:
    # Fallback for different path configurations
    sys.path.append(str(current_dir.parent.parent))
    from src.llm_layer.model_manual_test_llm import LLM
    from src.aws_layer.aws_bedrock_connector import AWSBedrockConnector
    from src.metrics.metrics_manager import MetricsManager


class TestLLM(unittest.TestCase):
    """Test cases for the LLM class"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Mock environment variables
        self.env_vars = {
            'LLM_FAMILY': 'AWS',
            'LLM_TEMPERATURE': '0.7',
            'LLM_MODEL_ID': 'us.anthropic.claude-3-5-sonnet-20241022-v2:0',
            'LLM_MAX_TOKENS': '150000'
        }

        # Create a mock metrics manager
        self.mock_metrics_manager = Mock(spec=MetricsManager)

        # Patch environment variables
        self.env_patcher = patch.dict(os.environ, self.env_vars)
        self.env_patcher.start()

    def tearDown(self):
        """Clean up after each test method."""
        self.env_patcher.stop()

    @patch('src.llm_layer.model_manual_test_llm.AWSBedrockConnector')
    @patch('src.llm_layer.model_manual_test_llm.MetricsManager')
    def test_init_with_defaults(self, mock_metrics_manager_class, mock_aws_bedrock_class):
        """Test LLM initialization with default parameters"""
        # Arrange
        mock_metrics_instance = Mock(spec=MetricsManager)
        mock_metrics_manager_class.return_value = mock_metrics_instance
        mock_aws_instance = Mock(spec=AWSBedrockConnector)
        mock_aws_bedrock_class.return_value = mock_aws_instance

        # Act
        llm = LLM()

        # Assert
        self.assertEqual(llm.config_path, '../Config')
        self.assertIsInstance(llm.config_parser, configparser.ConfigParser)
        self.assertEqual(llm.llm_family, 'AWS')
        self.assertEqual(llm.TEMPERATURE, 0.7)  # Changed from string to float
        self.assertEqual(llm.MODEL_ID, 'us.anthropic.claude-3-5-sonnet-20241022-v2:0')
        self.assertEqual(llm.MAX_TOKENS, 150000)  # Changed from string to int
        mock_aws_bedrock_class.assert_called_once_with(
            model_id='us.anthropic.claude-3-5-sonnet-20241022-v2:0',
            metrics_manager=mock_metrics_instance
        )

    @patch('src.llm_layer.model_manual_test_llm.AWSBedrockConnector')
    def test_init_with_provided_metrics_manager(self, mock_aws_bedrock_class):
        """Test LLM initialization with provided metrics manager"""
        # Arrange
        mock_aws_instance = Mock(spec=AWSBedrockConnector)
        mock_aws_bedrock_class.return_value = mock_aws_instance

        # Act
        llm = LLM(metrics_manager=self.mock_metrics_manager)

        # Assert
        self.assertEqual(llm.metrics_manager, self.mock_metrics_manager)
        mock_aws_bedrock_class.assert_called_once_with(
            model_id='us.anthropic.claude-3-5-sonnet-20241022-v2:0',
            metrics_manager=self.mock_metrics_manager
        )

    @patch('src.llm_layer.model_manual_test_llm.AWSBedrockConnector')
    @patch('src.llm_layer.model_manual_test_llm.MetricsManager')
    def test_init_with_llm_family_parameter(self, mock_metrics_manager_class, mock_aws_bedrock_class):
        """Test LLM initialization with llm_family parameter"""
        # Arrange
        mock_metrics_instance = Mock(spec=MetricsManager)
        mock_metrics_manager_class.return_value = mock_metrics_instance
        mock_aws_instance = Mock(spec=AWSBedrockConnector)
        mock_aws_bedrock_class.return_value = mock_aws_instance

        # Act
        llm = LLM(llm_family="CUSTOM_AWS")

        # Assert
        # Note: The llm_family parameter is not currently used in the implementation
        # but we test that it can be passed without error
        self.assertEqual(llm.llm_family, 'AWS')  # Still gets from environment

    @patch('src.llm_layer.model_manual_test_llm.AWSBedrockConnector')
    @patch('src.llm_layer.model_manual_test_llm.MetricsManager')
    @patch.dict(os.environ, {}, clear=True)
    def test_init_with_missing_env_variables(self, mock_metrics_manager_class, mock_aws_bedrock_class):
        """Test LLM initialization with missing environment variables"""
        # Arrange
        mock_metrics_instance = Mock(spec=MetricsManager)
        mock_metrics_manager_class.return_value = mock_metrics_instance
        mock_aws_instance = Mock(spec=AWSBedrockConnector)
        mock_aws_bedrock_class.return_value = mock_aws_instance

        # Act
        llm = LLM()

        # Assert - Update based on actual default values in implementation
        self.assertIsNone(llm.llm_family)
        # The implementation sets default values when env vars are missing
        self.assertEqual(llm.TEMPERATURE, 0.05)  # Default value when env var is missing
        self.assertEqual(llm.MODEL_ID, 'anthropic.claude-3-sonnet-20240229-v1:0')  # Default MODEL_ID
        self.assertEqual(llm.MAX_TOKENS, 200000)  # Default MAX_TOKENS from env or fallback

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('src.llm_layer.model_manual_test_llm.AWSBedrockConnector')
    @patch('src.llm_layer.model_manual_test_llm.MetricsManager')
    def test_create_default_config_aws(self, mock_metrics_manager_class, mock_aws_bedrock_class,
                                       mock_exists, mock_makedirs, mock_file_open):
        """Test creation of default AWS configuration file"""
        # Arrange
        mock_metrics_instance = Mock(spec=MetricsManager)
        mock_metrics_manager_class.return_value = mock_metrics_instance
        mock_aws_instance = Mock(spec=AWSBedrockConnector)
        mock_aws_bedrock_class.return_value = mock_aws_instance
        mock_exists.return_value = False

        # Reset the mock to isolate just our call
        mock_file_open.reset_mock()

        llm = LLM()

        # Act
        llm.create_default_config_aws()

        # Assert
        mock_makedirs.assert_called_once_with('../Config', exist_ok=True)
        # Check that our specific call was made - use os-agnostic path check
        expected_calls = [call for call in mock_file_open.call_args_list
                          if 'ConfigAWS.properties' in str(call)]
        self.assertTrue(len(expected_calls) > 0, "ConfigAWS.properties file should be opened")

    @patch('src.llm_layer.model_manual_test_llm.AWSBedrockConnector')
    @patch('src.llm_layer.model_manual_test_llm.MetricsManager')
    def test_send_request_multimodal_success(self, mock_metrics_manager_class, mock_aws_bedrock_class):
        """Test successful multimodal request"""
        # Arrange
        mock_metrics_instance = Mock(spec=MetricsManager)
        mock_metrics_manager_class.return_value = mock_metrics_instance
        mock_aws_instance = Mock(spec=AWSBedrockConnector)
        mock_aws_instance.generate_response_multimodal.return_value = "Test response"
        mock_aws_bedrock_class.return_value = mock_aws_instance

        llm = LLM()
        template_prompt = "Analyze this image: {context}"
        image_path = "/path/to/image.jpg"
        input_variables = ["context"]
        input_variables_dict = {"context": "test context"}

        # Act
        result = llm.send_request_multimodal(
            template_prompt=template_prompt,
            image_path=image_path,
            call_type="test_call",
            input_variables=input_variables,
            input_variables_dict=input_variables_dict
        )

        # Assert
        self.assertEqual(result, "Test response")
        mock_aws_instance.generate_response_multimodal.assert_called_once_with(
            prompt="Analyze this image: test context",
            image_path=image_path,
            temperature=0.7,  # Changed from string to float
            max_tokens=150000,  # Changed from string to int
            call_type="test_call"
        )

    @patch('src.llm_layer.model_manual_test_llm.AWSBedrockConnector')
    @patch('src.llm_layer.model_manual_test_llm.MetricsManager')
    def test_send_request_multimodal_without_variables(self, mock_metrics_manager_class, mock_aws_bedrock_class):
        """Test multimodal request without input variables"""
        # Arrange
        mock_metrics_instance = Mock(spec=MetricsManager)
        mock_metrics_manager_class.return_value = mock_metrics_instance
        mock_aws_instance = Mock(spec=AWSBedrockConnector)
        mock_aws_instance.generate_response_multimodal.return_value = "Test response"
        mock_aws_bedrock_class.return_value = mock_aws_instance

        llm = LLM()
        template_prompt = "Analyze this image"
        image_path = "/path/to/image.jpg"

        # Act
        result = llm.send_request_multimodal(
            template_prompt=template_prompt,
            image_path=image_path
        )

        # Assert
        self.assertEqual(result, "Test response")
        mock_aws_instance.generate_response_multimodal.assert_called_once_with(
            prompt="Analyze this image",
            image_path=image_path,
            temperature=0.7,  # Changed from string to float
            max_tokens=150000,  # Changed from string to int
            call_type="default"
        )

    @patch('src.llm_layer.model_manual_test_llm.AWSBedrockConnector')
    @patch('src.llm_layer.model_manual_test_llm.MetricsManager')
    def test_send_request_multimodal_partial_variables(self, mock_metrics_manager_class, mock_aws_bedrock_class):
        """Test multimodal request with partial variable replacement"""
        # Arrange
        mock_metrics_instance = Mock(spec=MetricsManager)
        mock_metrics_manager_class.return_value = mock_metrics_instance
        mock_aws_instance = Mock(spec=AWSBedrockConnector)
        mock_aws_instance.generate_response_multimodal.return_value = "Test response"
        mock_aws_bedrock_class.return_value = mock_aws_instance

        llm = LLM()
        template_prompt = "Analyze this image: {context} and {missing_var}"
        image_path = "/path/to/image.jpg"
        input_variables = ["context", "missing_var"]
        input_variables_dict = {"context": "test context"}  # missing_var not provided

        # Act
        result = llm.send_request_multimodal(
            template_prompt=template_prompt,
            image_path=image_path,
            input_variables=input_variables,
            input_variables_dict=input_variables_dict
        )

        # Assert
        self.assertEqual(result, "Test response")
        # Only context should be replaced, missing_var should remain as placeholder
        expected_prompt = "Analyze this image: test context and {missing_var}"
        mock_aws_instance.generate_response_multimodal.assert_called_once_with(
            prompt=expected_prompt,
            image_path=image_path,
            temperature=0.7,  # Changed from string to float
            max_tokens=150000,  # Changed from string to int
            call_type="default"
        )

    @patch('src.llm_layer.model_manual_test_llm.AWSBedrockConnector')
    @patch('src.llm_layer.model_manual_test_llm.MetricsManager')
    @patch('src.llm_layer.model_manual_test_llm.logging')
    def test_send_request_multimodal_exception_handling(self, mock_logging, mock_metrics_manager_class,
                                                        mock_aws_bedrock_class):
        """Test multimodal request exception handling"""
        # Arrange
        mock_metrics_instance = Mock(spec=MetricsManager)
        mock_metrics_manager_class.return_value = mock_metrics_instance
        mock_aws_instance = Mock(spec=AWSBedrockConnector)
        mock_aws_instance.generate_response_multimodal.side_effect = Exception("Connection error")
        mock_aws_bedrock_class.return_value = mock_aws_instance

        llm = LLM()
        template_prompt = "Analyze this image"
        image_path = "/path/to/image.jpg"

        # Act
        result = llm.send_request_multimodal(
            template_prompt=template_prompt,
            image_path=image_path
        )

        # Assert
        self.assertIn("Error: Unable to get multimodal response from LLM", result)
        self.assertIn("Connection error", result)
        mock_logging.error.assert_called_once()

    @patch('src.llm_layer.model_manual_test_llm.AWSBedrockConnector')
    @patch('src.llm_layer.model_manual_test_llm.MetricsManager')
    def test_send_request_multimodal_empty_variables_dict(self, mock_metrics_manager_class, mock_aws_bedrock_class):
        """Test multimodal request with empty variables dictionary"""
        # Arrange
        mock_metrics_instance = Mock(spec=MetricsManager)
        mock_metrics_manager_class.return_value = mock_metrics_instance
        mock_aws_instance = Mock(spec=AWSBedrockConnector)
        mock_aws_instance.generate_response_multimodal.return_value = "Test response"
        mock_aws_bedrock_class.return_value = mock_aws_instance

        llm = LLM()
        template_prompt = "Analyze this image: {context}"
        image_path = "/path/to/image.jpg"
        input_variables = ["context"]
        input_variables_dict = {}  # Empty dictionary

        # Act
        result = llm.send_request_multimodal(
            template_prompt=template_prompt,
            image_path=image_path,
            input_variables=input_variables,
            input_variables_dict=input_variables_dict
        )

        # Assert
        self.assertEqual(result, "Test response")
        # Template should remain unchanged since no matching variables in dict
        mock_aws_instance.generate_response_multimodal.assert_called_once_with(
            prompt="Analyze this image: {context}",
            image_path=image_path,
            temperature=0.7,  # Changed from string to float
            max_tokens=150000,  # Changed from string to int
            call_type="default"
        )

    @patch('src.llm_layer.model_manual_test_llm.AWSBedrockConnector')
    @patch('src.llm_layer.model_manual_test_llm.MetricsManager')
    def test_send_request_multimodal_multiple_variables(self, mock_metrics_manager_class, mock_aws_bedrock_class):
        """Test multimodal request with multiple variable replacements"""
        # Arrange
        mock_metrics_instance = Mock(spec=MetricsManager)
        mock_metrics_manager_class.return_value = mock_metrics_instance
        mock_aws_instance = Mock(spec=AWSBedrockConnector)
        mock_aws_instance.generate_response_multimodal.return_value = "Test response"
        mock_aws_bedrock_class.return_value = mock_aws_instance

        llm = LLM()
        template_prompt = "Analyze this image: {context} with {style} and {format}"
        image_path = "/path/to/image.jpg"
        input_variables = ["context", "style", "format"]
        input_variables_dict = {
            "context": "business document",
            "style": "detailed analysis",
            "format": "JSON output"
        }

        # Act
        result = llm.send_request_multimodal(
            template_prompt=template_prompt,
            image_path=image_path,
            input_variables=input_variables,
            input_variables_dict=input_variables_dict
        )

        # Assert
        self.assertEqual(result, "Test response")
        expected_prompt = "Analyze this image: business document with detailed analysis and JSON output"
        mock_aws_instance.generate_response_multimodal.assert_called_once_with(
            prompt=expected_prompt,
            image_path=image_path,
            temperature=0.7,  # Changed from string to float
            max_tokens=150000,  # Changed from string to int
            call_type="default"
        )

    @patch('src.llm_layer.model_manual_test_llm.AWSBedrockConnector')
    @patch('src.llm_layer.model_manual_test_llm.MetricsManager')
    def test_attributes_after_initialization(self, mock_metrics_manager_class, mock_aws_bedrock_class):
        """Test that all attributes are properly set after initialization"""
        # Arrange
        mock_metrics_instance = Mock(spec=MetricsManager)
        mock_metrics_manager_class.return_value = mock_metrics_instance
        mock_aws_instance = Mock(spec=AWSBedrockConnector)
        mock_aws_bedrock_class.return_value = mock_aws_instance

        # Act
        llm = LLM()

        # Assert
        self.assertTrue(hasattr(llm, 'config_path'))
        self.assertTrue(hasattr(llm, 'config_parser'))
        self.assertTrue(hasattr(llm, 'metrics_manager'))
        self.assertTrue(hasattr(llm, 'llm_family'))
        self.assertTrue(hasattr(llm, 'TEMPERATURE'))
        self.assertTrue(hasattr(llm, 'MODEL_ID'))
        self.assertTrue(hasattr(llm, 'MAX_TOKENS'))
        self.assertTrue(hasattr(llm, 'aws_bedrock'))

    @patch('src.llm_layer.model_manual_test_llm.AWSBedrockConnector')
    @patch('src.llm_layer.model_manual_test_llm.MetricsManager')
    def test_config_parser_type(self, mock_metrics_manager_class, mock_aws_bedrock_class):
        """Test that config_parser is properly initialized as ConfigParser"""
        # Arrange
        mock_metrics_instance = Mock(spec=MetricsManager)
        mock_metrics_manager_class.return_value = mock_metrics_instance
        mock_aws_instance = Mock(spec=AWSBedrockConnector)
        mock_aws_bedrock_class.return_value = mock_aws_instance

        # Act
        llm = LLM()

        # Assert
        self.assertIsInstance(llm.config_parser, configparser.ConfigParser)

    @patch('src.llm_layer.model_manual_test_llm.AWSBedrockConnector')
    @patch('src.llm_layer.model_manual_test_llm.MetricsManager')
    @patch('builtins.print')  # Mock print to test debug output
    def test_debug_print_statement(self, mock_print, mock_metrics_manager_class, mock_aws_bedrock_class):
        """Test that debug print statement is called during initialization"""
        # Arrange
        mock_metrics_instance = Mock(spec=MetricsManager)
        mock_metrics_manager_class.return_value = mock_metrics_instance
        mock_aws_instance = Mock(spec=AWSBedrockConnector)
        mock_aws_bedrock_class.return_value = mock_aws_instance

        # Act
        llm = LLM(metrics_manager=mock_metrics_instance)

        # Assert
        mock_print.assert_called_once()
        call_args = mock_print.call_args[0][0]
        self.assertIn("DEBUG: LLM received metrics_manager:", call_args)

    @patch('src.llm_layer.model_manual_test_llm.AWSBedrockConnector')
    @patch('src.llm_layer.model_manual_test_llm.MetricsManager')
    def test_send_request_multimodal_with_none_inputs(self, mock_metrics_manager_class, mock_aws_bedrock_class):
        """Test multimodal request with None inputs for optional parameters"""
        # Arrange
        mock_metrics_instance = Mock(spec=MetricsManager)
        mock_metrics_manager_class.return_value = mock_metrics_instance
        mock_aws_instance = Mock(spec=AWSBedrockConnector)
        mock_aws_instance.generate_response_multimodal.return_value = "Test response"
        mock_aws_bedrock_class.return_value = mock_aws_instance

        llm = LLM()
        template_prompt = "Analyze this image"
        image_path = "/path/to/image.jpg"

        # Act
        result = llm.send_request_multimodal(
            template_prompt=template_prompt,
            image_path=image_path,
            call_type=None,
            input_variables=None,
            input_variables_dict=None
        )

        # Assert
        self.assertEqual(result, "Test response")
        mock_aws_instance.generate_response_multimodal.assert_called_once_with(
            prompt="Analyze this image",
            image_path=image_path,
            temperature=0.7,  # Changed from string to float
            max_tokens=150000,  # Changed from string to int
            call_type=None
        )


class TestLLMIntegration(unittest.TestCase):
    """Integration tests for LLM class"""

    def setUp(self):
        """Set up test fixtures for integration tests."""
        self.env_vars = {
            'LLM_FAMILY': 'AWS',
            'LLM_TEMPERATURE': '0.5',
            'LLM_MODEL_ID': 'test-model-id',
            'LLM_MAX_TOKENS': '100000'
        }
        self.env_patcher = patch.dict(os.environ, self.env_vars)
        self.env_patcher.start()

    def tearDown(self):
        """Clean up after integration tests."""
        self.env_patcher.stop()

    @patch('src.llm_layer.model_manual_test_llm.AWSBedrockConnector')
    @patch('src.llm_layer.model_manual_test_llm.MetricsManager')
    def test_full_workflow_integration(self, mock_metrics_manager_class, mock_aws_bedrock_class):
        """Test full workflow integration"""
        # Arrange
        mock_metrics_instance = Mock(spec=MetricsManager)
        mock_metrics_manager_class.return_value = mock_metrics_instance
        mock_aws_instance = Mock(spec=AWSBedrockConnector)
        mock_aws_instance.generate_response_multimodal.return_value = "Integrated response"
        mock_aws_bedrock_class.return_value = mock_aws_instance

        # Act
        llm = LLM()
        result = llm.send_request_multimodal(
            template_prompt="Process {document} with {requirements}",
            image_path="/test/path/document.jpg",
            call_type="integration_test",
            input_variables=["document", "requirements"],
            input_variables_dict={
                "document": "user story document",
                "requirements": "quality analysis"
            }
        )

        # Assert
        self.assertEqual(result, "Integrated response")
        self.assertEqual(llm.TEMPERATURE, 0.5)  # Changed from string to float
        self.assertEqual(llm.MODEL_ID, 'test-model-id')
        self.assertEqual(llm.MAX_TOKENS, 100000)  # Changed from string to int


if __name__ == '__main__':
    # Configure logging for tests
    logging.basicConfig(level=logging.DEBUG)

    # Run the tests
    unittest.main(verbosity=2)
