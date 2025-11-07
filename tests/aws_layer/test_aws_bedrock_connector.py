import pytest
from unittest.mock import Mock, patch, MagicMock, ANY
import json
import boto3
import io
import os
from PIL import Image
from src.aws_layer.aws_bedrock_connector import AWSBedrockConnector


class TestAWSBedrockConnector:
    @pytest.fixture
    def connector(self, metrics_manager_mock):
        with patch('boto3.Session') as mock_session:
            mock_client = Mock()
            mock_session.return_value.client.return_value = mock_client
            connector = AWSBedrockConnector(model_id="anthropic.claude-3-sonnet-20240229-v1:0",
                                            metrics_manager=metrics_manager_mock)
            connector.bedrock_runtime = mock_client
            connector.has_tokenizer = False
            return connector

    def test_initialization(self, connector):
        assert connector is not None
        assert connector.model_id == "anthropic.claude-3-sonnet-20240229-v1:0"

    @patch('boto3.Session')
    def test_initialization_with_credentials(self, mock_session):
        with patch.dict('os.environ', {
            'AWS_ACCESS_KEY_ID': 'test_key',
            'AWS_SECRET_ACCESS_KEY': 'test_secret',
            'AWS_REGION': 'us-east-1'
        }):
            connector = AWSBedrockConnector()
            mock_session.assert_called_with(
                aws_access_key_id='test_key',
                aws_secret_access_key='test_secret',
                region_name='us-east-1'
            )

    @patch('boto3.Session')
    def test_initialization_with_session_token(self, mock_session):
        """Test initialization with session token - covers lines 89-92"""
        with patch.dict('os.environ', {
            'AWS_ACCESS_KEY_ID': 'test_key',
            'AWS_SECRET_ACCESS_KEY': 'test_secret',
            'AWS_SESSION_TOKEN': 'test_token',
            'AWS_REGION': 'us-east-1'
        }):
            connector = AWSBedrockConnector()
            mock_session.assert_called_with(
                aws_access_key_id='test_key',
                aws_secret_access_key='test_secret',
                aws_session_token='test_token',
                region_name='us-east-1'
            )

    @patch('boto3.Session')
    def test_initialization_error_handling(self, mock_session):
        """Test error handling during initialization - covers lines 94, 98"""
        mock_session.side_effect = Exception("AWS Error")

        # Should log error but not crash
        with patch('logging.error') as mock_log:
            connector = AWSBedrockConnector()
            mock_log.assert_called_with(ANY)

    def test_count_tokens(self, connector):
        # Test token counting without tokenizer
        result = connector.count_tokens("This is a test sentence.")
        assert result == 6  # 24 chars / 4 = 6 tokens (approximate)

        # Test with empty text
        result = connector.count_tokens("")
        assert result == 0

        # Test with None
        result = connector.count_tokens(None)
        assert result == 0

    def test_generate_response_with_guardrails(self, connector):
        """Test guardrails functionality - covers lines 160-212"""
        # Set up guardrails configuration
        connector.use_guardrails = True

        # Mock the response from AWS Bedrock with guardrail action
        mock_response = {
            'body': Mock(),
        }
        mock_response['body'].read.return_value = json.dumps({
            'content': [{'type': 'text', 'text': 'Test response'}],
            'guardrailAction': 'BLOCKED',
            'usage': {'input_tokens': 10, 'output_tokens': 5}
        })

        connector.bedrock_runtime.invoke_model.return_value = mock_response

        # Mock circuit breaker to directly call the function
        def mock_execute(func, *args, **kwargs):
            return func(*args, **kwargs)

        connector.circuit_breaker.execute = mock_execute

        result = connector.generate_response("Test prompt")

        assert result == "Test response"
        connector.metrics_manager.record_llm_call.assert_called_once_with(
            call_type="default",
            input_tokens=10,
            output_tokens=5,
            latency=ANY,
            model_name=connector.model_id,
            guardrail_triggered=True
        )

    def test_generate_response_with_guardrail_region(self, connector):
        """Test guardrails with different region - covers lines 175-183"""
        # Set up guardrails configuration with different region
        connector.use_guardrails = True

        # Mock environment variables
        with patch.dict('os.environ', {
            'guardrail_id': 'test-guardrail',
            'region': 'us-west-2'  # Different from default
        }):
            # Mock the response
            mock_response = {
                'body': Mock(),
            }
            mock_response['body'].read.return_value = json.dumps({
                'content': [{'type': 'text', 'text': 'Test response'}],
                'guardrailAction': 'NONE'
            })

            connector.bedrock_runtime.invoke_model.return_value = mock_response

            # Mock circuit breaker
            def mock_execute(func, *args, **kwargs):
                return func(*args, **kwargs)

            connector.circuit_breaker.execute = mock_execute

            result = connector.generate_response("Test prompt")

            assert result == "Test response"
            # Should not record guardrail triggered
            connector.metrics_manager.record_llm_call.assert_called_once_with(
                call_type="default",
                input_tokens=ANY,
                output_tokens=ANY,
                latency=ANY,
                model_name=connector.model_id,
                guardrail_triggered=False
            )

    def test_generate_response_streaming(self, connector):
        """Test streaming response generation - covers lines 316-344"""
        # Create a mock response for streaming
        mock_response = {
            'body': Mock()
        }

        # Set up the mock to return a streaming response
        mock_response['body'].read.return_value = json.dumps({
            'completion': 'Test streaming response'
        })

        connector.bedrock_runtime.invoke_model.return_value = mock_response

        # Call the streaming method
        result = list(connector.generate_response_streaming("Test prompt", 0.7, 500, "streaming"))

        # Check the result
        assert "".join(result) == "Test streaming response"

        # Verify metrics were recorded
        connector.metrics_manager.record_llm_call.assert_called_once()

    def test_generate_response_streaming_error(self, connector):
        """Test error handling in streaming - covers lines 316-344"""
        # Make the API call fail
        connector.bedrock_runtime.invoke_model.side_effect = Exception("API Error")

        # Call the streaming method
        result = list(connector.generate_response_streaming("Test prompt", 0.7, 500, "streaming"))

        # Check that we get an error message
        assert any("Error:" in chunk for chunk in result)

        # Verify error was recorded
        connector.metrics_manager.record_error.assert_called_once()

    def test_multimodal_edge_cases(self, connector, mock_image_path):
        """Test edge cases in multimodal processing - covers lines 351-370"""
        # Test with None image path
        result = connector.generate_response_multimodal("Test prompt", None, 0.7, 500, "multimodal")

        # Should fall back to text-only processing
        assert isinstance(result, str)

        # Test with image processing error
        with patch.object(connector, '_resize_and_encode_image', side_effect=Exception("Image processing error")):
            result = connector.generate_response_multimodal("Test prompt", mock_image_path, 0.7, 500, "multimodal")

            # Should handle the error and return an error message
            assert "Error:" in result
            connector.metrics_manager.record_llm_call.assert_called()

    def test_resize_and_encode_image_with_alpha(self, connector, mock_image_path):
        """Test image processing with alpha channel - covers part of lines 351-370"""
        with patch('PIL.Image.open') as mock_open:
            mock_img = Mock()
            mock_img.size = (1000, 800)
            mock_img.mode = 'RGBA'  # Image with alpha channel
            mock_img.format = 'PNG'
            mock_img.resize.return_value = mock_img
            mock_img.convert.return_value = mock_img
            mock_img.save.side_effect = lambda buffer, format, quality: buffer.write(b'fake_image_data')

            mock_open.return_value.__enter__.return_value = mock_img

            encoded_string, media_type = connector._resize_and_encode_image(mock_image_path)

            assert isinstance(encoded_string, str)
            assert media_type == "image/png"
            mock_img.convert.assert_called_with('RGB')  # Should convert RGBA to RGB

    def test_resize_and_encode_image_large(self, connector, mock_image_path):
        """Test resizing large images - covers part of lines 351-370"""
        with patch('PIL.Image.open') as mock_open:
            mock_img = Mock()
            mock_img.size = (3000, 2000)  # Large image
            mock_img.mode = 'RGB'
            mock_img.format = 'JPEG'
            mock_img.resize.return_value = mock_img
            mock_img.save.side_effect = lambda buffer, format, quality: buffer.write(b'fake_image_data')

            mock_open.return_value.__enter__.return_value = mock_img

            encoded_string, media_type = connector._resize_and_encode_image(mock_image_path)

            assert isinstance(encoded_string, str)
            # Should resize the image
            mock_img.resize.assert_called_once_with((1568, 1045), ANY)

    def test_resize_and_encode_image_error_fallback(self, connector, mock_image_path):
        """Test fallback encoding when resize fails - covers error handling in lines 351-370"""
        with patch('PIL.Image.open') as mock_open, patch('builtins.open') as mock_file_open:
            # Make PIL.Image.open fail
            mock_open.side_effect = Exception("Image processing error")

            # Set up the fallback file read
            mock_file = Mock()
            mock_file.read.return_value = b'fake_image_data'
            mock_file_open.return_value.__enter__.return_value = mock_file

            # Set up base64 encoding
            with patch('base64.b64encode') as mock_b64encode:
                mock_b64encode.return_value = b'base64_encoded_string'

                # This should use the fallback method
                encoded_string, media_type = connector._resize_and_encode_image(mock_image_path)

                assert encoded_string == 'base64_encoded_string'
                # Media type should be based on the mock_img.format
                assert "image/" in media_type

    def test_generate_response_with_usage_extraction(self, connector):
        """Test token usage extraction - covers line 233"""
        # Mock the response with usage information
        mock_response = {
            'body': Mock(),
        }
        mock_response['body'].read.return_value = json.dumps({
            'content': [{'type': 'text', 'text': 'Test response'}],
            'usage': {'input_tokens': 25, 'output_tokens': 15}
        })

        connector.bedrock_runtime.invoke_model.return_value = mock_response

        # Mock circuit breaker
        def mock_execute(func, *args, **kwargs):
            return func(*args, **kwargs)

        connector.circuit_breaker.execute = mock_execute

        result = connector.generate_response("Test prompt")

        assert result == "Test response"
        # Should extract token usage from response
        connector.metrics_manager.record_llm_call.assert_called_once_with(
            call_type="default",
            input_tokens=25,  # From response
            output_tokens=15,  # From response
            latency=ANY,
            model_name=connector.model_id,
            guardrail_triggered=False
        )

    def test_generate_response_without_usage(self, connector):
        """Test fallback token counting - covers line 233"""
        # Mock response without usage information
        mock_response = {
            'body': Mock(),
        }
        mock_response['body'].read.return_value = json.dumps({
            'content': [{'type': 'text', 'text': 'Test response'}]
            # No usage field
        })

        connector.bedrock_runtime.invoke_model.return_value = mock_response

        # Mock circuit breaker
        def mock_execute(func, *args, **kwargs):
            return func(*args, **kwargs)

        connector.circuit_breaker.execute = mock_execute

        # Mock token counting
        connector.count_tokens = Mock(return_value=10)

        result = connector.generate_response("Test prompt")

        assert result == "Test response"
        # Should fall back to counting tokens
        connector.metrics_manager.record_llm_call.assert_called_once()
        connector.count_tokens.assert_called()

    def test_generate_response_with_timestamp(self, connector):
        """Test timestamp addition - covers lines 138-142"""
        # Mock the response
        mock_response = {
            'body': Mock(),
        }
        mock_response['body'].read.return_value = json.dumps({
            'content': [{'type': 'text', 'text': 'Test response'}]
        })

        connector.bedrock_runtime.invoke_model.return_value = mock_response

        # Capture the actual request body
        def mock_execute(func, *args, **kwargs):
            # Store the request body for inspection
            self.request_body = kwargs.get('body', '')
            return func(*args, **kwargs)

        connector.circuit_breaker.execute = mock_execute

        result = connector.generate_response("Test prompt")

        assert result == "Test response"
        # Check that timestamp was added to the prompt
        assert "Timestamp:" in self.request_body

    def test_generate_response_with_circuit_breaker(self, connector):
        """Test circuit breaker integration - covers lines 102-103"""
        # Create a real circuit breaker
        from src.aws_layer.circuit_breaker import CircuitBreaker
        connector.circuit_breaker = CircuitBreaker(name="test", failure_threshold=2, reset_timeout=0.1)

        # Mock bedrock to fail
        connector.bedrock_runtime.invoke_model.side_effect = Exception("API Error")

        # First call - should fail but not trip breaker
        result = connector.generate_response("Test prompt")
        assert "Error:" in result
        assert connector.circuit_breaker.state == "CLOSED"

        # Second call - should trip breaker
        result = connector.generate_response("Test prompt")
        assert "Error:" in result
        assert connector.circuit_breaker.state == "OPEN"

        # Third call - should fail fast with circuit breaker error
        result = connector.generate_response("Test prompt")
        assert "Error:" in result
        assert "Circuit breaker" in result