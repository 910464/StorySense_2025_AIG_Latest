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
    def connector(self, global_global_metrics_manager_mock):
        with patch('boto3.Session') as mock_session:
            mock_client = Mock()
            mock_session.return_value.client.return_value = mock_client
            connector = AWSBedrockConnector(model_id="anthropic.claude-3-sonnet-20240229-v1:0",
                                            metrics_manager=global_global_metrics_manager_mock)
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
        mock_body = Mock()
        mock_body.read.return_value = json.dumps({
            'content': [{'type': 'text', 'text': 'Test response'}],
            'guardrailAction': 'BLOCKED',
            'usage': {'input_tokens': 10, 'output_tokens': 5}
        })
        mock_response = {'body': mock_body}

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

        # Mock environment variables and boto3.client for different region
        with patch.dict('os.environ', {
            'guardrail_id': 'test-guardrail',
            'region': 'us-west-2'  # Different from default
        }), patch('boto3.client') as mock_boto_client:
            # Mock the response
            mock_body = Mock()
            mock_body.read.return_value = json.dumps({
                'content': [{'type': 'text', 'text': 'Test response'}],
                'guardrailAction': 'NONE'
            })
            mock_response = {'body': mock_body}

            # Mock the different region boto3 client
            mock_regional_client = Mock()
            mock_regional_client.invoke_model.return_value = mock_response
            mock_boto_client.return_value = mock_regional_client

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
            assert media_type == "image/jpeg"
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
                # Assuming mock_image_path has .jpg extension for this test
                encoded_string, media_type = connector._resize_and_encode_image(mock_image_path)

                assert encoded_string == 'base64_encoded_string'
                # Media type should be based on file extension
                assert "image/" in media_type

    def test_generate_response_with_usage_extraction(self, connector):
        """Test token usage extraction - covers line 233"""
        # Mock the response with usage information
        mock_body = Mock()
        mock_body.read.return_value = json.dumps({
            'content': [{'type': 'text', 'text': 'Test response'}],
            'usage': {'input_tokens': 25, 'output_tokens': 15}
        })
        mock_response = {'body': mock_body}

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
        mock_body = Mock()
        mock_body.read.return_value = json.dumps({
            'content': [{'type': 'text', 'text': 'Test response'}]
            # No usage field
        })
        mock_response = {'body': mock_body}

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
        mock_body = Mock()
        mock_body.read.return_value = json.dumps({
            'content': [{'type': 'text', 'text': 'Test response'}]
        })
        mock_response = {'body': mock_body}

        # Capture the invoke_model call
        request_body_capture = []
        def mock_invoke_model(*args, **kwargs):
            request_body_capture.append(kwargs.get('body', ''))
            return mock_response
        
        connector.bedrock_runtime.invoke_model = mock_invoke_model

        # Mock circuit breaker
        def mock_execute(func, *args, **kwargs):
            return func(*args, **kwargs)

        connector.circuit_breaker.execute = mock_execute

        result = connector.generate_response("Test prompt")

        assert result == "Test response"
        # Check that timestamp was added to the prompt
        assert len(request_body_capture) > 0
        captured_body = request_body_capture[0]
        assert "Timestamp:" in captured_body

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

    @patch('boto3.Session')
    def test_model_validation_success(self, mock_session, global_global_metrics_manager_mock):
        """Test successful model validation - covers lines 88-91, 93"""
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client
        
        # Mock successful model listing with our model present
        mock_client.list_foundation_models.return_value = {
            'modelSummaries': [
                {'modelId': 'anthropic.claude-3-sonnet-20240229-v1:0'},
                {'modelId': 'other-model'}
            ]
        }
        
        with patch('logging.info') as mock_log:
            connector = AWSBedrockConnector(
                model_id="anthropic.claude-3-sonnet-20240229-v1:0",
                metrics_manager=global_global_metrics_manager_mock
            )
            # Should log that model was found
            mock_log.assert_any_call("Found requested model: anthropic.claude-3-sonnet-20240229-v1:0")

    @patch('boto3.Session')
    def test_model_validation_not_found(self, mock_session, global_global_metrics_manager_mock):
        """Test model not found warning - covers line 97"""
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client
        
        # Mock model listing without our model
        mock_client.list_foundation_models.return_value = {
            'modelSummaries': [
                {'modelId': 'other-model-1'},
                {'modelId': 'other-model-2'}
            ]
        }
        
        with patch('logging.warning') as mock_warning, patch('logging.info') as mock_info:
            connector = AWSBedrockConnector(
                model_id="anthropic.claude-3-sonnet-20240229-v1:0",
                metrics_manager=global_global_metrics_manager_mock
            )
            # Should warn that model was not found
            mock_warning.assert_any_call("Model anthropic.claude-3-sonnet-20240229-v1:0 not found in available models!")
            # Should list available models
            mock_info.assert_any_call("Available models:")

    def test_resize_image_height_greater_than_width(self, connector, mock_image_path):
        """Test image resizing when height > width - covers lines 119-120"""
        with patch('PIL.Image.open') as mock_open:
            mock_img = Mock()
            # Make image larger than max_long_side (1568) to trigger resize
            mock_img.size = (1000, 2000)  # Height > Width and max height > 1568
            mock_img.mode = 'RGB'
            mock_img.format = 'JPEG'
            
            # Create resized image mock
            resized_mock = Mock()
            resized_mock.save.side_effect = lambda buffer, format, quality: buffer.write(b'fake_image_data')
            resized_mock.mode = 'RGB'
            mock_img.resize.return_value = resized_mock

            mock_open.return_value.__enter__.return_value = mock_img

            encoded_string, media_type = connector._resize_and_encode_image(mock_image_path)

            assert isinstance(encoded_string, str)
            # Should resize based on height being the long side (max(1000, 2000) > 1568 triggers resize)
            expected_new_height = 1568
            expected_new_width = int(1000 * (1568 / 2000))
            mock_img.resize.assert_called_once_with((expected_new_width, expected_new_height), ANY)

    def test_count_tokens_with_tokenizer(self, connector):
        """Test token counting with tokenizer - covers line 244"""
        # Set up connector with tokenizer
        connector.has_tokenizer = True
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = ['token1', 'token2', 'token3']
        connector.tokenizer = mock_tokenizer

        result = connector.count_tokens("Test text")
        
        assert result == 3
        mock_tokenizer.encode.assert_called_once_with("Test text")

    def test_generate_response_with_env_defaults(self, connector):
        """Test environment variable defaults - covers lines 269-272, 272-276"""
        with patch.dict('os.environ', {
            'LLM_TEMPERATURE': '0.8',
            'LLM_MAX_TOKENS': '1000'
        }):
            # Mock the response
            mock_body = Mock()
            mock_body.read.return_value = json.dumps({
                'content': [{'type': 'text', 'text': 'Test response'}]
            })
            mock_response = {'body': mock_body}

            # Capture the invoke_model call to check parameters
            request_body_capture = []
            def mock_invoke_model(*args, **kwargs):
                request_body_capture.append(json.loads(kwargs.get('body', '{}')))
                return mock_response
            
            connector.bedrock_runtime.invoke_model = mock_invoke_model

            # Mock circuit breaker
            def mock_execute(func, *args, **kwargs):
                return func(*args, **kwargs)

            connector.circuit_breaker.execute = mock_execute

            # Call without specifying temperature or max_tokens
            result = connector.generate_response("Test prompt")

            assert result == "Test response"
            # Check that environment values were used
            assert len(request_body_capture) > 0
            captured_body = request_body_capture[0]
            assert captured_body['max_tokens'] == 1000

    def test_generate_response_fallback_formats(self, connector):
        """Test response format fallbacks - covers lines 344-352"""
        test_cases = [
            # Test completion format fallback
            ({'completion': 'Test completion response'}, 'Test completion response'),
            # Test text format fallback
            ({'text': 'Test text response'}, 'Test text response'),
            # Test unexpected format fallback
            ({'unexpected': 'format'}, '{"unexpected": "format"}')
        ]

        for response_data, expected_result in test_cases:
            # Mock the response
            mock_body = Mock()
            mock_body.read.return_value = json.dumps(response_data)
            mock_response = {'body': mock_body}

            connector.bedrock_runtime.invoke_model.return_value = mock_response

            # Mock circuit breaker
            def mock_execute(func, *args, **kwargs):
                return func(*args, **kwargs)

            connector.circuit_breaker.execute = mock_execute

            with patch('logging.warning') as mock_warning:
                result = connector.generate_response("Test prompt")
                
                if isinstance(response_data, dict) and 'unexpected' in response_data:
                    # Should log warning for unexpected format
                    mock_warning.assert_called()
                    assert expected_result in result
                else:
                    assert result == expected_result

        # Test direct string response as JSON (proper format)
        mock_body = Mock()
        mock_body.read.return_value = json.dumps("Direct string response")
        mock_response = {'body': mock_body}
        
        connector.bedrock_runtime.invoke_model.return_value = mock_response
        
        # Mock circuit breaker
        def mock_execute(func, *args, **kwargs):
            return func(*args, **kwargs)

        connector.circuit_breaker.execute = mock_execute
        
        result = connector.generate_response("Test prompt")
        assert result == "Direct string response"

    def test_generate_response_empty_result_logging(self, connector):
        """Test empty result logging - covers line 357"""
        # Mock response with content that results in empty string
        mock_body = Mock()
        mock_body.read.return_value = json.dumps({
            'content': [{'type': 'text', 'text': ''}]  # Empty text content
        })
        mock_response = {'body': mock_body}

        connector.bedrock_runtime.invoke_model.return_value = mock_response

        # Mock circuit breaker
        def mock_execute(func, *args, **kwargs):
            return func(*args, **kwargs)

        connector.circuit_breaker.execute = mock_execute

        with patch('logging.error') as mock_error:
            result = connector.generate_response("Test prompt")
            
            # Should log error for empty result
            mock_error.assert_called()
            assert len(result) == 0

    def test_truncate_prompt(self, connector):
        """Test prompt truncation - covers lines 413-414"""
        # Test normal prompt (no truncation)
        normal_prompt = "This is a normal prompt"
        result = connector._truncate_prompt(normal_prompt)
        assert result == normal_prompt

        # Test long prompt (should be truncated)
        long_prompt = "x" * 10001  # Longer than default max_length
        with patch('logging.warning') as mock_warning:
            result = connector._truncate_prompt(long_prompt, max_length=1000)
            
            mock_warning.assert_called_with("Prompt is too long (10001 chars). Truncating to 1000 chars.")
            assert len(result) == 1020  # 1000 + len("\n\n[PROMPT TRUNCATED]")
            assert result.endswith("\n\n[PROMPT TRUNCATED]")

    def test_multimodal_generate_response_success(self, connector, mock_image_path):
        """Test successful multimodal response generation - covers lines 166-221"""
        # Mock image processing
        with patch.object(connector, '_resize_and_encode_image', return_value=('base64data', 'image/jpeg')):
            # Mock the response
            mock_body = Mock()
            mock_body.read.return_value = json.dumps({
                'content': [{'type': 'text', 'text': 'Multimodal response'}],
                'usage': {'input_tokens': 30, 'output_tokens': 20}
            })
            mock_response = {'body': mock_body}

            connector.bedrock_runtime.invoke_model.return_value = mock_response

            # Mock circuit breaker
            def mock_execute(func, *args, **kwargs):
                return func(*args, **kwargs)

            connector.circuit_breaker.execute = mock_execute

            result = connector.generate_response_multimodal(
                "Describe this image", 
                mock_image_path, 
                temperature=0.7, 
                max_tokens=500, 
                call_type="multimodal"
            )

            assert result == "Multimodal response"
            # Check that record_llm_call was called (guardrail_triggered might not be in the call for multimodal)
            connector.metrics_manager.record_llm_call.assert_called_once()
            call_args = connector.metrics_manager.record_llm_call.call_args
            assert call_args[1]['call_type'] == 'multimodal'
            assert call_args[1]['input_tokens'] == 30
            assert call_args[1]['output_tokens'] == 20
            assert call_args[1]['model_name'] == connector.model_id
