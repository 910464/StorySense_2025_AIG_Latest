import boto3
import json
import os
import logging
import time
import asyncio
import aiohttp
from datetime import datetime
import traceback
import base64
from PIL import Image
import io
from src.context_handler.context_storage_handler.pgvector_connector import PGVectorConnector
# from src.configuration_handler.env_manager import EnvManager
from src.metrics.metrics_manager import MetricsManager
from src.aws_layer.circuit_breaker import CircuitBreaker

class AWSBedrockConnector:
    def __init__(self, model_id=None, metrics_manager=None):
        """
        Initialize AWS Bedrock connector

        Args:
            model_id (str): AWS Bedrock model ID
            metrics_manager: Optional metrics manager instance
        """
        # Initialize environment manager
        # self.env_manager = EnvManager()

        # Initialize or use provided metrics manager
        self.metrics_manager = metrics_manager or MetricsManager()

        # Initialize circuit breaker
        self.circuit_breaker = CircuitBreaker(name="bedrock", failure_threshold=3, reset_timeout=30)

        # Get model ID from environment or parameter
        self.model_id = os.getenv('LLM_MODEL_ID') or os.getenv('MODEL_ID', 'anthropic.claude-3-sonnet-20240229-v1:0')
        
        # Initialize guardrails configuration
        self.use_guardrails = False  # Default to False since guardrails aren't always needed

        # Initialize AWS session with region from environment
        region_name = os.getenv('AWS_REGION', 'us-east-1')

        # Create a more robust AWS session
        try:
            # Get credentials from environment variables
            aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
            aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
            aws_session_token = os.getenv('AWS_SESSION_TOKEN')

            # Log credential information for debugging
            logging.info(f"AWS credentials found: {'Yes' if aws_access_key and aws_secret_key else 'No'}")
            logging.info(f"AWS session token found: {'Yes' if aws_session_token else 'No'}")

            # Create session with explicit credentials if available
            if aws_access_key and aws_secret_key:
                session_kwargs = {
                    'aws_access_key_id': aws_access_key,
                    'aws_secret_access_key': aws_secret_key,
                    'region_name': region_name
                }

                if aws_session_token:
                    session_kwargs['aws_session_token'] = aws_session_token
                    logging.info("Including AWS session token in credentials")

                self.session = boto3.Session(**session_kwargs)
                logging.info("Created AWS session with explicit credentials")
            else:
                # Try to use default credentials (AWS CLI profile, EC2 instance profile, etc.)
                self.session = boto3.Session(region_name=region_name)
                logging.info("Created AWS session with default credential provider chain")

            # Create bedrock runtime client
            self.bedrock_runtime = self.session.client('bedrock-runtime')

            # Test if we can access the bedrock service
            try:
                bedrock = self.session.client('bedrock')
                models = bedrock.list_foundation_models()
                logging.info(
                    f"Successfully connected to AWS Bedrock. Found {len(models.get('modelSummaries', []))} models.")

                # Check if our model is available
                model_found = False
                for model in models.get('modelSummaries', []):
                    if model.get('modelId') == self.model_id:
                        model_found = True
                        logging.info(f"Found requested model: {self.model_id}")
                        break

                if not model_found:
                    logging.warning(f"Model {self.model_id} not found in available models!")
                    logging.info("Available models:")
                    for model in models.get('modelSummaries', []):
                        logging.info(f"- {model.get('modelId')}")
            except Exception as e:
                logging.warning(f"Could not list foundation models: {e}")

        except Exception as e:
            logging.error(f"Error creating AWS session: {e}")

    def _resize_and_encode_image(self, image_path, max_long_side=1568, quality=85):
        """
        Resizes an image to fit within a max dimension and encodes it to base64.
        """
        try:
            with Image.open(image_path) as img:
                # Get original dimensions
                width, height = img.size

                # Determine the scaling factor
                if max(width, height) > max_long_side:
                    if width > height:
                        new_width = max_long_side
                        new_height = int(height * (max_long_side / width))
                    else:
                        new_height = max_long_side
                        new_width = int(width * (max_long_side / height))
                    
                    # Resize the image
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                # Convert to RGB if it has an alpha channel (e.g., RGBA, P)
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')

                # Save the (potentially resized) image to an in-memory buffer
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=quality)
                
                # Encode to base64
                encoded_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
                return encoded_string, "image/jpeg"

        except Exception as e:
            logging.error(f"Error resizing and encoding image {image_path}: {e}")
            # Fallback to original encoding if resizing fails
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8'), f"image/{img.format.lower()}"

    def generate_response_multimodal(self, prompt, image_path, temperature, max_tokens, call_type):
        """
        Generate a response from a multimodal model on AWS Bedrock.
        If image_path is None, it will fall back to the text-only streaming generation.
        """
        if not image_path:
            # If no image is provided, use the text-based generation
            logging.info("No image provided, falling back to text generation.")
            # Use the regular generate_response method for text-only requests
            return self.generate_response(prompt, temperature, max_tokens, call_type)

        start_time = time.time()
        try:
            # Truncate prompt before sending
            prompt = self._truncate_prompt(prompt)
            
            image_data, media_type = self._resize_and_encode_image(image_path)

            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_data,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt,
                            }
                        ],
                    }
                ],
            })

            # Use circuit breaker to execute the model invocation
            def invoke_model_with_breaker():
                response = self.bedrock_runtime.invoke_model(
                    body=body,
                    modelId=self.model_id,
                    accept='application/json',
                    contentType='application/json'
                )
                return json.loads(response.get('body').read())

            response_body = self.circuit_breaker.execute(invoke_model_with_breaker)
            
            output_text = response_body['content'][0]['text']
            
            input_tokens = response_body.get('usage', {}).get('input_tokens', 0)
            output_tokens = response_body.get('usage', {}).get('output_tokens', 0)

            # Debug logging for multimodal response
            logging.info(f"Bedrock multimodal response length: {len(output_text)}")
            logging.info(f"Response starts with: {output_text[:100]}...")
            logging.info(f"Response ends with: {output_text[-100:]}")

            # Record metrics
            self.metrics_manager.record_llm_call(
                call_type=call_type,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency=time.time() - start_time,
                model_name=self.model_id
            )
            return output_text

        except Exception as e:
            # Record metrics
            self.metrics_manager.record_llm_call(
                call_type=call_type,
                input_tokens=0, # Or an estimate if possible
                output_tokens=0,
                latency=time.time() - start_time,
                model_name=self.model_id,
                guardrail_triggered=False # Assuming no guardrail for multimodal
            )
            logging.error(f"Error generating multimodal response: {e}")
            logging.error(f"Full error details: {traceback.format_exc()}")
            return f"Error: {e}"

    def count_tokens(self, text):
        """Count tokens in text using tiktoken if available, otherwise estimate"""
        if not text:
            return 0

        # Check if we have a tokenizer (this attribute may not exist)
        if hasattr(self, 'has_tokenizer') and self.has_tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Rough estimation (4 chars ≈ 1 token for most models)
            return len(text) // 4

    def generate_response(self, prompt, temperature=None, max_tokens=None, call_type="default"):
        """
        Generate a response from Claude using AWS Bedrock with circuit breaker protection

        Args:
            prompt (str): The prompt to send to Claude
            temperature (float): Controls randomness (0-1)
            max_tokens (int): Maximum tokens in response
            call_type (str): Type of call (e.g., "analysis", "validation")

        Returns:
            str: The generated response
        """
        start_time = time.time()
        input_tokens = self.count_tokens(prompt)

        try:
            # Use circuit breaker to protect against repeated failures
            def bedrock_call(temperature=None, max_tokens=None):
                # Get temperature and max_tokens from environment if not provided
                if temperature is None:
                    temperature = float(os.getenv('LLM_TEMPERATURE', 0.05))

                if max_tokens is None:
                    max_tokens = int(os.getenv('LLM_MAX_TOKENS', 200000))

                # Add a timestamp to make each prompt unique
                from datetime import datetime
                unique_prompt = prompt + f"\n\nTimestamp: {datetime.now().isoformat()}"

                request_body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": max_tokens,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": unique_prompt
                                }
                            ]
                        }
                    ],
                    "temperature": temperature
                }

                # Apply guardrails if configured
                if hasattr(self, 'use_guardrails') and self.use_guardrails:
                    # guardrail_config = os.get('default', {})
                    guardrail_id = os.getenv('guardrail_id')
                    guardrail_region = os.getenv('region')

                    # if guardrail_id:
                    #     logging.info(f"Applying guardrail {guardrail_id} to request")

                    #     # Add guardrail configuration to the request
                    #     request_body["guardrailConfig"] = {
                    #         "guardrailId": guardrail_id,
                    #         "guardrailVersion": "DRAFT"  # or "LATEST" for the latest published version
                    #     }

                    #     # Create a bedrock client in the guardrail's region if different
                    #     if guardrail_region != self.session.region_name:
                    #         bedrock_runtime = boto3.client('bedrock-runtime', region_name=guardrail_region)
                    #     else:
                    #         bedrock_runtime = self.bedrock_runtime
                    # else:
                    #     bedrock_runtime = self.bedrock_runtime
                else:
                    bedrock_runtime = self.bedrock_runtime

                # Invoke the model with or without guardrails
                response = bedrock_runtime.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(request_body)
                )

                response_body = json.loads(response.get('body').read())

                # Check if guardrail was triggered
                guardrail_triggered = False
                if 'guardrailAction' in response_body:
                    action = response_body['guardrailAction']
                    if action != 'NONE':
                        logging.warning(f"Guardrail triggered with action: {action}")
                        guardrail_triggered = True

                # Extract text from response with better error handling
                result = ""
                if 'content' in response_body and len(response_body['content']) > 0:
                    for content_block in response_body['content']:
                        if content_block.get('type') == 'text':
                            result += content_block.get('text', '')
                else:
                    # Fallback: try to get text from different response formats
                    if 'completion' in response_body:
                        result = response_body['completion']
                    elif 'text' in response_body:
                        result = response_body['text']
                    elif isinstance(response_body, str):
                        result = response_body
                    else:
                        logging.warning(f"Unexpected response format: {list(response_body.keys())}")
                        result = json.dumps(response_body)

                # Log the extracted result for debugging
                logging.info(f"Extracted result length: {len(result)}")
                if len(result) == 0:
                    logging.error(f"Empty result from response_body: {response_body}")

                # Extract token usage from response
                actual_input_tokens = input_tokens  # Use our calculated input tokens
                actual_output_tokens = 0

                if 'usage' in response_body:
                    actual_input_tokens = response_body['usage'].get('input_tokens', input_tokens)
                    actual_output_tokens = response_body['usage'].get('output_tokens', self.count_tokens(result))
                else:
                    # Fallback to counting tokens if usage not provided
                    actual_output_tokens = self.count_tokens(result)

                return result, guardrail_triggered, actual_input_tokens, actual_output_tokens

            # Execute with circuit breaker protection
            result, guardrail_triggered, actual_input_tokens, actual_output_tokens = self.circuit_breaker.execute(
                bedrock_call)

            # Calculate metrics
            end_time = time.time()
            latency = end_time - start_time

            # Record metrics using the metrics manager with actual token counts
            self.metrics_manager.record_llm_call(
                call_type=call_type,
                input_tokens=actual_input_tokens,
                output_tokens=actual_output_tokens,
                latency=latency,
                model_name=self.model_id,
                guardrail_triggered=guardrail_triggered
            )

            # Add debug print
            print(
                f"DEBUG: Recorded LLM call - Input: {actual_input_tokens}, Output: {actual_output_tokens}, "
                f"Total calls: {self.metrics_manager.metrics['llm']['total_calls']}, "
                f"Total tokens: {self.metrics_manager.metrics['llm']['total_tokens']}")

            return result

        except Exception as e:
            end_time = time.time()
            latency = end_time - start_time

            # Record error
            self.metrics_manager.record_error('llm_call_error', str(e))

            logging.error(f"Error generating response from AWS Bedrock: {str(e)}")
            return f"Error: {str(e)}"

    def _truncate_prompt(self, prompt, max_length=200000):
        """
        Truncates the prompt if it's too long.
        """
        if len(prompt) > max_length:
            logging.warning(f"Prompt is too long ({len(prompt)} chars). Truncating to {max_length} chars.")
            return prompt[:max_length] + "\n\n[PROMPT TRUNCATED]"
        return prompt