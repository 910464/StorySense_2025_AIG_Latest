import boto3
import json
import os
import logging
import numpy as np
import time
from datetime import datetime
from langchain.embeddings.base import Embeddings
# from src.configuration_handler.env_manager import EnvManager
from src.configuration_handler.config_loader import load_configuration
from src.embedding_handler.embedding_cache import EmbeddingCache


class AWSTitanEmbeddings(Embeddings):
    """AWS Titan Embeddings wrapper for LangChain"""

    def __init__(self, model_id=None, local_storage_path=None):
        """
        Initialize AWS Titan Embeddings

        Args:
            model_id (str): AWS Bedrock model ID for embeddings
            local_storage_path (str): Path to store embeddings locally
        """
        # Initialize environment manager
        # self.env_manager = EnvManager()

        # Initialize embedding cache
        self.cache = EmbeddingCache(max_size=1000)

        # Get model ID and storage path from environment or parameters
        self.model_id = os.getenv('EMBEDDING_MODEL_NAME', 'amazon.titan-embed-text-v1')
        self.local_storage_path = os.getenv('LOCAL_EMBEDDINGS_PATH','../Data/LocalEmbeddings')

        # Create a more robust AWS session
        try:
            # First try to get credentials from environment variables
            aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
            aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
            aws_session_token = os.getenv('AWS_SESSION_TOKEN')
            region_name = os.getenv('AWS_REGION', 'us-east-1')

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

                session = boto3.Session(**session_kwargs)
                logging.info("Created AWS session with explicit credentials")
            else:
                # Try to use default credentials (AWS CLI profile, EC2 instance profile, etc.)
                session = boto3.Session(region_name=region_name)
                logging.info("Created AWS session with default credential provider chain")

            # Create bedrock runtime client
            self.bedrock_runtime = session.client('bedrock-runtime')

            # Debug AWS credentials
            # self.env_manager.debug_aws_credentials()

        except Exception as e:
            logging.error(f"Error creating AWS session: {e}")
            # Create a fallback session as last resort
            self.bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
            logging.warning("Using fallback AWS session - embeddings may fail")

        os.makedirs(self.local_storage_path, exist_ok=True)
        logging.info(f"Initialized AWS Titan Embeddings with model: {self.model_id}")

    def embed_documents(self, texts):
        """
        Generate embeddings for a list of documents

        Args:
            texts (List[str]): List of texts to embed

        Returns:
            List[List[float]]: List of embeddings
        """
        return self.batch_embed_texts(texts)

    def embed_query(self, text):
        """
        Generate embedding for a query text with caching

        Args:
            text (str): Text to embed

        Returns:
            List[float]: Embedding
        """
        # Check cache first
        cached_embedding = self.cache.get(text)
        if cached_embedding is not None:
            return cached_embedding

        # Generate embedding if not in cache
        embedding = self._embed_text(text)

        # Store in cache
        self.cache.put(text, embedding)

        # Store locally with query prefix
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._store_locally(text, embedding, f"query_{timestamp}")

        return embedding

    def _embed_text(self, text):
        """Internal method to get embedding from AWS Titan with fallback options"""
        max_retries = 3
        retry_count = 0

        # Try primary model (AWS Titan)
        while retry_count < max_retries:
            try:
                request_body = {
                    "inputText": text
                }

                response = self.bedrock_runtime.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(request_body)
                )

                response_body = json.loads(response.get('body').read())
                embedding = response_body.get('embedding')

                if embedding and len(embedding) > 0:
                    return embedding
                else:
                    logging.error("No embedding returned from AWS Titan")
                    break  # Break to try fallback

            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    logging.error(f"Error generating embedding from AWS Titan after {max_retries} attempts: {str(e)}")
                    break  # Break to try fallback

                logging.warning(f"Embedding attempt {retry_count} failed: {str(e)}. Retrying...")
                time.sleep(1)  # Wait before retrying

        # Try fallback model if available
        try:
            # Check if sentence-transformers is installed
            import importlib
            if importlib.util.find_spec("sentence_transformers"):
                logging.info("Using fallback embedding model (sentence-transformers)")
                from sentence_transformers import SentenceTransformer

                # Use a singleton pattern for the model to avoid reloading
                if not hasattr(self, '_fallback_model'):
                    self._fallback_model = SentenceTransformer('all-MiniLM-L6-v2')

                # Generate embedding
                embedding = self._fallback_model.encode(text).tolist()

                # Resize to match Titan's dimensions if needed
                if len(embedding) != 1536:
                    logging.warning(f"Fallback embedding size ({len(embedding)}) doesn't match Titan (1536)")
                    # Pad or truncate to match expected size
                    if len(embedding) < 1536:
                        embedding = embedding + [0.0] * (1536 - len(embedding))
                    else:
                        embedding = embedding[:1536]

                return embedding
        except Exception as e:
            logging.error(f"Fallback embedding model failed: {e}")

        # Last resort: return zero vector
        logging.error("All embedding models failed, returning zero vector")
        return [0.0] * 1536

    def _store_locally(self, text, embedding, identifier):
        """Store embedding locally"""
        try:
            output_file = f"{self.local_storage_path}/{identifier}.json"

            with open(output_file, 'w') as f:
                json.dump({
                    "text": text[:1000] + ("..." if len(text) > 1000 else ""),  # Truncate long texts
                    "embedding": embedding,
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2)

        except Exception as e:
            logging.error(f"Error storing embedding locally: {str(e)}")

    def batch_embed_texts(self, texts, batch_size=20):
        """Process embeddings in batches to optimize API calls"""
        all_embeddings = []

        # Check cache for all texts first
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            cached_embedding = self.cache.get(text)
            if cached_embedding is not None:
                all_embeddings.append(cached_embedding)
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Process uncached texts in batches
        for i in range(0, len(uncached_texts), batch_size):
            batch = uncached_texts[i:i + batch_size]
            batch_embeddings = []

            # Process each text individually (Titan doesn't support batch embedding)
            for text in batch:
                embedding = self._embed_text(text)
                batch_embeddings.append(embedding)
                # Store in cache
                self.cache.put(text, embedding)

            # Insert batch results into the right positions
            for j, embedding in enumerate(batch_embeddings):
                idx = uncached_indices[i + j]
                # Extend all_embeddings list if needed
                while len(all_embeddings) <= idx:
                    all_embeddings.append(None)
                all_embeddings[idx] = embedding

        return all_embeddings