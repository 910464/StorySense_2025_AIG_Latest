import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
import json
import numpy as np
import os

from src.aws_layer.aws_titan_embedding import AWSTitanEmbeddings


class TestAWSTitanEmbeddings:
    @pytest.fixture
    def embeddings(self, metrics_manager_mock):
        with patch('boto3.Session') as mock_session:
            mock_client = Mock()
            mock_session.return_value.client.return_value = mock_client
            embeddings = AWSTitanEmbeddings(model_id="amazon.titan-embed-text-v1")
            embeddings.bedrock_runtime = mock_client
            embeddings.cache = Mock()
            return embeddings

    def test_initialization(self, embeddings):
        assert embeddings is not None
        assert embeddings.model_name == "amazon.titan-embed-text-v1"

    @patch('boto3.Session')
    def test_initialization_with_credentials(self, mock_session):
        with patch.dict('os.environ', {
            'AWS_ACCESS_KEY_ID': 'test_key',
            'AWS_SECRET_ACCESS_KEY': 'test_secret',
            'AWS_REGION': 'us-east-1'
        }):
            embeddings = AWSTitanEmbeddings()
            mock_session.assert_called_with(
                aws_access_key_id='test_key',
                aws_secret_access_key='test_secret',
                region_name='us-east-1'
            )

    @patch('boto3.Session')
    def test_initialization_with_session_token(self, mock_session):
        with patch.dict('os.environ', {
            'AWS_ACCESS_KEY_ID': 'test_key',
            'AWS_SECRET_ACCESS_KEY': 'test_secret',
            'AWS_SESSION_TOKEN': 'test_token',
            'AWS_REGION': 'us-east-1'
        }):
            embeddings = AWSTitanEmbeddings()
            mock_session.assert_called_with(
                aws_access_key_id='test_key',
                aws_secret_access_key='test_secret',
                aws_session_token='test_token',
                region_name='us-east-1'
            )

    @patch('boto3.Session')
    def test_initialization_error(self, mock_session):
        mock_session.side_effect = Exception("AWS Error")

        # Should create a fallback session
        with patch('boto3.client') as mock_client:
            embeddings = AWSTitanEmbeddings()
            mock_client.assert_called_with('bedrock-runtime', region_name='us-east-1')

    def test_embed_query(self, embeddings):
        # Mock the cache
        embeddings.cache.get.return_value = None

        # Mock the _embed_text method
        embeddings._embed_text = Mock(return_value=[0.1] * 1536)

        # Mock the _store_locally method
        embeddings._store_locally = Mock()

        result = embeddings.embed_query("Test query")

        assert len(result) == 1536
        assert isinstance(result, list)
        embeddings._embed_text.assert_called_once_with("Test query")
        embeddings.cache.put.assert_called_once()
        embeddings._store_locally.assert_called_once()

    def test_embed_query_with_cache(self, embeddings):
        # Mock cache hit
        cached_embedding = [0.2] * 1536
        embeddings.cache.get.return_value = cached_embedding

        # Mock the _embed_text method to verify it's not called
        embeddings._embed_text = Mock()

        result = embeddings.embed_query("Test query")

        assert result == cached_embedding
        embeddings._embed_text.assert_not_called()

    def test_embed_documents(self, embeddings):
        # Mock the batch_embed_texts method
        embeddings.batch_embed_texts = Mock(return_value=[[0.1] * 1536, [0.2] * 1536])

        result = embeddings.embed_documents(["Doc 1", "Doc 2"])

        assert len(result) == 2
        assert len(result[0]) == 1536
        embeddings.batch_embed_texts.assert_called_once_with(["Doc 1", "Doc 2"])

    def test_batch_embed_texts(self, embeddings):
        # Mock cache behavior
        embeddings.cache.get.side_effect = [None, [0.2] * 1536, None]

        # Mock _embed_text
        embeddings._embed_text = Mock(side_effect=[[0.1] * 1536, [0.3] * 1536])

        result = embeddings.batch_embed_texts(["Doc 1", "Doc 2", "Doc 3"], batch_size=2)

        assert len(result) == 3
        assert result[1] == [0.2] * 1536  # From cache
        assert embeddings._embed_text.call_count == 2  # Called for Doc 1 and Doc 3

    def test_batch_embed_texts_with_errors(self, embeddings):
        # Test batch processing with some errors
        embeddings.cache.get.side_effect = [None, None, None]

        # First call succeeds, second fails, third succeeds
        embeddings._embed_text = Mock(side_effect=[
            [0.1] * 1536,
            Exception("Embedding error"),
            [0.3] * 1536
        ])

        # Should still return three embeddings, with zeros for the failed one
        result = embeddings.batch_embed_texts(["Doc 1", "Doc 2", "Doc 3"])

        assert len(result) == 3
        assert all(x == 0.1 for x in result[0][:10])  # First embedding
        assert result[1] is None  # Failed embedding should be None
        assert all(x == 0.3 for x in result[2][:10])  # Third embedding

    def test_embed_text(self, embeddings):
        # Mock the AWS response
        mock_response = {
            'body': Mock()
        }
        mock_response['body'].read.return_value = json.dumps({
            'embedding': [0.1] * 1536
        })

        embeddings.bedrock_runtime.invoke_model.return_value = mock_response

        result = embeddings._embed_text("Test text")

        assert len(result) == 1536
        embeddings.bedrock_runtime.invoke_model.assert_called_once()

    def test_embed_text_with_retry(self, embeddings):
        # Test retry logic
        embeddings.bedrock_runtime.invoke_model.side_effect = [
            Exception("API Error"),  # First call fails
            Mock(body=Mock(read=lambda: json.dumps({'embedding': [0.1] * 1536})))  # Second call succeeds
        ]

        result = embeddings._embed_text("Test text")

        assert len(result) == 1536
        assert embeddings.bedrock_runtime.invoke_model.call_count == 2

    def test_embed_text_with_fallback(self, embeddings):
        # Test fallback when AWS fails
        embeddings.bedrock_runtime.invoke_model.side_effect = [Exception("API Error")] * 3

        # Mock sentence-transformers import
        with patch('importlib.util.find_spec', return_value=True), \
                patch('sentence_transformers.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.encode.return_value = np.array([0.1] * 384)  # Smaller dimension
            mock_st.return_value = mock_model

            result = embeddings._embed_text("Test text")

            assert len(result) == 1536  # Should be padded to match Titan's dimensions
            assert mock_model.encode.call_count == 1

    def test_embed_text_with_fallback_larger_dim(self, embeddings):
        # Test fallback with larger dimension that needs truncation
        embeddings.bedrock_runtime.invoke_model.side_effect = [Exception("API Error")] * 3

        # Mock sentence-transformers import
        with patch('importlib.util.find_spec', return_value=True), \
                patch('sentence_transformers.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.encode.return_value = np.array([0.1] * 2048)  # Larger dimension
            mock_st.return_value = mock_model

            result = embeddings._embed_text("Test text")

            assert len(result) == 1536  # Should be truncated to match Titan's dimensions
            assert mock_model.encode.call_count == 1

    def test_embed_text_all_failures(self, embeddings):
        # Test when both AWS and fallback fail
        embeddings.bedrock_runtime.invoke_model.side_effect = [Exception("API Error")] * 3

        with patch('importlib.util.find_spec', return_value=False):
            result = embeddings._embed_text("Test text")

            assert len(result) == 1536
            assert all(x == 0.0 for x in result)  # Should return zero vector

    def test_store_locally(self, embeddings, temp_file_path):
        # Test successful local storage
        embeddings.local_storage_path = os.path.dirname(temp_file_path)

        with patch('builtins.open', mock_open()) as mock_file:
            embeddings._store_locally("Test text", [0.1] * 1536, "test_identifier")
            mock_file.assert_called_once_with(f"{embeddings.local_storage_path}/test_identifier.json", 'w')
            mock_file().write.assert_called_once()

    def test_store_locally_error(self, embeddings):
        # Test error handling in local storage
        embeddings.local_storage_path = "/nonexistent/path"

        with patch('builtins.open', side_effect=Exception("File error")):
            # Should not raise exception
            embeddings._store_locally("Test text", [0.1] * 1536, "test_identifier")

    def test_cache_statistics(self, embeddings):
        # Test cache statistics
        embeddings.cache.stats.return_value = {
            "size": 10,
            "max_size": 1000,
            "hits": 5,
            "misses": 3,
            "hit_ratio": 0.625
        }

        stats = embeddings.cache.stats()
        assert stats["hit_ratio"] == 0.625
        assert stats["size"] == 10