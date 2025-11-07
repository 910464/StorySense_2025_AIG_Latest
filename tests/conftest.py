import pytest
from unittest.mock import Mock, patch
import os
import json
import pandas as pd


@pytest.fixture
def metrics_manager_mock():
    """Mock metrics manager for testing"""
    mock = Mock()
    mock.record_llm_call = Mock()
    mock.record_error = Mock()
    mock.record_vector_operation = Mock()
    mock.record_user_story_metrics = Mock()
    mock.get_metrics_summary.return_value = {
        'llm_calls': 10,
        'llm_tokens': 1000,
        'total_duration': 5.0,
        'story_count': 2,
        'vector_queries': 5,
        'peak_memory_mb': 150.5,
        'total_estimated_cost': 0.25,
        'total_estimated_cost_inr': 20.75,
        'llm_input_tokens': 500,
        'llm_output_tokens': 500,
        'llm_avg_latency': 2.5,
        'error_count': 0,
        'vector_db': {
            'query_operations': 5,
            'store_operations': 2,
            'total_vectors_stored': 100,
            'avg_query_time': 0.5,
            'avg_store_time': 1.2
        },
        'system': {
            'cpu_percent': 25.0,
            'peak_memory': 150.5
        },
        'cost_breakdown': {
            'llm': {
                'input_cost': 0.05,
                'output_cost': 0.20,
                'input_cost_inr': 4.15,
                'output_cost_inr': 16.60
            }
        },
        'batches': {}
    }
    return mock


@pytest.fixture
def aws_session_mock():
    """Mock AWS session for testing"""
    with patch('boto3.Session') as mock_session:
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client
        yield mock_session, mock_client


@pytest.fixture
def sample_user_stories():
    """Sample user stories dataframe for testing"""
    return pd.DataFrame({
        'ID': ['US001', 'US002'],
        'Description': [
            'As a user, I want to login so that I can access my account',
            'As an admin, I want to view all users so that I can manage them'
        ],
        'AcceptanceCriteria': [
            '1. User can enter credentials\n2. System validates credentials',
            '1. Admin can see user list\n2. Admin can filter users'
        ]
    })


@pytest.fixture
def sample_context():
    """Sample context dataframe for testing"""
    return pd.DataFrame({
        'text': [
            'Login should use secure authentication',
            'User management requires admin privileges'
        ]
    })


@pytest.fixture
def temp_file_path():
    """Create a temporary file for testing"""
    import tempfile

    temp = tempfile.NamedTemporaryFile(delete=False)
    yield temp.name

    # Cleanup
    if os.path.exists(temp.name):
        os.unlink(temp.name)


@pytest.fixture
def mock_image_path():
    """Create a mock image path for testing"""
    import tempfile

    temp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    temp.write(b'fake image data')
    temp.close()

    yield temp.name

    # Cleanup
    if os.path.exists(temp.name):
        os.unlink(temp.name)


@pytest.fixture
def mock_db_connector():
    """Mock database connector for testing"""
    mock = Mock()
    mock.cursor = Mock()
    mock.conn = Mock()
    mock.reconnect_if_needed = Mock()
    mock.diagnose_database = Mock(return_value=True)
    mock.close = Mock()
    return mock


@pytest.fixture
def mock_context_results():
    """Mock context search results for testing"""
    return {
        'business_rules': {
            'context': 'Business rule context',
            'documents': {0.85: 'Document 1', 0.75: 'Document 2'},
            'metadata': {0.85: {'source_file': 'rules.pdf', 'file_type': 'pdf'},
                         0.75: {'source_file': 'policy.docx', 'file_type': 'docx'}},
            'file_types': {0.85: {'type': 'pdf', 'source': 'rules.pdf'},
                           0.75: {'type': 'docx', 'source': 'policy.docx'}},
            'threshold': 0.7,
            'document_count': 2
        },
        'requirements': {
            'context': 'Requirements context',
            'documents': {0.9: 'Document 3'},
            'metadata': {0.9: {'source_file': 'reqs.xlsx', 'file_type': 'xlsx'}},
            'file_types': {0.9: {'type': 'xlsx', 'source': 'reqs.xlsx'}},
            'threshold': 0.7,
            'document_count': 1
        },
        'file_type_stats': {
            'pdf': 1,
            'docx': 1,
            'xlsx': 1
        }
    }