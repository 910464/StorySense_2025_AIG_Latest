# import pytest
# from unittest.mock import Mock, patch
# import os
# import json
# import pandas as pd
#
#
# @pytest.fixture
# def metrics_manager_mock():
#     """Mock metrics manager for testing"""
#     mock = Mock()
#     mock.record_llm_call = Mock()
#     mock.record_error = Mock()
#     mock.record_vector_operation = Mock()
#     mock.record_user_story_metrics = Mock()
#     mock.get_metrics_summary.return_value = {
#         'llm_calls': 10,
#         'llm_tokens': 1000,
#         'total_duration': 5.0,
#         'story_count': 2,
#         'vector_queries': 5,
#         'peak_memory_mb': 150.5,
#         'total_estimated_cost': 0.25,
#         'total_estimated_cost_inr': 20.75,
#         'llm_input_tokens': 500,
#         'llm_output_tokens': 500,
#         'llm_avg_latency': 2.5,
#         'error_count': 0,
#         'vector_db': {
#             'query_operations': 5,
#             'store_operations': 2,
#             'total_vectors_stored': 100,
#             'avg_query_time': 0.5,
#             'avg_store_time': 1.2
#         },
#         'system': {
#             'cpu_percent': 25.0,
#             'peak_memory': 150.5
#         },
#         'cost_breakdown': {
#             'llm': {
#                 'input_cost': 0.05,
#                 'output_cost': 0.20,
#                 'input_cost_inr': 4.15,
#                 'output_cost_inr': 16.60
#             }
#         },
#         'batches': {}
#     }
#     return mock
#
#
# @pytest.fixture
# def aws_session_mock():
#     """Mock AWS session for testing"""
#     with patch('boto3.Session') as mock_session:
#         mock_client = Mock()
#         mock_session.return_value.client.return_value = mock_client
#         yield mock_session, mock_client
#
#
# @pytest.fixture
# def sample_user_stories():
#     """Sample user stories dataframe for testing"""
#     return pd.DataFrame({
#         'ID': ['US001', 'US002'],
#         'Description': [
#             'As a user, I want to login so that I can access my account',
#             'As an admin, I want to view all users so that I can manage them'
#         ],
#         'AcceptanceCriteria': [
#             '1. User can enter credentials\n2. System validates credentials',
#             '1. Admin can see user list\n2. Admin can filter users'
#         ]
#     })
#
#
# @pytest.fixture
# def sample_context():
#     """Sample context dataframe for testing"""
#     return pd.DataFrame({
#         'text': [
#             'Login should use secure authentication',
#             'User management requires admin privileges'
#         ]
#     })
#
#
# @pytest.fixture
# def temp_file_path():
#     """Create a temporary file for testing"""
#     import tempfile
#
#     temp = tempfile.NamedTemporaryFile(delete=False)
#     yield temp.name
#
#     # Cleanup
#     if os.path.exists(temp.name):
#         os.unlink(temp.name)
#
#
# @pytest.fixture
# def mock_image_path():
#     """Create a mock image path for testing"""
#     import tempfile
#
#     temp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
#     temp.write(b'fake image data')
#     temp.close()
#
#     yield temp.name
#
#     # Cleanup
#     if os.path.exists(temp.name):
#         os.unlink(temp.name)
#
#
# @pytest.fixture
# def mock_db_connector():
#     """Mock database connector for testing"""
#     mock = Mock()
#     mock.cursor = Mock()
#     mock.conn = Mock()
#     mock.reconnect_if_needed = Mock()
#     mock.diagnose_database = Mock(return_value=True)
#     mock.close = Mock()
#     return mock
#
#
# @pytest.fixture
# def mock_context_results():
#     """Mock context search results for testing"""
#     return {
#         'business_rules': {
#             'context': 'Business rule context',
#             'documents': {0.85: 'Document 1', 0.75: 'Document 2'},
#             'metadata': {0.85: {'source_file': 'rules.pdf', 'file_type': 'pdf'},
#                          0.75: {'source_file': 'policy.docx', 'file_type': 'docx'}},
#             'file_types': {0.85: {'type': 'pdf', 'source': 'rules.pdf'},
#                            0.75: {'type': 'docx', 'source': 'policy.docx'}},
#             'threshold': 0.7,
#             'document_count': 2
#         },
#         'requirements': {
#             'context': 'Requirements context',
#             'documents': {0.9: 'Document 3'},
#             'metadata': {0.9: {'source_file': 'reqs.xlsx', 'file_type': 'xlsx'}},
#             'file_types': {0.9: {'type': 'xlsx', 'source': 'reqs.xlsx'}},
#             'threshold': 0.7,
#             'document_count': 1
#         },
#         'file_type_stats': {
#             'pdf': 1,
#             'docx': 1,
#             'xlsx': 1
#         }
#     }

import pytest
import sys
from unittest.mock import Mock, patch, mock_open
import os
import json
import tempfile
import pandas as pd
from fastapi import APIRouter


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


# ============================================================================
# NEW FIXTURES FOR MAIN_SERVICE.PY AND MAIN.PY TESTING
# ============================================================================

@pytest.fixture(scope="function")
def mock_main_service_router():
    """Mock the main_service_router module to avoid import errors"""
    mock_router = APIRouter(tags=["StorySense Generator"])

    # Create a mock module
    mock_module = Mock()
    mock_module.story_sense_router = mock_router

    # Add to sys.modules before any imports
    sys.modules['main_service_router'] = mock_module

    yield mock_router

    # Cleanup
    if 'main_service_router' in sys.modules:
        del sys.modules['main_service_router']


@pytest.fixture
def mock_config_loader():
    """Mock configuration loader"""
    with patch('src.configuration_handler.config_loader.load_configuration') as mock:
        mock.return_value = True
        yield mock


@pytest.fixture
def mock_uvicorn():
    """Mock uvicorn.run"""
    with patch('uvicorn.run') as mock:
        yield mock


@pytest.fixture
def mock_fastapi_app():
    """Mock FastAPI app"""
    from fastapi import FastAPI
    app = FastAPI(
        title="Story Sense Analyser",
        description="API for generating metrics report",
        version="1.0"
    )
    return app


@pytest.fixture
def clean_sys_modules():
    """Clean up sys.modules before and after test"""
    modules_to_clean = [
        'src.interface_layer.main',
        'src.interface_layer.main_service',
        'main_service_router'
    ]

    # Clean before test
    for module in modules_to_clean:
        if module in sys.modules:
            del sys.modules[module]

    yield

    # Clean after test
    for module in modules_to_clean:
        if module in sys.modules:
            del sys.modules[module]


@pytest.fixture
def mock_environment_variables():
    """Mock environment variables for testing"""
    env_vars = {
        'LLM_FAMILY': 'AWS',
        'LLM_MODEL_ID': 'anthropic.claude-3-sonnet-20240229-v1:0',
        'LLM_TEMPERATURE': '0.05',
        'LLM_MAX_TOKENS': '100000',
        'AWS_ACCESS_KEY_ID': 'test_key',
        'AWS_SECRET_ACCESS_KEY': 'test_secret',
        'AWS_REGION': 'us-east-1',
        'DB_HOST': 'test-host',
        'DB_PORT': '5432',
        'DB_NAME': 'testdb',
        'DB_USER': 'testuser',
        'DB_PASSWORD': 'testpass',
        'SIMILARITY_METRIC': 'cosine',
        'SIMILARITY_THRESHOLD': '0.7'
    }

    with patch.dict('os.environ', env_vars, clear=False):
        yield env_vars


@pytest.fixture
def mock_load_dotenv():
    """Mock python-dotenv load_dotenv function"""
    with patch('dotenv.load_dotenv') as mock:
        mock.return_value = True
        yield mock


@pytest.fixture(autouse=True)
def setup_test_environment(mock_main_service_router):
    """
    Auto-use fixture to set up test environment for all tests
    This ensures main_service_router is always mocked
    """
    # This fixture runs automatically for all tests
    # It uses mock_main_service_router to ensure the router is mocked
    pass


@pytest.fixture
def fastapi_test_client(mock_main_service_router):
    """Create a FastAPI test client with mocked router"""
    from fastapi.testclient import TestClient
    from fastapi import FastAPI
    from starlette.middleware.cors import CORSMiddleware

    app = FastAPI(
        title="Story Sense Analyser",
        description="API for generating metrics report",
        version="1.0"
    )

    # Set up CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include the mocked router
    app.include_router(mock_main_service_router, prefix="/api")

    return TestClient(app)


@pytest.fixture
def mock_boto3_session():
    """Mock boto3 Session for AWS testing"""
    with patch('boto3.Session') as mock_session:
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client
        yield mock_session


@pytest.fixture
def mock_psycopg2_connect():
    """Mock psycopg2 connection for database testing"""
    with patch('psycopg2.connect') as mock_connect:
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        yield mock_connect, mock_conn, mock_cursor


@pytest.fixture
def temp_config_file():
    """Create a temporary configuration file for testing"""
    import tempfile

    config_content = """
# LLM Configuration
LLM_FAMILY=AWS
LLM_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0
LLM_TEMPERATURE=0.05
LLM_MAX_TOKENS=100000

# AWS Credentials
AWS_ACCESS_KEY_ID=test_key
AWS_SECRET_ACCESS_KEY=test_secret
AWS_REGION=us-east-1

# Database Configuration
DB_HOST=test-host
DB_PORT=5432
DB_NAME=testdb
DB_USER=testuser
DB_PASSWORD=testpass

# Vector Database Configuration
SIMILARITY_METRIC=cosine
SIMILARITY_THRESHOLD=0.7
"""

    temp = tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False)
    temp.write(config_content)
    temp.close()

    yield temp.name

    # Cleanup
    if os.path.exists(temp.name):
        os.unlink(temp.name)


@pytest.fixture
def mock_file_operations():
    """Mock file operations for testing"""
    with patch('builtins.open', create=True) as mock_open, \
            patch('os.path.exists') as mock_exists, \
            patch('os.makedirs') as mock_makedirs:
        mock_exists.return_value = True

        yield {
            'open': mock_open,
            'exists': mock_exists,
            'makedirs': mock_makedirs
        }


@pytest.fixture
def capture_logs():
    """Capture log messages for testing"""
    import logging
    from io import StringIO

    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.DEBUG)

    # Get root logger
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    yield log_capture

    # Cleanup
    logger.removeHandler(handler)


@pytest.fixture
def mock_metrics_manager():
    """Mock MetricsManager for testing"""
    with patch('src.metrics.metrics_manager.MetricsManager') as mock_class:
        mock_instance = Mock()
        mock_instance.record_llm_call = Mock()
        mock_instance.record_error = Mock()
        mock_instance.record_vector_operation = Mock()
        mock_instance.record_user_story_metrics = Mock()
        mock_instance.get_metrics_summary.return_value = {
            'llm_calls': 10,
            'llm_tokens': 1000,
            'total_duration': 5.0
        }
        mock_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_circuit_breaker():
    """Mock CircuitBreaker for testing"""
    with patch('src.aws_layer.circuit_breaker.CircuitBreaker') as mock_class:
        mock_instance = Mock()
        mock_instance.execute = Mock(side_effect=lambda func, *args, **kwargs: func(*args, **kwargs))
        mock_instance.state = "CLOSED"
        mock_instance.failure_count = 0
        mock_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture(scope="function", autouse=True)
def mock_main_service_router_autouse():
    """Auto-mock the main_service_router to avoid import errors"""
    from fastapi import APIRouter

    # Create a mock router
    mock_router = APIRouter(tags=["StorySense Generator"])

    # Create a mock module
    mock_module = Mock()
    mock_module.story_sense_router = mock_router

    # Add to sys.modules
    sys.modules['main_service_router'] = mock_module

    # Also mock src.interface_layer.main if needed
    if 'src.interface_layer.main' not in sys.modules:
        mock_main = Mock()
        mock_main.app = Mock()
        sys.modules['src.interface_layer.main'] = mock_main

    yield mock_router

    # Cleanup is optional since it's function-scoped


@pytest.fixture
def mock_main_module():
    """Mock the main module"""
    mock_app = Mock()
    mock_main = Mock()
    mock_main.app = mock_app

    sys.modules['src.interface_layer.main'] = mock_main

    yield mock_main

    # Cleanup
    if 'src.interface_layer.main' in sys.modules:
        del sys.modules['src.interface_layer.main']


@pytest.fixture
def mock_storysense_processor():
    """Mock StorySenseProcessor"""
    with patch('src.html_report.storysense_processor.StorySenseProcessor') as mock_class:
        mock_instance = Mock()
        mock_instance.analyze_stories_in_batches = Mock(return_value=[])
        mock_instance.analyze_stories_with_context_in_batches = Mock(return_value=[])
        mock_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_enhanced_context_processor():
    """Mock EnhancedContextProcessor"""
    with patch(
            'src.context_handler.context_file_handler.enhanced_context_processor.EnhancedContextProcessor') as mock_class:
        mock_instance = Mock()
        mock_instance.process_all_context_files = Mock(return_value={
            'processed_files': 5,
            'skipped_files': 2,
            'failed_files': 0,
            'total_chunks': 50
        })
        mock_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_context_manager():
    """Mock ContextManager"""
    with patch('src.context_handler.context_file_handler.context_manager.ContextManager') as mock_class:
        mock_instance = Mock()
        mock_instance.check_and_process_context_library = Mock(return_value={
            'status': 'context_processed',
            'message': 'Success',
            'has_context': True,
            'processed_files': 5
        })
        mock_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def sample_excel_file(sample_user_stories):
    """Create a sample Excel file for testing"""
    temp_file = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)
    sample_user_stories.to_excel(temp_file.name, index=False)
    yield temp_file.name
    if os.path.exists(temp_file.name):
        os.unlink(temp_file.name)


@pytest.fixture
def sample_csv_file(sample_user_stories):
    """Create a sample CSV file for testing"""
    temp_file = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
    sample_user_stories.to_csv(temp_file.name, index=False)
    yield temp_file.name
    if os.path.exists(temp_file.name):
        os.unlink(temp_file.name)


@pytest.fixture
def sample_context_file(sample_context):
    """Create a sample context file for testing"""
    temp_file = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)
    sample_context.to_excel(temp_file.name, index=False)
    yield temp_file.name
    if os.path.exists(temp_file.name):
        os.unlink(temp_file.name)


@pytest.fixture(autouse=True)
def mock_threading_enumerate():
    """Mock threading.enumerate to avoid thread-related issues in tests"""
    import threading
    original_enumerate = threading.enumerate

    yield

    # Restore original
    threading.enumerate = original_enumerate


@pytest.fixture(autouse=True)
def mock_config_files_autouse():
    """Auto-mock configuration files for all tests"""
    config_io_content = """[Input]
input_file_path = ../Input/UserStories.xlsx
additional_context_path = ../Input/AdditionalContext.xlsx

[Output]
output_file_path = ../Output/StorySense
retrieval_context = ../Output/RetrievalContext
num_context_retrieve = 8

[Processing]
batch_size = 5
parallel_processing = false

[Context]
context_library_path = ../Input/ContextLibrary
"""

    config_content = """[LLM]
LLM_FAMILY = AWS
TEMPERATURE = 0.05

[Guardrails]
guardrail_id = test-guardrail
region = us-east-1
"""

    def mock_open_handler(file, mode='r', *args, **kwargs):
        if 'ConfigIO.properties' in str(file):
            return mock_open(read_data=config_io_content)()
        elif 'Config.properties' in str(file):
            return mock_open(read_data=config_content)()
        else:
            return mock_open(read_data="")()

    with patch('builtins.open', mock_open_handler), \
            patch('os.path.exists', return_value=True), \
            patch('os.makedirs'):
        yield


@pytest.fixture
def temp_excel_file(sample_user_stories):
    """Create a temporary Excel file"""
    temp_file = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)
    sample_user_stories.to_excel(temp_file.name, index=False)
    yield temp_file.name
    if os.path.exists(temp_file.name):
        os.unlink(temp_file.name)


@pytest.fixture
def temp_csv_file(sample_user_stories):
    """Create a temporary CSV file"""
    temp_file = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
    sample_user_stories.to_csv(temp_file.name, index=False)
    yield temp_file.name
    if os.path.exists(temp_file.name):
        os.unlink(temp_file.name)


@pytest.fixture
def temp_context_file(sample_context):
    """Create a temporary context file"""
    temp_file = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)
    sample_context.to_excel(temp_file.name, index=False)
    yield temp_file.name
    if os.path.exists(temp_file.name):
        os.unlink(temp_file.name)


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_aws: mark test as requiring AWS credentials"
    )
    config.addinivalue_line(
        "markers", "requires_db: mark test as requiring database connection"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically"""
    for item in items:
        # Add 'unit' marker to all tests by default
        if not any(marker.name in ['integration', 'slow'] for marker in item.iter_markers()):
            item.add_marker(pytest.mark.unit)

        # Add 'requires_aws' marker to AWS-related tests
        if 'aws' in item.nodeid.lower() or 'bedrock' in item.nodeid.lower():
            item.add_marker(pytest.mark.requires_aws)

        # Add 'requires_db' marker to database-related tests
        if 'db' in item.nodeid.lower() or 'pgvector' in item.nodeid.lower():
            item.add_marker(pytest.mark.requires_db)