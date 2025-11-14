# tests/context_handler/conftest.py

import pytest
from unittest.mock import Mock


@pytest.fixture
def global_metrics_manager_mock():
    """Mock metrics manager for tests"""
    mock_metrics = Mock()
    mock_metrics.record_metric = Mock()
    mock_metrics.record_error = Mock()
    mock_metrics.record_timing = Mock()
    mock_metrics.get_metrics = Mock(return_value={})
    return mock_metrics
