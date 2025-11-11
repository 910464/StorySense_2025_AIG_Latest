"""
Unit tests for the MetricsManager class in metrics_manager.py
This module provides comprehensive test coverage for the MetricsManager class functionality.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open
import pytest
import os
import json
import time
import threading
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add the src directory to the Python path
current_dir = Path(__file__).parent
src_dir = current_dir.parent.parent / "src"
sys.path.insert(0, str(src_dir))

try:
    from src.metrics.metrics_manager import MetricsManager
except ImportError:
    # Fallback for different path configurations
    sys.path.append(str(current_dir.parent.parent))
    from src.metrics.metrics_manager import MetricsManager


class TestMetricsManager(unittest.TestCase):
    """Test cases for the MetricsManager class"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Mock external dependencies
        self.mock_psutil_process = Mock()
        self.mock_psutil_process.memory_info.return_value.rss = 1024 * 1024 * 100  # 100MB
        
        # Mock boto3 client
        self.mock_cloudwatch_client = Mock()
        
        # Start all patches
        self.patches = [
            patch('src.metrics.metrics_manager.psutil.Process', return_value=self.mock_psutil_process),
            patch('src.metrics.metrics_manager.psutil.cpu_percent', return_value=50.0),
            patch('src.metrics.metrics_manager.boto3.client', return_value=self.mock_cloudwatch_client),
            patch('src.metrics.metrics_manager.time.time', return_value=1000.0),
        ]
        
        for p in self.patches:
            p.start()

    def tearDown(self):
        """Clean up after each test method."""
        for p in self.patches:
            p.stop()

    def test_init_default_parameters(self):
        """Test MetricsManager initialization with default parameters"""
        # Act
        with patch('src.metrics.metrics_manager.threading.Thread') as mock_thread:
            metrics_manager = MetricsManager()

        # Assert
        self.assertEqual(metrics_manager.app_name, "StorySense")
        self.assertEqual(metrics_manager.start_time, 1000.0)
        self.assertIsInstance(metrics_manager.metrics, dict)
        self.assertEqual(metrics_manager.metrics['app'], "StorySense")
        self.assertEqual(metrics_manager.metrics['version'], '1.0.0')
        self.assertIn('start_time', metrics_manager.metrics)
        self.assertEqual(metrics_manager.usd_to_inr_rate, 83.5)
        mock_thread.assert_called()

    def test_init_custom_app_name(self):
        """Test MetricsManager initialization with custom app name"""
        # Act
        with patch('src.metrics.metrics_manager.threading.Thread'):
            metrics_manager = MetricsManager(app_name="CustomApp")

        # Assert
        self.assertEqual(metrics_manager.app_name, "CustomApp")
        self.assertEqual(metrics_manager.metrics['app'], "CustomApp")

    def test_init_metrics_structure(self):
        """Test that metrics structure is properly initialized"""
        # Act
        with patch('src.metrics.metrics_manager.threading.Thread'):
            metrics_manager = MetricsManager()

        # Assert
        expected_keys = ['app', 'version', 'start_time', 'end_time', 'total_duration',
                        'user_stories', 'batches', 'llm', 'vector_db', 'system', 'errors', 'costs']
        for key in expected_keys:
            self.assertIn(key, metrics_manager.metrics)

        # Check nested structures
        self.assertIn('total_calls', metrics_manager.metrics['llm'])
        self.assertIn('store_operations', metrics_manager.metrics['vector_db'])
        self.assertIn('start_memory', metrics_manager.metrics['system'])
        self.assertIn('total', metrics_manager.metrics['errors'])
        self.assertIn('llm', metrics_manager.metrics['costs'])

    def test_claude_pricing_structure(self):
        """Test that Claude pricing is properly configured"""
        # Act
        with patch('src.metrics.metrics_manager.threading.Thread'):
            metrics_manager = MetricsManager()

        # Assert
        self.assertIsInstance(metrics_manager.claude_pricing, dict)
        self.assertIn('claude-3-sonnet', metrics_manager.claude_pricing)
        self.assertIn('input', metrics_manager.claude_pricing['claude-3-sonnet'])
        self.assertIn('output', metrics_manager.claude_pricing['claude-3-sonnet'])
        self.assertEqual(metrics_manager.claude_pricing['claude-3-sonnet']['input'], 0.003)
        self.assertEqual(metrics_manager.claude_pricing['claude-3-sonnet']['output'], 0.015)

    @patch('src.metrics.metrics_manager.os.getenv')
    def test_cloudwatch_initialization_success(self, mock_getenv):
        """Test successful CloudWatch client initialization"""
        # Arrange
        mock_getenv.return_value = 'us-east-1'

        # Act
        with patch('src.metrics.metrics_manager.threading.Thread'):
            metrics_manager = MetricsManager()

        # Assert
        self.assertIsNotNone(metrics_manager.cloudwatch)

    @patch('src.metrics.metrics_manager.boto3.client')
    def test_cloudwatch_initialization_failure(self, mock_boto3_client):
        """Test CloudWatch client initialization failure"""
        # Arrange
        mock_boto3_client.side_effect = Exception("AWS credentials not found")

        # Act
        with patch('src.metrics.metrics_manager.threading.Thread'):
            with patch('src.metrics.metrics_manager.logging.warning') as mock_warning:
                metrics_manager = MetricsManager()

        # Assert
        self.assertIsNone(metrics_manager.cloudwatch)
        mock_warning.assert_called_once()

    def test_record_user_story_metrics(self):
        """Test recording user story metrics"""
        # Arrange
        with patch('src.metrics.metrics_manager.threading.Thread'):
            metrics_manager = MetricsManager()
        
        # Act
        metrics_manager.record_user_story_metrics(
            us_id="US001",
            processing_time=5.5,
            context_count=3,
            context_quality="good",
            overall_score=8.5
        )

        # Assert
        self.assertIn("US001", metrics_manager.metrics['user_stories'])
        story_metrics = metrics_manager.metrics['user_stories']["US001"]
        self.assertEqual(story_metrics['processing_time'], 5.5)
        self.assertEqual(story_metrics['context_count'], 3)
        self.assertEqual(story_metrics['context_quality'], "good")
        self.assertEqual(story_metrics['overall_score'], 8.5)

    def test_record_batch_metrics(self):
        """Test recording batch metrics"""
        # Arrange
        with patch('src.metrics.metrics_manager.threading.Thread'):
            metrics_manager = MetricsManager()

        # Act
        metrics_manager.record_batch_metrics(
            batch_num=1,
            story_count=10,
            processing_time=30.0
        )

        # Assert
        self.assertIn("1", metrics_manager.metrics['batches'])
        batch_metrics = metrics_manager.metrics['batches']["1"]
        self.assertEqual(batch_metrics['story_count'], 10)
        self.assertEqual(batch_metrics['processing_time'], 30.0)
        self.assertEqual(batch_metrics['stories_per_second'], 10/30.0)

    def test_record_batch_metrics_zero_time(self):
        """Test recording batch metrics with zero processing time"""
        # Arrange
        with patch('src.metrics.metrics_manager.threading.Thread'):
            metrics_manager = MetricsManager()

        # Act
        metrics_manager.record_batch_metrics(
            batch_num=1,
            story_count=10,
            processing_time=0
        )

        # Assert
        batch_metrics = metrics_manager.metrics['batches']["1"]
        self.assertEqual(batch_metrics['stories_per_second'], 0)

    def test_calculate_llm_cost_known_model(self):
        """Test LLM cost calculation for known model"""
        # Arrange
        with patch('src.metrics.metrics_manager.threading.Thread'):
            metrics_manager = MetricsManager()

        # Act
        cost_data = metrics_manager._calculate_llm_cost("claude-3-sonnet", 1000, 500)

        # Assert
        expected_input_cost = (1000 / 1000) * 0.003  # 0.003
        expected_output_cost = (500 / 1000) * 0.015  # 0.0075
        expected_total_cost = expected_input_cost + expected_output_cost

        self.assertEqual(cost_data['input_cost'], expected_input_cost)
        self.assertEqual(cost_data['output_cost'], expected_output_cost)
        self.assertEqual(cost_data['total_cost'], expected_total_cost)
        self.assertEqual(cost_data['total_cost_inr'], expected_total_cost * 83.5)

    def test_calculate_llm_cost_unknown_model(self):
        """Test LLM cost calculation for unknown model using defaults"""
        # Arrange
        with patch('src.metrics.metrics_manager.threading.Thread'):
            metrics_manager = MetricsManager()

        # Act
        cost_data = metrics_manager._calculate_llm_cost("unknown-model", 1000, 500)

        # Assert
        expected_input_cost = (1000 / 1000) * 0.003  # Default pricing
        expected_output_cost = (500 / 1000) * 0.015
        expected_total_cost = expected_input_cost + expected_output_cost

        self.assertEqual(cost_data['input_cost'], expected_input_cost)
        self.assertEqual(cost_data['output_cost'], expected_output_cost)
        self.assertEqual(cost_data['total_cost'], expected_total_cost)

    def test_record_llm_call_basic(self):
        """Test recording basic LLM call"""
        # Arrange
        with patch('src.metrics.metrics_manager.threading.Thread'):
            metrics_manager = MetricsManager()

        # Act
        metrics_manager.record_llm_call(
            call_type="analysis",
            input_tokens=1000,
            output_tokens=500,
            latency=2.5,
            model_name="claude-3-sonnet"
        )

        # Assert
        llm_metrics = metrics_manager.metrics['llm']
        self.assertEqual(llm_metrics['total_calls'], 1)
        self.assertEqual(llm_metrics['input_tokens'], 1000)
        self.assertEqual(llm_metrics['output_tokens'], 500)
        self.assertEqual(llm_metrics['total_tokens'], 1500)
        self.assertEqual(llm_metrics['total_latency'], 2.5)
        self.assertEqual(llm_metrics['avg_latency'], 2.5)
        self.assertEqual(llm_metrics['guardrail_interventions'], 0)

    def test_record_llm_call_with_guardrail(self):
        """Test recording LLM call with guardrail intervention"""
        # Arrange
        with patch('src.metrics.metrics_manager.threading.Thread'):
            metrics_manager = MetricsManager()

        # Act
        metrics_manager.record_llm_call(
            call_type="analysis",
            input_tokens=1000,
            output_tokens=500,
            latency=2.5,
            guardrail_triggered=True
        )

        # Assert
        self.assertEqual(metrics_manager.metrics['llm']['guardrail_interventions'], 1)

    def test_record_llm_call_updates_costs(self):
        """Test that LLM call recording updates cost metrics"""
        # Arrange
        with patch('src.metrics.metrics_manager.threading.Thread'):
            metrics_manager = MetricsManager()

        # Act
        metrics_manager.record_llm_call(
            call_type="analysis",
            input_tokens=1000,
            output_tokens=500,
            latency=2.5,
            model_name="claude-3-sonnet"
        )

        # Assert
        costs = metrics_manager.metrics['costs']['llm']
        self.assertGreater(costs['total_cost'], 0)
        self.assertGreater(costs['input_cost'], 0)
        self.assertGreater(costs['output_cost'], 0)
        self.assertGreater(costs['total_cost_inr'], 0)

    def test_record_llm_call_by_model_tracking(self):
        """Test that LLM calls are tracked by model"""
        # Arrange
        with patch('src.metrics.metrics_manager.threading.Thread'):
            metrics_manager = MetricsManager()

        # Act
        metrics_manager.record_llm_call(
            call_type="analysis",
            input_tokens=1000,
            output_tokens=500,
            latency=2.5,
            model_name="claude-3-sonnet"
        )

        # Assert
        model_costs = metrics_manager.metrics['costs']['llm']['by_model']['claude-3-sonnet']
        self.assertEqual(model_costs['calls'], 1)
        self.assertEqual(model_costs['input_tokens'], 1000)
        self.assertEqual(model_costs['output_tokens'], 500)
        self.assertGreater(model_costs['total_cost'], 0)

    def test_record_llm_call_by_type_tracking(self):
        """Test that LLM calls are tracked by call type"""
        # Arrange
        with patch('src.metrics.metrics_manager.threading.Thread'):
            metrics_manager = MetricsManager()

        # Act
        metrics_manager.record_llm_call(
            call_type="analysis",
            input_tokens=1000,
            output_tokens=500,
            latency=2.5
        )

        # Assert
        call_type_metrics = metrics_manager.metrics['llm']['calls_by_type']['analysis']
        self.assertEqual(call_type_metrics['calls'], 1)
        self.assertEqual(call_type_metrics['input_tokens'], 1000)
        self.assertEqual(call_type_metrics['output_tokens'], 500)
        self.assertEqual(call_type_metrics['total_tokens'], 1500)
        self.assertEqual(call_type_metrics['latency'], 2.5)

    def test_record_llm_call_average_latency_calculation(self):
        """Test average latency calculation with multiple calls"""
        # Arrange
        with patch('src.metrics.metrics_manager.threading.Thread'):
            metrics_manager = MetricsManager()

        # Act
        metrics_manager.record_llm_call("analysis", 1000, 500, 2.0)
        metrics_manager.record_llm_call("validation", 500, 250, 4.0)

        # Assert
        expected_avg_latency = (2.0 + 4.0) / 2
        self.assertEqual(metrics_manager.metrics['llm']['avg_latency'], expected_avg_latency)

    def test_record_vector_operation_store(self):
        """Test recording vector store operation"""
        # Arrange
        with patch('src.metrics.metrics_manager.threading.Thread'):
            metrics_manager = MetricsManager()

        # Act
        metrics_manager.record_vector_operation("store", 100, 1.5)

        # Assert
        vector_metrics = metrics_manager.metrics['vector_db']
        self.assertEqual(vector_metrics['store_operations'], 1)
        self.assertEqual(vector_metrics['total_vectors_stored'], 100)
        self.assertEqual(vector_metrics['total_store_time'], 1.5)
        self.assertEqual(vector_metrics['avg_store_time'], 1.5)

    def test_record_vector_operation_query(self):
        """Test recording vector query operation"""
        # Arrange
        with patch('src.metrics.metrics_manager.threading.Thread'):
            metrics_manager = MetricsManager()

        # Act
        metrics_manager.record_vector_operation("query", 10, 0.5)

        # Assert
        vector_metrics = metrics_manager.metrics['vector_db']
        self.assertEqual(vector_metrics['query_operations'], 1)
        self.assertEqual(vector_metrics['total_query_time'], 0.5)
        self.assertEqual(vector_metrics['avg_query_time'], 0.5)

    def test_record_vector_operation_multiple_stores(self):
        """Test average calculation with multiple store operations"""
        # Arrange
        with patch('src.metrics.metrics_manager.threading.Thread'):
            metrics_manager = MetricsManager()

        # Act
        metrics_manager.record_vector_operation("store", 100, 1.0)
        metrics_manager.record_vector_operation("store", 200, 3.0)

        # Assert
        vector_metrics = metrics_manager.metrics['vector_db']
        self.assertEqual(vector_metrics['store_operations'], 2)
        self.assertEqual(vector_metrics['total_vectors_stored'], 300)
        self.assertEqual(vector_metrics['total_store_time'], 4.0)
        self.assertEqual(vector_metrics['avg_store_time'], 2.0)

    def test_record_error_new_type(self):
        """Test recording error with new error type"""
        # Arrange
        with patch('src.metrics.metrics_manager.threading.Thread'):
            metrics_manager = MetricsManager()

        # Act
        metrics_manager.record_error("validation_error", "Invalid input format")

        # Assert
        error_metrics = metrics_manager.metrics['errors']
        self.assertEqual(error_metrics['total'], 1)
        self.assertIn("validation_error", error_metrics['by_type'])
        self.assertEqual(error_metrics['by_type']['validation_error']['count'], 1)
        self.assertIn("Invalid input format", error_metrics['by_type']['validation_error']['messages'])

    def test_record_error_existing_type(self):
        """Test recording multiple errors of the same type"""
        # Arrange
        with patch('src.metrics.metrics_manager.threading.Thread'):
            metrics_manager = MetricsManager()

        # Act
        metrics_manager.record_error("validation_error", "Error 1")
        metrics_manager.record_error("validation_error", "Error 2")

        # Assert
        error_metrics = metrics_manager.metrics['errors']
        self.assertEqual(error_metrics['total'], 2)
        self.assertEqual(error_metrics['by_type']['validation_error']['count'], 2)
        self.assertEqual(len(error_metrics['by_type']['validation_error']['messages']), 2)

    def test_update_total_cost(self):
        """Test total cost update calculation"""
        # Arrange
        with patch('src.metrics.metrics_manager.threading.Thread'):
            metrics_manager = MetricsManager()
        
        # Set up some cost data
        metrics_manager.metrics['costs']['llm']['total_cost'] = 10.50
        metrics_manager.metrics['costs']['llm']['total_cost_inr'] = 877.75

        # Act
        metrics_manager._update_total_cost()

        # Assert
        self.assertEqual(metrics_manager.metrics['costs']['total_estimated_cost'], 10.50)
        self.assertEqual(metrics_manager.metrics['costs']['total_estimated_cost_inr'], 877.75)

    @patch('src.metrics.metrics_manager.datetime')
    def test_send_to_cloudwatch_success(self, mock_datetime):
        """Test successful CloudWatch metrics sending"""
        # Arrange
        mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0, 0)
        
        with patch('src.metrics.metrics_manager.threading.Thread'):
            metrics_manager = MetricsManager()
        
        # Act
        metrics_manager._send_to_cloudwatch("Test", {"MetricA": 10, "MetricB": 20.5})

        # Assert
        self.mock_cloudwatch_client.put_metric_data.assert_called_once()
        call_args = self.mock_cloudwatch_client.put_metric_data.call_args
        self.assertEqual(call_args[1]['Namespace'], 'StorySense/StorySense')
        self.assertEqual(len(call_args[1]['MetricData']), 2)

    def test_send_to_cloudwatch_no_client(self):
        """Test CloudWatch sending when client is not available"""
        # Arrange
        with patch('src.metrics.metrics_manager.threading.Thread'):
            metrics_manager = MetricsManager()
        metrics_manager.cloudwatch = None

        # Act
        result = metrics_manager._send_to_cloudwatch("Test", {"MetricA": 10})

        # Assert
        # Should return without error when no client available
        self.assertIsNone(result)

    @patch('src.metrics.metrics_manager.datetime')
    def test_send_to_cloudwatch_exception_handling(self, mock_datetime):
        """Test CloudWatch exception handling"""
        # Arrange
        mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0, 0)
        self.mock_cloudwatch_client.put_metric_data.side_effect = Exception("CloudWatch error")
        
        with patch('src.metrics.metrics_manager.threading.Thread'):
            metrics_manager = MetricsManager()

        # Act
        with patch('src.metrics.metrics_manager.logging.warning') as mock_warning:
            metrics_manager._send_to_cloudwatch("Test", {"MetricA": 10})

        # Assert
        mock_warning.assert_called_once()

    def test_send_to_cloudwatch_filters_non_numeric(self):
        """Test that CloudWatch sending filters out non-numeric values"""
        # Arrange
        with patch('src.metrics.metrics_manager.threading.Thread'):
            metrics_manager = MetricsManager()

        # Act
        metrics_manager._send_to_cloudwatch("Test", {
            "NumericMetric": 10,
            "StringMetric": "test",
            "FloatMetric": 20.5,
            "BoolMetric": True
        })

        # Assert
        call_args = self.mock_cloudwatch_client.put_metric_data.call_args
        metric_data = call_args[1]['MetricData']
        # Should only include numeric metrics (int, float)
        self.assertEqual(len(metric_data), 2)  # NumericMetric and FloatMetric

    @patch('src.metrics.metrics_manager.time.time')
    @patch('src.metrics.metrics_manager.datetime')
    def test_get_metrics_summary(self, mock_datetime, mock_time):
        """Test metrics summary generation"""
        # Arrange
        mock_time.return_value = 1100.0  # 100 seconds later
        mock_datetime.now.return_value.isoformat.return_value = "2024-01-01T12:00:00"
        
        with patch('src.metrics.metrics_manager.threading.Thread'):
            metrics_manager = MetricsManager()
        
        # Add some test data
        metrics_manager.metrics['user_stories']['US001'] = {'processing_time': 5.0}
        metrics_manager.metrics['llm']['total_calls'] = 5
        metrics_manager.metrics['llm']['total_tokens'] = 1000

        # Act
        summary = metrics_manager.get_metrics_summary()

        # Assert
        self.assertIn('start_time', summary)
        self.assertIn('end_time', summary)
        self.assertEqual(summary['story_count'], 1)
        self.assertEqual(summary['llm_calls'], 5)
        self.assertEqual(summary['llm_tokens'], 1000)
        self.assertIn('system', summary)
        self.assertIn('cost_breakdown', summary)

    def test_get_metrics_summary_no_stories(self):
        """Test metrics summary with no user stories"""
        # Arrange
        with patch('src.metrics.metrics_manager.threading.Thread'):
            metrics_manager = MetricsManager()

        # Act
        summary = metrics_manager.get_metrics_summary()

        # Assert
        self.assertEqual(summary['story_count'], 0)
        self.assertEqual(summary['avg_story_time'], 0)

    @patch('builtins.open', new_callable=mock_open)
    @patch('src.metrics.metrics_manager.os.makedirs')
    @patch('src.metrics.metrics_manager.datetime')
    @patch('src.metrics.metrics_manager.time.time')
    def test_save_metrics(self, mock_time, mock_datetime, mock_makedirs, mock_file):
        """Test saving metrics to file"""
        # Arrange
        mock_time.return_value = 1100.0
        mock_datetime.now.return_value.strftime.return_value = "20240101_120000"
        mock_datetime.now.return_value.isoformat.return_value = "2024-01-01T12:00:00"
        
        with patch('src.metrics.metrics_manager.threading.Thread'):
            metrics_manager = MetricsManager()

        # Act
        result = metrics_manager.save_metrics("test_output")

        # Assert
        mock_makedirs.assert_called_once_with("test_output", exist_ok=True)
        mock_file.assert_called_once_with("test_output/metrics_20240101_120000.json", 'w')
        self.assertEqual(result, "test_output/metrics_20240101_120000.json")

    @patch('builtins.open', new_callable=mock_open)
    @patch('src.metrics.metrics_manager.os.makedirs')  
    @patch('src.metrics.metrics_manager.json.dump')
    def test_save_metrics_json_content(self, mock_json_dump, mock_makedirs, mock_file):
        """Test that metrics are properly serialized to JSON"""
        # Arrange
        with patch('src.metrics.metrics_manager.threading.Thread'):
            metrics_manager = MetricsManager()

        # Act
        metrics_manager.save_metrics()

        # Assert
        mock_json_dump.assert_called_once()
        args = mock_json_dump.call_args
        self.assertEqual(args[0][0], metrics_manager.metrics)  # First argument should be the metrics
        self.assertEqual(args[1]['indent'], 2)
        self.assertEqual(args[1]['default'], str)

    def test_monitor_system_resources_thread_creation(self):
        """Test that system monitoring thread is created"""
        # Arrange & Act
        with patch('src.metrics.metrics_manager.threading.Thread') as mock_thread:
            mock_thread_instance = Mock()
            mock_thread.return_value = mock_thread_instance
            
            metrics_manager = MetricsManager()

        # Assert
        mock_thread.assert_called_once()
        mock_thread_instance.start.assert_called_once()
        self.assertTrue(mock_thread_instance.daemon)

    @patch('src.metrics.metrics_manager.time.sleep')
    def test_monitor_system_resources_updates_metrics(self, mock_sleep):
        """Test system resource monitoring updates metrics"""
        # Arrange
        mock_sleep.side_effect = [None, Exception("Stop loop")]  # Stop after one iteration
        
        with patch('src.metrics.metrics_manager.threading.Thread'):
            metrics_manager = MetricsManager()

        # Act
        with self.assertRaises(Exception):
            metrics_manager._monitor_system_resources()

        # Assert that CPU and memory were checked
        # The exact assertions depend on the mocked psutil behavior

    def test_destructor_stops_monitoring(self):
        """Test that destructor stops monitoring thread"""
        # Arrange
        with patch('src.metrics.metrics_manager.threading.Thread') as mock_thread:
            mock_thread_instance = Mock()
            mock_thread_instance.is_alive.return_value = True
            mock_thread.return_value = mock_thread_instance
            
            metrics_manager = MetricsManager()

        # Act
        metrics_manager.__del__()

        # Assert
        self.assertTrue(metrics_manager.stop_monitoring)
        mock_thread_instance.join.assert_called_once_with(timeout=1.0)

    def test_destructor_thread_not_alive(self):
        """Test destructor when monitoring thread is not alive"""
        # Arrange
        with patch('src.metrics.metrics_manager.threading.Thread') as mock_thread:
            mock_thread_instance = Mock()
            mock_thread_instance.is_alive.return_value = False
            mock_thread.return_value = mock_thread_instance
            
            metrics_manager = MetricsManager()

        # Act
        metrics_manager.__del__()

        # Assert
        self.assertTrue(metrics_manager.stop_monitoring)
        mock_thread_instance.join.assert_not_called()

    def test_system_memory_tracking(self):
        """Test that system memory is properly tracked"""
        # Arrange
        self.mock_psutil_process.memory_info.return_value.rss = 1024 * 1024 * 150  # 150MB
        
        with patch('src.metrics.metrics_manager.threading.Thread'):
            metrics_manager = MetricsManager()

        # Assert
        expected_memory_mb = 150  # 150MB
        self.assertEqual(metrics_manager.metrics['system']['start_memory'], expected_memory_mb)

    def test_integration_full_workflow(self):
        """Test a complete workflow integration"""
        # Arrange
        with patch('src.metrics.metrics_manager.threading.Thread'):
            metrics_manager = MetricsManager("TestApp")

        # Act - Simulate a complete workflow
        metrics_manager.record_user_story_metrics("US001", 5.0, 2, "good", 8.5)
        metrics_manager.record_llm_call("analysis", 1000, 500, 2.0, "claude-3-sonnet")
        metrics_manager.record_vector_operation("store", 100, 1.0)
        metrics_manager.record_batch_metrics(1, 1, 5.0)
        metrics_manager.record_error("test_error", "Test error message")

        summary = metrics_manager.get_metrics_summary()

        # Assert
        self.assertEqual(summary['story_count'], 1)
        self.assertEqual(summary['llm_calls'], 1)
        self.assertEqual(summary['error_count'], 1)
        self.assertGreater(summary['total_estimated_cost'], 0)


class TestMetricsManagerThreading(unittest.TestCase):
    """Test cases for MetricsManager threading functionality"""

    def setUp(self):
        """Set up test fixtures for threading tests."""
        self.patches = [
            patch('src.metrics.metrics_manager.psutil.Process'),
            patch('src.metrics.metrics_manager.psutil.cpu_percent', return_value=50.0),
            patch('src.metrics.metrics_manager.boto3.client'),
            patch('src.metrics.metrics_manager.time.time', return_value=1000.0),
        ]
        
        for p in self.patches:
            p.start()

    def tearDown(self):
        """Clean up after threading tests."""
        for p in self.patches:
            p.stop()

    @patch('src.metrics.metrics_manager.time.sleep')
    def test_monitor_system_resources_loop(self, mock_sleep):
        """Test the system monitoring loop behavior"""
        # Arrange
        call_count = [0]
        
        def side_effect(*args):
            call_count[0] += 1
            if call_count[0] >= 2:  # Stop after 2 iterations
                raise Exception("Stop monitoring")
        
        mock_sleep.side_effect = side_effect
        
        with patch('src.metrics.metrics_manager.threading.Thread'):
            metrics_manager = MetricsManager()

        # Act & Assert
        with self.assertRaises(Exception):
            metrics_manager._monitor_system_resources()
        
        self.assertEqual(call_count[0], 2)

    def test_stop_monitoring_flag(self):
        """Test that stop_monitoring flag controls thread execution"""
        # Arrange
        with patch('src.metrics.metrics_manager.threading.Thread'):
            metrics_manager = MetricsManager()
        
        metrics_manager.stop_monitoring = True

        # Act
        with patch('src.metrics.metrics_manager.time.sleep') as mock_sleep:
            metrics_manager._monitor_system_resources()

        # Assert
        mock_sleep.assert_not_called()  # Should exit immediately


if __name__ == '__main__':
    # Configure logging for tests
    logging.basicConfig(level=logging.DEBUG)
    
    # Run the tests
    unittest.main(verbosity=2)
