import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open, call
import pytest
import os
import json
import time
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
import sys
import threading

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
    """Comprehensive test suite for MetricsManager class"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Mock psutil to avoid system dependencies
        self.psutil_patcher = patch('src.metrics.metrics_manager.psutil')
        self.mock_psutil = self.psutil_patcher.start()
        
        # Configure psutil mock
        mock_process = Mock()
        mock_process.memory_info.return_value.rss = 104857600  # 100MB in bytes
        self.mock_psutil.Process.return_value = mock_process
        self.mock_psutil.cpu_percent.return_value = 25.5
        
        # Mock boto3 to avoid AWS dependencies
        self.boto3_patcher = patch('src.metrics.metrics_manager.boto3')
        self.mock_boto3 = self.boto3_patcher.start()
        
        # Mock threading to control background threads
        self.threading_patcher = patch('src.metrics.metrics_manager.threading.Thread')
        self.mock_thread = self.threading_patcher.start()
        
        # Mock time.sleep to speed up tests
        self.sleep_patcher = patch('time.sleep')
        self.mock_sleep = self.sleep_patcher.start()
        
        # Mock datetime for consistent testing
        self.datetime_patcher = patch('src.metrics.metrics_manager.datetime')
        self.mock_datetime = self.datetime_patcher.start()
        self.mock_datetime.now.return_value.isoformat.return_value = "2024-01-01T12:00:00"
        self.mock_datetime.fromtimestamp.return_value.isoformat.return_value = "2024-01-01T10:00:00"
        self.mock_datetime.utcnow.return_value = datetime(2024, 1, 1, 12, 0, 0)

    def tearDown(self):
        """Clean up after each test method."""
        self.psutil_patcher.stop()
        self.boto3_patcher.stop()
        self.threading_patcher.stop()
        self.sleep_patcher.stop()
        self.datetime_patcher.stop()

    # ==================== Initialization Tests ====================

    def test_init_default_values(self):
        """Test MetricsManager initialization with default values"""
        with patch.dict(os.environ, {}, clear=True):
            manager = MetricsManager()
            
            # Check basic initialization
            self.assertEqual(manager.app_name, "StorySense")
            self.assertIsInstance(manager.start_time, float)
            self.assertEqual(manager.usd_to_inr_rate, 83.5)
            self.assertFalse(manager.stop_monitoring)
            
            # Check metrics structure
            self.assertIn('app', manager.metrics)
            self.assertIn('version', manager.metrics)
            self.assertIn('llm', manager.metrics)
            self.assertIn('vector_db', manager.metrics)
            self.assertIn('system', manager.metrics)
            self.assertIn('errors', manager.metrics)
            self.assertIn('costs', manager.metrics)
            
            # Check LLM metrics initialization
            self.assertEqual(manager.metrics['llm']['total_calls'], 0)
            self.assertEqual(manager.metrics['llm']['total_tokens'], 0)
            self.assertEqual(manager.metrics['llm']['guardrail_interventions'], 0)

    def test_init_custom_app_name(self):
        """Test MetricsManager initialization with custom app name"""
        custom_name = "CustomApp"
        manager = MetricsManager(app_name=custom_name)
        
        self.assertEqual(manager.app_name, custom_name)
        self.assertEqual(manager.metrics['app'], custom_name)

    @patch.dict(os.environ, {'AWS_REGION': 'us-west-2'})
    def test_init_with_aws_credentials(self):
        """Test initialization with AWS credentials"""
        mock_cloudwatch = Mock()
        self.mock_boto3.client.return_value = mock_cloudwatch
        
        manager = MetricsManager()
        
        self.mock_boto3.client.assert_called_once_with('cloudwatch', region_name='us-west-2')
        self.assertEqual(manager.cloudwatch, mock_cloudwatch)

    def test_init_boto3_exception(self):
        """Test initialization when boto3 fails"""
        self.mock_boto3.client.side_effect = Exception("AWS credentials not found")
        
        with patch('src.metrics.metrics_manager.logging') as mock_logging:
            manager = MetricsManager()
            
            self.assertIsNone(manager.cloudwatch)
            mock_logging.warning.assert_called_once()

    def test_claude_pricing_structure(self):
        """Test that Claude pricing is properly initialized"""
        manager = MetricsManager()
        
        self.assertIn('claude-3-sonnet', manager.claude_pricing)
        self.assertIn('claude-3-5-sonnet', manager.claude_pricing)
        self.assertIn('claude-3-opus', manager.claude_pricing)
        
        for model, pricing in manager.claude_pricing.items():
            self.assertIn('input', pricing)
            self.assertIn('output', pricing)
            self.assertIsInstance(pricing['input'], float)
            self.assertIsInstance(pricing['output'], float)

    # ==================== System Monitoring Tests ====================

    # def test_monitor_system_resources(self):
    #     """Test _monitor_system_resources method"""
    #     manager = MetricsManager()
        
    #     # Test the monitoring loop (simulate a few iterations)
    #     manager.stop_monitoring = False
        
    #     # Mock to stop after first iteration
    #     def stop_after_one_call(*args, **kwargs):
    #         manager.stop_monitoring = True
    #         return 30.0
        
    #     self.mock_psutil.cpu_percent.side_effect = stop_after_one_call
        
    #     # Run monitoring once
    #     manager._monitor_system_resources()
        
    #     # Verify CPU was updated
    #     self.mock_psutil.cpu_percent.assert_called()

    # def test_monitor_system_resources_exception(self):
    #     """Test _monitor_system_resources exception handling"""
    #     manager = MetricsManager()
        
    #     # Make psutil.cpu_percent raise an exception
    #     self.mock_psutil.cpu_percent.side_effect = Exception("System error")
        
    #     with patch('src.metrics.metrics_manager.logging') as mock_logging:
    #         manager.stop_monitoring = False
            
    #         # Simulate one iteration then stop
    #         def stop_monitoring(*args, **kwargs):
    #             manager.stop_monitoring = True
    #             raise Exception("System error")
            
    #         self.mock_psutil.cpu_percent.side_effect = stop_monitoring
            
    #         manager._monitor_system_resources()
            
    #         # Verify error was logged
    #         mock_logging.error.assert_called()

    # ==================== User Story Metrics Tests ====================

    def test_record_user_story_metrics(self):
        """Test recording user story metrics"""
        manager = MetricsManager()
        
        us_id = "US001"
        processing_time = 5.5
        context_count = 3
        context_quality = "high"
        overall_score = 8.5
        
        manager.record_user_story_metrics(
            us_id, processing_time, context_count, context_quality, overall_score
        )
        
        # Verify metrics were recorded
        self.assertIn(us_id, manager.metrics['user_stories'])
        story_metrics = manager.metrics['user_stories'][us_id]
        self.assertEqual(story_metrics['processing_time'], processing_time)
        self.assertEqual(story_metrics['context_count'], context_count)
        self.assertEqual(story_metrics['context_quality'], context_quality)
        self.assertEqual(story_metrics['overall_score'], overall_score)

    def test_record_user_story_metrics_defaults(self):
        """Test recording user story metrics with default values"""
        manager = MetricsManager()
        
        us_id = "US002"
        processing_time = 3.2
        
        manager.record_user_story_metrics(us_id, processing_time)
        
        story_metrics = manager.metrics['user_stories'][us_id]
        self.assertEqual(story_metrics['processing_time'], processing_time)
        self.assertEqual(story_metrics['context_count'], 0)
        self.assertEqual(story_metrics['context_quality'], "none")
        self.assertEqual(story_metrics['overall_score'], 0)

    # ==================== Batch Metrics Tests ====================

    def test_record_batch_metrics(self):
        """Test recording batch metrics"""
        manager = MetricsManager()
        
        batch_num = 1
        story_count = 10
        processing_time = 25.5
        
        manager.record_batch_metrics(batch_num, story_count, processing_time)
        
        # Verify metrics were recorded
        self.assertIn(str(batch_num), manager.metrics['batches'])
        batch_metrics = manager.metrics['batches'][str(batch_num)]
        self.assertEqual(batch_metrics['story_count'], story_count)
        self.assertEqual(batch_metrics['processing_time'], processing_time)
        self.assertEqual(batch_metrics['stories_per_second'], story_count / processing_time)

    def test_record_batch_metrics_zero_time(self):
        """Test recording batch metrics with zero processing time"""
        manager = MetricsManager()
        
        batch_num = 2
        story_count = 5
        processing_time = 0
        
        manager.record_batch_metrics(batch_num, story_count, processing_time)
        
        batch_metrics = manager.metrics['batches'][str(batch_num)]
        self.assertEqual(batch_metrics['stories_per_second'], 0)

    # ==================== LLM Cost Calculation Tests ====================

    def test_calculate_llm_cost_known_model(self):
        """Test LLM cost calculation for known model"""
        manager = MetricsManager()
        
        model_name = "claude-3-sonnet"
        input_tokens = 1000
        output_tokens = 500
        
        cost_data = manager._calculate_llm_cost(model_name, input_tokens, output_tokens)
        
        # Expected costs
        expected_input_cost = (1000 / 1000) * 0.003  # 0.003
        expected_output_cost = (500 / 1000) * 0.015  # 0.0075
        expected_total_cost = expected_input_cost + expected_output_cost  # 0.0105
        
        self.assertEqual(cost_data['input_cost'], expected_input_cost)
        self.assertEqual(cost_data['output_cost'], expected_output_cost)
        self.assertEqual(cost_data['total_cost'], expected_total_cost)
        
        # Check INR costs
        self.assertEqual(cost_data['input_cost_inr'], expected_input_cost * 83.5)
        self.assertEqual(cost_data['output_cost_inr'], expected_output_cost * 83.5)
        self.assertEqual(cost_data['total_cost_inr'], expected_total_cost * 83.5)

    def test_calculate_llm_cost_unknown_model(self):
        """Test LLM cost calculation for unknown model"""
        manager = MetricsManager()
        
        model_name = "unknown-model"
        input_tokens = 2000
        output_tokens = 1000
        
        cost_data = manager._calculate_llm_cost(model_name, input_tokens, output_tokens)
        
        # Should use default pricing
        expected_input_cost = (2000 / 1000) * 0.003  # 0.006
        expected_output_cost = (1000 / 1000) * 0.015  # 0.015
        expected_total_cost = expected_input_cost + expected_output_cost  # 0.021
        
        self.assertEqual(cost_data['input_cost'], expected_input_cost)
        self.assertEqual(cost_data['output_cost'], expected_output_cost)
        self.assertEqual(cost_data['total_cost'], expected_total_cost)

    # ==================== LLM Call Recording Tests ====================

    def test_record_llm_call_basic(self):
        """Test basic LLM call recording"""
        manager = MetricsManager()
        
        call_type = "analysis"
        input_tokens = 1000
        output_tokens = 500
        latency = 2.5
        model_name = "claude-3-sonnet"
        
        manager.record_llm_call(call_type, input_tokens, output_tokens, latency, model_name)
        
        # Check LLM metrics
        llm_metrics = manager.metrics['llm']
        self.assertEqual(llm_metrics['total_calls'], 1)
        self.assertEqual(llm_metrics['input_tokens'], input_tokens)
        self.assertEqual(llm_metrics['output_tokens'], output_tokens)
        self.assertEqual(llm_metrics['total_tokens'], input_tokens + output_tokens)
        self.assertEqual(llm_metrics['total_latency'], latency)
        self.assertEqual(llm_metrics['avg_latency'], latency)
        self.assertEqual(llm_metrics['guardrail_interventions'], 0)
        
        # Check call type tracking
        self.assertIn(call_type, llm_metrics['calls_by_type'])
        call_type_metrics = llm_metrics['calls_by_type'][call_type]
        self.assertEqual(call_type_metrics['calls'], 1)
        self.assertEqual(call_type_metrics['input_tokens'], input_tokens)
        self.assertEqual(call_type_metrics['output_tokens'], output_tokens)
        
        # Check model tracking
        self.assertIn(model_name, manager.metrics['costs']['llm']['by_model'])

    def test_record_llm_call_with_guardrail(self):
        """Test LLM call recording with guardrail triggered"""
        manager = MetricsManager()
        
        call_type = "validation"
        input_tokens = 500
        output_tokens = 200
        latency = 1.8
        model_name = "claude-3-opus"
        
        manager.record_llm_call(
            call_type, input_tokens, output_tokens, latency, model_name, guardrail_triggered=True
        )
        
        # Check guardrail metrics
        self.assertEqual(manager.metrics['llm']['guardrail_interventions'], 1)
        self.assertEqual(
            manager.metrics['llm']['calls_by_type'][call_type]['guardrail_interventions'], 1
        )

    def test_record_llm_call_multiple_calls(self):
        """Test multiple LLM call recordings"""
        manager = MetricsManager()
        
        # First call
        manager.record_llm_call("analysis", 1000, 500, 2.0, "claude-3-sonnet")
        
        # Second call
        manager.record_llm_call("validation", 800, 300, 1.5, "claude-3-sonnet")
        
        # Check aggregated metrics
        llm_metrics = manager.metrics['llm']
        self.assertEqual(llm_metrics['total_calls'], 2)
        self.assertEqual(llm_metrics['input_tokens'], 1800)
        self.assertEqual(llm_metrics['output_tokens'], 800)
        self.assertEqual(llm_metrics['total_tokens'], 2600)
        self.assertEqual(llm_metrics['total_latency'], 3.5)
        self.assertEqual(llm_metrics['avg_latency'], 1.75)

    def test_record_llm_call_cost_calculation(self):
        """Test LLM call cost calculation integration"""
        manager = MetricsManager()
        
        input_tokens = 1000
        output_tokens = 500
        model_name = "claude-3-sonnet"
        
        manager.record_llm_call("test", input_tokens, output_tokens, 1.0, model_name)
        
        # Check cost metrics were updated
        costs = manager.metrics['costs']['llm']
        self.assertGreater(costs['total_cost'], 0)
        self.assertGreater(costs['input_cost'], 0)
        self.assertGreater(costs['output_cost'], 0)
        self.assertGreater(costs['total_cost_inr'], 0)
        
        # Check model-specific costs
        self.assertIn(model_name, costs['by_model'])
        model_costs = costs['by_model'][model_name]
        self.assertEqual(model_costs['calls'], 1)
        self.assertEqual(model_costs['input_tokens'], input_tokens)
        self.assertEqual(model_costs['output_tokens'], output_tokens)

    # ==================== Vector Database Tests ====================

    def test_record_vector_operation_store(self):
        """Test recording vector store operations"""
        manager = MetricsManager()
        
        operation_type = "store"
        item_count = 100
        duration = 5.5
        
        manager.record_vector_operation(operation_type, item_count, duration)
        
        vector_metrics = manager.metrics['vector_db']
        self.assertEqual(vector_metrics['store_operations'], 1)
        self.assertEqual(vector_metrics['total_vectors_stored'], item_count)
        self.assertEqual(vector_metrics['total_store_time'], duration)
        self.assertEqual(vector_metrics['avg_store_time'], duration)

    def test_record_vector_operation_query(self):
        """Test recording vector query operations"""
        manager = MetricsManager()
        
        operation_type = "query"
        item_count = 50
        duration = 2.3
        
        manager.record_vector_operation(operation_type, item_count, duration)
        
        vector_metrics = manager.metrics['vector_db']
        self.assertEqual(vector_metrics['query_operations'], 1)
        self.assertEqual(vector_metrics['total_query_time'], duration)
        self.assertEqual(vector_metrics['avg_query_time'], duration)

    def test_record_vector_operation_multiple_stores(self):
        """Test multiple vector store operations"""
        manager = MetricsManager()
        
        # First store operation
        manager.record_vector_operation("store", 100, 5.0)
        
        # Second store operation
        manager.record_vector_operation("store", 200, 7.0)
        
        vector_metrics = manager.metrics['vector_db']
        self.assertEqual(vector_metrics['store_operations'], 2)
        self.assertEqual(vector_metrics['total_vectors_stored'], 300)
        self.assertEqual(vector_metrics['total_store_time'], 12.0)
        self.assertEqual(vector_metrics['avg_store_time'], 6.0)

    def test_record_vector_operation_multiple_queries(self):
        """Test multiple vector query operations"""
        manager = MetricsManager()
        
        # First query
        manager.record_vector_operation("query", 10, 1.0)
        
        # Second query
        manager.record_vector_operation("query", 20, 3.0)
        
        vector_metrics = manager.metrics['vector_db']
        self.assertEqual(vector_metrics['query_operations'], 2)
        self.assertEqual(vector_metrics['total_query_time'], 4.0)
        self.assertEqual(vector_metrics['avg_query_time'], 2.0)

    # ==================== Error Recording Tests ====================

    def test_record_error_new_type(self):
        """Test recording error of new type"""
        manager = MetricsManager()
        
        error_type = "ConnectionError"
        error_message = "Failed to connect to database"
        
        manager.record_error(error_type, error_message)
        
        error_metrics = manager.metrics['errors']
        self.assertEqual(error_metrics['total'], 1)
        self.assertIn(error_type, error_metrics['by_type'])
        
        type_metrics = error_metrics['by_type'][error_type]
        self.assertEqual(type_metrics['count'], 1)
        self.assertIn(error_message, type_metrics['messages'])

    def test_record_error_existing_type(self):
        """Test recording error of existing type"""
        manager = MetricsManager()
        
        error_type = "ValidationError"
        first_message = "Invalid input format"
        second_message = "Missing required field"
        
        # Record first error
        manager.record_error(error_type, first_message)
        
        # Record second error of same type
        manager.record_error(error_type, second_message)
        
        error_metrics = manager.metrics['errors']
        self.assertEqual(error_metrics['total'], 2)
        
        type_metrics = error_metrics['by_type'][error_type]
        self.assertEqual(type_metrics['count'], 2)
        self.assertIn(first_message, type_metrics['messages'])
        self.assertIn(second_message, type_metrics['messages'])

    def test_record_multiple_error_types(self):
        """Test recording multiple error types"""
        manager = MetricsManager()
        
        # Record different error types
        manager.record_error("NetworkError", "Timeout occurred")
        manager.record_error("ParseError", "Invalid JSON")
        manager.record_error("NetworkError", "Connection refused")
        
        error_metrics = manager.metrics['errors']
        self.assertEqual(error_metrics['total'], 3)
        self.assertEqual(len(error_metrics['by_type']), 2)
        
        # Check NetworkError count
        self.assertEqual(error_metrics['by_type']['NetworkError']['count'], 2)
        
        # Check ParseError count
        self.assertEqual(error_metrics['by_type']['ParseError']['count'], 1)

    # ==================== Cost Update Tests ====================

    def test_update_total_cost(self):
        """Test _update_total_cost method"""
        manager = MetricsManager()
        
        # Set some LLM costs
        manager.metrics['costs']['llm']['total_cost'] = 10.50
        manager.metrics['costs']['llm']['total_cost_inr'] = 877.5
        
        manager._update_total_cost()
        
        self.assertEqual(manager.metrics['costs']['total_estimated_cost'], 10.50)
        self.assertEqual(manager.metrics['costs']['total_estimated_cost_inr'], 877.5)

    # ==================== CloudWatch Integration Tests ====================

    def test_send_to_cloudwatch_success(self):
        """Test successful CloudWatch metric sending"""
        mock_cloudwatch = Mock()
        self.mock_boto3.client.return_value = mock_cloudwatch
        
        manager = MetricsManager()
        manager.cloudwatch = mock_cloudwatch
        
        metric_type = "TestMetric"
        metrics_data = {
            'TestValue': 123.45,
            'TestCount': 10,
            'TestString': 'ignored'  # Should be filtered out
        }
        
        manager._send_to_cloudwatch(metric_type, metrics_data)
        
        # Verify CloudWatch was called
        mock_cloudwatch.put_metric_data.assert_called_once()
        call_args = mock_cloudwatch.put_metric_data.call_args[1]
        
        self.assertEqual(call_args['Namespace'], f"StorySense/{manager.app_name}")
        self.assertEqual(len(call_args['MetricData']), 2)  # Only numeric values

    def test_send_to_cloudwatch_no_client(self):
        """Test CloudWatch sending when client is None"""
        manager = MetricsManager()
        manager.cloudwatch = None
        
        # Should not raise exception
        manager._send_to_cloudwatch("Test", {'value': 123})

    def test_send_to_cloudwatch_exception(self):
        """Test CloudWatch sending exception handling"""
        mock_cloudwatch = Mock()
        mock_cloudwatch.put_metric_data.side_effect = Exception("CloudWatch error")
        
        manager = MetricsManager()
        manager.cloudwatch = mock_cloudwatch
        
        with patch('src.metrics.metrics_manager.logging') as mock_logging:
            manager._send_to_cloudwatch("Test", {'value': 123})
            
            mock_logging.warning.assert_called_once()

    # ==================== Metrics Summary Tests ====================

    def test_get_metrics_summary_no_stories(self):
        """Test metrics summary with no user stories"""
        manager = MetricsManager()
        
        summary = manager.get_metrics_summary()
        
        self.assertIn('start_time', summary)
        self.assertIn('end_time', summary)
        self.assertIn('total_duration', summary)
        self.assertEqual(summary['story_count'], 0)
        self.assertEqual(summary['avg_story_time'], 0)

    def test_get_metrics_summary_with_stories(self):
        """Test metrics summary with user stories"""
        manager = MetricsManager()
        
        # Add some user stories
        manager.record_user_story_metrics("US001", 5.0, 2, "high", 8.5)
        manager.record_user_story_metrics("US002", 3.0, 1, "medium", 7.0)
        
        # Add some LLM calls
        manager.record_llm_call("analysis", 1000, 500, 2.0, "claude-3-sonnet")
        
        summary = manager.get_metrics_summary()
        
        self.assertEqual(summary['story_count'], 2)
        self.assertGreater(summary['avg_story_time'], 0)
        self.assertEqual(summary['llm_calls'], 1)
        self.assertEqual(summary['llm_tokens'], 1500)
        self.assertGreater(summary['total_estimated_cost'], 0)

    def test_get_metrics_summary_end_time_already_set(self):
        """Test metrics summary when end_time is already set"""
        manager = MetricsManager()
        
        # Pre-set end time
        manager.metrics['end_time'] = "2024-01-01T15:00:00"
        manager.metrics['total_duration'] = 100.0
        
        summary = manager.get_metrics_summary()
        
        self.assertEqual(summary['end_time'], "2024-01-01T15:00:00")
        self.assertEqual(summary['total_duration'], 100.0)

    # ==================== Save Metrics Tests ====================

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.makedirs')
    def test_save_metrics_success(self, mock_makedirs, mock_file_open):
        """Test successful metrics saving"""
        manager = MetricsManager()
        
        # Add some test data
        manager.record_user_story_metrics("US001", 5.0)
        
        output_dir = "/test/output"
        result_file = manager.save_metrics(output_dir)
        
        # Verify directory creation
        mock_makedirs.assert_called_once_with(output_dir, exist_ok=True)
        
        # Verify file was opened for writing
        mock_file_open.assert_called_once()
        
        # Verify file path format
        self.assertIn("metrics_", result_file)
        self.assertIn(".json", result_file)

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.makedirs')
    def test_save_metrics_default_directory(self, mock_makedirs, mock_file_open):
        """Test metrics saving with default directory"""
        manager = MetricsManager()
        
        result_file = manager.save_metrics()
        
        # Verify default directory was used
        mock_makedirs.assert_called_once_with("../Output/Metrics", exist_ok=True)

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.makedirs')
    def test_save_metrics_end_time_calculation(self, mock_makedirs, mock_file_open):
        """Test that save_metrics calculates end_time if not set"""
        manager = MetricsManager()
        
        # Ensure end_time is not set
        self.assertIsNone(manager.metrics['end_time'])
        
        manager.save_metrics()
        
        # Verify end_time and duration were set
        self.assertIsNotNone(manager.metrics['end_time'])
        self.assertGreater(manager.metrics['total_duration'], 0)

    # ==================== Destructor Tests ====================

    def test_destructor_cleanup(self):
        """Test __del__ method cleanup"""
        manager = MetricsManager()
        
        # Mock thread
        mock_thread = Mock()
        mock_thread.is_alive.return_value = True
        manager.monitor_thread = mock_thread
        
        # Call destructor
        manager.__del__()
        
        # Verify cleanup
        self.assertTrue(manager.stop_monitoring)
        mock_thread.join.assert_called_once_with(timeout=1.0)

    def test_destructor_no_thread(self):
        """Test __del__ method when thread doesn't exist"""
        manager = MetricsManager()
        
        # Remove monitor_thread attribute
        delattr(manager, 'monitor_thread')
        
        # Should not raise exception
        manager.__del__()
        
        self.assertTrue(manager.stop_monitoring)

    def test_destructor_dead_thread(self):
        """Test __del__ method when thread is already dead"""
        manager = MetricsManager()
        
        # Mock dead thread
        mock_thread = Mock()
        mock_thread.is_alive.return_value = False
        manager.monitor_thread = mock_thread
        
        manager.__del__()
        
        # join should not be called on dead thread
        mock_thread.join.assert_not_called()

    # ==================== Integration Tests ====================

    def test_full_workflow_integration(self):
        """Test complete workflow integration"""
        manager = MetricsManager(app_name="IntegrationTest")
        
        # Record various metrics
        manager.record_user_story_metrics("US001", 5.0, 2, "high", 8.5)
        manager.record_batch_metrics(1, 10, 50.0)
        manager.record_llm_call("analysis", 1000, 500, 2.0, "claude-3-sonnet")
        manager.record_vector_operation("store", 100, 3.0)
        manager.record_error("TestError", "Test error message")
        
        # Get summary
        summary = manager.get_metrics_summary()
        
        # Verify all metrics are present
        self.assertEqual(summary['story_count'], 1)
        self.assertEqual(summary['llm_calls'], 1)
        self.assertEqual(summary['error_count'], 1)
        self.assertGreater(summary['total_estimated_cost'], 0)
        
        # Verify batch metrics
        self.assertIn('1', manager.metrics['batches'])
        
        # Verify vector metrics
        self.assertEqual(manager.metrics['vector_db']['store_operations'], 1)

    # ==================== Edge Cases and Error Handling ====================

    def test_record_llm_call_inr_cost_initialization(self):
        """Test INR cost initialization in record_llm_call"""
        manager = MetricsManager()
        
        # Remove INR cost keys to test initialization
        del manager.metrics['costs']['llm']['total_cost_inr']
        del manager.metrics['costs']['llm']['input_cost_inr']
        del manager.metrics['costs']['llm']['output_cost_inr']
        
        manager.record_llm_call("test", 1000, 500, 1.0, "claude-3-sonnet")
        
        # Verify INR costs were initialized and updated
        self.assertIn('total_cost_inr', manager.metrics['costs']['llm'])
        self.assertGreater(manager.metrics['costs']['llm']['total_cost_inr'], 0)

    def test_record_llm_call_existing_call_type(self):
        """Test recording LLM call with existing call type (covers branch 253->263)"""
        manager = MetricsManager()
        
        call_type = "existing_type"
        
        # First call - creates the call type
        manager.record_llm_call(call_type, 500, 250, 1.0, "claude-3-sonnet")
        
        # Second call - uses existing call type (this covers the missing branch)
        manager.record_llm_call(call_type, 800, 400, 1.5, "claude-3-sonnet")
        
        # Verify the call type metrics were updated correctly
        call_type_metrics = manager.metrics['llm']['calls_by_type'][call_type]
        self.assertEqual(call_type_metrics['calls'], 2)
        self.assertEqual(call_type_metrics['input_tokens'], 1300)
        self.assertEqual(call_type_metrics['output_tokens'], 650)

    def test_save_metrics_with_existing_end_time(self):
        """Test save_metrics when end_time is already set (covers branch 427->432)"""
        manager = MetricsManager()
        
        # Pre-set end_time to test the else branch
        manager.metrics['end_time'] = "2024-01-01T15:00:00"
        manager.metrics['total_duration'] = 100.0
        
        with patch('builtins.open', mock_open()) as mock_file_open, \
             patch('os.makedirs') as mock_makedirs:
            
            result_file = manager.save_metrics()
            
            # Verify that end_time and total_duration were NOT recalculated
            self.assertEqual(manager.metrics['end_time'], "2024-01-01T15:00:00")
            self.assertEqual(manager.metrics['total_duration'], 100.0)
            
            # Verify file operations still occurred
            mock_makedirs.assert_called_once()
            mock_file_open.assert_called_once()

    def test_cost_calculation_zero_tokens(self):
        """Test cost calculation with zero tokens"""
        manager = MetricsManager()
        
        cost_data = manager._calculate_llm_cost("claude-3-sonnet", 0, 0)
        
        self.assertEqual(cost_data['input_cost'], 0.0)
        self.assertEqual(cost_data['output_cost'], 0.0)
        self.assertEqual(cost_data['total_cost'], 0.0)
        self.assertEqual(cost_data['total_cost_inr'], 0.0)

    def test_vector_operation_invalid_type(self):
        """Test vector operation with invalid operation type"""
        manager = MetricsManager()
        
        # Should not crash with invalid operation type
        manager.record_vector_operation("invalid_type", 10, 1.0)
        
        # Verify no store or query operations were recorded
        self.assertEqual(manager.metrics['vector_db']['store_operations'], 0)
        self.assertEqual(manager.metrics['vector_db']['query_operations'], 0)

    @patch('src.metrics.metrics_manager.json.dump')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.makedirs')
    def test_save_metrics_json_serialization(self, mock_makedirs, mock_file_open, mock_json_dump):
        """Test JSON serialization in save_metrics"""
        manager = MetricsManager()
        
        manager.save_metrics()
        
        # Verify json.dump was called with correct parameters
        mock_json_dump.assert_called_once()
        call_args = mock_json_dump.call_args
        self.assertEqual(call_args[0][0], manager.metrics)  # First arg is metrics
        self.assertEqual(call_args[1]['indent'], 2)
        self.assertEqual(call_args[1]['default'], str)


if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Run the tests
    unittest.main(verbosity=2)
