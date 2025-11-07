import pytest
import os
import json
from unittest.mock import Mock, patch, MagicMock, ANY, mock_open
from datetime import datetime
from src.context_handler.context_storage_handler.metrics_reporter import MetricsReporter


class TestMetricsReporter:
    @pytest.fixture
    def metrics_reporter(self, metrics_manager_mock):
        """Create a MetricsReporter instance with mocked dependencies"""
        with patch('boto3.client') as mock_boto_client:
            reporter = MetricsReporter(collection_name='test_collection', metrics_manager=metrics_manager_mock)
            reporter.cloudwatch = mock_boto_client.return_value
            return reporter

    def test_initialization(self, metrics_reporter, metrics_manager_mock):
        """Test initialization of MetricsReporter"""
        assert metrics_reporter.collection_name == 'test_collection'
        assert metrics_reporter.metrics_manager == metrics_manager_mock
        assert metrics_reporter.metrics['vector_store_operations'] == 0
        assert metrics_reporter.metrics['vector_query_operations'] == 0
        assert metrics_reporter.metrics['total_store_time'] == 0
        assert metrics_reporter.metrics['total_query_time'] == 0
        assert metrics_reporter.metrics['avg_store_time'] == 0
        assert metrics_reporter.metrics['avg_query_time'] == 0
        assert metrics_reporter.metrics['total_vectors_stored'] == 0
        assert metrics_reporter.metrics['total_queries'] == 0
        assert metrics_reporter.metrics['query_latencies'] == []
        assert metrics_reporter.metrics['store_latencies'] == []

    def test_initialization_without_cloudwatch(self):
        """Test initialization when CloudWatch client cannot be created"""
        with patch('boto3.client', side_effect=Exception("AWS Error")):
            with patch('logging.warning') as mock_log:
                reporter = MetricsReporter(collection_name='test_collection')
                assert reporter.cloudwatch is None
                mock_log.assert_called_once_with("Could not initialize CloudWatch client for MetricsReporter")

    def test_record_vector_operation_query(self, metrics_reporter, metrics_manager_mock):
        """Test recording a query operation"""
        metrics_reporter.record_vector_operation('query', 5, 1.5)

        # Check internal metrics were updated
        assert metrics_reporter.metrics['vector_query_operations'] == 1
        assert metrics_reporter.metrics['total_query_time'] == 1.5
        assert metrics_reporter.metrics['total_queries'] == 5
        assert metrics_reporter.metrics['query_latencies'] == [1.5]
        assert metrics_reporter.metrics['avg_query_time'] == 1.5

        # Check metrics manager was called
        metrics_manager_mock.record_vector_operation.assert_called_once_with(
            operation_type='query',
            item_count=5,
            duration=1.5
        )

        # Check CloudWatch metrics were sent
        metrics_reporter.cloudwatch.put_metric_data.assert_called_once()

    def test_record_vector_operation_store(self, metrics_reporter, metrics_manager_mock):
        """Test recording a store operation"""
        metrics_reporter.record_vector_operation('store', 10, 2.0)

        # Check internal metrics were updated
        assert metrics_reporter.metrics['vector_store_operations'] == 1
        assert metrics_reporter.metrics['total_store_time'] == 2.0
        assert metrics_reporter.metrics['total_vectors_stored'] == 10
        assert metrics_reporter.metrics['store_latencies'] == [2.0]
        assert metrics_reporter.metrics['avg_store_time'] == 2.0

        # Check metrics manager was called
        metrics_manager_mock.record_vector_operation.assert_called_once_with(
            operation_type='store',
            item_count=10,
            duration=2.0
        )

        # Check CloudWatch metrics were sent
        metrics_reporter.cloudwatch.put_metric_data.assert_called_once()

    def test_record_vector_operation_multiple_queries(self, metrics_reporter):
        """Test recording multiple query operations"""
        metrics_reporter.record_vector_operation('query', 5, 1.5)
        metrics_reporter.record_vector_operation('query', 3, 0.5)

        # Check internal metrics were updated correctly
        assert metrics_reporter.metrics['vector_query_operations'] == 2
        assert metrics_reporter.metrics['total_query_time'] == 2.0
        assert metrics_reporter.metrics['total_queries'] == 8
        assert metrics_reporter.metrics['query_latencies'] == [1.5, 0.5]
        assert metrics_reporter.metrics['avg_query_time'] == 1.0  # (1.5 + 0.5) / 2

    def test_record_vector_operation_multiple_stores(self, metrics_reporter):
        """Test recording multiple store operations"""
        metrics_reporter.record_vector_operation('store', 10, 2.0)
        metrics_reporter.record_vector_operation('store', 5, 1.0)

        # Check internal metrics were updated correctly
        assert metrics_reporter.metrics['vector_store_operations'] == 2
        assert metrics_reporter.metrics['total_store_time'] == 3.0
        assert metrics_reporter.metrics['total_vectors_stored'] == 15
        assert metrics_reporter.metrics['store_latencies'] == [2.0, 1.0]
        assert metrics_reporter.metrics['avg_store_time'] == 1.5  # (2.0 + 1.0) / 2

    def test_record_vector_operation_metrics_manager_error(self, metrics_reporter, metrics_manager_mock):
        """Test handling of errors from metrics manager"""
        metrics_manager_mock.record_vector_operation.side_effect = Exception("Metrics error")

        # Should not raise exception
        with patch('logging.warning') as mock_log:
            metrics_reporter.record_vector_operation('query', 5, 1.5)

            # Internal metrics should still be updated
            assert metrics_reporter.metrics['vector_query_operations'] == 1
            mock_log.assert_called_once_with("Failed to record metrics in MetricsManager: Metrics error")

    def test_record_vector_operation_cloudwatch_error(self, metrics_reporter):
        """Test handling of errors when sending to CloudWatch"""
        metrics_reporter.cloudwatch.put_metric_data.side_effect = Exception("CloudWatch error")

        # Should not raise exception
        with patch('logging.warning') as mock_log:
            metrics_reporter.record_vector_operation('query', 5, 1.5)

            # Internal metrics should still be updated
            assert metrics_reporter.metrics['vector_query_operations'] == 1
            mock_log.assert_called_once_with("Failed to send metrics to CloudWatch: CloudWatch error")

    def test_record_vector_operation_cloudwatch_error_with_metrics_manager(self, metrics_reporter,
                                                                           metrics_manager_mock):
        """Test CloudWatch error handling with metrics manager"""
        metrics_reporter.cloudwatch.put_metric_data.side_effect = Exception("CloudWatch error")

        # Should record error in metrics manager
        metrics_reporter.record_vector_operation('query', 5, 1.5)

        # Check error was recorded
        metrics_manager_mock.record_error.assert_called_once_with('cloudwatch_error', "CloudWatch error")

    def test_record_vector_operation_cloudwatch_error_without_metrics_manager(self):
        """Test CloudWatch error handling without metrics manager"""
        with patch('boto3.client') as mock_boto_client:
            reporter = MetricsReporter(collection_name='test_collection', metrics_manager=None)
            reporter.cloudwatch = mock_boto_client.return_value
            reporter.cloudwatch.put_metric_data.side_effect = Exception("CloudWatch error")

            # Should not raise exception even without metrics manager
            with patch('logging.warning') as mock_log:
                reporter.record_vector_operation('query', 5, 1.5)
                mock_log.assert_called_once_with("Failed to send metrics to CloudWatch: CloudWatch error")

    def test_send_metrics_to_cloudwatch_query(self, metrics_reporter):
        """Test sending query metrics to CloudWatch"""
        metrics_reporter._send_metrics_to_cloudwatch('query', 1.5, 5)

        # Check CloudWatch was called with correct parameters
        metrics_reporter.cloudwatch.put_metric_data.assert_called_once()
        call_args = metrics_reporter.cloudwatch.put_metric_data.call_args[1]

        assert call_args['Namespace'] == 'StorySense/PGVector'
        assert len(call_args['MetricData']) == 2

        # Check first metric (duration)
        assert call_args['MetricData'][0]['MetricName'] == 'queryDuration'
        assert call_args['MetricData'][0]['Value'] == 1.5
        assert call_args['MetricData'][0]['Unit'] == 'Seconds'

        # Check second metric (item count)
        assert call_args['MetricData'][1]['MetricName'] == 'queryItemCount'
        assert call_args['MetricData'][1]['Value'] == 5
        assert call_args['MetricData'][1]['Unit'] == 'Count'

        # Check dimensions
        for metric in call_args['MetricData']:
            assert len(metric['Dimensions']) == 2
            assert metric['Dimensions'][0]['Name'] == 'Collection'
            assert metric['Dimensions'][0]['Value'] == 'test_collection'
            assert metric['Dimensions'][1]['Name'] == 'Operation'
            assert metric['Dimensions'][1]['Value'] == 'query'

    def test_send_metrics_to_cloudwatch_store(self, metrics_reporter):
        """Test sending store metrics to CloudWatch"""
        metrics_reporter._send_metrics_to_cloudwatch('store', 2.0, 10)

        # Check CloudWatch was called with correct parameters
        metrics_reporter.cloudwatch.put_metric_data.assert_called_once()
        call_args = metrics_reporter.cloudwatch.put_metric_data.call_args[1]

        assert call_args['Namespace'] == 'StorySense/PGVector'
        assert len(call_args['MetricData']) == 2

        # Check first metric (duration)
        assert call_args['MetricData'][0]['MetricName'] == 'storeDuration'
        assert call_args['MetricData'][0]['Value'] == 2.0
        assert call_args['MetricData'][0]['Unit'] == 'Seconds'

        # Check second metric (item count)
        assert call_args['MetricData'][1]['MetricName'] == 'storeItemCount'
        assert call_args['MetricData'][1]['Value'] == 10
        assert call_args['MetricData'][1]['Unit'] == 'Count'

    def test_send_metrics_to_cloudwatch_no_client(self):
        """Test sending metrics when CloudWatch client is not available"""
        reporter = MetricsReporter(collection_name='test_collection')
        reporter.cloudwatch = None

        # Should return early without error
        reporter._send_metrics_to_cloudwatch('query', 1.5, 5)
        # No assertions needed - just checking it doesn't raise an exception

    def test_get_metrics(self, metrics_reporter):
        """Test getting metrics"""
        # Record some operations
        metrics_reporter.record_vector_operation('query', 5, 1.5)
        metrics_reporter.record_vector_operation('store', 10, 2.0)

        # Get metrics
        metrics = metrics_reporter.get_metrics()

        # Check metrics are returned correctly
        assert metrics['vector_query_operations'] == 1
        assert metrics['vector_store_operations'] == 1
        assert metrics['total_query_time'] == 1.5
        assert metrics['total_store_time'] == 2.0
        assert metrics['total_queries'] == 5
        assert metrics['total_vectors_stored'] == 10
        assert metrics['query_latencies'] == [1.5]
        assert metrics['store_latencies'] == [2.0]
        assert metrics['avg_query_time'] == 1.5
        assert metrics['avg_store_time'] == 2.0

    def test_save_metrics(self, metrics_reporter, temp_dir):
        """Test saving metrics to a file"""
        # Record some operations
        metrics_reporter.record_vector_operation('query', 5, 1.5)
        metrics_reporter.record_vector_operation('store', 10, 2.0)

        # Save metrics
        with patch('datetime.now') as mock_now:
            mock_now.return_value.strftime.return_value = "20240101_120000"
            output_file = metrics_reporter.save_metrics(output_dir=temp_dir)

        # Check file was created
        expected_file = f"{temp_dir}/pgvector_metrics_20240101_120000.json"
        assert output_file == expected_file
        assert os.path.exists(expected_file)

        # Check file contents
        with open(expected_file, 'r') as f:
            saved_metrics = json.load(f)

        assert saved_metrics['vector_query_operations'] == 1
        assert saved_metrics['vector_store_operations'] == 1
        assert saved_metrics['total_queries'] == 5
        assert saved_metrics['total_vectors_stored'] == 10

    def test_save_metrics_directory_creation(self, metrics_reporter):
        """Test that save_metrics creates the output directory if it doesn't exist"""
        # Use a non-existent directory
        output_dir = "/tmp/nonexistent_metrics_dir"
        if os.path.exists(output_dir):
            import shutil
            shutil.rmtree(output_dir)

        with patch('os.makedirs') as mock_makedirs, \
                patch('builtins.open', mock_open()), \
                patch('json.dump'), \
                patch('logging.info'):
            metrics_reporter.save_metrics(output_dir=output_dir)
            mock_makedirs.assert_called_once_with(output_dir, exist_ok=True)

    def test_save_metrics_error_handling(self, metrics_reporter):
        """Test error handling when saving metrics"""
        with patch('os.makedirs', side_effect=Exception("Directory error")), \
                patch('logging.info') as mock_log:
            # Should not raise exception
            metrics_reporter.save_metrics(output_dir="/bad/path")
            mock_log.assert_not_called()  # Log info shouldn't be called on error

