import os
import json
import logging
from datetime import datetime

import boto3
# from src.configuration_handler.env_manager import EnvManager


class MetricsReporter:
    def __init__(self, collection_name='default', metrics_manager=None):
        self.collection_name = collection_name
        self.metrics_manager = metrics_manager

        self.metrics = {
            'vector_store_operations': 0,
            'vector_query_operations': 0,
            'total_store_time': 0,
            'total_query_time': 0,
            'avg_store_time': 0,
            'avg_query_time': 0,
            'total_vectors_stored': 0,
            'total_queries': 0,
            'query_latencies': [],
            'store_latencies': []
        }

        # self.env_manager = EnvManager()
        self.cloudwatch = None
        try:
            self.cloudwatch = boto3.client('cloudwatch',
                                           region_name=os.getenv('AWS_REGION'))
            logging.info("CloudWatch client initialized for PGVector monitoring")
        except Exception:
            logging.warning("Could not initialize CloudWatch client for MetricsReporter")

    def record_vector_operation(self, operation_type, item_count, duration):
        # Update local metrics
        if operation_type == 'query':
            self.metrics['vector_query_operations'] += 1
            self.metrics['total_query_time'] += duration
            self.metrics['total_queries'] += item_count
            self.metrics['query_latencies'].append(duration)
            self.metrics['avg_query_time'] = (self.metrics['total_query_time'] /
                                             max(1, len(self.metrics['query_latencies'])))
        elif operation_type == 'store':
            self.metrics['vector_store_operations'] += 1
            self.metrics['total_store_time'] += duration
            self.metrics['total_vectors_stored'] += item_count
            self.metrics['store_latencies'].append(duration)
            self.metrics['avg_store_time'] = (self.metrics['total_store_time'] /
                                            max(1, len(self.metrics['store_latencies'])))

        # Record via external metrics manager if provided
        if self.metrics_manager:
            try:
                self.metrics_manager.record_vector_operation(operation_type=operation_type,
                                                              item_count=item_count,
                                                              duration=duration)
            except Exception as e:
                logging.warning(f"Failed to record metrics in MetricsManager: {e}")

        # Send to CloudWatch (best-effort)
        self._send_metrics_to_cloudwatch(operation_type, duration, item_count)

    def _send_metrics_to_cloudwatch(self, operation_type, duration, item_count):
        if not self.cloudwatch:
            return
        try:
            namespace = "StorySense/PGVector"
            timestamp = datetime.now()

            metrics_data = [
                {
                    'MetricName': f'{operation_type}Duration',
                    'Dimensions': [
                        {'Name': 'Collection', 'Value': self.collection_name},
                        {'Name': 'Operation', 'Value': operation_type}
                    ],
                    'Value': duration,
                    'Unit': 'Seconds',
                    'Timestamp': timestamp
                },
                {
                    'MetricName': f'{operation_type}ItemCount',
                    'Dimensions': [
                        {'Name': 'Collection', 'Value': self.collection_name},
                        {'Name': 'Operation', 'Value': operation_type}
                    ],
                    'Value': item_count,
                    'Unit': 'Count',
                    'Timestamp': timestamp
                }
            ]

            self.cloudwatch.put_metric_data(Namespace=namespace, MetricData=metrics_data)
            logging.info(f"Sent {operation_type} metrics to CloudWatch: duration={duration:.2f}s, items={item_count}")
        except Exception as e:
            logging.warning(f"Failed to send metrics to CloudWatch: {e}")
            if self.metrics_manager:
                try:
                    self.metrics_manager.record_error('cloudwatch_error', str(e))
                except Exception:
                    pass

    def get_metrics(self):
        return self.metrics

    def save_metrics(self, output_dir="../Output/Metrics"):
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = f"{output_dir}/pgvector_metrics_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        logging.info(f"PGVector metrics saved to {metrics_file}")
        return metrics_file
