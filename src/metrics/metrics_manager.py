import time
import json
import os
import logging
import boto3
from datetime import datetime
import psutil
import threading
import requests


class MetricsManager:
    """Centralized metrics collection and reporting for StorySense"""

    def __init__(self, app_name="StorySense"):
        self.app_name = app_name
        self.start_time = time.time()
        self.metrics = {
            'app': app_name,
            'version': '1.0.0',
            'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
            'end_time': None,
            'total_duration': 0,
            'user_stories': {},
            'batches': {},
            'llm': {
                'total_calls': 0,
                'total_tokens': 0,
                'input_tokens': 0,
                'output_tokens': 0,
                'total_latency': 0,
                'avg_latency': 0,
                'calls_by_type': {},
                'guardrail_interventions': 0
            },
            'vector_db': {
                'store_operations': 0,
                'query_operations': 0,
                'total_vectors_stored': 0,
                'total_query_time': 0,
                'total_store_time': 0,
                'avg_query_time': 0,
                'avg_store_time': 0
            },
            'system': {
                'start_memory': psutil.Process().memory_info().rss / (1024 * 1024),  # MB
                'peak_memory': 0,
                'end_memory': 0,
                'cpu_percent': 0
            },
            'errors': {
                'total': 0,
                'by_type': {}
            },
            'costs': {
                'llm': {
                    'total_cost': 0,
                    'input_cost': 0,
                    'output_cost': 0,
                    'total_cost_inr': 0,
                    'input_cost_inr': 0,
                    'output_cost_inr': 0,
                    'by_model': {}
                },
                'total_estimated_cost': 0,
                'total_estimated_cost_inr': 0
            }
        }

        # Claude pricing (per 1K tokens)
        self.claude_pricing = {
            'claude-3-sonnet': {
                'input': 0.003,  # \$0.003 per 1K input tokens
                'output': 0.015  # \$0.015 per 1K output tokens
            },
            'claude-3-5-sonnet': {
                'input': 0.003,  # \$0.003 per 1K input tokens
                'output': 0.015  # \$0.015 per 1K output tokens
            },
            'claude-3-opus': {
                'input': 0.015,  # \$0.015 per 1K input tokens
                'output': 0.075  # \$0.075 per 1K output tokens
            },
            'us.anthropic.claude-3-5-sonnet-20241022-v2:0': {
                'input': 0.003,
                'output': 0.015
            },
            'claude-3-7-sonnet': {
                'input': 0.003,  # \$0.003 per 1K input tokens
                'output': 0.015  # \$0.015 per 1K output tokens
            },
            'claude-3-sonnet-4': {
                'input': 0.003,  # \$0.003 per 1K input tokens
                'output': 0.015  # \$0.015 per 1K output tokens
            }
        }

        # USD to INR conversion rate (update as needed)
        self.usd_to_inr_rate = 83.5  # Example rate, update to current rate

        # Start CPU monitoring thread
        self.stop_monitoring = False
        self.monitor_thread = threading.Thread(target=self._monitor_system_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

        # Initialize CloudWatch client if AWS credentials are available
        self.cloudwatch = None
        try:
            self.cloudwatch = boto3.client('cloudwatch', region_name=os.getenv('AWS_REGION'))
            logging.info("CloudWatch client initialized for metrics")
        except Exception as e:
            logging.warning(f"Could not initialize CloudWatch client: {e}")

    def _monitor_system_resources(self):
        """Background thread to monitor system resources"""
        while not self.stop_monitoring:
            try:
                # Update CPU usage
                self.metrics['system']['cpu_percent'] = psutil.cpu_percent(interval=1)

                # Update peak memory
                current_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
                self.metrics['system']['peak_memory'] = max(
                    self.metrics['system']['peak_memory'],
                    current_memory
                )

                # Sleep for a bit
                time.sleep(5)
            except Exception as e:
                logging.error(f"Error monitoring system resources: {e}")

    def record_user_story_metrics(self, us_id, processing_time, context_count=0,
                                  context_quality="none", overall_score=0):
        """Record metrics for a user story"""
        self.metrics['user_stories'][us_id] = {
            'processing_time': processing_time,
            'context_count': context_count,
            'context_quality': context_quality,
            'overall_score': overall_score
        }

        # Send to CloudWatch if available
        self._send_to_cloudwatch('UserStory', {
            'ProcessingTime': processing_time,
            'ContextCount': context_count,
            'OverallScore': overall_score
        })

    def record_batch_metrics(self, batch_num, story_count, processing_time):
        """Record metrics for a batch of user stories"""
        self.metrics['batches'][str(batch_num)] = {
            'story_count': story_count,
            'processing_time': processing_time,
            'stories_per_second': story_count / processing_time if processing_time > 0 else 0
        }

        # Send to CloudWatch if available
        self._send_to_cloudwatch('Batch', {
            'BatchNumber': batch_num,
            'StoryCount': story_count,
            'ProcessingTime': processing_time,
            'StoriesPerSecond': story_count / processing_time if processing_time > 0 else 0
        })

    def _calculate_llm_cost(self, model_name, input_tokens, output_tokens):
        """Calculate LLM cost with both USD and INR"""
        # Default pricing if model not found
        input_cost_per_1k = 0.003
        output_cost_per_1k = 0.015

        # Get model-specific pricing if available
        if model_name in self.claude_pricing:
            input_cost_per_1k = self.claude_pricing[model_name]['input']
            output_cost_per_1k = self.claude_pricing[model_name]['output']

        # Calculate costs in USD
        input_cost_usd = (input_tokens / 1000) * input_cost_per_1k
        output_cost_usd = (output_tokens / 1000) * output_cost_per_1k
        total_cost_usd = input_cost_usd + output_cost_usd

        # Calculate costs in INR
        input_cost_inr = input_cost_usd * self.usd_to_inr_rate
        output_cost_inr = output_cost_usd * self.usd_to_inr_rate
        total_cost_inr = total_cost_usd * self.usd_to_inr_rate

        return {
            'input_cost': input_cost_usd,
            'output_cost': output_cost_usd,
            'total_cost': total_cost_usd,
            'input_cost_inr': input_cost_inr,
            'output_cost_inr': output_cost_inr,
            'total_cost_inr': total_cost_inr
        }

    def record_llm_call(self, call_type, input_tokens, output_tokens, latency,
                        model_name="claude-3-sonnet", guardrail_triggered=False):
        """Record metrics for an LLM call with cost calculation in USD and INR"""
        self.metrics['llm']['total_calls'] += 1
        self.metrics['llm']['input_tokens'] += input_tokens
        self.metrics['llm']['output_tokens'] += output_tokens
        self.metrics['llm']['total_tokens'] += (input_tokens + output_tokens)
        self.metrics['llm']['total_latency'] += latency

        if guardrail_triggered:
            self.metrics['llm']['guardrail_interventions'] += 1

        # Calculate cost
        cost_data = self._calculate_llm_cost(model_name, input_tokens, output_tokens)

        # Update cost metrics
        self.metrics['costs']['llm']['total_cost'] += cost_data['total_cost']
        self.metrics['costs']['llm']['input_cost'] += cost_data['input_cost']
        self.metrics['costs']['llm']['output_cost'] += cost_data['output_cost']

        # Add INR costs
        if 'total_cost_inr' not in self.metrics['costs']['llm']:
            self.metrics['costs']['llm']['total_cost_inr'] = 0
            self.metrics['costs']['llm']['input_cost_inr'] = 0
            self.metrics['costs']['llm']['output_cost_inr'] = 0

        self.metrics['costs']['llm']['total_cost_inr'] += cost_data['total_cost_inr']
        self.metrics['costs']['llm']['input_cost_inr'] += cost_data['input_cost_inr']
        self.metrics['costs']['llm']['output_cost_inr'] += cost_data['output_cost_inr']

        # Track by model
        if model_name not in self.metrics['costs']['llm']['by_model']:
            self.metrics['costs']['llm']['by_model'][model_name] = {
                'calls': 0,
                'input_tokens': 0,
                'output_tokens': 0,
                'total_cost': 0,
                'input_cost': 0,
                'output_cost': 0,
                'total_cost_inr': 0,
                'input_cost_inr': 0,
                'output_cost_inr': 0
            }

        model_costs = self.metrics['costs']['llm']['by_model'][model_name]
        model_costs['calls'] += 1
        model_costs['input_tokens'] += input_tokens
        model_costs['output_tokens'] += output_tokens
        model_costs['total_cost'] += cost_data['total_cost']
        model_costs['input_cost'] += cost_data['input_cost']
        model_costs['output_cost'] += cost_data['output_cost']
        model_costs['total_cost_inr'] += cost_data['total_cost_inr']
        model_costs['input_cost_inr'] += cost_data['input_cost_inr']
        model_costs['output_cost_inr'] += cost_data['output_cost_inr']

        # Record by call type
        if call_type not in self.metrics['llm']['calls_by_type']:
            self.metrics['llm']['calls_by_type'][call_type] = {
                'calls': 0,
                'input_tokens': 0,
                'output_tokens': 0,
                'total_tokens': 0,
                'latency': 0,
                'guardrail_interventions': 0
            }

        self.metrics['llm']['calls_by_type'][call_type]['calls'] += 1
        self.metrics['llm']['calls_by_type'][call_type]['input_tokens'] += input_tokens
        self.metrics['llm']['calls_by_type'][call_type]['output_tokens'] += output_tokens
        self.metrics['llm']['calls_by_type'][call_type]['total_tokens'] += (input_tokens + output_tokens)
        self.metrics['llm']['calls_by_type'][call_type]['latency'] += latency

        if guardrail_triggered:
            self.metrics['llm']['calls_by_type'][call_type]['guardrail_interventions'] += 1

        # Update average latency
        self.metrics['llm']['avg_latency'] = (
                self.metrics['llm']['total_latency'] / self.metrics['llm']['total_calls']
        )

        # Update total estimated cost
        self._update_total_cost()

        # Send to CloudWatch if available
        self._send_to_cloudwatch('LLM', {
            'CallType': call_type,
            'InputTokens': input_tokens,
            'OutputTokens': output_tokens,
            'TotalTokens': input_tokens + output_tokens,
            'Latency': latency,
            'CostUSD': cost_data['total_cost'],
            'CostINR': cost_data['total_cost_inr']
        })

    def record_vector_operation(self, operation_type, item_count, duration):
        """Record metrics for vector database operations"""
        if operation_type == 'store':
            self.metrics['vector_db']['store_operations'] += 1
            self.metrics['vector_db']['total_vectors_stored'] += item_count
            self.metrics['vector_db']['total_store_time'] += duration
            self.metrics['vector_db']['avg_store_time'] = (
                    self.metrics['vector_db']['total_store_time'] / self.metrics['vector_db']['store_operations']
            )
        elif operation_type == 'query':
            self.metrics['vector_db']['query_operations'] += 1
            self.metrics['vector_db']['total_query_time'] += duration
            self.metrics['vector_db']['avg_query_time'] = (
                    self.metrics['vector_db']['total_query_time'] / self.metrics['vector_db']['query_operations']
            )

        # Send to CloudWatch if available
        self._send_to_cloudwatch('VectorDB', {
            'OperationType': operation_type,
            'ItemCount': item_count,
            'Duration': duration
        })

    def record_error(self, error_type, error_message):
        """Record an error"""
        self.metrics['errors']['total'] += 1

        if error_type not in self.metrics['errors']['by_type']:
            self.metrics['errors']['by_type'][error_type] = {
                'count': 0,
                'messages': []
            }

        self.metrics['errors']['by_type'][error_type]['count'] += 1
        self.metrics['errors']['by_type'][error_type]['messages'].append(error_message)

        # Send to CloudWatch if available
        self._send_to_cloudwatch('Error', {
            'ErrorType': error_type,
            'ErrorMessage': error_message
        })

    def _update_total_cost(self):
        """Update total estimated cost"""
        # LLM costs
        llm_cost = self.metrics['costs']['llm']['total_cost']
        llm_cost_inr = self.metrics['costs']['llm']['total_cost_inr']

        # Set total cost to just LLM cost (removing database costs)
        self.metrics['costs']['total_estimated_cost'] = llm_cost

        # Add INR total cost
        self.metrics['costs']['total_estimated_cost_inr'] = llm_cost_inr

    def _send_to_cloudwatch(self, metric_type, metrics_data):
        """Send metrics to CloudWatch if available"""
        if not self.cloudwatch:
            return

        try:
            namespace = f"StorySense/{self.app_name}"
            # Use UTC time to avoid timezone issues with CloudWatch
            timestamp = datetime.utcnow()

            metric_data = []
            for name, value in metrics_data.items():
                if isinstance(value, (int, float)):
                    metric_data.append({
                        'MetricName': name,
                        'Dimensions': [
                            {'Name': 'Type', 'Value': metric_type}
                        ],
                        'Value': value,
                        'Timestamp': timestamp
                    })

            if metric_data:
                self.cloudwatch.put_metric_data(
                    Namespace=namespace,
                    MetricData=metric_data
                )
        except Exception as e:
            logging.warning(f"Failed to send metrics to CloudWatch: {e}")

    def get_metrics_summary(self):
        """Get a summary of metrics for reporting"""
        # Calculate end time and duration if not already set
        if not self.metrics['end_time']:
            self.metrics['end_time'] = datetime.now().isoformat()
            self.metrics['total_duration'] = time.time() - self.start_time

        # Update system metrics
        self.metrics['system']['end_memory'] = psutil.Process().memory_info().rss / (1024 * 1024)

        # Create summary
        summary = {
            'start_time': self.metrics['start_time'],
            'end_time': self.metrics['end_time'],
            'total_duration': self.metrics['total_duration'],
            'story_count': len(self.metrics['user_stories']),
            'avg_story_time': self.metrics['total_duration'] / len(self.metrics['user_stories']) if self.metrics[
                'user_stories'] else 0,
            'llm_calls': self.metrics['llm']['total_calls'],
            'llm_tokens': self.metrics['llm']['total_tokens'],
            'llm_input_tokens': self.metrics['llm']['input_tokens'],
            'llm_output_tokens': self.metrics['llm']['output_tokens'],
            'llm_avg_latency': self.metrics['llm']['avg_latency'],
            'peak_memory_mb': self.metrics['system']['peak_memory'],
            'error_count': self.metrics['errors']['total'],
            'total_estimated_cost': self.metrics['costs']['total_estimated_cost'],
            'total_estimated_cost_inr': self.metrics['costs']['total_estimated_cost_inr'],
            'estimated_llm_cost': self.metrics['costs']['llm']['total_cost'],
            'estimated_llm_cost_inr': self.metrics['costs']['llm']['total_cost_inr'],
            'vector_queries': self.metrics['vector_db']['query_operations'],
            'cost_breakdown': {
                'llm': {
                    'input_cost': self.metrics['costs']['llm']['input_cost'],
                    'output_cost': self.metrics['costs']['llm']['output_cost'],
                    'input_cost_inr': self.metrics['costs']['llm']['input_cost_inr'],
                    'output_cost_inr': self.metrics['costs']['llm']['output_cost_inr']
                }
            },
            'system': self.metrics['system'],
            'batches': self.metrics['batches'],
            'vector_db': self.metrics['vector_db']
        }

        return summary

    def save_metrics(self, output_dir="../Output/Metrics"):
        """Save metrics to a JSON file"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = f"{output_dir}/metrics_{timestamp}.json"

        # Calculate end time and duration if not already set
        if not self.metrics['end_time']:
            self.metrics['end_time'] = datetime.now().isoformat()
            self.metrics['total_duration'] = time.time() - self.start_time

        # Update system metrics
        self.metrics['system']['end_memory'] = psutil.Process().memory_info().rss / (1024 * 1024)

        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)

        logging.info(f"Metrics saved to {metrics_file}")
        return metrics_file

    def __del__(self):
        """Clean up when object is destroyed"""
        self.stop_monitoring = True
        if hasattr(self, 'monitor_thread') and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)

# import time
# import json
# import os
# import logging
# import boto3
# import requests
# import numpy as np
# from datetime import datetime
# import psutil
# import threading
#
#
# class MetricsManager:
#     """Centralized metrics collection and reporting for StorySense"""
#
#     def __init__(self, app_name="StorySense"):
#         self.app_name = app_name
#         self.start_time = time.time()
#         self.metrics = {
#             'app': app_name,
#             'version': '1.0.0',
#             'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
#             'end_time': None,
#             'total_duration': 0,
#             'user_stories': {},
#             'batches': {},
#             'llm': {
#                 'total_calls': 0,
#                 'total_tokens': 0,
#                 'input_tokens': 0,
#                 'output_tokens': 0,
#                 'total_latency': 0,
#                 'avg_latency': 0,
#                 'calls_by_type': {},
#                 'guardrail_interventions': 0
#             },
#             'vector_db': {
#                 'store_operations': 0,
#                 'query_operations': 0,
#                 'total_vectors_stored': 0,
#                 'total_query_time': 0,
#                 'total_store_time': 0,
#                 'avg_query_time': 0,
#                 'avg_store_time': 0
#             },
#             'system': {
#                 'start_memory': psutil.Process().memory_info().rss / (1024 * 1024),  # MB
#                 'peak_memory': 0,
#                 'end_memory': 0,
#                 'cpu_percent': 0
#             },
#             'errors': {
#                 'total': 0,
#                 'by_type': {}
#             },
#             'costs': {
#                 'llm': {
#                     'total_cost': 0,
#                     'input_cost': 0,
#                     'output_cost': 0,
#                     'total_cost_inr': 0,
#                     'input_cost_inr': 0,
#                     'output_cost_inr': 0,
#                     'by_model': {}
#                 },
#                 'total_estimated_cost': 0,
#                 'total_estimated_cost_inr': 0
#             }
#         }
#
#         # Claude pricing (per 1K tokens)
#         self.claude_pricing = {
#             'claude-3-sonnet': {
#                 'input': 0.003,  # \$0.003 per 1K input tokens
#                 'output': 0.015  # \$0.015 per 1K output tokens
#             },
#             'claude-3-5-sonnet': {
#                 'input': 0.003,  # \$0.003 per 1K input tokens
#                 'output': 0.015  # \$0.015 per 1K output tokens
#             },
#             'claude-3-opus': {
#                 'input': 0.015,  # \$0.015 per 1K input tokens
#                 'output': 0.075  # \$0.075 per 1K output tokens
#             },
#             'anthropic.claude-3-sonnet-20240229-v1:0': {
#                 'input': 0.003,
#                 'output': 0.015
#             },
#             'claude-3-7-sonnet': {
#                 'input': 0.003,  # \$0.003 per 1K input tokens
#                 'output': 0.015  # \$0.015 per 1K output tokens
#             },
#             'claude-3-sonnet-4': {
#                 'input': 0.003,  # \$0.003 per 1K input tokens
#                 'output': 0.015  # \$0.015 per 1K output tokens
#             }
#         }
#
#         # USD to INR conversion rate (update as needed)
#         self.usd_to_inr_rate = 83.5  # Default fallback rate
#         self.update_currency_rate()  # Try to update from API
#         self.start_currency_rate_updater(24)  # Update every 24 hours
#
#         # Flag to control background threads
#         self.stop_monitoring = False
#
#         # Start CPU monitoring thread
#         self.monitor_thread = threading.Thread(target=self._monitor_system_resources)
#         self.monitor_thread.daemon = True
#         self.monitor_thread.start()
#
#         # Initialize CloudWatch client if AWS credentials are available
#         self.cloudwatch = None
#         try:
#             self.cloudwatch = boto3.client('cloudwatch', region_name=os.environ.get('AWS_REGION', 'us-east-1'))
#             logging.info("CloudWatch client initialized for metrics")
#         except Exception as e:
#             logging.warning(f"Could not initialize CloudWatch client: {e}")
#
#     def update_currency_rate(self):
#         """Update USD to INR conversion rate from a public API"""
#         try:
#             # Try multiple APIs in case one fails
#             apis = [
#                 "https://open.er-api.com/v6/latest/USD",
#                 "https://api.exchangerate.host/latest?base=USD",
#                 "https://api.exchangerate-api.com/v4/latest/USD"
#             ]
#
#             for api_url in apis:
#                 try:
#                     response = requests.get(api_url, timeout=5)
#                     if response.status_code == 200:
#                         data = response.json()
#
#                         # Different APIs have different response structures
#                         if 'rates' in data and 'INR' in data['rates']:
#                             self.usd_to_inr_rate = data['rates']['INR']
#                             logging.info(f"Updated USD to INR rate: {self.usd_to_inr_rate}")
#                             return True
#                 except Exception as e:
#                     logging.warning(f"Failed to fetch exchange rate from {api_url}: {e}")
#                     continue
#
#             # If all APIs fail, use the fallback rate
#             logging.warning("Could not update currency rate from any API, using default rate")
#             return False
#         except Exception as e:
#             logging.error(f"Error updating currency rate: {e}")
#             return False
#
#     def start_currency_rate_updater(self, interval_hours=24):
#         """Start a background thread to periodically update the currency rate"""
#
#         def update_periodically():
#             while not self.stop_monitoring:
#                 try:
#                     self.update_currency_rate()
#                     # Sleep for the specified interval
#                     for _ in range(int(interval_hours * 3600)):
#                         if self.stop_monitoring:
#                             break
#                         time.sleep(1)
#                 except Exception as e:
#                     logging.error(f"Error in currency rate updater: {e}")
#                     time.sleep(3600)  # Wait an hour before retrying
#
#         thread = threading.Thread(target=update_periodically, daemon=True)
#         thread.start()
#         logging.info(f"Currency rate updater started (interval: {interval_hours} hours)")
#
#     def _monitor_system_resources(self):
#         """Background thread to monitor system resources"""
#         while not self.stop_monitoring:
#             try:
#                 # Update CPU usage
#                 self.metrics['system']['cpu_percent'] = psutil.cpu_percent(interval=1)
#
#                 # Update peak memory
#                 current_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
#                 self.metrics['system']['peak_memory'] = max(
#                     self.metrics['system']['peak_memory'],
#                     current_memory
#                 )
#
#                 # Sleep for a bit
#                 time.sleep(5)
#             except Exception as e:
#                 logging.error(f"Error monitoring system resources: {e}")
#
#     def record_user_story_metrics(self, us_id, processing_time, context_count=0,
#                                   context_quality="none", overall_score=0):
#         """Record metrics for a user story"""
#         self.metrics['user_stories'][us_id] = {
#             'processing_time': processing_time,
#             'context_count': context_count,
#             'context_quality': context_quality,
#             'overall_score': overall_score
#         }
#
#         # Send to CloudWatch if available
#         self._send_to_cloudwatch('UserStory', {
#             'ProcessingTime': processing_time,
#             'ContextCount': context_count,
#             'OverallScore': overall_score
#         })
#
#     def record_batch_metrics(self, batch_num, story_count, processing_time):
#         """Record metrics for a batch of user stories"""
#         self.metrics['batches'][str(batch_num)] = {
#             'story_count': story_count,
#             'processing_time': processing_time,
#             'stories_per_second': story_count / processing_time if processing_time > 0 else 0
#         }
#
#         # Send to CloudWatch if available
#         self._send_to_cloudwatch('Batch', {
#             'BatchNumber': batch_num,
#             'StoryCount': story_count,
#             'ProcessingTime': processing_time,
#             'StoriesPerSecond': story_count / processing_time if processing_time > 0 else 0
#         })
#
#     def _calculate_llm_cost(self, model_name, input_tokens, output_tokens):
#         """Calculate LLM cost with both USD and INR"""
#         # Default pricing if model not found
#         input_cost_per_1k = 0.003
#         output_cost_per_1k = 0.015
#
#         # Get model-specific pricing if available
#         if model_name in self.claude_pricing:
#             input_cost_per_1k = self.claude_pricing[model_name]['input']
#             output_cost_per_1k = self.claude_pricing[model_name]['output']
#
#         # Calculate costs in USD
#         input_cost_usd = (input_tokens / 1000) * input_cost_per_1k
#         output_cost_usd = (output_tokens / 1000) * output_cost_per_1k
#         total_cost_usd = input_cost_usd + output_cost_usd
#
#         # Calculate costs in INR
#         input_cost_inr = input_cost_usd * self.usd_to_inr_rate
#         output_cost_inr = output_cost_usd * self.usd_to_inr_rate
#         total_cost_inr = total_cost_usd * self.usd_to_inr_rate
#
#         return {
#             'input_cost': input_cost_usd,
#             'output_cost': output_cost_usd,
#             'total_cost': total_cost_usd,
#             'input_cost_inr': input_cost_inr,
#             'output_cost_inr': output_cost_inr,
#             'total_cost_inr': total_cost_inr
#         }
#
#     def record_llm_call(self, call_type, input_tokens, output_tokens, latency,
#                         model_name="claude-3-sonnet", guardrail_triggered=False):
#         """Record metrics for an LLM call with cost calculation in USD and INR"""
#         self.metrics['llm']['total_calls'] += 1
#         self.metrics['llm']['input_tokens'] += input_tokens
#         self.metrics['llm']['output_tokens'] += output_tokens
#         self.metrics['llm']['total_tokens'] += (input_tokens + output_tokens)
#         self.metrics['llm']['total_latency'] += latency
#
#         if guardrail_triggered:
#             self.metrics['llm']['guardrail_interventions'] += 1
#
#         # Calculate cost
#         cost_data = self._calculate_llm_cost(model_name, input_tokens, output_tokens)
#
#         # Update cost metrics
#         self.metrics['costs']['llm']['total_cost'] += cost_data['total_cost']
#         self.metrics['costs']['llm']['input_cost'] += cost_data['input_cost']
#         self.metrics['costs']['llm']['output_cost'] += cost_data['output_cost']
#
#         # Add INR costs
#         if 'total_cost_inr' not in self.metrics['costs']['llm']:
#             self.metrics['costs']['llm']['total_cost_inr'] = 0
#             self.metrics['costs']['llm']['input_cost_inr'] = 0
#             self.metrics['costs']['llm']['output_cost_inr'] = 0
#
#         self.metrics['costs']['llm']['total_cost_inr'] += cost_data['total_cost_inr']
#         self.metrics['costs']['llm']['input_cost_inr'] += cost_data['input_cost_inr']
#         self.metrics['costs']['llm']['output_cost_inr'] += cost_data['output_cost_inr']
#
#         # Track by model
#         if model_name not in self.metrics['costs']['llm']['by_model']:
#             self.metrics['costs']['llm']['by_model'][model_name] = {
#                 'calls': 0,
#                 'input_tokens': 0,
#                 'output_tokens': 0,
#                 'total_cost': 0,
#                 'input_cost': 0,
#                 'output_cost': 0,
#                 'total_cost_inr': 0,
#                 'input_cost_inr': 0,
#                 'output_cost_inr': 0
#             }
#
#         model_costs = self.metrics['costs']['llm']['by_model'][model_name]
#         model_costs['calls'] += 1
#         model_costs['input_tokens'] += input_tokens
#         model_costs['output_tokens'] += output_tokens
#         model_costs['total_cost'] += cost_data['total_cost']
#         model_costs['input_cost'] += cost_data['input_cost']
#         model_costs['output_cost'] += cost_data['output_cost']
#         model_costs['total_cost_inr'] += cost_data['total_cost_inr']
#         model_costs['input_cost_inr'] += cost_data['input_cost_inr']
#         model_costs['output_cost_inr'] += cost_data['output_cost_inr']
#
#         # Record by call type
#         if call_type not in self.metrics['llm']['calls_by_type']:
#             self.metrics['llm']['calls_by_type'][call_type] = {
#                 'calls': 0,
#                 'input_tokens': 0,
#                 'output_tokens': 0,
#                 'total_tokens': 0,
#                 'latency': 0,
#                 'guardrail_interventions': 0
#             }
#
#         self.metrics['llm']['calls_by_type'][call_type]['calls'] += 1
#         self.metrics['llm']['calls_by_type'][call_type]['input_tokens'] += input_tokens
#         self.metrics['llm']['calls_by_type'][call_type]['output_tokens'] += output_tokens
#         self.metrics['llm']['calls_by_type'][call_type]['total_tokens'] += (input_tokens + output_tokens)
#         self.metrics['llm']['calls_by_type'][call_type]['latency'] += latency
#
#         if guardrail_triggered:
#             self.metrics['llm']['calls_by_type'][call_type]['guardrail_interventions'] += 1
#
#         # Update average latency
#         self.metrics['llm']['avg_latency'] = (
#                 self.metrics['llm']['total_latency'] / self.metrics['llm']['total_calls']
#         )
#
#         # Update total estimated cost
#         self._update_total_cost()
#
#         # Send to CloudWatch if available
#         self._send_to_cloudwatch('LLM', {
#             'CallType': call_type,
#             'InputTokens': input_tokens,
#             'OutputTokens': output_tokens,
#             'TotalTokens': input_tokens + output_tokens,
#             'Latency': latency,
#             'CostUSD': cost_data['total_cost'],
#             'CostINR': cost_data['total_cost_inr']
#         })
#
#     def record_vector_operation(self, operation_type, item_count, duration):
#         """Record metrics for vector database operations"""
#         if operation_type == 'store':
#             self.metrics['vector_db']['store_operations'] += 1
#             self.metrics['vector_db']['total_vectors_stored'] += item_count
#             self.metrics['vector_db']['total_store_time'] += duration
#             self.metrics['vector_db']['avg_store_time'] = (
#                     self.metrics['vector_db']['total_store_time'] / self.metrics['vector_db']['store_operations']
#             )
#         elif operation_type == 'query':
#             self.metrics['vector_db']['query_operations'] += 1
#             self.metrics['vector_db']['total_query_time'] += duration
#             self.metrics['vector_db']['avg_query_time'] = (
#                     self.metrics['vector_db']['total_query_time'] / self.metrics['vector_db']['query_operations']
#             )
#
#         # Send to CloudWatch if available
#         self._send_to_cloudwatch('VectorDB', {
#             'OperationType': operation_type,
#             'ItemCount': item_count,
#             'Duration': duration
#         })
#
#     def record_error(self, error_type, error_message):
#         """Record an error"""
#         self.metrics['errors']['total'] += 1
#
#         if error_type not in self.metrics['errors']['by_type']:
#             self.metrics['errors']['by_type'][error_type] = {
#                 'count': 0,
#                 'messages': []
#             }
#
#         self.metrics['errors']['by_type'][error_type]['count'] += 1
#         self.metrics['errors']['by_type'][error_type]['messages'].append(error_message)
#
#         # Send to CloudWatch if available
#         self._send_to_cloudwatch('Error', {
#             'ErrorType': error_type,
#             'ErrorMessage': error_message
#         })
#
#     def _update_total_cost(self):
#         """Update total estimated cost"""
#         # LLM costs
#         llm_cost = self.metrics['costs']['llm']['total_cost']
#         llm_cost_inr = self.metrics['costs']['llm']['total_cost_inr']
#
#         # Set total cost to just LLM cost (removing database costs)
#         self.metrics['costs']['total_estimated_cost'] = llm_cost
#
#         # Add INR total cost
#         self.metrics['costs']['total_estimated_cost_inr'] = llm_cost_inr
#
#     def _send_to_cloudwatch(self, metric_type, metrics_data):
#         """Send metrics to CloudWatch if available"""
#         if not self.cloudwatch:
#             return
#
#         try:
#             namespace = f"StorySense/{self.app_name}"
#             timestamp = datetime.now()
#
#             metric_data = []
#             for name, value in metrics_data.items():
#                 if isinstance(value, (int, float)):
#                     metric_data.append({
#                         'MetricName': name,
#                         'Dimensions': [
#                             {'Name': 'Type', 'Value': metric_type}
#                         ],
#                         'Value': value,
#                         'Timestamp': timestamp
#                     })
#
#             if metric_data:
#                 self.cloudwatch.put_metric_data(
#                     Namespace=namespace,
#                     MetricData=metric_data
#                 )
#         except Exception as e:
#             logging.warning(f"Failed to send metrics to CloudWatch: {e}")
#
#     def get_metrics_summary(self):
#         """Get a summary of metrics for reporting"""
#         # Calculate end time and duration if not already set
#         if not self.metrics['end_time']:
#             self.metrics['end_time'] = datetime.now().isoformat()
#             self.metrics['total_duration'] = time.time() - self.start_time
#
#         # Update system metrics
#         self.metrics['system']['end_memory'] = psutil.Process().memory_info().rss / (1024 * 1024)
#
#         # Create summary
#         summary = {
#             'start_time': self.metrics['start_time'],
#             'end_time': self.metrics['end_time'],
#             'total_duration': self.metrics['total_duration'],
#             'story_count': len(self.metrics['user_stories']),
#             'avg_story_time': self.metrics['total_duration'] / len(self.metrics['user_stories']) if self.metrics[
#                 'user_stories'] else 0,
#             'llm_calls': self.metrics['llm']['total_calls'],
#             'llm_tokens': self.metrics['llm']['total_tokens'],
#             'llm_input_tokens': self.metrics['llm']['input_tokens'],
#             'llm_output_tokens': self.metrics['llm']['output_tokens'],
#             'llm_avg_latency': self.metrics['llm']['avg_latency'],
#             'peak_memory_mb': self.metrics['system']['peak_memory'],
#             'error_count': self.metrics['errors']['total'],
#             'total_estimated_cost': self.metrics['costs']['total_estimated_cost'],
#             'total_estimated_cost_inr': self.metrics['costs']['total_estimated_cost_inr'],
#             'estimated_llm_cost': self.metrics['costs']['llm']['total_cost'],
#             'estimated_llm_cost_inr': self.metrics['costs']['llm']['total_cost_inr'],
#             'vector_queries': self.metrics['vector_db']['query_operations'],
#             'cost_breakdown': {
#                 'llm': {
#                     'input_cost': self.metrics['costs']['llm']['input_cost'],
#                     'output_cost': self.metrics['costs']['llm']['output_cost'],
#                     'input_cost_inr': self.metrics['costs']['llm']['input_cost_inr'],
#                     'output_cost_inr': self.metrics['costs']['llm']['output_cost_inr']
#                 }
#             },
#             'system': self.metrics['system'],
#             'batches': self.metrics['batches'],
#             'vector_db': self.metrics['vector_db'],
#             'currency_rate': {
#                 'usd_to_inr': self.usd_to_inr_rate,
#                 'last_updated': datetime.now().isoformat()
#             }
#         }
#
#         return summary
#
#     def save_metrics(self, output_dir="../Output/Metrics"):
#         """Save metrics to a JSON file"""
#         os.makedirs(output_dir, exist_ok=True)
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         metrics_file = f"{output_dir}/metrics_{timestamp}.json"
#
#         # Calculate end time and duration if not already set
#         if not self.metrics['end_time']:
#             self.metrics['end_time'] = datetime.now().isoformat()
#             self.metrics['total_duration'] = time.time() - self.start_time
#
#         # Update system metrics
#         self.metrics['system']['end_memory'] = psutil.Process().memory_info().rss / (1024 * 1024)
#
#         # Add currency rate information
#         self.metrics['currency_rate'] = {
#             'usd_to_inr': self.usd_to_inr_rate,
#             'last_updated': datetime.now().isoformat()
#         }
#
#         with open(metrics_file, 'w') as f:
#             json.dump(self.metrics, f, indent=2, default=str)
#
#         logging.info(f"Metrics saved to {metrics_file}")
#         return metrics_file
#
#     def __del__(self):
#         """Clean up when object is destroyed"""
#         self.stop_monitoring = True
#         if hasattr(self, 'monitor_thread') and self.monitor_thread.is_alive():
#             self.monitor_thread.join(timeout=1.0)