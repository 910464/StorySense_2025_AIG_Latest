import datetime
import shutil
import os.path
import json
import time
import psutil
from src.context_handler.context_storage_handler.pgvector_connector import PGVectorConnector
from src.prompt_layer.storysense_analyzer import StorySenseAnalyzer
from colorama import init, Fore, Style
import configparser
import pandas as pd
# from src.configuration_handler.env_manager import EnvManager
from concurrent.futures import ThreadPoolExecutor
from src.metrics.metrics_manager  import MetricsManager
from src.context_handler.context_file_handler.context_manager import ContextManager
import logging
import hashlib
import base64


class StorySenseProcessor:
    def __init__(self, metrics_manager=None):
        self.config_path = '../Config'
        self.saved_contexts_path = '../Data/SavedContexts'
        self.csv_file = self.saved_contexts_path + '/Contexts.csv'
        self.embed_data_mtc_path = self.saved_contexts_path + '/EmbedDataMTC'
        self.cleaned_mtc_csv_path = self.csv_file

        # Initialize environment manager
        # self.env_manager = EnvManager()

        # Initialize or use provided metrics manager
        print(f"DEBUG: StorySenseProcessor received metrics_manager: {id(metrics_manager)}")
        self.metrics_manager = metrics_manager or MetricsManager()

        # Create necessary directories
        os.makedirs(self.saved_contexts_path, exist_ok=True)
        os.makedirs(self.embed_data_mtc_path, exist_ok=True)

        # Create default Config.properties if it doesn't exist
        if not os.path.exists(os.path.join(self.config_path, 'Config.properties')):
            self.create_default_config()

        self.config_parser_io = configparser.ConfigParser()
        self.config_parser_io.read(os.path.join(self.config_path, 'ConfigIO.properties'))

        # Initialize config parser for LLM settings
        self.config_parser_llm = configparser.ConfigParser()
        self.config_parser_llm.read(os.path.join(self.config_path, 'Config.properties'))

        # Initialize context manager for utilize historical contexts
        self.context_manager = ContextManager(metrics_manager=self.metrics_manager)
        self.context_available = False

        # Set default values if config sections/options are missing
        try:
            self.num_context_retrieve = int(self.config_parser_io.get('Output', 'num_context_retrieve'))
        except (configparser.NoSectionError, configparser.NoOptionError):
            self.num_context_retrieve = 8

        # Get LLM family from environment or config
        self.llm_family = os.getenv('LLM_FAMILY') or self.config_parser_llm.get('LLM', 'LLM_FAMILY',
                                                                                               fallback='AWS')

        # Pass metrics_manager to StorySenseAnalyzer
        self.story_analyzer = StorySenseAnalyzer(self.llm_family, metrics_manager=self.metrics_manager)
        init()

        # Initialize timing trackers for telemetry
        self._context_retrieval_time = 0
        self._llm_analysis_time = 0
        self._report_generation_time = 0
        self._last_db_query_time = 0
        self._active_db_connections = 0
        self._cache_hit_rate = 0
        self._warnings = []

    def _update_memory_usage(self):
        """Update peak memory usage"""
        current_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        self.metrics_manager.metrics['system']['peak_memory'] = max(
            self.metrics_manager.metrics['system']['peak_memory'],
            current_memory
        )

    def create_default_config(self):
        """Create a default Config.properties file"""
        config = configparser.ConfigParser()
        config['AdvancedConfigurations'] = {
            'embedding_model_name': 'amazon.titan-embed-text-v1',
            'embedding_model_path': '../Data/ExternalEmbeddingModel',
            'external_model_threshold': '0.7',  # Increased threshold for better matches
            'default_model_threshold': '0.50',
            'local_embeddings_path': '../Data/LocalEmbeddings'
        }
        config['LLM'] = {
            'LLM_FAMILY': 'AWS',
            'TEMPERATURE': '0.05'
        }

        os.makedirs(self.config_path, exist_ok=True)
        with open(os.path.join(self.config_path, 'Config.properties'), 'w') as f:
            config.write(f)
        print(Fore.GREEN + "Created default Config.properties file" + Style.RESET_ALL)

    def collect_realtime_metrics(self):
        """Collect comprehensive real-time metrics"""
        # Get metrics from metrics_manager
        llm_metrics = self.metrics_manager.metrics['llm']

        return {
            # Performance Metrics
            'processing_start_time': time.time(),
            'context_retrieval_time': self._context_retrieval_time,
            'llm_analysis_time': self._llm_analysis_time,
            'report_generation_time': self._report_generation_time,

            # Resource Metrics
            'current_memory_mb': psutil.Process().memory_info().rss / (1024 * 1024),
            'peak_memory_mb': self.metrics_manager.metrics['system']['peak_memory'],
            'cpu_percent': psutil.cpu_percent(),
            'disk_usage_percent': psutil.disk_usage('/').percent,

            # Token and Cost Metrics
            'input_tokens': llm_metrics.get('input_tokens', 0),
            'output_tokens': llm_metrics.get('output_tokens', 0),
            'total_tokens': llm_metrics.get('total_tokens', 0),
            'estimated_cost': self.calculate_estimated_cost(),

            # Database Metrics
            'db_query_time': self._last_db_query_time,
            'db_connections': self._active_db_connections,
            'cache_hit_rate': self._cache_hit_rate,

            # Error Tracking
            'errors': self.metrics_manager.metrics['errors'].get('by_type', {}),
            'warnings': self._warnings,

            # Quality Metrics
            'context_quality_distribution': self.get_context_quality_distribution(),
            'average_processing_time': self.get_average_processing_time(),
            'success_rate': self.calculate_success_rate()
        }

    def calculate_estimated_cost(self):
        """Calculate estimated cost based on token usage"""
        llm_metrics = self.metrics_manager.metrics['llm']
        input_tokens = llm_metrics.get('input_tokens', 0)
        output_tokens = llm_metrics.get('output_tokens', 0)

        # Claude pricing (as of 2025)
        input_cost_per_1k = 0.003  # \$0.003 per 1K input tokens
        output_cost_per_1k = 0.015  # \$0.015 per 1K output tokens

        input_cost = (input_tokens / 1000) * input_cost_per_1k
        output_cost = (output_tokens / 1000) * output_cost_per_1k

        # USD cost
        usd_cost = input_cost + output_cost

        # INR cost (using conversion rate from metrics_manager)
        inr_cost = usd_cost * self.metrics_manager.usd_to_inr_rate

        return {
            'usd': usd_cost,
            'inr': inr_cost
        }

    def get_context_quality_distribution(self):
        """Get distribution of context quality across processed stories"""
        quality_counts = {'high': 0, 'medium': 0, 'low': 0, 'none': 0}
        for story_data in self.metrics_manager.metrics['user_stories'].values():
            quality = story_data.get('context_quality', 'none')
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
        return quality_counts

    def get_average_processing_time(self):
        """Get average processing time per story"""
        stories = self.metrics_manager.metrics['user_stories']
        if not stories:
            return 0
        total_time = sum(story['processing_time'] for story in stories.values())
        return total_time / len(stories)

    def calculate_success_rate(self):
        """Calculate success rate based on errors"""
        total_operations = self.metrics_manager.metrics['llm']['total_calls']
        total_errors = self.metrics_manager.metrics['errors']['total']
        if total_operations == 0:
            return 1.0
        return max(0, (total_operations - total_errors) / total_operations)

    def calculate_percentage(self, value, total):
        """Calculate percentage for progress bars"""
        if total == 0:
            return 0
        return min(100, (value / total) * 100)

    def get_score_interpretation(self, score):
        """Get human-readable interpretation of score"""
        if score >= 8:
            return "🎉 Excellent! Your story is well-crafted and ready for development."
        elif score >= 6:
            return "👍 Good quality! Minor improvements will make it even better."
        elif score >= 4:
            return "⚠️ Fair quality. Some significant improvements are needed."
        else:
            return "🔧 Needs major improvements before development should begin."

    def generate_report(self, analysis_results, us_id, output_file):
        """Generate and save the enhanced analysis report"""
        if analysis_results:
            # Output File name
            current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")[:-3]
            output_path = f"{output_file}/story_sense_{us_id}_{current_time}.json"
            html_output_path = f"{output_file}/story_sense_{us_id}_{current_time}.html"

            # Create output directory if it doesn't exist
            os.makedirs(output_file, exist_ok=True)

            # Ensure the recommendation field exists
            if 'recommendation' not in analysis_results:
                analysis_results['recommendation'] = {
                    "recommended": "no",
                    "descriptionDiff": "No changes recommended",
                    "acceptanceCriteriaDiff": "No changes recommended",
                    "newUserStory": analysis_results['userstory']
                }

            # Writing the output into a JSON file
            with open(output_path, "w", encoding="utf-8") as file:
                json.dump(analysis_results, file, indent=4)

            # Generate enhanced HTML report
            self.generate_html_report(analysis_results, html_output_path)

            print(Fore.GREEN + f"\nStory Sense analysis completed successfully for User Story {us_id}\n" +
                  Style.RESET_ALL)
            return output_path
        return None

    def generate_html_report(self, analysis, html_path):
        """Generate an HTML report from the analysis results"""
        # Get metrics for this analysis
        metrics_data = self.collect_realtime_metrics()

        # Generate file type section if available
        file_type_section = ""
        if analysis.get('context_file_types'):
            file_type_section = """
            <div class="file-type-section">
                <h4>📄 Context File Types</h4>
                <div class="file-type-grid">
            """

            for file_type, count in analysis.get('context_file_types', {}).items():
                file_type_class = self._get_file_type_class(file_type)
                file_type_section += f"""
                <div class="file-type-item {file_type_class}">
                    <div class="file-type-icon">{self._get_file_type_icon(file_type)}</div>
                    <div class="file-type-name">{file_type}</div>
                    <div class="file-type-count">{count}</div>
                </div>
                """

            file_type_section += """
                </div>
            </div>
            """

        html_content = f"""<!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Story Sense Analysis - User Story {analysis['us_id']}</title>
            <style>
                /* Modern CSS Variables */
                :root {{
                    --primary-color: #3498db;
                    --primary-dark: #2980b9;
                    --success-color: #27ae60;
                    --warning-color: #f39c12;
                    --danger-color: #e74c3c;
                    --dark-color: #2c3e50;
                    --light-bg: #f8f9fa;
                    --white: #ffffff;
                    --gray-100: #f1f5f9;
                    --gray-200: #e2e8f0;
                    --gray-300: #cbd5e1;
                    --gray-600: #475569;
                    --gray-700: #334155;
                    --gray-800: #1e293b;
                    --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
                    --shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
                    --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1);
                    --border-radius: 12px;
                    --border-radius-sm: 8px;
                    --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                }}

                * {{
                    box-sizing: border-box;
                    margin: 0;
                    padding: 0;
                }}

                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    background-color: #f5f7fa;
                    padding: 20px;
                }}

                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: var(--white);
                    border-radius: var(--border-radius);
                    box-shadow: var(--shadow-lg);
                    overflow: hidden;
                }}

                /* Header Styles */
                .header {{
                    background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
                    color: white;
                    padding: 40px;
                    text-align: center;
                    position: relative;
                }}

                .header h1 {{
                    margin: 0 0 20px 0;
                    font-size: 2.5rem;
                    font-weight: 700;
                }}

                .header-info {{
                    display: flex;
                    justify-content: center;
                    gap: 40px;
                    margin-bottom: 20px;
                    flex-wrap: wrap;
                }}

                .header-item {{
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    gap: 5px;
                }}

                .header-label {{
                    font-size: 0.9rem;
                    opacity: 0.9;
                }}

                .header-value {{
                    font-size: 1.2rem;
                    font-weight: 600;
                }}

                /* Section Styles */
                .section {{
                    padding: 40px;
                    border-bottom: 1px solid var(--gray-200);
                }}

                .section:last-child {{
                    border-bottom: none;
                }}

                .section h2 {{
                    color: var(--dark-color);
                    font-size: 1.8rem;
                    margin: 0 0 30px 0;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }}

                /* User Story Section */
                .story-content {{
                    background-color: var(--light-bg);
                    padding: 20px;
                    border-radius: var(--border-radius);
                    margin-bottom: 20px;
                    white-space: pre-wrap;
                    font-family: monospace;
                    border-left: 4px solid var(--primary-color);
                }}

                /* Recommendations Section */
                .recommendation-box {{
                    background-color: #f0f7ff;
                    padding: 20px;
                    border-radius: var(--border-radius);
                    margin-bottom: 20px;
                    border-left: 4px solid var(--primary-color);
                }}

                .recommendation-header {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 20px;
                    padding-bottom: 10px;
                    border-bottom: 1px solid var(--gray-200);
                }}

                .recommendation-title {{
                    font-size: 1.3rem;
                    font-weight: 600;
                    color: var(--dark-color);
                    margin: 0;
                }}

                .recommendation-badge {{
                    padding: 5px 15px;
                    border-radius: 20px;
                    font-weight: 600;
                    font-size: 0.9rem;
                }}

                .yes {{
                    background-color: var(--success-color);
                    color: white;
                }}

                .no {{
                    background-color: var(--gray-300);
                    color: var(--gray-700);
                }}

                .recommendation-section {{
                    margin-bottom: 20px;
                }}

                .recommendation-section h4 {{
                    color: var(--dark-color);
                    margin-bottom: 10px;
                    font-size: 1.1rem;
                }}

                .diff-box {{
                    background-color: var(--light-bg);
                    padding: 15px;
                    border-radius: var(--border-radius-sm);
                    margin-bottom: 15px;
                    white-space: pre-wrap;
                    font-family: monospace;
                    font-size: 14px;
                    border-left: 3px solid var(--gray-300);
                }}

                .improved-story {{
                    background-color: #e8f8f5;
                    padding: 20px;
                    border-radius: var(--border-radius);
                    margin-top: 20px;
                    border-left: 4px solid var(--success-color);
                }}

                .improved-story h4 {{
                    color: var(--success-color);
                    margin-bottom: 15px;
                }}

                /* Scores Section */
                .scores-overview {{
                    display: flex;
                    justify-content: space-between;
                    margin-bottom: 30px;
                }}

                .overall-score {{
                    flex: 0 0 250px;
                    text-align: center;
                    padding: 30px;
                    background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
                    border-radius: var(--border-radius);
                    color: white;
                }}

                .score-circle {{
                    width: 120px;
                    height: 120px;
                    border-radius: 50%;
                    background: rgba(255, 255, 255, 0.1);
                    margin: 0 auto 15px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    position: relative;
                }}

                .score-circle::before {{
                    content: '';
                    position: absolute;
                    top: 5px;
                    left: 5px;
                    right: 5px;
                    bottom: 5px;
                    border-radius: 50%;
                    border: 3px solid white;
                }}

                .score-number {{
                    font-size: 2.5rem;
                    font-weight: 700;
                }}

                .score-label {{
                    font-size: 1rem;
                    opacity: 0.9;
                }}

                .score-interpretation {{
                    font-size: 1.1rem;
                    margin-top: 15px;
                }}

                .dimension-scores {{
                    flex: 1;
                    margin-left: 30px;
                }}

                .dimensions-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                    gap: 15px;
                }}

                .dimension-card {{
                    background: white;
                    border-radius: var(--border-radius-sm);
                    padding: 15px;
                    box-shadow: var(--shadow-sm);
                    transition: var(--transition);
                    border: 1px solid var(--gray-200);
                }}

                .dimension-card:hover {{
                    transform: translateY(-3px);
                    box-shadow: var(--shadow);
                }}

                .dimension-header {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 10px;
                }}

                .dimension-name {{
                    font-size: 0.9rem;
                    font-weight: 600;
                    color: var(--dark-color);
                    display: flex;
                    align-items: center;
                    gap: 5px;
                }}

                .dimension-score {{
                    font-size: 1.1rem;
                    font-weight: 700;
                }}

                .dimension-bar {{
                    height: 6px;
                    background: var(--gray-200);
                    border-radius: 3px;
                    overflow: hidden;
                }}

                .dimension-fill {{
                    height: 100%;
                    border-radius: 3px;
                    transition: width 1s ease-in-out;
                }}

                .good {{
                    color: var(--success-color);
                }}

                .medium {{
                    color: var(--warning-color);
                }}

                .poor {{
                    color: var(--danger-color);
                }}

                .bg-good {{
                    background-color: var(--success-color);
                }}

                .bg-medium {{
                    background-color: var(--warning-color);
                }}

                .bg-poor {{
                    background-color: var(--danger-color);
                }}

                /* Detailed Analysis Section */
                .parameter-analysis {{
                    margin-bottom: 40px;
                }}

                .parameter-card {{
                    background: white;
                    border-radius: var(--border-radius);
                    margin-bottom: 20px;
                    box-shadow: var(--shadow-sm);
                    border: 1px solid var(--gray-200);
                    overflow: hidden;
                }}

                .parameter-header {{
                    padding: 15px 20px;
                    background: var(--light-bg);
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    cursor: pointer;
                    transition: var(--transition);
                    border-bottom: 1px solid var(--gray-200);
                }}

                .parameter-header:hover {{
                    background: var(--gray-200);
                }}

                .parameter-title {{
                    font-size: 1.1rem;
                    font-weight: 600;
                    color: var(--dark-color);
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    margin: 0;
                }}

                .parameter-score {{
                    font-size: 1.1rem;
                    font-weight: 700;
                    padding: 3px 10px;
                    border-radius: 15px;
                }}

                .parameter-content {{
                    padding: 20px;
                    display: none;
                }}

                .parameter-content.active {{
                    display: block;
                }}

                .parameter-justification {{
                    margin-bottom: 20px;
                }}

                .parameter-recommendations {{
                    background: var(--light-bg);
                    padding: 15px;
                    border-radius: var(--border-radius-sm);
                    border-left: 3px solid var(--primary-color);
                }}

                .parameter-recommendations h4 {{
                    color: var(--primary-color);
                    margin-top: 0;
                    margin-bottom: 10px;
                    font-size: 1rem;
                }}

                .parameter-recommendations ul {{
                    margin: 0;
                    padding-left: 20px;
                }}

                .parameter-recommendations li {{
                    margin-bottom: 5px;
                }}

                /* Insights and Risks Section */
                .insights-risks-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }}

                .insights-card, .risks-card, .actions-card {{
                    background: white;
                    border-radius: var(--border-radius);
                    padding: 20px;
                    box-shadow: var(--shadow-sm);
                    border: 1px solid var(--gray-200);
                }}

                .insights-card h3, .risks-card h3, .actions-card h3 {{
                    color: var(--dark-color);
                    margin-bottom: 15px;
                    font-size: 1.2rem;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }}

                .insight-item, .risk-item, .action-item {{
                    display: flex;
                    margin-bottom: 10px;
                    padding-bottom: 10px;
                    border-bottom: 1px solid var(--gray-200);
                }}

                .insight-item:last-child, .risk-item:last-child, .action-item:last-child {{
                    border-bottom: none;
                    padding-bottom: 0;
                    margin-bottom: 0;
                }}

                .item-icon {{
                    flex: 0 0 30px;
                    font-size: 1.2rem;
                    color: var(--primary-color);
                }}

                .item-text {{
                    flex: 1;
                }}

                /* Parameter Education Section */
                .education-section {{
                    background: var(--light-bg);
                    border-radius: var(--border-radius);
                    padding: 30px;
                    margin-top: 40px;
                }}

                .education-intro {{
                    text-align: center;
                    max-width: 800px;
                    margin: 0 auto 30px;
                    font-size: 1.1rem;
                    color: var(--gray-700);
                }}

                .parameter-education-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
                    gap: 25px;
                }}

                .education-card {{
                    background: white;
                    border-radius: var(--border-radius);
                    box-shadow: var(--shadow-sm);
                    overflow: hidden;
                    border: 1px solid var(--gray-200);
                }}

                .education-header {{
                    padding: 15px 20px;
                    background: var(--light-bg);
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    cursor: pointer;
                    transition: var(--transition);
                    border-bottom: 1px solid var(--gray-200);
                }}

                .education-header:hover {{
                    background: var(--gray-200);
                }}

                .education-title {{
                    font-size: 1.1rem;
                    font-weight: 600;
                    color: var(--dark-color);
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    margin: 0;
                }}

                .learn-more-btn {{
                    background: var(--primary-color);
                    color: white;
                    border: none;
                    padding: 5px 15px;
                    border-radius: 20px;
                    font-size: 0.8rem;
                    font-weight: 600;
                    cursor: pointer;
                    transition: var(--transition);
                }}

                .learn-more-btn:hover {{
                    background: var(--primary-dark);
                }}

                .education-content {{
                    padding: 0;
                    max-height: 0;
                    overflow: hidden;
                    transition: max-height 0.5s ease, padding 0.3s ease;
                }}

                .education-content.active {{
                    padding: 20px;
                    max-height: 2000px;
                }}

                .example-comparison {{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 20px;
                    margin: 20px 0;
                }}

                .good-example, .poor-example {{
                    padding: 15px;
                    border-radius: var(--border-radius-sm);
                }}

                .good-example {{
                    background-color: #d4edda;
                    border-left: 4px solid var(--success-color);
                }}

                .poor-example {{
                    background-color: #f8d7da;
                    border-left: 4px solid var(--danger-color);
                }}

                .example-text {{
                    background: rgba(255, 255, 255, 0.7);
                    padding: 10px;
                    border-radius: 5px;
                    margin: 10px 0;
                    font-style: italic;
                }}

                .improvement-tips {{
                    background: var(--light-bg);
                    padding: 15px;
                    border-radius: var(--border-radius-sm);
                    margin-top: 20px;
                    border-left: 3px solid var(--warning-color);
                }}

                /* Context Information Section */
                .context-info {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    margin-top: 20px;
                }}

                .context-item {{
                    background: white;
                    border-radius: var(--border-radius-sm);
                    padding: 15px;
                    box-shadow: var(--shadow-sm);
                    text-align: center;
                    border: 1px solid var(--gray-200);
                }}

                .context-label {{
                    font-size: 0.9rem;
                    color: var(--gray-600);
                    margin-bottom: 5px;
                }}

                .context-value {{
                    font-size: 1.2rem;
                    font-weight: 600;
                    color: var(--primary-color);
                }}

                .context-high {{
                    color: var(--success-color);
                }}

                .context-medium {{
                    color: var(--warning-color);
                }}

                .context-low {{
                    color: var(--danger-color);
                }}

                .context-none {{
                    color: var(--gray-600);
                }}

                /* File Type Section */
                .file-type-section {{
                    margin-top: 20px;
                    background: var(--light-bg);
                    border-radius: var(--border-radius);
                    padding: 20px;
                    border: 1px solid var(--gray-200);
                }}

                .file-type-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
                    gap: 15px;
                    margin-top: 15px;
                }}

                .file-type-item {{
                    background: white;
                    border-radius: var(--border-radius-sm);
                    padding: 15px 10px;
                    text-align: center;
                    box-shadow: var(--shadow-sm);
                    border: 1px solid var(--gray-200);
                    transition: var(--transition);
                }}

                .file-type-item:hover {{
                    transform: translateY(-3px);
                    box-shadow: var(--shadow);
                }}

                .file-type-icon {{
                    font-size: 24px;
                    margin-bottom: 5px;
                }}

                .file-type-name {{
                    font-size: 0.9rem;
                    font-weight: 600;
                    color: var(--dark-color);
                    margin-bottom: 5px;
                }}

                .file-type-count {{
                    font-size: 1.1rem;
                    font-weight: 700;
                    color: var(--primary-color);
                }}

                /* File type specific colors */
                .file-pdf .file-type-icon {{ color: #e74c3c; }}
                .file-doc .file-type-icon {{ color: #3498db; }}
                .file-excel .file-type-icon {{ color: #27ae60; }}
                .file-ppt .file-type-icon {{ color: #e67e22; }}
                .file-text .file-type-icon {{ color: #95a5a6; }}
                .file-image .file-type-icon {{ color: #9b59b6; }}
                .file-story .file-type-icon {{ color: #f39c12; }}

                /* Context Analysis Section */
                .context-analysis {{
                    margin-top: 20px;
                    background: var(--light-bg);
                    border-radius: var(--border-radius);
                    padding: 20px;
                    border: 1px solid var(--gray-200);
                }}

                .context-analysis h4 {{
                    margin-top: 0;
                    margin-bottom: 15px;
                    color: var(--dark-color);
                }}

                .context-analysis h5 {{
                    color: var(--primary-color);
                    margin-bottom: 10px;
                    font-size: 1rem;
                }}

                .useful-types ul {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 10px;
                    list-style: none;
                    padding: 0;
                    margin: 0 0 15px 0;
                }}

                .useful-types li {{
                    background: white;
                    padding: 5px 12px;
                    border-radius: 20px;
                    font-size: 0.9rem;
                    box-shadow: var(--shadow-sm);
                    border: 1px solid var(--gray-200);
                }}

                .quality-assessment p {{
                    background: white;
                    padding: 15px;
                    border-radius: var(--border-radius-sm);
                    margin: 0;
                    border-left: 3px solid var(--primary-color);
                }}

                /* Responsive Design */
                @media (max-width: 768px) {{
                    .section {{
                        padding: 30px 20px;
                    }}

                    .scores-overview {{
                        flex-direction: column;
                    }}

                    .overall-score {{
                        margin-bottom: 20px;
                    }}

                    .dimension-scores {{
                        margin-left: 0;
                    }}

                    .dimensions-grid {{
                        grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
                    }}

                    .example-comparison {{
                        grid-template-columns: 1fr;
                    }}
                }}

                @media (max-width: 480px) {{
                    .header {{
                        padding: 30px 20px;
                    }}

                    .header h1 {{
                        font-size: 2rem;
                    }}

                    .score-circle {{
                        width: 100px;
                        height: 100px;
                    }}

                    .score-number {{
                        font-size: 2rem;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Story Sense Analysis Report</h1>
                    <div class="header-info">
                        <div class="header-item">
                            <span class="header-label">User Story ID</span>
                            <span class="header-value">{analysis['us_id']}</span>
                        </div>
                        <div class="header-item">
                            <span class="header-label">Overall Score</span>
                            <span class="header-value">{analysis['overall_score']:.1f}/10</span>
                        </div>
                        <div class="header-item">
                            <span class="header-label">Generated On</span>
                            <span class="header-value">{datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}</span>
                        </div>
                    </div>
                </div>

                <div class="section">
                    <h2>📝 User Story</h2>
                    <div class="story-content">{analysis['userstory']}</div>
                </div>

                <div class="section">
                    <h2>💡 Recommended Improvements</h2>
                    <div class="recommendation-box">
                        <div class="recommendation-header">
                            <h3 class="recommendation-title">Changes Recommended</h3>
                            <span class="recommendation-badge {self._get_recommendation_class(analysis['recommendation']['recommended'])}">{analysis['recommendation']['recommended'].upper()}</span>
                        </div>

                        <div class="recommendation-section">
                            <h4>Description Changes:</h4>
                            <div class="diff-box">{analysis['recommendation']['descriptionDiff']}</div>
                        </div>

                        <div class="recommendation-section">
                            <h4>Acceptance Criteria Changes:</h4>
                            <div class="diff-box">{analysis['recommendation']['acceptanceCriteriaDiff']}</div>
                        </div>

                        <div class="improved-story">
                            <h4>Improved User Story:</h4>
                            <div class="story-content">{analysis['recommendation']['newUserStory']}</div>
                        </div>
                    </div>
                </div>

                <div class="section">
                    <h2>📊 Quality Score Analysis</h2>

                    <div class="scores-overview">
                        <div class="overall-score">
                            <div class="score-circle">
                                <div class="score-number">{analysis['overall_score']:.1f}</div>
                            </div>
                            <div class="score-label">Overall Score</div>
                            <div class="score-interpretation">{self.get_score_interpretation(analysis['overall_score'])}</div>
                        </div>

                        <div class="dimension-scores">
                            <div class="dimensions-grid">
                                {self._generate_dimension_cards(analysis)}
                            </div>
                        </div>
                    </div>
                </div>

                <div class="section">
                    <h2>🔍 Detailed Analysis</h2>

                    <div class="parameter-analysis">
                        {self._generate_parameter_cards(analysis)}
                    </div>
                </div>

                <div class="section">
                    <h2>💎 Key Insights & Recommendations</h2>

                    <div class="insights-risks-grid">
                        <div class="insights-card">
                            <h3>🔍 Key Insights</h3>
                            {self._generate_insight_items(analysis.get('key_insights', ['No insights available']))}
                        </div>

                        <div class="risks-card">
                            <h3>⚠️ Potential Risks</h3>
                            {self._generate_risk_items(analysis.get('potential_risks', ['No risks identified']))}
                        </div>
                    </div>

                    <div class="actions-card">
                        <h3>🎯 Action Items</h3>
                        {self._generate_action_items(analysis.get('recommendations', ['No action items available']))}
                    </div>
                </div>

                <div class="section">
                    <h2>📚 Context Information</h2>

                    <div class="context-info">
                        <div class="context-item">
                            <div class="context-label">Context Quality</div>
                            <div class="context-value context-{analysis.get('context_quality', 'none')}">{analysis.get('context_quality', 'None').title()}</div>
                        </div>

                        <div class="context-item">
                            <div class="context-label">Context Documents</div>
                            <div class="context-value">{analysis.get('context_count', 0)}</div>
                        </div>

                        <div class="context-item">
                            <div class="context-label">Processing Time</div>
                            <div class="context-value">{analysis.get('processing_time', 0):.2f}s</div>
                        </div>

                        <div class="context-item">
                            <div class="context-label">Analysis Model</div>
                            <div class="context-value">Claude 3 Sonnet</div>
                        </div>
                    </div>

                    {file_type_section}

                    {self._generate_context_analysis_section(analysis)}
                </div>

                <div class="section">
                    <h2>📖 Understanding the Analysis Parameters</h2>

                    <p class="education-intro">
                        This analysis evaluates your user story against 10 key quality parameters based on the INVEST criteria and industry best practices.
                        Click on each parameter below to learn what it means, see examples, and understand how to improve your stories.
                    </p>

                    <div class="parameter-education-grid">
                        {self._generate_education_cards()}
                    </div>
                </div>
            </div>

            <script>
                // Initialize interactive features
                document.addEventListener('DOMContentLoaded', function() {{
                    // Initialize parameter cards
                    initializeParameterCards();

                    // Initialize education cards
                    initializeEducationCards();

                    // Animate dimension bars
                    animateDimensionBars();
                }});

                function initializeParameterCards() {{
                    const parameterHeaders = document.querySelectorAll('.parameter-header');

                    parameterHeaders.forEach(header => {{
                        header.addEventListener('click', function() {{
                            const content = this.nextElementSibling;
                            const isActive = content.classList.contains('active');

                            // Close all parameter contents
                            document.querySelectorAll('.parameter-content').forEach(item => {{
                                item.classList.remove('active');
                            }});

                            // Toggle current content
                            if (!isActive) {{
                                content.classList.add('active');
                            }}
                        }});
                    }});

                    // Open the first parameter card by default
                    const firstContent = document.querySelector('.parameter-content');
                    if (firstContent) {{
                        firstContent.classList.add('active');
                    }}
                }}

                function initializeEducationCards() {{
                    const educationHeaders = document.querySelectorAll('.education-header');

                    educationHeaders.forEach(header => {{
                        header.addEventListener('click', function() {{
                            const content = this.nextElementSibling;
                            const isActive = content.classList.contains('active');
                            const button = this.querySelector('.learn-more-btn');

                            // Close all education contents
                            document.querySelectorAll('.education-content').forEach(item => {{
                                item.classList.remove('active');
                            }});

                            document.querySelectorAll('.learn-more-btn').forEach(btn => {{
                                btn.textContent = 'Learn More';
                            }});

                            // Toggle current content
                            if (!isActive) {{
                                content.classList.add('active');
                                if (button) {{
                                    button.textContent = 'Show Less';
                                }}
                            }}
                        }});
                    }});
                }}

                function animateDimensionBars() {{
                    const dimensionFills = document.querySelectorAll('.dimension-fill');

                    setTimeout(() => {{
                        dimensionFills.forEach(fill => {{
                            const width = fill.getAttribute('data-width');
                            fill.style.width = width;
                        }});
                    }}, 300);
                }}
            </script>
        </body>
        </html>
        """

        with open(html_path, "w", encoding="utf-8") as file:
            file.write(html_content)

    def _get_file_type_icon(self, file_type):
        """Get icon for file type"""
        icons = {
            'pdf': '📄',
            'docx': '📝',
            'doc': '📝',
            'xlsx': '📊',
            'xls': '📊',
            'pptx': '📊',
            'ppt': '📊',
            'txt': '📃',
            'md': '📃',
            'image': '🖼️',
            'user_story': '📋',
            'unknown': '❓'
        }
        return icons.get(file_type.lower(), '📄')

    def _get_file_type_class(self, file_type):
        """Get CSS class for file type"""
        classes = {
            'pdf': 'file-pdf',
            'docx': 'file-doc',
            'doc': 'file-doc',
            'xlsx': 'file-excel',
            'xls': 'file-excel',
            'pptx': 'file-ppt',
            'ppt': 'file-ppt',
            'txt': 'file-text',
            'md': 'file-text',
            'image': 'file-image',
            'user_story': 'file-story',
            'unknown': 'file-unknown'
        }
        return classes.get(file_type.lower(), 'file-unknown')

    def _generate_context_analysis_section(self, analysis):
        """Generate HTML for context analysis section"""
        if not analysis.get('context_analysis'):
            return ""

        context_analysis = analysis.get('context_analysis', {})
        most_useful_types = context_analysis.get('most_useful_file_types', [])
        quality_assessment = context_analysis.get('context_quality_assessment', 'No assessment provided')

        html = """
        <div class="context-analysis">
            <h4>🔍 Context Analysis</h4>
        """

        if most_useful_types:
            html += """
            <div class="useful-types">
                <h5>Most Useful File Types:</h5>
                <ul>
            """
            for file_type in most_useful_types:
                html += f"""
                <li>{self._get_file_type_icon(file_type)} {file_type}</li>
                """
            html += """
                </ul>
            </div>
            """

        html += f"""
            <div class="quality-assessment">
                <h5>Quality Assessment:</h5>
                <p>{quality_assessment}</p>
            </div>
        </div>
        """

        return html

    def _generate_dimension_cards(self, analysis):
        """Generate dimension cards for the scores overview section"""
        dimensions = [
            ('user_centered', 'User-Centered', '👤'),
            ('independent', 'Independent', '🔗'),
            ('negotiable', 'Negotiable', '💬'),
            ('valuable', 'Valuable', '💎'),
            ('estimable', 'Estimable', '📏'),
            ('small', 'Small', '📦'),
            ('testable', 'Testable', '🧪'),
            ('acceptance_criteria', 'Acceptance Criteria', '✅'),
            ('prioritized', 'Prioritized', '🎯'),
            ('collaboration', 'Collaboration', '🤝')
        ]

        cards_html = ""
        for param_key, param_name, icon in dimensions:
            score = analysis.get(f'{param_key}_score', 0)
            score_class = self._get_score_class(score)

            cards_html += f"""
            <div class="dimension-card">
                <div class="dimension-header">
                    <div class="dimension-name">{icon} {param_name}</div>
                    <div class="dimension-score {score_class}">{score}/10</div>
                </div>
                <div class="dimension-bar">
                    <div class="dimension-fill bg-{score_class}" data-width="{score * 10}%" style="width: 0%"></div>
                </div>
            </div>
            """

        return cards_html

    def _generate_parameter_cards(self, analysis):
        """Generate parameter cards for the detailed analysis section"""
        parameters = [
            ('user_centered', 'User-Centered', '👤'),
            ('independent', 'Independent', '🔗'),
            ('negotiable', 'Negotiable', '💬'),
            ('valuable', 'Valuable', '💎'),
            ('estimable', 'Estimable', '📏'),
            ('small', 'Small', '📦'),
            ('testable', 'Testable', '🧪'),
            ('acceptance_criteria', 'Acceptance Criteria', '✅'),
            ('prioritized', 'Prioritized', '🎯'),
            ('collaboration', 'Collaboration', '🤝')
        ]

        cards_html = ""
        for param_key, param_name, icon in parameters:
            score = analysis.get(f'{param_key}_score', 0)
            justification = analysis.get(f'{param_key}_justification', 'No analysis provided')
            recommendations = analysis.get(f'{param_key}_recommendations', ['No recommendations provided'])
            score_class = self._get_score_class(score)

            recommendations_html = ""
            if score < 10 and recommendations and recommendations != ['No recommendations provided']:
                recommendations_html = f"""
                <div class="parameter-recommendations">
                    <h4>💡 Recommendations for Improvement:</h4>
                    <ul>
                        {self._generate_list_items(recommendations)}
                    </ul>
                </div>
                """
            elif score >= 10:
                recommendations_html = f"""
                <div class="parameter-recommendations">
                    <h4>💡 Excellent Work!</h4>
                    <p>This aspect of your user story is excellent! No improvements needed.</p>
                </div>
                """

            cards_html += f"""
            <div class="parameter-card">
                <div class="parameter-header">
                    <h3 class="parameter-title">{icon} {param_name}</h3>
                    <span class="parameter-score {score_class}">{score}/10</span>
                </div>
                <div class="parameter-content">
                    <div class="parameter-justification">
                        <p>{justification}</p>
                    </div>
                    {recommendations_html}
                </div>
            </div>
            """

        return cards_html

    def _generate_education_cards(self):
        """Generate education cards for the parameters education section"""
        parameters = [
            ('user-centered', 'User-Centered', '👤',
             'A user-centered story focuses on the person who will actually use the feature, clearly identifying who they are and why they need this functionality.'),
            ('independent', 'Independent', '🔗',
             'An independent story can be developed, tested, and delivered without waiting for other stories to be completed first.'),
            ('negotiable', 'Negotiable', '💬',
             'A negotiable story focuses on the "what" and "why" rather than the "how," leaving room for discussion about implementation details.'),
            ('valuable', 'Valuable', '💎',
             'A valuable story delivers clear, measurable benefit to users or the business.'),
            ('estimable', 'Estimable', '📏',
             'An estimable story contains enough detail and clarity that developers can reasonably estimate the effort required.'),
            ('small', 'Small', '📦',
             'A small story is appropriately sized to be completed within a single iteration or sprint.'),
            ('testable', 'Testable', '🧪',
             'A testable story has clear, verifiable criteria that define when the story is complete.'),
            ('acceptance-criteria', 'Acceptance Criteria', '✅',
             'Good acceptance criteria are specific, measurable conditions that must be met for the story to be considered complete.'),
            ('prioritized', 'Prioritized', '🎯',
             'A prioritized story has a clear understanding of its importance relative to other work.'),
            ('collaboration', 'Collaboration & Understanding', '🤝',
             'A story that promotes collaboration is written in language that all team members can understand.')
        ]

        cards_html = ""
        for param_id, param_name, icon, description in parameters:
            cards_html += f"""
            <div class="education-card">
                <div class="education-header">
                    <h3 class="education-title">{icon} {param_name}</h3>
                    <button class="learn-more-btn">Learn More</button>
                </div>
                <div class="education-content">
                    <h4>What does this mean?</h4>
                    <p>{description}</p>

                    {self._generate_education_examples(param_id)}

                    {self._generate_education_tips(param_id)}
                </div>
            </div>
            """

        return cards_html

    def _generate_education_examples(self, param_id):
        """Generate examples for education cards"""
        examples = {
            'user-centered': {
                'good': {
                    'text': '"As a busy working parent, I want to quickly reorder my usual groceries from my purchase history so that I can save time during my weekly shopping and spend more time with my family."',
                    'explanation': 'This clearly identifies the specific user type (busy working parent), their context, and their motivation (save time for family).'
                },
                'poor': {
                    'text': '"As a user, I want a button that calls the API endpoint to retrieve order history from the database."',
                    'explanation': 'This focuses on technical implementation rather than user needs and uses the generic term "user" without context.'
                }
            },
            'independent': {
                'good': {
                    'text': '"As a registered user, I want to reset my password via email so that I can regain access to my account if I forget my credentials."',
                    'explanation': 'This can be implemented independently and provides immediate value without requiring other features.'
                },
                'poor': {
                    'text': '"As a user, I want to share my completed profile with friends so that they can see my information."',
                    'explanation': 'This depends on profile creation, friend management, and sharing features being completed first.'
                }
            },
            'negotiable': {
                'good': {
                    'text': '"As a customer, I want to be notified when my order status changes so that I can track my purchase progress."',
                    'explanation': 'This describes the need without specifying whether notifications should be email, SMS, push notifications, or in-app alerts.'
                },
                'poor': {
                    'text': '"As a customer, I want a blue popup window with Arial font to appear in the top-right corner showing order status updates every 30 seconds."',
                    'explanation': 'This dictates specific implementation details, leaving no room for better solutions or user experience improvements.'
                }
            },
            'valuable': {
                'good': {
                    'text': '"As a mobile user, I want to use biometric authentication so that I can access my banking app quickly and securely without typing passwords."',
                    'explanation': 'Clear value: faster access, better security, improved user experience for a critical use case.'
                },
                'poor': {
                    'text': '"As a user, I want the logo to be 2 pixels larger so that it looks better."',
                    'explanation': 'The value is subjective and minimal - unlikely to significantly impact user experience or business metrics.'
                }
            },
            'estimable': {
                'good': {
                    'text': '"As a customer service rep, I want to search for customers by email address so that I can quickly find their account information during support calls."',
                    'explanation': 'Clear scope: search functionality, specific field (email), defined use case. Developers can estimate this.'
                },
                'poor': {
                    'text': '"As a user, I want the system to be more intelligent and intuitive."',
                    'explanation': 'Too vague - "intelligent" and "intuitive" are subjective and don\'t define specific functionality to implement.'
                }
            },
            'small': {
                'good': {
                    'text': '"As a user, I want to update my email address in my profile so that I receive notifications at my current email."',
                    'explanation': 'Focused on one specific action: updating email address. Can be completed in a few days.'
                },
                'poor': {
                    'text': '"As a user, I want a complete customer relationship management system so that I can manage all my business relationships."',
                    'explanation': 'This is an epic, not a story. It would take months to implement and should be broken into dozens of smaller stories.'
                }
            },
            'testable': {
                'good': {
                    'text': '"As a user, I want to receive an email confirmation within 5 minutes of placing an order so that I know my purchase was successful."',
                    'explanation': 'Specific, measurable criteria: email sent, within 5 minutes, after order placement. Easy to test.'
                },
                'poor': {
                    'text': '"As a user, I want the website to feel more responsive and modern."',
                    'explanation': '"Feel more responsive" and "modern" are subjective - there\'s no clear way to verify when this is complete.'
                }
            },
            'acceptance-criteria': {
                'good': {
                    'text': '1. User can enter email address in login form\n2. System validates email format before submission\n3. Invalid emails show error message "Please enter a valid email"\n4. Valid emails proceed to password field\n5. Email field remembers last successful login',
                    'explanation': 'Specific, testable conditions that clearly define what needs to be built.'
                },
                'poor': {
                    'text': '1. Login should work properly\n2. Users should be happy with the experience\n3. No bugs',
                    'explanation': 'Too vague - "work properly," "happy," and "no bugs" don\'t provide clear guidance for implementation or testing.'
                }
            },
            'prioritized': {
                'good': {
                    'text': 'Priority: High - Critical for Q1 launch\nBusiness Value: Reduces customer support calls by 40%\nUser Impact: Affects 80% of daily active users',
                    'explanation': 'Clear rationale for priority based on measurable business impact and user benefit.'
                },
                'poor': {
                    'text': 'Priority: Medium\nReason: Would be nice to have',
                    'explanation': 'No clear justification for priority - "nice to have" doesn\'t help with planning or resource allocation.'
                }
            },
            'collaboration': {
                'good': {
                    'text': '"As a customer service representative, I want to quickly access a customer\'s recent order history during phone calls so that I can provide faster, more helpful support without putting customers on hold."',
                    'explanation': 'Clear language that business stakeholders, developers, and testers can all understand and discuss.'
                },
                'poor': {
                    'text': '"As a system administrator, I want to implement a RESTful API endpoint with OAuth 2.0 authentication that queries the PostgreSQL database using indexed joins."',
                    'explanation': 'Too technical - business stakeholders can\'t meaningfully participate in discussions about this story.'
                }
            }
        }

        if param_id in examples:
            example = examples[param_id]
            return f"""
            <div class="example-comparison">
                <div class="good-example">
                    <h5>✅ Good Example</h5>
                    <div class="example-text">{example['good']['text']}</div>
                    <p>{example['good']['explanation']}</p>
                </div>

                <div class="poor-example">
                    <h5>❌ Poor Example</h5>
                    <div class="example-text">{example['poor']['text']}</div>
                    <p>{example['poor']['explanation']}</p>
                </div>
            </div>
            """

        return ""

    def _generate_education_tips(self, param_id):
        """Generate improvement tips for education cards"""
        tips = {
            'user-centered': [
                'Be specific about who the user is (role, context, characteristics)',
                'Focus on the user\'s goal, not the system\'s functionality',
                'Explain the user\'s motivation clearly in the "so that" clause',
                'Avoid technical jargon and system-focused language'
            ],
            'independent': [
                'Avoid references to other unfinished features',
                'Ensure the story provides value on its own',
                'Break down large features into smaller, independent pieces',
                'Consider what minimum functionality is needed for the story to be useful'
            ],
            'negotiable': [
                'Focus on the outcome, not the implementation method',
                'Avoid specifying UI elements, colors, or exact layouts',
                'Leave technical decisions to the development team',
                'Use acceptance criteria to define boundaries, not exact solutions'
            ],
            'valuable': [
                'Clearly articulate the benefit in the "so that" clause',
                'Consider both user value and business value',
                'Quantify the impact when possible (time saved, errors reduced, etc.)',
                'Ensure the value justifies the development effort'
            ],
            'estimable': [
                'Provide enough context about the functionality needed',
                'Define clear boundaries with acceptance criteria',
                'Avoid vague terms like "better," "faster," or "more user-friendly"',
                'Include relevant business rules or constraints'
            ],
            'small': [
                'Focus on one specific piece of functionality',
                'Break large features into smaller, deliverable pieces',
                'Aim for stories that can be completed in 1-3 days',
                'Consider the minimum viable version of the feature'
            ],
            'testable': [
                'Use specific, measurable language',
                'Define clear success criteria',
                'Avoid subjective terms like "better," "nice," or "user-friendly"',
                'Think about how you would demonstrate the feature working'
            ],
            'acceptance-criteria': [
                'Write 3-7 specific, testable conditions',
                'Use "Given-When-Then" format when helpful',
                'Include both positive and negative test cases',
                'Define error conditions and edge cases',
                'Specify any business rules or constraints'
            ],
            'prioritized': [
                'Clearly state the priority level (High/Medium/Low or numerical)',
                'Explain the rationale behind the priority',
                'Consider business value, user impact, and technical risk',
                'Relate to business goals or strategic initiatives',
                'Include any time constraints or dependencies'
            ],
            'collaboration': [
                'Use business language, not technical jargon',
                'Write for your least technical stakeholder',
                'Focus on user outcomes rather than system behavior',
                'Include context that helps everyone understand the need',
                'Encourage questions and discussion during story review'
            ]
        }

        if param_id in tips:
            tip_items = tips[param_id]
            tip_html = "".join([f"<li>{tip}</li>" for tip in tip_items])

            return f"""
            <div class="improvement-tips">
                <h5>💡 How to Improve:</h5>
                <ul>
                    {tip_html}
                </ul>
            </div>
            """

        return ""

    def _generate_insight_items(self, insights):
        """Generate insight items with icons"""
        items_html = ""
        for insight in insights:
            if insight and insight != 'No insights available':
                items_html += f"""
                <div class="insight-item">
                    <div class="item-icon">💡</div>
                    <div class="item-text">{insight}</div>
                </div>
                """

        if not items_html:
            items_html = """
            <div class="insight-item">
                <div class="item-icon">ℹ️</div>
                <div class="item-text">No specific insights available for this user story.</div>
            </div>
            """

        return items_html

    def _generate_risk_items(self, risks):
        """Generate risk items with icons"""
        items_html = ""
        for risk in risks:
            if risk and risk != 'No risks identified':
                items_html += f"""
                <div class="risk-item">
                    <div class="item-icon">⚠️</div>
                    <div class="item-text">{risk}</div>
                </div>
                """

        if not items_html:
            items_html = """
            <div class="risk-item">
                <div class="item-icon">✅</div>
                <div class="item-text">No significant risks identified for this user story.</div>
            </div>
            """

        return items_html

    def _generate_action_items(self, actions):
        """Generate action items with priority indicators"""
        items_html = ""
        for i, action in enumerate(actions):
            if action and action != 'No action items available':
                priority_class = 'high' if i < 2 else 'medium' if i < 4 else 'low'
                items_html += f"""
                <div class="action-item {priority_class}">
                    <div class="item-icon">🎯</div>
                    <div class="item-text">{action}</div>
                </div>
                """

        if not items_html:
            items_html = """
            <div class="action-item">
                <div class="item-icon">✓</div>
                <div class="item-text">Continue with the current user story - it meets quality standards.</div>
            </div>
            """

        return items_html

    def _generate_list_items(self, items):
        """Generate HTML list items from a list of strings"""
        if not items:
            return "<li>No items available</li>"

        return "".join([f"<li>{item}</li>" for item in items])

    def _get_score_class(self, score):
        """Return CSS class based on score value"""
        if score >= 7:
            return "good"
        elif score >= 5:
            return "medium"
        else:
            return "poor"

    def _get_recommendation_class(self, recommended):
        """Return CSS class based on recommendation value"""
        return "yes" if recommended.lower() == "yes" else "no"

    def generate_combined_html_report(self, all_analyses, output_file):
        """
        Generate a single HTML report for all user stories with cost metrics and enhanced features
        """
        # Sort analyses by overall score (descending)
        sorted_analyses = sorted(all_analyses, key=lambda x: x['overall_score'], reverse=True)

        # Get metrics summary
        metrics_summary = self.metrics_manager.get_metrics_summary()

        # Calculate average score
        avg_score = sum(analysis['overall_score'] for analysis in all_analyses) / len(
            all_analyses) if all_analyses else 0

        # Create the HTML content
        html_content = f"""<!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Story Sense Analysis Report</title>
            <style>
                /* Modern CSS Variables */
                :root {{
                    --primary-color: #3498db;
                    --primary-dark: #2980b9;
                    --success-color: #27ae60;
                    --warning-color: #f39c12;
                    --danger-color: #e74c3c;
                    --dark-color: #2c3e50;
                    --light-bg: #f8f9fa;
                    --white: #ffffff;
                    --gray-100: #f1f5f9;
                    --gray-200: #e2e8f0;
                    --gray-300: #cbd5e1;
                    --gray-600: #475569;
                    --gray-700: #334155;
                    --gray-800: #1e293b;
                    --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
                    --shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
                    --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1);
                    --border-radius: 12px;
                    --border-radius-sm: 8px;
                    --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                }}

                * {{
                    box-sizing: border-box;
                    margin: 0;
                    padding: 0;
                }}

                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    background-color: #f5f7fa;
                    padding: 20px;
                }}

                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: var(--white);
                    border-radius: var(--border-radius);
                    box-shadow: var(--shadow-lg);
                    overflow: hidden;
                }}

                /* Header Styles */
                .header {{
                    background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
                    color: white;
                    padding: 40px;
                    text-align: center;
                    position: relative;
                }}

                .header h1 {{
                    margin: 0 0 20px 0;
                    font-size: 2.5rem;
                    font-weight: 700;
                }}

                .header-info {{
                    display: flex;
                    justify-content: center;
                    gap: 40px;
                    margin-bottom: 20px;
                    flex-wrap: wrap;
                }}

                .header-item {{
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    gap: 5px;
                }}

                .header-label {{
                    font-size: 0.9rem;
                    opacity: 0.9;
                }}

                .header-value {{
                    font-size: 1.2rem;
                    font-weight: 600;
                }}

                /* Section Styles */
                .section {{
                    padding: 40px;
                    border-bottom: 1px solid var(--gray-200);
                }}

                .section:last-child {{
                    border-bottom: none;
                }}

                .section h2 {{
                    color: var(--dark-color);
                    font-size: 1.8rem;
                    margin: 0 0 30px 0;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }}

                /* Metrics Section */
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }}

                .metric-card {{
                    background: white;
                    border-radius: var(--border-radius);
                    padding: 20px;
                    box-shadow: var(--shadow-sm);
                    text-align: center;
                    transition: var(--transition);
                    border: 1px solid var(--gray-200);
                }}

                .metric-card:hover {{
                    transform: translateY(-5px);
                    box-shadow: var(--shadow);
                }}

                .metric-value {{
                    font-size: 1.8rem;
                    font-weight: 700;
                    color: var(--primary-color);
                    margin: 10px 0;
                }}

                .metric-label {{
                    color: var(--gray-600);
                    font-size: 0.9rem;
                    margin: 0;
                }}

                .metrics-section {{
                    margin-bottom: 30px;
                    padding: 25px;
                    background: var(--light-bg);
                    border-radius: var(--border-radius);
                    border: 1px solid var(--gray-200);
                }}

                .metrics-section h3 {{
                    color: var(--dark-color);
                    margin-top: 0;
                    margin-bottom: 20px;
                    font-size: 1.3rem;
                    font-weight: 600;
                }}

                /* Cost Metrics */
                .cost-card {{
                    background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
                    color: white;
                    border: none;
                }}

                .cost-card .metric-value {{
                    color: white;
                    font-size: 2rem;
                }}

                .cost-card .metric-label {{
                    color: rgba(255, 255, 255, 0.9);
                }}

                /* Ranking Table */
                .ranking-table-container {{
                    overflow-x: auto;
                    margin-bottom: 30px;
                    border-radius: var(--border-radius);
                    box-shadow: var(--shadow);
                }}

                .ranking-table {{
                    width: 100%;
                    border-collapse: collapse;
                    font-size: 0.9rem;
                }}

                .ranking-table th, .ranking-table td {{
                    padding: 12px 15px;
                    text-align: center;
                    border-bottom: 1px solid var(--gray-200);
                }}

                .ranking-table th {{
                    background-color: var(--primary-color);
                    color: white;
                    font-weight: 600;
                    position: sticky;
                    top: 0;
                }}

                .ranking-table tr:nth-child(even) {{
                    background-color: var(--gray-100);
                }}

                .ranking-table tr:hover {{
                    background-color: var(--gray-200);
                }}

                .ranking-table a {{
                    color: var(--primary-color);
                    text-decoration: none;
                    font-weight: 500;
                }}

                .ranking-table a:hover {{
                    text-decoration: underline;
                }}

                /* Table of Contents */
                .toc {{
                    background: var(--light-bg);
                    border-radius: var(--border-radius);
                    padding: 30px;
                    margin-bottom: 40px;
                    border: 1px solid var(--gray-200);
                }}

                .toc-list {{
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                    gap: 15px;
                    list-style-type: none;
                    padding: 0;
                    margin: 0;
                }}

                .toc-item {{
                    background: var(--white);
                    padding: 15px;
                    border-radius: var(--border-radius-sm);
                    box-shadow: var(--shadow-sm);
                    transition: var(--transition);
                    border: 1px solid var(--gray-200);
                }}

                .toc-item:hover {{
                    transform: translateY(-3px);
                    box-shadow: var(--shadow);
                    border-color: var(--primary-color);
                }}

                .toc-link {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    color: var(--dark-color);
                    text-decoration: none;
                }}

                .toc-score {{
                    background: var(--primary-color);
                    color: white;
                    padding: 3px 10px;
                    border-radius: 20px;
                    font-size: 0.8rem;
                    font-weight: 600;
                }}

                /* Story Detail Section */
                .story-detail {{
                    margin-bottom: 60px;
                    padding: 30px;
                    background: var(--white);
                    border-radius: var(--border-radius);
                    box-shadow: var(--shadow);
                    border: 1px solid var(--gray-200);
                }}

                .story-detail-header {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 20px;
                    padding-bottom: 15px;
                    border-bottom: 2px solid var(--gray-200);
                }}

                .story-detail-title {{
                    font-size: 1.5rem;
                    font-weight: 700;
                    color: var(--dark-color);
                    margin: 0;
                }}

                .story-detail-score {{
                    font-size: 1.8rem;
                    font-weight: 700;
                    padding: 5px 15px;
                    border-radius: var(--border-radius-sm);
                    color: white;
                }}

                .story-detail-content {{
                    background: var(--light-bg);
                    padding: 20px;
                    border-radius: var(--border-radius-sm);
                    margin-bottom: 25px;
                    white-space: pre-wrap;
                    font-family: monospace;
                    border-left: 4px solid var(--primary-color);
                }}

                /* Parameter Analysis Section - Dropdown Style */
                .parameter-analysis {{
                    margin-bottom: 30px;
                }}

                .parameter-analysis h4 {{
                    color: var(--dark-color);
                    font-size: 1.2rem;
                    margin-bottom: 15px;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }}

                .parameter-card {{
                    background: white;
                    border-radius: var(--border-radius);
                    margin-bottom: 20px;
                    box-shadow: var(--shadow-sm);
                    border: 1px solid var(--gray-200);
                    overflow: hidden;
                }}

                .parameter-header {{
                    padding: 15px 20px;
                    background: var(--light-bg);
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    cursor: pointer;
                    transition: var(--transition);
                    border-bottom: 1px solid var(--gray-200);
                }}

                .parameter-header:hover {{
                    background: var(--gray-200);
                }}

                .parameter-title {{
                    font-size: 1.1rem;
                    font-weight: 600;
                    color: var(--dark-color);
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    margin: 0;
                }}

                .parameter-score {{
                    font-size: 1.1rem;
                    font-weight: 700;
                    padding: 3px 10px;
                    border-radius: 15px;
                }}

                .parameter-content {{
                    padding: 20px;
                    display: none;
                }}

                .parameter-content.active {{
                    display: block;
                }}

                .parameter-justification {{
                    margin-bottom: 20px;
                }}

                .parameter-recommendations {{
                    background: var(--light-bg);
                    padding: 15px;
                    border-radius: var(--border-radius-sm);
                    border-left: 3px solid var(--primary-color);
                }}

                .parameter-recommendations h5 {{
                    color: var(--primary-color);
                    margin-top: 0;
                    margin-bottom: 10px;
                    font-size: 1rem;
                }}

                .parameter-recommendations ul {{
                    margin: 0;
                    padding-left: 20px;
                }}

                .parameter-recommendations li {{
                    margin-bottom: 5px;
                }}

                /* Recommendations Section */
                .recommendation-box {{
                    background-color: #f0f7ff;
                    padding: 20px;
                    border-radius: var(--border-radius);
                    margin-bottom: 20px;
                    border-left: 4px solid var(--primary-color);
                }}

                .recommendation-header {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 20px;
                    padding-bottom: 10px;
                    border-bottom: 1px solid var(--gray-200);
                }}

                .recommendation-title {{
                    font-size: 1.3rem;
                    font-weight: 600;
                    color: var(--dark-color);
                    margin: 0;
                }}

                .recommendation-badge {{
                    padding: 5px 15px;
                    border-radius: 20px;
                    font-weight: 600;
                    font-size: 0.9rem;
                }}

                .yes {{
                    background-color: var(--success-color);
                    color: white;
                }}

                .no {{
                    background-color: var(--gray-300);
                    color: var(--gray-700);
                }}

                .recommendation-section {{
                    margin-bottom: 20px;
                }}

                .recommendation-section h5 {{
                    color: var(--dark-color);
                    margin-bottom: 10px;
                    font-size: 1.1rem;
                }}

                .diff-box {{
                    background-color: var(--light-bg);
                    padding: 15px;
                    border-radius: var(--border-radius-sm);
                    margin-bottom: 15px;
                    white-space: pre-wrap;
                    font-family: monospace;
                    font-size: 14px;
                    border-left: 3px solid var(--gray-300);
                }}

                .improved-story {{
                    background-color: #e8f8f5;
                    padding: 20px;
                    border-radius: var(--border-radius);
                    margin-top: 20px;
                    border-left: 4px solid var(--success-color);
                    border: 2px solid var(--success-color);
                }}

                .improved-story h5 {{
                    color: var(--success-color);
                    margin-bottom: 15px;
                    font-size: 1.1rem;
                    font-weight: 600;
                }}

                .improved-story-content {{
                    background: rgba(255, 255, 255, 0.8);
                    padding: 15px;
                    border-radius: var(--border-radius-sm);
                    white-space: pre-wrap;
                    font-family: monospace;
                    font-size: 14px;
                    max-height: 400px;
                    overflow-y: auto;
                    border: 1px solid rgba(39, 174, 96, 0.3);
                }}

                /* Score Classes */
                .good {{
                    color: var(--success-color);
                }}

                .medium {{
                    color: var(--warning-color);
                }}

                .poor {{
                    color: var(--danger-color);
                }}

                .bg-good {{
                    background-color: var(--success-color);
                }}

                .bg-medium {{
                    background-color: var(--warning-color);
                }}

                .bg-poor {{
                    background-color: var(--danger-color);
                }}

                /* Story Divider */
                .story-divider {{
                    margin: 60px 0;
                    border-top: 3px dashed var(--gray-300);
                }}

                /* Parameter Education Section */
                .education-section {{
                    padding: 40px;
                    background: var(--light-bg);
                    border-radius: var(--border-radius);
                    margin-bottom: 40px;
                }}

                .education-intro {{
                    text-align: center;
                    max-width: 800px;
                    margin: 0 auto 30px auto;
                    font-size: 1.1rem;
                    color: var(--gray-700);
                }}

                .parameter-education-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
                    gap: 25px;
                }}

                .education-card {{
                    background: white;
                    border-radius: var(--border-radius);
                    box-shadow: var(--shadow-sm);
                    overflow: hidden;
                    border: 1px solid var(--gray-200);
                }}

                .education-header {{
                    padding: 15px 20px;
                    background: var(--light-bg);
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    cursor: pointer;
                    transition: var(--transition);
                    border-bottom: 1px solid var(--gray-200);
                }}

                .education-header:hover {{
                    background: var(--gray-200);
                }}

                .education-title {{
                    font-size: 1.1rem;
                    font-weight: 600;
                    color: var(--dark-color);
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    margin: 0;
                }}

                .learn-more-btn {{
                    background: var(--primary-color);
                    color: white;
                    border: none;
                    padding: 5px 15px;
                    border-radius: 20px;
                    font-size: 0.8rem;
                    font-weight: 600;
                    cursor: pointer;
                    transition: var(--transition);
                }}

                .learn-more-btn:hover {{
                    background: var(--primary-dark);
                }}

                .education-content {{
                    padding: 0;
                    max-height: 0;
                    overflow: hidden;
                    transition: max-height 0.5s ease, padding 0.3s ease;
                }}

                .education-content.active {{
                    padding: 20px;
                    max-height: 2000px;
                }}

                .example-comparison {{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 20px;
                    margin: 20px 0;
                }}

                .good-example, .poor-example {{
                    padding: 15px;
                    border-radius: var(--border-radius-sm);
                }}

                .good-example {{
                    background-color: #d4edda;
                    border-left: 4px solid var(--success-color);
                }}

                .poor-example {{
                    background-color: #f8d7da;
                    border-left: 4px solid var(--danger-color);
                }}

                .example-text {{
                    background: rgba(255, 255, 255, 0.7);
                    padding: 10px;
                    border-radius: 5px;
                    margin: 10px 0;
                    font-style: italic;
                }}

                .improvement-tips {{
                    background: var(--light-bg);
                    padding: 15px;
                    border-radius: var(--border-radius-sm);
                    margin-top: 20px;
                    border-left: 3px solid var(--warning-color);
                }}

                /* Responsive Design */
                @media (max-width: 768px) {{
                    .section {{
                        padding: 30px 20px;
                    }}

                    .header {{
                        padding: 30px 20px;
                    }}

                    .header h1 {{
                        font-size: 2rem;
                    }}

                    .header-info {{
                        flex-direction: column;
                        gap: 20px;
                    }}

                    .parameter-education-grid {{
                        grid-template-columns: 1fr;
                    }}

                    .example-comparison {{
                        grid-template-columns: 1fr;
                    }}
                }}

                @media (max-width: 480px) {{
                    .header {{
                        padding: 30px 20px;
                    }}

                    .header h1 {{
                        font-size: 2rem;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📊 STORY SENSE ANALYSIS REPORT</h1>

                    <div class="header-info">
                        <div class="header-item">
                            <span class="header-label">Total User Stories Analyzed:</span>
                            <span class="header-value">{len(all_analyses)}</span>
                        </div>

                        <div class="header-item">
                            <span class="header-label">Generated on:</span>
                            <span class="header-value">{datetime.datetime.now().strftime("%B %d, %Y at %I:%M %p")}</span>
                        </div>
                    </div>

                    <p>Comprehensive analysis of all user stories with quality scores, recommendations, and insights</p>
                </div>

                <div class="section">
                    <h2>📈 Performance Metrics</h2>

                    <div class="metrics-grid">
                        <div class="metric-card">
                            <p class="metric-label">Average Quality Score</p>
                            <p class="metric-value">{avg_score:.1f}</p>
                            <p class="metric-label">out of 10</p>
                        </div>

                        <div class="metric-card">
                            <p class="metric-label">Total Processing Time</p>
                            <p class="metric-value">{metrics_summary.get('total_duration', 0):.1f}</p>
                            <p class="metric-label">seconds</p>
                        </div>

                        <div class="metric-card">
                            <p class="metric-label">LLM Tokens Used</p>
                            <p class="metric-value">{metrics_summary.get('llm_tokens', 0):,}</p>
                            <p class="metric-label">total tokens</p>
                        </div>

                        <div class="metric-card">
                            <p class="metric-label">Context Documents</p>
                            <p class="metric-value">{metrics_summary.get('vector_queries', 0)}</p>
                            <p class="metric-label">queries made</p>
                        </div>  
                    </div>
                    
                    
                    <div class="metrics-section">
                        <h3>💰 Cost Analysis</h3>
                        <div class="metrics-grid">
                            <div class="metric-card cost-card">
                                <p class="metric-label">Total Estimated Cost</p>
                                <p class="metric-value">\\${metrics_summary.get('total_estimated_cost', 0):.4f}</p>
                                <p class="metric-label">USD</p>
                            </div>
                            <div class="metric-card cost-card">
                                <p class="metric-label">Total Estimated Cost</p>
                                <p class="metric-value">₹{metrics_summary.get('total_estimated_cost_inr', 0):.2f}</p>
                                <p class="metric-label">INR</p>
                            </div>
                            <div class="metric-card cost-card">
                                <p class="metric-label">Cost per Story</p>
                                <p class="metric-value">\\${(metrics_summary.get('total_estimated_cost', 0) / len(all_analyses) if len(all_analyses) > 0 else 0):.4f}</p>
                                <p class="metric-label">USD</p>
                            </div>
                            <div class="metric-card cost-card">
                                <p class="metric-label">Cost per Story</p>
                                <p class="metric-value">₹{(metrics_summary.get('total_estimated_cost_inr', 0) / len(all_analyses) if len(all_analyses) > 0 else 0):.2f}</p>
                                <p class="metric-label">INR</p>
                            </div>
                        </div>

                        <div class="metrics-grid" style="margin-top: 20px;">
                            <div class="metric-card">
                                <p class="metric-label">Input Token Cost</p>
                                <p class="metric-value">\\${metrics_summary.get('cost_breakdown', {}).get('llm', {}).get('input_cost', 0):.4f}</p>
                                <p class="metric-label">USD</p>
                            </div>
                            <div class="metric-card">
                                <p class="metric-label">Output Token Cost</p>
                                <p class="metric-value">\\${metrics_summary.get('cost_breakdown', {}).get('llm', {}).get('output_cost', 0):.4f}</p>
                                <p class="metric-label">USD</p>
                            </div>
                            <div class="metric-card">
                                <p class="metric-label">Input Token Cost</p>
                                <p class="metric-value">₹{metrics_summary.get('cost_breakdown', {}).get('llm', {}).get('input_cost_inr', 0):.2f}</p>
                                <p class="metric-label">INR</p>
                            </div>
                            <div class="metric-card">
                                <p class="metric-label">Output Token Cost</p>
                                <p class="metric-value">₹{metrics_summary.get('cost_breakdown', {}).get('llm', {}).get('output_cost_inr', 0):.2f}</p>
                                <p class="metric-label">INR</p>
                            </div>
                        </div>
                    </div>

                    <div class="metrics-section">
                        <h3>LLM Performance</h3>
                        <div class="metrics-grid">
                            <div class="metric-card">
                                <p class="metric-label">Total LLM Calls</p>
                                <p class="metric-value">{metrics_summary.get('llm_calls', 0)}</p>
                            </div>
                            <div class="metric-card">
                                <p class="metric-label">Input Tokens</p>
                                <p class="metric-value">{metrics_summary.get('llm_input_tokens', 0):,}</p>
                            </div>
                            <div class="metric-card">
                                <p class="metric-label">Output Tokens</p>
                                <p class="metric-value">{metrics_summary.get('llm_output_tokens', 0):,}</p>
                            </div>
                            <div class="metric-card">
                                <p class="metric-label">Avg. LLM Latency</p>
                                <p class="metric-value">{metrics_summary.get('llm_avg_latency', 0):.2f}s</p>
                            </div>
                        </div>
                    </div>

                    <div class="metrics-section">
                        <h3>PGVector Performance</h3>
                        <div class="metrics-grid">
                            <div class="metric-card">
                                <p class="metric-label">Vector Queries</p>
                                <p class="metric-value">{metrics_summary.get('vector_queries', 0)}</p>
                            </div>
                            <div class="metric-card">
                                <p class="metric-label">Avg. Query Time</p>
                                <p class="metric-value">{metrics_summary.get('vector_db', {}).get('avg_query_time', 0):.2f}s</p>
                            </div>
                            <div class="metric-card">
                                <p class="metric-label">Vectors Stored</p>
                                <p class="metric-value">{metrics_summary.get('vector_db', {}).get('total_vectors_stored', 0)}</p>
                            </div>
                            <div class="metric-card">
                                <p class="metric-label">Avg. Store Time</p>
                                <p class="metric-value">{metrics_summary.get('vector_db', {}).get('avg_store_time', 0):.2f}s</p>
                            </div>
                        </div>
                    </div>

                    <div class="metrics-section">
                        <h3>System Resources</h3>
                        <div class="metrics-grid">
                            <div class="metric-card">
                                <p class="metric-label">Peak Memory Usage</p>
                                <p class="metric-value">{metrics_summary.get('peak_memory_mb', 0):.2f} MB</p>
                            </div>
                            <div class="metric-card">
                                <p class="metric-label">CPU Usage</p>
                                <p class="metric-value">{metrics_summary.get('system', {}).get('cpu_percent', 0):.1f}%</p>
                            </div>
                            <div class="metric-card">
                                <p class="metric-label">Total Batches</p>
                                <p class="metric-value">{len(metrics_summary.get('batches', {}))}</p>
                            </div>
                            <div class="metric-card">
                                <p class="metric-label">Total Errors</p>
                                <p class="metric-value">{metrics_summary.get('error_count', 0)}</p>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="section">
                    <h2>🏆 User Stories Ranking</h2>

                    <p>All user stories ranked by overall quality score (highest to lowest):</p>

                    <div class="ranking-table-container">
                        <table class="ranking-table">
                            <thead>
                                <tr>
                                    <th>Rank</th>
                                    <th>Story ID</th>
                                    <th>Overall</th>
                                    <th>User-Centered</th>
                                    <th>Independent</th>
                                    <th>Negotiable</th>
                                    <th>Valuable</th>
                                    <th>Estimable</th>
                                    <th>Small</th>
                                    <th>Testable</th>
                                    <th>Acceptance</th>
                                    <th>Prioritized</th>
                                    <th>Collaboration</th>
                                    <th>Needs Changes</th>
                                    <th>Time (s)</th>
                                </tr>
                            </thead>
                            <tbody>
                                {self._generate_ranking_rows(sorted_analyses)}
                            </tbody>
                        </table>
                    </div>
                </div>

                <div class="section toc">
                    <h2>📋 Table of Contents</h2>

                    <ul class="toc-list">
                        {self._generate_toc_items(sorted_analyses)}
                    </ul>
                </div>

                <div class="section">
                    <h2>📖 Detailed Story Analyses</h2>

                    {self._generate_detailed_stories(sorted_analyses)}
                </div>

                <div class="section">
                    <h2>📚 Understanding the Analysis Parameters</h2>

                    <p class="education-intro">
                        This analysis evaluates your user story against 10 key quality parameters based on the INVEST criteria and industry best practices.
                        Click on each parameter below to learn what it means, see examples, and understand how to improve your stories.
                    </p>

                    <div class="parameter-education-grid">
                        {self._generate_education_cards()}
                    </div>
                </div>
            </div>

            <script>
                // Initialize interactive features
                document.addEventListener('DOMContentLoaded', function() {{
                    // Initialize parameter cards for quality analysis
                    initializeParameterCards();

                    // Initialize education cards
                    initializeEducationCards();
                }});

                function initializeParameterCards() {{
                    const parameterHeaders = document.querySelectorAll('.parameter-header');

                    parameterHeaders.forEach(header => {{
                        header.addEventListener('click', function() {{
                            const content = this.nextElementSibling;
                            const isActive = content.classList.contains('active');

                            // Close all parameter contents in the same story
                            const storyDetail = this.closest('.story-detail');
                            if (storyDetail) {{
                                storyDetail.querySelectorAll('.parameter-content').forEach(item => {{
                                    item.classList.remove('active');
                                }});
                            }}

                            // Toggle current content
                            if (!isActive) {{
                                content.classList.add('active');
                            }}
                        }});
                    }});
                }}

                function initializeEducationCards() {{
                    const educationHeaders = document.querySelectorAll('.education-header');

                    educationHeaders.forEach(header => {{
                        header.addEventListener('click', function() {{
                            const content = this.nextElementSibling;
                            const isActive = content.classList.contains('active');
                            const button = this.querySelector('.learn-more-btn');

                            // Close all education contents
                            document.querySelectorAll('.education-content').forEach(item => {{
                                item.classList.remove('active');
                            }});

                            document.querySelectorAll('.learn-more-btn').forEach(btn => {{
                                btn.textContent = 'Learn More';
                            }});

                            // Toggle current content
                            if (!isActive) {{
                                content.classList.add('active');
                                if (button) {{
                                    button.textContent = 'Show Less';
                                }}
                            }}
                        }});
                    }});
                }}
            </script>
        </body>
        </html>
        """

        # Create output directory if it doesn't exist
        os.makedirs(output_file, exist_ok=True)

        # Write the HTML content to a file
        current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        html_report_path = f"{output_file}/combined_story_sense_report_{current_time}.html"

        with open(html_report_path, "w", encoding="utf-8") as file:
            file.write(html_content)

        print(Fore.GREEN + f"\n📊 Combined HTML report generated: {html_report_path}\n" + Style.RESET_ALL)
        return html_report_path

    def _generate_ranking_rows(self, sorted_analyses):
        """Generate ranking table rows"""
        rows_html = ""
        for rank, analysis in enumerate(sorted_analyses, 1):
            processing_time = analysis.get('processing_time', 0)
            rows_html += f"""
            <tr>
                <td>{rank}</td>
                <td><a href="#story-{analysis['us_id']}">{analysis['us_id']}</a></td>
                <td class="{self._get_score_class(analysis['overall_score'])}">{analysis['overall_score']:.1f}</td>
                <td class="{self._get_score_class(analysis['user_centered_score'])}">{analysis['user_centered_score']}</td>
                <td class="{self._get_score_class(analysis['independent_score'])}">{analysis['independent_score']}</td>
                <td class="{self._get_score_class(analysis['negotiable_score'])}">{analysis['negotiable_score']}</td>
                <td class="{self._get_score_class(analysis['valuable_score'])}">{analysis['valuable_score']}</td>
                <td class="{self._get_score_class(analysis['estimable_score'])}">{analysis['estimable_score']}</td>
                <td class="{self._get_score_class(analysis['small_score'])}">{analysis['small_score']}</td>
                <td class="{self._get_score_class(analysis['testable_score'])}">{analysis['testable_score']}</td>
                <td class="{self._get_score_class(analysis['acceptance_criteria_score'])}">{analysis['acceptance_criteria_score']}</td>
                <td class="{self._get_score_class(analysis['prioritized_score'])}">{analysis['prioritized_score']}</td>
                <td class="{self._get_score_class(analysis['collaboration_score'])}">{analysis['collaboration_score']}</td>
                <td class="{self._get_recommendation_class(analysis['recommendation']['recommended'])}">{analysis['recommendation']['recommended'].upper()}</td>
                <td>{processing_time:.2f}</td>
            </tr>
            """
        return rows_html

    def _generate_toc_items(self, sorted_analyses):
        """Generate table of contents items"""
        items_html = ""
        for analysis in sorted_analyses:
            score_class = self._get_score_class(analysis['overall_score'])
            items_html += f"""
            <li class="toc-item">
                <a href="#story-{analysis['us_id']}" class="toc-link">
                    <span>User Story {analysis['us_id']}</span>
                    <span class="toc-score {score_class}">{analysis['overall_score']:.1f}/10</span>
                </a>
            </li>
            """
        return items_html

    def _generate_detailed_stories(self, sorted_analyses):
        """Generate detailed story sections with dropdown quality analysis"""
        stories_html = ""
        for analysis in sorted_analyses:
            score_class = self._get_score_class(analysis['overall_score'])

            stories_html += f"""
            <div class="story-detail" id="story-{analysis['us_id']}">
                <div class="story-detail-header">
                    <h3 class="story-detail-title">User Story {analysis['us_id']}</h3>
                    <span class="story-detail-score bg-{score_class}">{analysis['overall_score']:.1f}/10</span>
                </div>

                <div class="story-detail-content">
                    {analysis['userstory']}
                </div>

                <div class="parameter-analysis">
                    <h4>💡 Recommended Improvements</h4>
                    <div class="recommendation-box">
                        <div class="recommendation-header">
                            <h3 class="recommendation-title">Changes Recommended</h3>
                            <span class="recommendation-badge {self._get_recommendation_class(analysis['recommendation']['recommended'])}">{analysis['recommendation']['recommended'].upper()}</span>
                        </div>

                        <div class="recommendation-section">
                            <h5>Description Changes:</h5>
                            <div class="diff-box">{analysis['recommendation']['descriptionDiff']}</div>
                        </div>

                        <div class="recommendation-section">
                            <h5>Acceptance Criteria Changes:</h5>
                            <div class="diff-box">{analysis['recommendation']['acceptanceCriteriaDiff']}</div>
                        </div>

                        <div class="improved-story">
                            <h5>✨ Improved User Story</h5>
                            <div class="improved-story-content">{analysis['recommendation']['newUserStory']}</div>
                        </div>
                    </div>
                </div>

                <div class="parameter-analysis">
                    <h4>📊 Quality Analysis</h4>
                    <div class="parameter-analysis">
                        {self._generate_parameter_cards_dropdown(analysis)}
                    </div>
                </div>

                <div class="parameter-analysis">
                    <h4>💎 Key Insights & Recommendations</h4>
                    <div class="parameter-grid">
                        <div class="parameter-card">
                            <div class="parameter-header">
                                <h5 class="parameter-name">🔍 Key Insights</h5>
                            </div>
                            <ul>
                                {self._generate_list_items(analysis.get('key_insights', ['No insights available']))}
                            </ul>
                        </div>

                        <div class="parameter-card">
                            <div class="parameter-header">
                                <h5 class="parameter-name">⚠️ Potential Risks</h5>
                            </div>
                            <ul>
                                {self._generate_list_items(analysis.get('potential_risks', ['No risks identified']))}
                            </ul>
                        </div>

                        <div class="parameter-card">
                            <div class="parameter-header">
                                <h5 class="parameter-name">🎯 Action Items</h5>
                            </div>
                            <ul>
                                {self._generate_list_items(analysis.get('recommendations', ['No action items available']))}
                            </ul>
                        </div>
                    </div>
                </div>
            </div>

            <div class="story-divider"></div>
            """

        return stories_html

    def _generate_parameter_cards_dropdown(self, analysis):
        """Generate parameter cards in dropdown format for combined report"""
        parameters = [
            ('user_centered', 'User-Centered', '👤'),
            ('independent', 'Independent', '🔗'),
            ('negotiable', 'Negotiable', '💬'),
            ('valuable', 'Valuable', '💎'),
            ('estimable', 'Estimable', '📏'),
            ('small', 'Small', '📦'),
            ('testable', 'Testable', '🧪'),
            ('acceptance_criteria', 'Acceptance Criteria', '✅'),
            ('prioritized', 'Prioritized', '🎯'),
            ('collaboration', 'Collaboration', '🤝')
        ]

        cards_html = ""
        for param_key, param_name, icon in parameters:
            score = analysis.get(f'{param_key}_score', 0)
            justification = analysis.get(f'{param_key}_justification', 'No analysis provided')
            recommendations = analysis.get(f'{param_key}_recommendations', ['No recommendations provided'])
            score_class = self._get_score_class(score)

            recommendations_html = ""
            if score < 10 and recommendations and recommendations != ['No recommendations provided']:
                recommendations_html = f"""
                <div class="parameter-recommendations">
                    <h5>💡 Recommendations for Improvement:</h5>
                    <ul>
                        {self._generate_list_items(recommendations)}
                    </ul>
                </div>
                """
            elif score >= 10:
                recommendations_html = f"""
                <div class="parameter-recommendations">
                    <h5>💡 Excellent Work!</h5>
                    <p>This aspect of your user story is excellent! No improvements needed.</p>
                </div>
                """

            cards_html += f"""
            <div class="parameter-card">
                <div class="parameter-header">
                    <h3 class="parameter-title">{icon} {param_name}</h3>
                    <span class="parameter-score {score_class}">{score}/10</span>
                </div>
                <div class="parameter-content">
                    <div class="parameter-justification">
                        <p>{justification}</p>
                    </div>
                    {recommendations_html}
                </div>
            </div>
            """

        return cards_html

    def prepare_context_for_storage(self, input_context):
        """Prepare context dataframe for storage by ensuring it has a 'text' column"""
        if 'text' not in input_context.columns:
            # Try to find a suitable column to use as text
            text_like_columns = [col for col in input_context.columns if
                                 any(name in col.lower() for name in ['text', 'content', 'description', 'context'])]

            if text_like_columns:
                # Use the first matching column
                input_context = input_context.rename(columns={text_like_columns[0]: 'text'})
                print(Fore.GREEN + f"Renamed column '{text_like_columns[0]}' to 'text'" + Style.RESET_ALL)
            elif len(input_context.columns) > 0:
                # Use the first column
                input_context = input_context.rename(columns={input_context.columns[0]: 'text'})
                print(Fore.GREEN + f"Renamed column '{input_context.columns[0]}' to 'text'" + Style.RESET_ALL)
            else:
                # Empty dataframe, add a dummy text column
                input_context['text'] = "No context available"
                print(Fore.YELLOW + "Added dummy 'text' column to empty context dataframe" + Style.RESET_ALL)

        return input_context

    def store_user_stories_in_pgvector(self, input_us):
        """Store user stories in PGVector for future reference"""
        # Create a collection name for user stories
        user_stories_collection = "user_stories"

        # Create a new PGVector connector for user stories
        pgvector_stories = PGVectorConnector(collection_name=user_stories_collection,
                                             metrics_manager=self.metrics_manager)

        # Prepare user stories for storage
        stories_for_db = []
        for _, row in input_us.iterrows():
            us_id = row['ID']
            description = row['Description']
            acceptance_criteria = row['AcceptanceCriteria']
            user_story_text = f"ID: {us_id}\nDescription: {description}\nAcceptance Criteria: {acceptance_criteria}"
            stories_for_db.append({"text": user_story_text, "metadata": {"id": us_id}})

        # Store user stories in PGVector
        success = pgvector_stories.vector_store_documents(stories_for_db)
        if success:
            print(Fore.GREEN + f"\nStored {len(stories_for_db)} user stories in PGVector\n" + Style.RESET_ALL)
        else:
            print(Fore.RED + "\nFailed to store user stories in PGVector\n" + Style.RESET_ALL)

        # Run diagnostics
        pgvector_stories.diagnose_database()

    def analyze_stories_in_batches(self, input_us, output_file, batch_size=5, parallel=False):
        """Analyze user stories in batches to handle large datasets efficiently"""
        # Start timing
        overall_start_time = time.time()

        # Calculate total batches
        total_stories = len(input_us)
        total_batches = (total_stories + batch_size - 1) // batch_size  # Ceiling division

        print(f"Processing {total_stories} user stories in {total_batches} batches of {batch_size}")

        # Collect all analyses
        all_analyses = []

        # Process in batches
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, total_stories)

            print(f"\nProcessing batch {batch_num + 1}/{total_batches} (stories {start_idx + 1}-{end_idx})")

            batch_analyses = []
            batch_start_time = time.time()

            # Process this batch
            if parallel and batch_size > 1:
                # Parallel processing using concurrent.futures
                def process_story(row_idx):
                    row = input_us.iloc[row_idx]
                    us_id = row['ID']
                    description = row['Description']
                    acceptance_criteria = row['AcceptanceCriteria']

                    user_story_description_updated = ("User Story Description:" + description +
                                                      "\nAcceptance Criteria : \n" + acceptance_criteria)

                    print(f"Analyzing User Story {us_id}...\n")
                    story_start_time = time.time()

                    analysis_results = self.story_analyzer.analyze_user_story(
                        {"us_id": us_id, "userstory": user_story_description_updated, "context": "",
                         "context_quality": "none", "context_count": 0}
                    )

                    story_end_time = time.time()
                    story_processing_time = story_end_time - story_start_time

                    # Record metrics for this user story
                    self.metrics_manager.record_user_story_metrics(
                        us_id=us_id,
                        processing_time=story_processing_time,
                        context_count=0,
                        context_quality='none',
                        overall_score=analysis_results['overall_score']
                    )

                    # Generate individual report
                    self.generate_report(analysis_results, us_id, output_file)

                    return analysis_results

                # Use ThreadPoolExecutor for parallel processing
                with ThreadPoolExecutor(max_workers=min(batch_size, 5)) as executor:
                    futures = [executor.submit(process_story, start_idx + i)
                               for i in range(end_idx - start_idx)]

                    for future in futures:
                        try:
                            analysis = future.result()
                            batch_analyses.append(analysis)
                        except Exception as e:
                            error_msg = f"Error in parallel processing: {e}"
                            logging.error(error_msg)
                            self.metrics_manager.record_error('parallel_processing_error', error_msg)

            else:
                # Sequential processing
                for i in range(start_idx, end_idx):
                    row = input_us.iloc[i]
                    us_id = row['ID']
                    description = row['Description']
                    acceptance_criteria = row['AcceptanceCriteria']

                    user_story_description_updated = ("User Story Description:" + description +
                                                      "\nAcceptance Criteria : \n" + acceptance_criteria)

                    print(f"Analyzing User Story {us_id}...\n")
                    story_start_time = time.time()

                    try:
                        analysis_results = self.story_analyzer.analyze_user_story(
                            {"us_id": us_id, "userstory": user_story_description_updated, "context": "",
                             "context_quality": "none", "context_count": 0}
                        )

                        story_end_time = time.time()
                        story_processing_time = story_end_time - story_start_time

                        # Record metrics for this user story
                        self.metrics_manager.record_user_story_metrics(
                            us_id=us_id,
                            processing_time=story_processing_time,
                            context_count=0,
                            context_quality='none',
                            overall_score=analysis_results['overall_score']
                        )

                        # Generate individual report
                        self.generate_report(analysis_results, us_id, output_file)
                        batch_analyses.append(analysis_results)
                    except Exception as e:
                        error_msg = f"Error processing user story {us_id}: {e}"
                        logging.error(error_msg)
                        self.metrics_manager.record_error('story_processing_error', error_msg)

            # Add batch results to overall results
            all_analyses.extend(batch_analyses)

            # Calculate batch metrics
            batch_time = time.time() - batch_start_time
            print(f"Batch {batch_num + 1} completed in {batch_time:.2f} seconds")

            # Record batch metrics
            self.metrics_manager.record_batch_metrics(
                batch_num=batch_num + 1,
                story_count=len(batch_analyses),
                processing_time=batch_time
            )

            # Update memory usage
            self._update_memory_usage()

            # Optional: Add a small delay between batches to prevent rate limiting
            if batch_num < total_batches - 1:
                time.sleep(1)

        # End overall timing
        overall_end_time = time.time()
        total_processing_time = overall_end_time - overall_start_time

        # Generate combined report
        self.generate_combined_html_report(all_analyses, output_file)

        # Save metrics
        metrics_file = self.metrics_manager.save_metrics(output_file + "/Metrics")

        # Print summary
        metrics_summary = self.metrics_manager.get_metrics_summary()
        print(f"\nProcessing Summary:")
        print(f"Total stories: {metrics_summary['story_count']}")
        print(f"Total time: {metrics_summary['total_duration']:.2f} seconds")
        print(f"Average time per story: {metrics_summary['avg_story_time']:.2f} seconds")
        print(f"LLM calls: {metrics_summary['llm_calls']}")
        print(f"Total tokens: {metrics_summary['llm_tokens']}")
        print(f"Peak memory: {metrics_summary['peak_memory_mb']:.2f} MB")
        print(f"Detailed metrics saved to: {metrics_file}")

        return all_analyses

    def analyze_stories_with_context_in_batches(self, input_us, input_context, output_file, batch_size=5,
                                                parallel=False):
        """Analyze user stories with context in batches"""
        # Start timing
        overall_start_time = time.time()

        if not os.path.exists(self.saved_contexts_path):
            os.makedirs(self.saved_contexts_path, exist_ok=True)
            # Create an empty file with .csv extension
            open(self.csv_file, 'a').close()
            os.makedirs(self.embed_data_mtc_path, exist_ok=True)

        # Process context library first using ContextManager
        context_status = self.context_manager.check_and_process_context_library()

        # Then process the input_context if provided (for backward compatibility)
        if input_context is not None and not input_context.empty:
            # Prepare context for storage by ensuring it has a 'text' column
            input_context = self.prepare_context_for_storage(input_context)

            # Save input context to CSV
            input_context.to_csv(self.cleaned_mtc_csv_path, index=False)
            print(Fore.GREEN + '\nSaved the input context in a CSV file \n' + Style.RESET_ALL)

            # Store context in PGVector
            vector_store_start = time.time()
            pgvector = PGVectorConnector(collection_name="additional_context", metrics_manager=self.metrics_manager)
            self.pgvector = pgvector  # Store for metrics access
            success = pgvector.vector_store(self.cleaned_mtc_csv_path)
            vector_store_time = time.time() - vector_store_start

            if success:
                print(Fore.GREEN + f"\nStored context in PGVector collection 'additional_context'\n" + Style.RESET_ALL)
            else:
                print(Fore.RED + "\nFailed to store context in PGVector\n" + Style.RESET_ALL)
                self.metrics_manager.record_error('vector_store_error', "Failed to store context in PGVector")

            # Run diagnostics
            pgvector.diagnose_database()

        # Store user stories in PGVector
        self.store_user_stories_in_pgvector(input_us)

        # Calculate total batches
        total_stories = len(input_us)
        total_batches = (total_stories + batch_size - 1) // batch_size  # Ceiling division

        print(f"Processing {total_stories} user stories with context in {total_batches} batches of {batch_size}")

        # Collect all analyses
        all_analyses = []

        # Process in batches
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, total_stories)

            print(f"\nProcessing batch {batch_num + 1}/{total_batches} (stories {start_idx + 1}-{end_idx})")

            batch_analyses = []
            batch_start_time = time.time()

            # Process this batch
            if parallel and batch_size > 1:
                # Parallel processing using concurrent.futures
                def process_story_with_context(row_idx):
                    row = input_us.iloc[row_idx]
                    us_id = row['ID']
                    description = row['Description']
                    acceptance_criteria = row['AcceptanceCriteria']

                    user_story_description_updated = ("User Story Description:" + description +
                                                      "\nAcceptance Criteria : \n" + acceptance_criteria)

                    print(f"Processing User Story {us_id} with context...\n")
                    story_start_time = time.time()

                    # Use ContextManager to search for relevant context
                    query_start_time = time.time()
                    context_results = self.context_manager.search_context(
                        query=user_story_description_updated,
                        k=self.num_context_retrieve
                    )
                    query_time = time.time() - query_start_time

                    # Format the context for analysis
                    formatted_context = self._format_context_for_analysis(context_results)
                    context_count = self._count_context_documents(context_results)
                    context_quality = self._determine_context_quality(context_results)

                    # For backward compatibility, also search in the additional_context collection
                    if hasattr(self, 'pgvector'):
                        additional_context, docs_with_similarity_score, threshold = self.pgvector.retrieval_context(
                            query=user_story_description_updated, k=self.num_context_retrieve)

                        # Add additional context if available
                        if additional_context:
                            if formatted_context:
                                formatted_context += "\n\n## ADDITIONAL CONTEXT:\n" + additional_context
                            else:
                                formatted_context = additional_context

                            # Update context count and quality
                            context_count += len(docs_with_similarity_score)
                            if len(docs_with_similarity_score) > 0:
                                context_quality = "high" if context_quality == "high" else "medium"

                    if formatted_context:
                        print(f"Found relevant context for User Story {us_id}")
                        print(f"Analyzing User Story {us_id} with context...\n")

                        analysis_results = self.story_analyzer.analyze_user_story(
                            {
                                "us_id": us_id,
                                "userstory": user_story_description_updated,
                                "context": formatted_context,
                                "context_quality": context_quality,
                                "context_count": context_count
                            }
                        )
                    else:
                        print(Fore.RED + f"\nSimilar context not found for User Story {us_id}\n" + Style.RESET_ALL)
                        print(Fore.GREEN + "\nAnalyzing User Story without context\n" + Style.RESET_ALL)

                        analysis_results = self.story_analyzer.analyze_user_story(
                            {
                                "us_id": us_id,
                                "userstory": user_story_description_updated,
                                "context": "",
                                "context_quality": "none",
                                "context_count": 0
                            }
                        )

                    story_end_time = time.time()
                    story_processing_time = story_end_time - story_start_time

                    # Record metrics for this user story
                    self.metrics_manager.record_user_story_metrics(
                        us_id=us_id,
                        processing_time=story_processing_time,
                        context_count=context_count,
                        context_quality=context_quality if formatted_context else 'none',
                        overall_score=analysis_results['overall_score']
                    )

                    # Generate individual report
                    self.generate_report(analysis_results, us_id, output_file)

                    return analysis_results

                # Use ThreadPoolExecutor for parallel processing
                with ThreadPoolExecutor(max_workers=min(batch_size, 5)) as executor:
                    futures = [executor.submit(process_story_with_context, start_idx + i)
                               for i in range(end_idx - start_idx)]

                    for future in futures:
                        try:
                            analysis = future.result()
                            batch_analyses.append(analysis)
                        except Exception as e:
                            error_msg = f"Error in parallel processing with context: {e}"
                            logging.error(error_msg)
                            self.metrics_manager.record_error('parallel_processing_error', error_msg)

            else:
                # Sequential processing
                for i in range(start_idx, end_idx):
                    row = input_us.iloc[i]
                    us_id = row['ID']
                    description = row['Description']
                    acceptance_criteria = row['AcceptanceCriteria']

                    user_story_description_updated = ("User Story Description:" + description +
                                                      "\nAcceptance Criteria : \n" + acceptance_criteria)

                    print(f"Processing User Story {us_id} with context...\n")
                    story_start_time = time.time()

                    try:
                        # Use ContextManager to search for relevant context
                        query_start_time = time.time()
                        try:
                            context_results = self.context_manager.search_context(
                                query=user_story_description_updated,
                                k=self.num_context_retrieve
                            )
                            query_time = time.time() - query_start_time

                            # Format the context for analysis
                            formatted_context, file_type_info = self._format_context_for_analysis(context_results)
                            context_count = self._count_context_documents(context_results)
                            context_quality = self._determine_context_quality(context_results)
                        except Exception as context_error:
                            logging.error(f"Error retrieving context: {context_error}")
                            formatted_context = ""
                            context_count = 0
                            context_quality = "none"
                            file_type_info = {}

                        # For backward compatibility, also search in the additional_context collection
                        additional_context = ""
                        if hasattr(self, 'pgvector'):
                            try:
                                retrieval_result = self.pgvector.retrieval_context(
                                    query=user_story_description_updated, k=self.num_context_retrieve)

                                # Handle different return value patterns
                                if isinstance(retrieval_result, tuple):
                                    if len(retrieval_result) >= 1:
                                        additional_context = retrieval_result[0]
                                    if len(retrieval_result) >= 2:
                                        docs_with_similarity_score = retrieval_result[1]
                                        # Update context count and quality
                                        if additional_context:
                                            context_count += len(docs_with_similarity_score)
                                            if len(docs_with_similarity_score) > 0:
                                                context_quality = "high" if context_quality == "high" else "medium"
                            except Exception as pgvector_error:
                                logging.error(f"Error retrieving from additional_context: {pgvector_error}")
                                additional_context = ""

                        # Add additional context if available
                        if additional_context:
                            if formatted_context:
                                formatted_context += "\n\n## ADDITIONAL CONTEXT:\n" + additional_context
                            else:
                                formatted_context = additional_context

                        # Skip user stories context retrieval since we only want additional context

                        # Proceed with LLM analysis regardless of context retrieval success
                        if formatted_context:
                            print(f"Found relevant context for User Story {us_id}")
                            print(f"Analyzing User Story {us_id} with context...\n")

                            analysis_results = self.story_analyzer.analyze_user_story(
                                {
                                    "us_id": us_id,
                                    "userstory": user_story_description_updated,
                                    "context": formatted_context,
                                    "context_quality": context_quality,
                                    "context_count": context_count,
                                    "context_file_types": file_type_info if 'file_type_info' in locals() else {}
                                }
                            )
                        else:
                            print(Fore.RED + f"\nSimilar context not found for User Story {us_id}\n" + Style.RESET_ALL)
                            print(Fore.GREEN + "\nAnalyzing User Story without context\n" + Style.RESET_ALL)

                            analysis_results = self.story_analyzer.analyze_user_story(
                                {
                                    "us_id": us_id,
                                    "userstory": user_story_description_updated,
                                    "context": "",
                                    "context_quality": "none",
                                    "context_count": 0
                                }
                            )

                        story_end_time = time.time()
                        story_processing_time = story_end_time - story_start_time

                        # Record metrics for this user story
                        self.metrics_manager.record_user_story_metrics(
                            us_id=us_id,
                            processing_time=story_processing_time,
                            context_count=context_count,
                            context_quality=context_quality if formatted_context else 'none',
                            overall_score=analysis_results['overall_score']
                        )

                        # Generate individual report
                        self.generate_report(analysis_results, us_id, output_file)
                        batch_analyses.append(analysis_results)
                    except Exception as e:
                        error_msg = f"Error processing user story {us_id} with context: {e}"
                        logging.error(error_msg)
                        self.metrics_manager.record_error('story_processing_error', error_msg)

                        # IMPORTANT: Add fallback LLM call when an error occurs
                        try:
                            print(
                                Fore.YELLOW + f"\nError occurred while processing with context. Falling back to no-context analysis.\n" + Style.RESET_ALL)
                            story_start_time = time.time()

                            analysis_results = self.story_analyzer.analyze_user_story(
                                {
                                    "us_id": us_id,
                                    "userstory": user_story_description_updated,
                                    "context": "",
                                    "context_quality": "none",
                                    "context_count": 0
                                }
                            )

                            story_end_time = time.time()
                            story_processing_time = story_end_time - story_start_time

                            # Record metrics for this user story
                            self.metrics_manager.record_user_story_metrics(
                                us_id=us_id,
                                processing_time=story_processing_time,
                                context_count=0,
                                context_quality='none',
                                overall_score=analysis_results['overall_score']
                            )

                            # Generate individual report
                            self.generate_report(analysis_results, us_id, output_file)
                            batch_analyses.append(analysis_results)
                        except Exception as fallback_error:
                            logging.error(f"Fallback analysis also failed: {fallback_error}")

            # Add batch results to overall results
            all_analyses.extend(batch_analyses)

            # Calculate batch metrics
            batch_time = time.time() - batch_start_time
            print(f"Batch {batch_num + 1} completed in {batch_time:.2f} seconds")

            # Record batch metrics
            self.metrics_manager.record_batch_metrics(
                batch_num=batch_num + 1,
                story_count=len(batch_analyses),
                processing_time=batch_time
            )

            # Update memory usage
            self._update_memory_usage()

            # Optional: Add a small delay between batches to prevent rate limiting
            if batch_num < total_batches - 1:
                time.sleep(1)

        # End overall timing
        overall_end_time = time.time()
        total_processing_time = overall_end_time - overall_start_time

        # Generate combined report
        self.generate_combined_html_report(all_analyses, output_file)

        # Save metrics
        metrics_file = self.metrics_manager.save_metrics(output_file + "/Metrics")

        # Print summary
        metrics_summary = self.metrics_manager.get_metrics_summary()
        print(f"\nProcessing Summary:")
        print(f"Total stories: {metrics_summary['story_count']}")
        print(f"Total time: {metrics_summary['total_duration']:.2f} seconds")
        print(f"Average time per story: {metrics_summary['avg_story_time']:.2f} seconds")
        print(f"LLM calls: {metrics_summary['llm_calls']}")
        print(f"Total tokens: {metrics_summary['llm_tokens']}")
        print(f"Vector queries: {metrics_summary['vector_queries']}")
        print(f"Peak memory: {metrics_summary['peak_memory_mb']:.2f} MB")
        print(f"Detailed metrics saved to: {metrics_file}")

        return all_analyses

    def _format_context_for_analysis(self, context_results):
        """Format context results for LLM analysis with file type information"""
        formatted_context = ""
        file_type_info = {}

        # Add each context type with its documents
        for context_type, result in context_results.items():
            if context_type == 'file_type_stats':
                continue  # Skip the stats entry

            if result.get('document_count', 0) > 0:
                formatted_context += f"\n## {context_type.replace('_', ' ').upper()} CONTEXT:\n"

                # Add file type information for each document
                for similarity, content in result.get('documents', {}).items():
                    file_info = result.get('file_types', {}).get(similarity, {})
                    file_type = file_info.get('type', 'unknown')
                    source_file = file_info.get('source', 'unknown')

                    # Store file type info for analytics
                    if file_type not in file_type_info:
                        file_type_info[file_type] = 0
                    file_type_info[file_type] += 1

                    # Add source information to the context
                    formatted_context += f"\n[Source: {source_file} (Type: {file_type})]\n"
                    formatted_context += content + "\n"

        # Add file type summary
        if file_type_info:
            formatted_context += "\n## CONTEXT SOURCES SUMMARY:\n"
            for file_type, count in file_type_info.items():
                formatted_context += f"- {file_type}: {count} documents\n"

        return formatted_context.strip(), file_type_info

    def _count_context_documents(self, context_results):
        """Count total documents across all context types"""
        return sum(result.get('document_count', 0) for result in context_results.values())

    def _determine_context_quality(self, context_results):
        """Determine overall context quality based on results and file types"""
        total_docs = self._count_context_documents(context_results)
        total_text = sum(len(result.get('context', '')) for result in context_results.values()
                         if isinstance(result, dict) and 'context' in result)

        # Get file type statistics
        file_type_stats = context_results.get('file_type_stats', {})

        # Define file type quality weights
        file_type_weights = {
            'pdf': 1.2,  # Higher quality for PDF documents
            'docx': 1.1,  # Good quality for Word documents
            'doc': 1.1,
            'xlsx': 1.0,  # Spreadsheets are average
            'xls': 1.0,
            'pptx': 0.9,  # Presentations slightly lower
            'ppt': 0.9,
            'txt': 0.8,  # Plain text is lower
            'md': 0.8,
            'image': 0.7,  # Images lowest unless they have OCR
            'unknown': 0.6
        }

        # Calculate weighted quality score
        if total_docs == 0:
            return "none"

        # Base quality score
        base_quality = 1.0
        if total_docs > 10 or total_text > 200000:
            base_quality = 2.0  # High
        elif total_docs > 5 or total_text > 100000:
            base_quality = 1.5  # Medium
        else:
            base_quality = 1.0  # Low

        # Apply file type weights
        weighted_score = 0
        total_weight = 0

        for file_type, count in file_type_stats.items():
            weight = file_type_weights.get(file_type, 0.7)
            weighted_score += weight * count
            total_weight += count

        if total_weight > 0:
            final_score = (base_quality * weighted_score) / total_weight
        else:
            final_score = base_quality

        # Convert score to quality level
        if final_score >= 1.8:
            return "high"
        elif final_score >= 1.3:
            return "medium"
        elif final_score > 0:
            return "low"
        else:
            return "none"

    # Keep existing methods, but modify analyze_stories to use the batch processing method
    def analyze_stories(self, input_us, output_file):
        """Analyze user stories without additional context - now uses batch processing"""
        # Get batch size from config or use default
        batch_size = 5
        try:
            batch_size = int(self.config_parser_io.get('Processing', 'batch_size', fallback='5'))
        except (configparser.NoSectionError, configparser.NoOptionError):
            pass

        # Get parallel processing flag from config or use default
        parallel = False
        try:
            parallel = self.config_parser_io.getboolean('Processing', 'parallel_processing', fallback=False)
        except (configparser.NoSectionError, configparser.NoOptionError):
            pass

        return self.analyze_stories_in_batches(input_us, output_file, batch_size, parallel)

    def analyze_stories_with_context(self, input_us, input_context, output_file):
        """Analyze user stories with additional context - now uses batch processing"""
        # Get batch size from config or use default
        batch_size = 5
        try:
            batch_size = int(self.config_parser_io.get('Processing', 'batch_size', fallback='5'))
        except (configparser.NoSectionError, configparser.NoOptionError):
            pass

        # Get parallel processing flag from config or use default
        parallel = False
        try:
            parallel = self.config_parser_io.getboolean('Processing', 'parallel_processing', fallback=False)
        except (configparser.NoSectionError, configparser.NoOptionError):
            pass

        return self.analyze_stories_with_context_in_batches(input_us, input_context, output_file, batch_size, parallel)