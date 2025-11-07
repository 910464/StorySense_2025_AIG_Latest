import os
import logging
from datetime import datetime

from src.context_handler.context_storage_handler.db_config_loader import DBConfigLoader
from src.context_handler.context_storage_handler.db_connector import PGDatabase
from src.context_handler.context_storage_handler.metrics_reporter import MetricsReporter
# from src.configuration_handler.env_manager import EnvManager
from src.metrics.metrics_manager import MetricsManager

# Keep same relative config path for backward compatibility
config_path = '../Config'


class PGVectorOrchestrator:
    """Holds shared components used by the modular PGVector classes.

    This class replaces the monolithic initialization logic from the original
    connector and exposes attributes used by Retriever, Store and Searcher.
    """

    def __init__(self, collection_name="default", config_file_path=config_path + '/Config.properties',
                 metrics_manager=None):
        self.collection_name = collection_name
        self.config_file_path = config_file_path
        self.metrics_manager = metrics_manager or MetricsManager()

        # Environment helper
        # self.env_manager = EnvManager()

        # Load configuration and embeddings
        self.config = DBConfigLoader(config_file_path=self.config_file_path,
                                   metrics_manager=self.metrics_manager)

        # Metrics reporter (local aggregator + CloudWatch best-effort)
        self.metrics_reporter = MetricsReporter(collection_name=self.collection_name,
                                               metrics_manager=self.metrics_manager)

        # Database connector handles engine / connection / schema
        self.db = PGDatabase(self.config, collection_name=self.collection_name)

        # Expose convenient attributes for backward compatibility
        self.embeddings = self.config.embeddings
        self.similarity_metric = self.config.similarity_metric
        self.threshold = self.config.threshold
        self.model_name = self.config.model_name
        self.local_storage_path = self.config.local_storage_path

        # Retrieval context output dir
        self.retrieval_dir = '../Output/RetrievalContext'
        try:
            os.makedirs(self.retrieval_dir, exist_ok=True)
        except Exception as e:
            logging.warning(f"Could not create retrieval directory {self.retrieval_dir}: {e}")

    def diagnose_database(self):
        return self.db.diagnose_database()

    def reconnect_if_needed(self):
        return self.db.reconnect_if_needed()

    def close(self):
        try:
            if hasattr(self, 'db') and self.db:
                self.db.close()
        except Exception as e:
            logging.error(f"Error closing database connections: {e}")

    # Helper passthroughs (used by callers relying on old API)
    def get_metrics(self):
        return self.metrics_reporter.get_metrics()

    def save_metrics(self, output_dir="../Output/Metrics"):
        return self.metrics_reporter.save_metrics(output_dir=output_dir)
