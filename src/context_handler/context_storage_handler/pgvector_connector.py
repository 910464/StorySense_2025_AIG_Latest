import os
import logging
import pandas as pd
import json
import time
from datetime import datetime

from src.context_handler.context_storage_handler.pgvector_orchestrator import PGVectorOrchestrator
from src.context_handler.context_storage_handler.pgvector_retriever import PGVectorRetriever
from src.context_handler.context_storage_handler.pgvector_store import PGVectorStore
from src.context_handler.context_storage_handler.pgvector_searcher import PGVectorSearcher

# Keep same relative config path for backward compatibility
config_path = '../Config'

# Create directory for retrieval context if it doesn't exist
retrieval_dir = '../Output/RetrievalContext'
os.makedirs(retrieval_dir, exist_ok=True)


class PGVectorConnector:
    """Facade preserving the original PGVectorConnector public API.

    Internally composes the modular components implemented in separate files.
    """

    def __init__(self, collection_name="default", config_file_path=config_path + '/Config.properties',
                 metrics_manager=None):
        self.collection_name = collection_name
        self.config_file_path = config_file_path
        self.metrics_manager = metrics_manager

        # Shared orchestrator holding config, DB, embeddings, metrics, env
        self.orch = PGVectorOrchestrator(collection_name=self.collection_name,
                                         config_file_path=self.config_file_path,
                                         metrics_manager=self.metrics_manager)

        # Components
        self.retriever = PGVectorRetriever(self.orch)
        self.store = PGVectorStore(self.orch)
        self.searcher = PGVectorSearcher(self.orch)

        # Backwards-compatible attributes
        self.embeddings = self.orch.embeddings
        self.similarity_metric = self.orch.similarity_metric
        self.threshold = self.orch.threshold
        self.model_name = self.orch.model_name
        self.local_storage_path = self.orch.local_storage_path

    # Database helpers
    def diagnose_database(self):
        return self.orch.diagnose_database()

    def reconnect_if_needed(self):
        return self.orch.reconnect_if_needed()

    # Search / retrieval / storage API preserved
    def search_all_collections(self, query, k=5):
        return self.searcher.search_all_collections(query, k=k)

    def retrieval_context(self, query, k):
        return self.retriever.retrieval_context(query, k)

    def vector_store(self, fpath):
        return self.store.vector_store(fpath)

    def vector_store_documents(self, documents):
        return self.store.vector_store_documents(documents)

    # Metrics
    def get_metrics(self):
        return self.orch.get_metrics()

    def save_metrics(self, output_dir="../Output/Metrics"):
        return self.orch.save_metrics(output_dir=output_dir)

    def __del__(self):
        try:
            self.orch.close()
        except Exception as e:
            logging.error(f"Error closing orchestrator: {e}")