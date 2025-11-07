import time
import logging

from src.context_handler.context_storage_handler.pgvector_orchestrator import PGVectorOrchestrator
from src.context_handler.context_storage_handler.pgvector_retriever import PGVectorRetriever


class PGVectorSearcher:
    """Handles high-level search across collections using query expansion.

    Mirrors the original search_all_collections behavior but is implemented
    in a standalone class that uses the Retriever component.
    """

    def __init__(self, orchestrator: PGVectorOrchestrator):
        self.orch = orchestrator
        self.retriever = PGVectorRetriever(self.orch)
        self.metrics_manager = orchestrator.metrics_manager
        self.collection_name = orchestrator.collection_name

    def search_all_collections(self, query, k=5):
        """
        Search across both user stories and additional context using query expansion.
        Returns: combined_context, combined_docs, combined_metadata, threshold
        """
        retry_count = 0
        while retry_count < 3:
            try:
                self.orch.reconnect_if_needed()

                # Query expansion
                from src.prompt_layer.query_expander import QueryExpander
                query_expander = QueryExpander(metrics_manager=self.metrics_manager)
                expansion_result = query_expander.expand_query(query)
                search_queries = query_expander.generate_search_queries(expansion_result)

                original_query = query
                combined_context = ""
                combined_docs = {}
                combined_metadata = {}

                for expanded_query in search_queries:
                    context, docs_with_score_context, docs_metadata_context, threshold = self.retriever.retrieval_context(
                        expanded_query, k // max(1, len(search_queries)) + 1)

                    # Search in user stories collection by creating a separate orchestrator + retriever
                    stories_collection = "user_stories"
                    stories_orch = PGVectorOrchestrator(collection_name=stories_collection,
                                                        config_file_path=self.orch.config_file_path,
                                                        metrics_manager=self.metrics_manager)
                    stories_retriever = PGVectorRetriever(stories_orch)
                    stories, docs_with_score_stories, docs_metadata_stories, _ = stories_retriever.retrieval_context(
                        expanded_query, k // max(1, len(search_queries)) + 1)

                    if context:
                        combined_context += f"CONTEXT FROM QUERY '{expanded_query}':\n{context}\n\n"
                    if stories:
                        combined_context += f"STORIES FROM QUERY '{expanded_query}':\n{stories}\n\n"

                    combined_docs.update(docs_with_score_context)
                    combined_docs.update(docs_with_score_stories)
                    combined_metadata.update(docs_metadata_context)
                    combined_metadata.update(docs_metadata_stories)

                logging.info(
                    f"Query expansion: Original query '{original_query}' expanded to {len(search_queries)} queries")

                # Limit to top k results overall
                if len(combined_docs) > k:
                    # Sort by similarity score (highest first)
                    sorted_docs = sorted(combined_docs.items(), key=lambda x: x[0], reverse=True)
                    combined_docs = dict(sorted_docs[:k])
                    combined_metadata = {score: combined_metadata[score] for score in combined_docs.keys() if
                                         score in combined_metadata}

                return combined_context, combined_docs, combined_metadata, self.orch.threshold

            except (Exception,) as e:
                retry_count += 1
                if retry_count >= 3:
                    logging.error(f"Failed to search collections after 3 attempts: {e}")
                    try:
                        self.metrics_manager.record_error('search_collections_error', str(e))
                    except Exception:
                        pass
                    return "", {}, {}, self.orch.threshold
                wait_time = 2 * retry_count
                logging.warning(f"Error in search_all_collections, retrying in {wait_time}s: {e}")
                time.sleep(wait_time)
                self.orch.db.connect_with_retry()
