import time
import logging
import json
from datetime import datetime

from src.context_handler.context_storage_handler.pgvector_orchestrator import PGVectorOrchestrator


class PGVectorRetriever:
    """Handles vector retrieval queries against a PGVector-backed table.

    This class encapsulates the retrieval_context method previously on the
    monolithic connector.
    """

    def __init__(self, orchestrator: PGVectorOrchestrator):
        self.orch = orchestrator
        self.db = orchestrator.db
        self.embeddings = orchestrator.embeddings
        self.similarity_metric = orchestrator.similarity_metric
        self.threshold = orchestrator.threshold
        self.metrics_reporter = orchestrator.metrics_reporter
        self.metrics_manager = orchestrator.metrics_manager
        self.collection_name = orchestrator.collection_name
        self.retrieval_dir = orchestrator.retrieval_dir

    def retrieval_context(self, query, k):
        """Retrieve similar documents from the configured collection."""
        start_time = time.time()
        retry_count = 0

        while retry_count < 3:
            try:
                self.db.reconnect_if_needed()

                # Get embedding for query
                query_embedding = self.embeddings.embed_query(query)
                embedding_str = f"[{','.join(str(x) for x in query_embedding)}]"

                cursor = self.db.cursor

                if self.similarity_metric == 'cosine':
                    cursor.execute(f"""
                    SELECT content, 1 - (embedding <=> %s::vector) as similarity, metadata
                    FROM document_embeddings
                    WHERE collection_name = %s
                    AND 1 - (embedding <=> %s::vector) > %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                    """, (embedding_str, self.collection_name, embedding_str, float(self.threshold) * 0.8,
                           embedding_str, k))
                elif self.similarity_metric == 'l2':
                    max_distance = 2.0 / float(self.threshold)
                    cursor.execute(f"""
                    SELECT content, 1 / (1 + (embedding <-> %s::vector)) as similarity, metadata
                    FROM document_embeddings
                    WHERE collection_name = %s
                    AND embedding <-> %s::vector < %s
                    ORDER BY embedding <-> %s::vector
                    LIMIT %s;
                    """, (embedding_str, self.collection_name, embedding_str, max_distance, embedding_str, k))
                elif self.similarity_metric == 'inner':
                    cursor.execute(f"""
                    SELECT content, (embedding <#> %s::vector) as similarity, metadata
                    FROM document_embeddings
                    WHERE collection_name = %s
                    AND embedding <#> %s::vector > %s
                    ORDER BY embedding <#> %s::vector DESC
                    LIMIT %s;
                    """, (embedding_str, self.collection_name, embedding_str, float(self.threshold) * 0.8,
                           embedding_str, k))

                results = cursor.fetchall()
                docs_with_similarity_score = {}
                docs_metadata = {}
                context = ''

                logging.info(f"Retrieved {len(results)} results for collection '{self.collection_name}'")

                for content, similarity, metadata in results:
                    docs_with_similarity_score[float(similarity)] = content
                    docs_metadata[float(similarity)] = metadata
                    context += content + '\n'
                    logging.info(f"Similarity: {similarity}, Content preview: {content[:50]}...")

                # Write retrieved context to a file for diagnostics
                try:
                    filename = f"retrieved_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
                    with open(f"{self.retrieval_dir}/{filename}", "a", encoding="utf-8") as file:
                        file.write(context)
                except Exception:
                    # Don't fail retrieval if write fails
                    logging.debug("Failed to write retrieval context to file")

                passing_results = [s for s in docs_with_similarity_score.keys() if s > float(self.threshold)]
                logging.info(f"Found {len(passing_results)} results passing threshold {self.threshold}")

                end_time = time.time()
                operation_time = end_time - start_time
                try:
                    self.metrics_reporter.record_vector_operation(operation_type='query', item_count=len(results),
                                                                  duration=operation_time)
                except Exception:
                    logging.debug("Failed to record metrics for retrieval")

                return context, docs_with_similarity_score, docs_metadata, self.threshold

            except (Exception,) as e:
                retry_count += 1
                if retry_count >= 3:
                    logging.error(f"Failed to retrieve context after 3 attempts: {e}")
                    try:
                        self.metrics_manager.record_error('vector_query_error', str(e))
                    except Exception:
                        pass
                    return "", {}, {}, self.threshold
                wait_time = 2 * retry_count
                logging.warning(f"Database error in retrieval_context, retrying ({retry_count}/3) in {wait_time}s: {e}")
                time.sleep(wait_time)
                self.db.connect_with_retry()
