import os
import time
import logging
import json
from psycopg2.extras import execute_values

from src.context_handler.context_storage_handler.pgvector_orchestrator import PGVectorOrchestrator


class PGVectorStore:
    """Handles storing documents into PGVector (chunking, dedupe, batched inserts)."""

    def __init__(self, orchestrator: PGVectorOrchestrator):
        self.orch = orchestrator
        self.db = orchestrator.db
        self.embeddings = orchestrator.embeddings
        self.metrics_reporter = orchestrator.metrics_reporter
        self.metrics_manager = orchestrator.metrics_manager
        # self.env_manager = orchestrator.env_manager
        self.collection_name = orchestrator.collection_name

    def vector_store(self, fpath):
        """Load additional context from a CSV file into the vector store."""
        start_time = time.time()
        retry_count = 0

        while retry_count < 3:
            try:
                self.db.reconnect_if_needed()

                import pandas as pd
                df = pd.read_csv(fpath, encoding="utf-8")
                if 'text' not in df.columns:
                    logging.error("CSV file must contain a 'text' column")
                    try:
                        self.metrics_manager.record_error('vector_store_error', "CSV file must contain a 'text' column")
                    except Exception:
                        pass
                    return False

                documents = []
                for _, row in df.iterrows():
                    doc = {"text": row['text']}
                    metadata = {}
                    for col in df.columns:
                        if col != 'text':
                            metadata[col] = row[col]
                    if metadata:
                        doc["metadata"] = metadata
                    documents.append(doc)

                result = self.vector_store_documents(documents)

                end_time = time.time()
                operation_time = end_time - start_time
                try:
                    self.metrics_reporter.record_vector_operation(operation_type='store', item_count=len(documents),
                                                                  duration=operation_time)
                except Exception:
                    logging.debug("Failed to record metrics for vector_store")
                return result

            except (Exception,) as e:
                retry_count += 1
                if retry_count >= 3:
                    logging.error(f"Failed to store vectors after 3 attempts: {e}")
                    try:
                        self.metrics_manager.record_error('vector_store_error', str(e))
                    except Exception:
                        pass
                    return False
                wait_time = 2 * retry_count
                logging.warning(f"Database error in vector_store, retrying ({retry_count}/3) in {wait_time}s: {e}")
                time.sleep(wait_time)
                self.db.connect_with_retry()

    def vector_store_documents(self, documents):
        """Store documents in PGVector (chunking, dedupe, batched inserts)."""
        start_time = time.time()
        retry_count = 0

        while retry_count < 3:
            try:
                self.db.reconnect_if_needed()

                from src.metrics.semantic_chunker import SemanticChunker

                chunk_size = int(os.getenv('CHUNK_SIZE', '512'))
                chunk_overlap = int(os.getenv('CHUNK_OVERLAP', '50'))
                chunker = SemanticChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

                chunked_documents = []
                for doc in documents:
                    chunks = chunker.chunk_document(doc)
                    chunked_documents.extend(chunks)

                documents_to_process = chunked_documents

                batch_size = 100
                total_stored = 0
                cursor = self.db.cursor

                for i in range(0, len(documents_to_process), batch_size):
                    batch = documents_to_process[i:i + batch_size]
                    data = []
                    for doc in batch:
                        text = doc['text']
                        metadata = json.dumps(doc.get('metadata', {}))

                        cursor.execute("""
                        SELECT id FROM document_embeddings 
                        WHERE collection_name = %s AND md5(content) = md5(%s)
                        """, (self.collection_name, text))

                        if cursor.fetchone():
                            logging.info(f"Document already exists in collection '{self.collection_name}', skipping")
                            continue

                        embedding = self.embeddings.embed_query(text)
                        embedding_str = f"[{','.join(str(x) for x in embedding)}]"

                        data.append((self.collection_name, text, metadata, embedding_str))

                    if not data:
                        logging.info(f"No new documents to store in batch {i // batch_size + 1}")
                        continue

                    execute_values(
                        cursor,
                        """
                        INSERT INTO document_embeddings (collection_name, content, metadata, embedding)
                        VALUES %s
                        """,
                        data,
                        template="(%s, %s, %s, %s::vector)"
                    )

                    self.db.conn.commit()
                    total_stored += len(data)
                    logging.info(f"Stored batch {i // batch_size + 1} with {len(data)} documents")

                end_time = time.time()
                operation_time = end_time - start_time

                try:
                    self.metrics_reporter.record_vector_operation(operation_type='store', item_count=total_stored,
                                                                  duration=operation_time)
                except Exception:
                    logging.debug("Failed to record metrics for vector_store_documents")

                logging.info(
                    f"Successfully stored {total_stored} documents in PGVector collection '{self.collection_name}'")
                return True

            except (Exception,) as e:
                retry_count += 1
                if retry_count >= 3:
                    logging.error(f"Failed to store documents after 3 attempts: {e}")
                    try:
                        self.metrics_manager.record_error('vector_store_error', str(e))
                    except Exception:
                        pass
                    return False
                wait_time = 2 * retry_count
                logging.warning(f"Database error in vector_store_documents, retrying ({retry_count}/3) in {wait_time}s: {e}")
                time.sleep(wait_time)
                self.db.connect_with_retry()
