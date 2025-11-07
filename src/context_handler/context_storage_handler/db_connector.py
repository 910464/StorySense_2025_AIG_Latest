import logging
import time
import os
from datetime import datetime
from sqlalchemy import create_engine
import psycopg2


class PGDatabase:
    def __init__(self, config_loader, collection_name='default', max_retries=3, retry_backoff=2):
        self.config = config_loader
        self.collection_name = collection_name
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

        self.connection_string = None
        self.engine = None
        self.conn = None
        self.cursor = None

        # Setup database and ensure schema
        self.setup_database()

    def _build_connection_string(self):
        ssl_param = f"?sslmode={self.config.ssl_mode}" if getattr(self.config, 'ssl_mode', None) else ""
        self.connection_string = (
            f"postgresql://{self.config.db_user}:{self.config.db_password}@{self.config.db_host}:"
            f"{self.config.db_port}/{self.config.db_name}{ssl_param}"
        )

    def setup_database(self):
        try:
            self._build_connection_string()

            # Create SQLAlchemy engine
            self.engine = create_engine(
                self.connection_string,
                pool_size=5,
                max_overflow=10,
                pool_timeout=30,
                pool_recycle=1800,
                connect_args={"application_name": "StorySense"}
            )

            # Connect with retry using psycopg2 for direct cursor operations
            self.connect_with_retry()

            # Create extension and tables/indexes if necessary
            self.cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_embeddings (
                id SERIAL PRIMARY KEY,
                collection_name VARCHAR(255) NOT NULL,
                content TEXT NOT NULL,
                metadata JSONB,
                embedding vector(1536),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            self.cursor.execute("CREATE INDEX IF NOT EXISTS collection_idx ON document_embeddings (collection_name);")

            self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS collection_content_idx ON document_embeddings (collection_name, md5(content));
            """)

            # Create vector index based on similarity metric
            if getattr(self.config, 'similarity_metric', 'cosine') == 'cosine':
                self.cursor.execute("""
                CREATE INDEX IF NOT EXISTS embedding_cosine_idx 
                ON document_embeddings 
                USING ivfflat (embedding vector_cosine_ops) 
                WITH (lists = 100);
                """)
            elif self.config.similarity_metric == 'l2':
                self.cursor.execute("""
                CREATE INDEX IF NOT EXISTS embedding_l2_idx 
                ON document_embeddings 
                USING ivfflat (embedding vector_l2_ops) 
                WITH (lists = 100);
                """)
            elif self.config.similarity_metric == 'inner':
                self.cursor.execute("""
                CREATE INDEX IF NOT EXISTS embedding_ip_idx 
                ON document_embeddings 
                USING ivfflat (embedding vector_ip_ops) 
                WITH (lists = 100);
                """)

            self.conn.commit()
        except Exception as e:
            logging.error(f"Error setting up database: {e}")
            if hasattr(self, 'conn') and self.conn:
                try:
                    self.conn.rollback()
                except Exception:
                    pass
            raise

    def connect_with_retry(self):
        retry_count = 0
        last_exception = None

        while retry_count < self.max_retries:
            try:
                self.conn = psycopg2.connect(
                    host=self.config.db_host,
                    port=self.config.db_port,
                    database=self.config.db_name,
                    user=self.config.db_user,
                    password=self.config.db_password,
                    sslmode=self.config.ssl_mode,
                    application_name="StorySense"
                )
                self.cursor = self.conn.cursor()
                logging.info("Successfully connected to Aurora PostgreSQL")
                return
            except Exception as e:
                retry_count += 1
                last_exception = e
                wait_time = self.retry_backoff * retry_count
                logging.warning(f"Connection attempt {retry_count} failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)

        logging.error(f"Failed to connect to database after {self.max_retries} attempts")
        raise last_exception

    def reconnect_if_needed(self):
        try:
            if not self.cursor:
                self.connect_with_retry()
                return
            self.cursor.execute("SELECT 1")
        except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
            logging.warning(f"Database connection lost: {e}. Reconnecting...")
            self.connect_with_retry()

    def diagnose_database(self):
        try:
            self.reconnect_if_needed()

            self.cursor.execute("SELECT collection_name, COUNT(*) FROM document_embeddings GROUP BY collection_name")
            collections = self.cursor.fetchall()
            logging.info(f"Collections in database: {collections}")
            print(f"Collections in database: {collections}")

            self.cursor.execute("""
            SELECT collection_name, content, metadata 
            FROM document_embeddings 
            ORDER BY id 
            LIMIT 3
            """)
            samples = self.cursor.fetchall()
            for i, (collection, content, metadata) in enumerate(samples):
                preview = content[:100] + "..." if len(content) > 100 else content
                logging.info(f"Sample {i + 1} from {collection}: {preview} Metadata: {metadata}")
                print(f"Sample {i + 1} from {collection}: {preview} Metadata: {metadata}")

            self.cursor.execute("""
            SELECT 
                collection_name, 
                vector_dims(embedding) as dimensions
            FROM document_embeddings
            LIMIT 1
            """)
            dimensions = self.cursor.fetchone()
            if dimensions:
                logging.info(f"Vector dimensions: {dimensions[1]} for collection {dimensions[0]}")
                print(f"Vector dimensions: {dimensions[1]} for collection {dimensions[0]}")
            else:
                logging.warning("No vectors found in database")
                print("No vectors found in database")

            logging.info(f"Using similarity metric: {self.config.similarity_metric}")
            print(f"Using similarity metric: {self.config.similarity_metric}")
            logging.info(f"Similarity threshold: {self.config.threshold}")
            print(f"Similarity threshold: {self.config.threshold}")

            return True
        except Exception as e:
            logging.error(f"Error in diagnose_database: {e}")
            print(f"Error in diagnose_database: {e}")
            if hasattr(self, 'conn') and self.conn:
                try:
                    self.conn.rollback()
                except Exception:
                    pass
            return False

    def close(self):
        try:
            if hasattr(self, 'cursor') and self.cursor:
                self.cursor.close()
            if hasattr(self, 'conn') and self.conn:
                self.conn.close()
        except Exception as e:
            logging.error(f"Error closing database connections: {e}")
