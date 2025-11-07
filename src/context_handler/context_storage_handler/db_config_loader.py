import os
import logging
import configparser
from datetime import datetime

# from src.configuration_handler.env_manager import EnvManager

# Keep same relative config path for backward compatibility
config_path = '../Config'


class DBConfigLoader:
    def __init__(self, config_file_path=config_path + '/Config.properties', metrics_manager=None):
        self.config_file_path = config_file_path
        # self.env_manager = env_manager or EnvManager()
        self.metrics_manager = metrics_manager

        # Defaults
        self.similarity_metric = 'cosine'
        self.threshold = 0.7
        self.db_host = None
        self.db_port = None
        self.db_name = None
        self.db_user = None
        self.db_password = None
        self.ssl_mode = None
        self.model_id = os.getenv('EMBEDDING_MODEL_NAME', 'amazon.titan-embed-text-v1')
        self.local_storage_path = os.getenv('LOCAL_EMBEDDINGS_PATH','../Data/LocalEmbeddings')
        self.embeddings = None

        # Load values
        self.load_config()

    def load_config(self):
        try:
            # Load vector and DB config from environment via EnvManager
            # vector_config = self.env_manager.get_vector_db_config()
            self.similarity_metric = os.getenv('SIMILARITY_METRIC')
            self.threshold = os.getenv('SIMILARITY_THRESHOLD')

            # db_config = self.env_manager.get_db_config()
            self.db_host = os.getenv('DB_HOST')
            self.db_port = os.getenv('DB_PORT')
            self.db_name = os.getenv('DB_NAME')
            self.db_user = os.getenv('DB_USER')
            self.db_password = os.getenv('DB_PASSWORD')
            self.ssl_mode = os.getenv('DB_SSL_MODE')

            # Fallback to config file if needed
            # if not self.db_host or not self.db_user:
            #     self._load_from_config_file()

            # Embedding model from env or config
            self.model_name = os.getenv('EMBEDDING_MODEL_NAME')
            self.local_storage_path = os.getenv('LOCAL_EMBEDDINGS_PATH')

            # Initialize embeddings provider
            from src.aws_layer.aws_titan_embedding import AWSTitanEmbeddings
            self.embeddings = AWSTitanEmbeddings(
                model_id=self.model_name,
                local_storage_path=self.local_storage_path
            )

        except Exception as e:
            logging.error(f"Error loading configuration: {e}")
            if self.metrics_manager:
                self.metrics_manager.record_error('config_error', str(e))

            from src.aws_layer.aws_titan_embedding import AWSTitanEmbeddings
            self.embeddings = AWSTitanEmbeddings(
                model_id=self.model_name,
                local_storage_path=self.local_storage_path
            )

    # def _load_from_config_file(self):
    #     # Read the properties defined in the config.properties file
    #     config = configparser.ConfigParser()
    #     config.read(self.config_file_path)

    #     # Ensure ConfigPG.properties exists
    #     pg_config_path = os.path.join(config_path, 'ConfigPG.properties')
    #     if not os.path.exists(pg_config_path):
    #         self.create_default_pg_config()

    #     config_parser_pg = configparser.ConfigParser()
    #     config_parser_pg.read(pg_config_path)

    #     # Load PG config
    #     self.db_host = config_parser_pg.get('PostgreSQL', 'host')
    #     self.db_port = config_parser_pg.get('PostgreSQL', 'port')
    #     self.db_name = config_parser_pg.get('PostgreSQL', 'database')
    #     self.db_user = config_parser_pg.get('PostgreSQL', 'user')
    #     self.db_password = config_parser_pg.get('PostgreSQL', 'password')
    #     self.similarity_metric = config_parser_pg.get('PostgreSQL', 'similarity_metric', fallback=self.similarity_metric)
    #     self.ssl_mode = config_parser_pg.get('PostgreSQL', 'ssl_mode', fallback='require')

    #     try:
    #         self.model_name = config.get('AdvancedConfigurations', 'embedding_model_name')
    #         self.threshold = float(config.get('AdvancedConfigurations', 'external_model_threshold'))
    #         self.local_storage_path = config.get('AdvancedConfigurations', 'local_embeddings_path', fallback=self.local_storage_path)
    #     except (configparser.NoSectionError, configparser.NoOptionError):
    #         # Keep defaults
    #         pass

    # def create_default_pg_config(self):
    #     config = configparser.ConfigParser()
    #     config['PostgreSQL'] = {
    #         'host': 'pgvector-aig-db.cluster-cxaiah6wepef.us-east-1.rds.amazonaws.com',
    #         'port': '5432',
    #         'database': 'dqgenai',
    #         'user': 'postgres',
    #         'password': 'PostAurAig25',
    #         'similarity_metric': 'cosine',
    #         'ssl_mode': 'require',
    #         'use_secrets_manager': 'false'
    #     }

    #     os.makedirs(config_path, exist_ok=True)
    #     with open(os.path.join(config_path, 'ConfigPG.properties'), 'w') as f:
    #         config.write(f)
    #     logging.info("Created default ConfigPG.properties file")
