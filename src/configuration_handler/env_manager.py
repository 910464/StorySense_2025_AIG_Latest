import os
import logging
import threading
import time
import boto3
from pathlib import Path
import json
import configparser
from src.configuration_handler.config_loader import load_configuration
from botocore.exceptions import ClientError

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnvManager:
    """Manages environment variables and secrets for StorySense"""

    # Define the root path of the 'src' directory relative to this file
    _SRC_ROOT = Path(__file__).parent.parent

    def __init__(self):

        # Load environment variables from a .env file
        load_configuration()


        # Initialize AWS clients if credentials are available
        self.secrets_manager = None
        self.ssm = None
        if self._has_aws_credentials():
            self._init_aws_clients()

    def _has_aws_credentials(self):
        """Check if AWS credentials are available (including session token)"""
        return ((os.environ.get('AWS_ACCESS_KEY_ID') and
                 os.environ.get('AWS_SECRET_ACCESS_KEY')) or
                os.path.exists(os.path.expanduser('~/.aws/credentials')))

    def _init_aws_clients(self):
        """Initialize AWS clients with session token if available"""
        try:
            # Create session with session token if available
            session_kwargs = {
                'region_name': os.environ.get('AWS_REGION', 'us-east-1')
            }

            # Add credentials if available in environment
            if os.environ.get('AWS_ACCESS_KEY_ID') and os.environ.get('AWS_SECRET_ACCESS_KEY'):
                session_kwargs['aws_access_key_id'] = os.environ.get('AWS_ACCESS_KEY_ID')
                session_kwargs['aws_secret_access_key'] = os.environ.get('AWS_SECRET_ACCESS_KEY')

                if os.environ.get('AWS_SESSION_TOKEN'):
                    session_kwargs['aws_session_token'] = os.environ.get('AWS_SESSION_TOKEN')

            session = boto3.Session(**session_kwargs)

            self.secrets_manager = session.client('secretsmanager')
            self.ssm = session.client('ssm')
            logger.info("AWS clients initialized with session token" if os.environ.get(
                'AWS_SESSION_TOKEN') else "AWS clients initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize AWS clients: {e}")

    def get_env(self, key, default=None):
        """Get environment variable with fallback to default"""
        return os.environ.get(key, default)

    def get_required_env(self, key):
        """Get required environment variable or raise exception"""
        value = os.environ.get(key)
        if value is None:
            raise ValueError(f"Required environment variable {key} is not set")
        return value

    def get_secret(self, secret_name):
        """Get secret from AWS Secrets Manager"""
        if not self.secrets_manager:
            logger.warning("AWS Secrets Manager client not initialized")
            return None

        try:
            response = self.secrets_manager.get_secret_value(SecretId=secret_name)
            if 'SecretString' in response:
                return json.loads(response['SecretString'])
            else:
                return None
        except ClientError as e:
            logger.error(f"Failed to get secret {secret_name}: {e}")
            return None

    def get_parameter(self, parameter_name, decrypt=True):
        """Get parameter from AWS Systems Manager Parameter Store"""
        if not self.ssm:
            logger.warning("AWS SSM client not initialized")
            return None

        try:
            response = self.ssm.get_parameter(Name=parameter_name, WithDecryption=decrypt)
            return response['Parameter']['Value']
        except ClientError as e:
            logger.error(f"Failed to get parameter {parameter_name}: {e}")
            return None

    def get_db_config(self):
        """Get database configuration from environment or secrets"""
        # Try to get from Secrets Manager first
        use_secrets = self.get_env('USE_SECRETS_MANAGER', 'false').lower() == 'true'
        secret_name = self.get_env('DB_SECRETS_NAME')

        if use_secrets and secret_name and self.secrets_manager:
            db_secret = self.get_secret(secret_name)
            if db_secret:
                logger.info(f"Using database configuration from Secrets Manager: {secret_name}")
                return {
                    'host': db_secret.get('host'),
                    'port': db_secret.get('port', 5432),
                    'database': db_secret.get('dbname'),
                    'user': db_secret.get('username'),
                    'password': db_secret.get('password'),
                    'sslmode': db_secret.get('sslmode', 'require')
                }

        # Fall back to environment variables
        logger.info("Using database configuration from environment variables")
        return {
            'host': self.get_env('DB_HOST'),
            'port': self.get_env('DB_PORT', 5432),
            'database': self.get_env('DB_NAME', 'storysense'),
            'user': self.get_env('DB_USER'),
            'password': self.get_env('DB_PASSWORD'),
            'sslmode': self.get_env('DB_SSL_MODE', 'require')
        }

    def get_llm_config(self):
        """Get LLM configuration from environment"""
        return {
            'llm_family': self.get_env('LLM_FAMILY', 'AWS'),
            'model_id': self.get_env('LLM_MODEL_ID', 'anthropic.claude-3-sonnet-20240229-v1:0'),
            'temperature': float(self.get_env('LLM_TEMPERATURE', 0.05)),
            'max_tokens': int(self.get_env('LLM_MAX_TOKENS', 10000))
        }

    def get_vector_db_config(self):
        """Get vector database configuration from environment"""
        return {
            'similarity_metric': self.get_env('SIMILARITY_METRIC', 'cosine'),
            'threshold': float(self.get_env('SIMILARITY_THRESHOLD', 0.7))
        }

    def get_guardrails_config(self):
        """Get guardrails configuration from environment or config"""
        # Check if guardrails are enabled
        guardrails_enabled = self.get_env('USE_GUARDRAILS', 'false').lower() == 'true'

        if not guardrails_enabled:
            return None

        # Try to get guardrails from environment variables
        guardrail_id = self.get_env('GUARDRAIL_ID')
        guardrail_region = self.get_env('GUARDRAIL_REGION', 'us-east-1')
        guardrail_description = self.get_env('GUARDRAIL_DESCRIPTION', 'Default guardrail')

        # If not in environment, try to get from config file
        if not guardrail_id:
            try:
                config = configparser.ConfigParser()
                # Construct a robust path to the config file
                config_path = self._SRC_ROOT / 'Config' / 'Config.properties'
                config.read(config_path)
                if config.has_section('Guardrails') and config.has_option('Guardrails', 'guardrail_id'):
                    guardrail_id = config.get('Guardrails', 'guardrail_id')
                    guardrail_region = config.get('Guardrails', 'region', fallback='us-east-1')
                    guardrail_description = config.get('Guardrails', 'description', fallback='Default guardrail')
            except Exception as e:
                logger.warning(f"Error reading guardrails from config: {e}")

        # If we have a guardrail ID, return the configuration
        if guardrail_id:
            return {
                "default": {
                    "guardrail_id": guardrail_id,
                    "region": guardrail_region,
                    "description": guardrail_description
                }
            }

        return None

    def get_aws_session(self):
        """Get AWS session with credentials including session token if available"""
        session_kwargs = {
            'region_name': os.environ.get('AWS_REGION', 'us-east-1')
        }

        # Add credentials if available in environment
        if os.environ.get('AWS_ACCESS_KEY_ID') and os.environ.get('AWS_SECRET_ACCESS_KEY'):
            session_kwargs['aws_access_key_id'] = os.environ.get('AWS_ACCESS_KEY_ID')
            session_kwargs['aws_secret_access_key'] = os.environ.get('AWS_SECRET_ACCESS_KEY')

            # Add session token if available
            if os.environ.get('AWS_SESSION_TOKEN'):
                session_kwargs['aws_session_token'] = os.environ.get('AWS_SESSION_TOKEN')

        return boto3.Session(**session_kwargs)

    def debug_aws_credentials(self):
        """Debug AWS credentials and print detailed information"""
        logger.info("Debugging AWS credentials...")

        # Check environment variables
        aws_key = os.environ.get('AWS_ACCESS_KEY_ID')
        aws_secret = os.environ.get('AWS_SECRET_ACCESS_KEY')
        aws_token = os.environ.get('AWS_SESSION_TOKEN')
        aws_region = os.environ.get('AWS_REGION')

        logger.info(f"Environment variables: "
                    f"AWS_ACCESS_KEY_ID: {'Set' if aws_key else 'Not set'}, "
                    f"AWS_SECRET_ACCESS_KEY: {'Set' if aws_secret else 'Not set'}, "
                    f"AWS_SESSION_TOKEN: {'Set' if aws_token else 'Not set'}, "
                    f"AWS_REGION: {aws_region or 'Not set'}")

        # Check AWS config files
        home = os.path.expanduser("~")
        aws_config = os.path.join(home, ".aws", "config")
        aws_credentials = os.path.join(home, ".aws", "credentials")

        logger.info(f"AWS config file exists: {os.path.exists(aws_config)}")
        logger.info(f"AWS credentials file exists: {os.path.exists(aws_credentials)}")

        # Try to get caller identity
        try:
            session = self.get_aws_session()
            sts = session.client('sts')
            identity = sts.get_caller_identity()
            logger.info(f"AWS credentials valid. Account: {identity['Account']}, "
                        f"User: {identity['UserId']}")
            return True
        except Exception as e:
            logger.error(f"AWS credential validation failed: {e}")
            return False

    def rotate_aws_credentials(self, new_credentials):
        """Update AWS credentials at runtime without restarting the application"""
        # Update environment variables
        os.environ['AWS_ACCESS_KEY_ID'] = new_credentials['aws_access_key_id']
        os.environ['AWS_SECRET_ACCESS_KEY'] = new_credentials['aws_secret_access_key']
        os.environ['AWS_SESSION_TOKEN'] = new_credentials.get('aws_session_token', '')

        # Reinitialize AWS clients
        self._init_aws_clients()

        logging.info("AWS credentials rotated successfully")

        # Return validation result
        return self.debug_aws_credentials()

    def start_credential_health_check(self, interval=300):
        """Start a background thread to periodically check credential health"""

        def check_credentials():
            while True:
                try:
                    # Test credentials
                    valid = self.debug_aws_credentials()
                    if not valid:
                        logging.warning("AWS credentials may be invalid or expired")
                        # Send notification or alert
                    time.sleep(interval)
                except Exception as e:
                    logging.error(f"Error in credential health check: {e}")

        thread = threading.Thread(target=check_credentials, daemon=True)
        thread.start()
        logging.info(f"Credential health check started (interval: {interval}s)")

    def encrypt_sensitive_data(self, data, key=None):
        """Encrypt sensitive data like credentials"""
        try:
            from cryptography.fernet import Fernet

            # Generate or use provided key
            if not key:
                key = Fernet.generate_key()

            f = Fernet(key)
            encrypted_data = f.encrypt(json.dumps(data).encode())

            return {
                "encrypted_data": encrypted_data.decode(),
                "key": key.decode()
            }
        except ImportError:
            logging.warning("cryptography package not installed, returning data unencrypted")
            return {
                "encrypted_data": None,
                "key": None,
                "unencrypted_data": data
            }

    def decrypt_sensitive_data(self, encrypted_data, key):
        """Decrypt sensitive data"""
        try:
            from cryptography.fernet import Fernet

            f = Fernet(key.encode())
            decrypted_data = f.decrypt(encrypted_data.encode())

            return json.loads(decrypted_data.decode())
        except ImportError:
            logging.warning("cryptography package not installed, cannot decrypt data")
            return None