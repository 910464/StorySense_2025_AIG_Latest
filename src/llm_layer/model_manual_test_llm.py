from src.aws_layer.aws_bedrock_connector import AWSBedrockConnector
from langchain.prompts.prompt import PromptTemplate
# from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import configparser
import os
import logging
# from src.configuration_handler.env_manager  import EnvManager
from src.metrics.metrics_manager import MetricsManager

# Set up logging
# logging.basicConfig(level=logging.INFO)


class LLM:
    def __init__(self, llm_family=None, metrics_manager=None):
        """
        :param llm_family: LLM Family chosen (AWS)
        :param metrics_manager: Optional metrics manager instance
        """
        self.config_path = '../Config'
        self.config_parser = configparser.ConfigParser()

        # Initialize environment manager
        # self.env_manager = EnvManager()

        # Use provided metrics_manager or create a new one
        print(f"DEBUG: LLM received metrics_manager: {id(metrics_manager)}")
        self.metrics_manager = metrics_manager or MetricsManager()

        # Get LLM configuration from environment
        # llm_config = self.env_manager.get_llm_config()
        self.llm_family = os.getenv('LLM_FAMILY')
        self.TEMPERATURE = os.getenv('LLM_TEMPERATURE')

        # # Create ConfigAWS.properties if it doesn't exist
        # if not os.path.exists(os.path.join(self.config_path, 'ConfigAWS.properties')):
        #     self.create_default_config_aws()

        # self.config_parser_aws = configparser.ConfigParser()
        # self.config_parser_aws.read(os.path.join(self.config_path, 'ConfigAWS.properties'))

        # Get AWS Bedrock model ID from environment or config
        self.MODEL_ID = os.getenv('LLM_MODEL_ID') or 'anthropic.claude-3-sonnet-20240229-v1:0'
        self.MAX_TOKENS = int(os.getenv('LLM_MAX_TOKENS', '200000'))

        # Validate configuration
        if not self.TEMPERATURE:
            self.TEMPERATURE = 0.05
        else:
            self.TEMPERATURE = float(self.TEMPERATURE)

        logging.info(f"LLM Configuration: Model={self.MODEL_ID}, Temp={self.TEMPERATURE}, MaxTokens={self.MAX_TOKENS}")

        # Initialize AWS Bedrock connector with metrics_manager
        self.aws_bedrock = AWSBedrockConnector(model_id=self.MODEL_ID, metrics_manager=self.metrics_manager)

    def create_default_config_aws(self):
        """Create a default ConfigAWS.properties file"""
        config = configparser.ConfigParser()
        config['AWS'] = {
            'MODEL_ID': 'us.anthropic.claude-3-5-sonnet-20241022-v2:0',
            'MAX_TOKENS': '150000'
        }
        config['StorySense'] = {
            'story_sense_prompt': 'I want you to act as a user story analysis expert. I am giving you a User Story Description and User Story Acceptance Criteria.',
            'story_sense_instruction': 'Analyze the user story for quality, completeness, clarity, and testability. Provide scores on a scale of 1-10 for each aspect, along with justifications, key insights, potential risks, and recommendations for improvement.',
            'story_sense_context': '{context}',
            'story_sense_userStory': '{UserStory}'
        }
        config['NoContext'] = {
            'story_sense_noContext_prompt': 'Act as a User Story Analysis Expert. I am giving you a User Story Description and User Story Acceptance Criteria.',
            'story_sense_noContext_instruction': 'Analyze the user story for quality, completeness, clarity, and testability. Provide scores on a scale of 1-10 for each aspect, along with justifications, key insights, potential risks, and recommendations for improvement.',
            'story_sense_noContext_userStory': '{UserStory}'
        }

        os.makedirs(self.config_path, exist_ok=True)
        with open(os.path.join(self.config_path, 'ConfigAWS.properties'), 'w') as f:
            config.write(f)
        logging.info("Created default ConfigAWS.properties file")

    def send_request_multimodal(self, template_prompt, image_path, call_type="default", input_variables=None, input_variables_dict=None):
        """
        :param template_prompt: prompt for LLM
        :param image_path: path to the image file
        :param call_type: type of call for metrics tracking
        :param input_variables: List of variable names in the template prompt
        :param input_variables_dict: Dictionary mapping variable names to their values
        :return: LLM response
        """
        try:
            # Replace variables in the template prompt with actual values
            if input_variables and input_variables_dict:
                for var in input_variables:
                    if var in input_variables_dict:
                        template_prompt = template_prompt.replace(f"{{{var}}}", input_variables_dict[var])

            # This method will need to be implemented in AWSBedrockConnector
            # to handle multimodal requests.
            output = self.aws_bedrock.generate_response_multimodal(
                prompt=template_prompt,
                image_path=image_path,
                temperature=self.TEMPERATURE,
                max_tokens=self.MAX_TOKENS,
                call_type=call_type
            )
            return output
        except Exception as e:
            logging.error(f"Error in send_request_multimodal: {e}")
            return f"Error: Unable to get multimodal response from LLM. Details: {str(e)}"
        
 #    def send_request(self, template_prompt, input_variables, input_variables_dict, call_type="default"):
 #
 #        """
 #        :param template_prompt: prompt for LLM
 #        :param input_variables: contains the variable names used in the prompt
 #        :param input_variables_dict: contains the value of variables used in the prompt
 #        :param call_type: type of call for metrics tracking (e.g., "analysis", "validation")
 #        :return: LLM response
 #        """
 #
 #    # Create the prompt template
 #    prompt = PromptTemplate(input_variables=input_variables,template=template_prompt)
 #
 #    # Format the prompt with the input variables
 #    formatted_prompt = prompt.format(**input_variables_dict)
 #
 #    try:
 #
 #        # Send the formatted prompt to AWS Bedrock with call_type for metrics
 #        output = self.aws_bedrock.generate_response(
 #            prompt=formatted_prompt,
 #            temperature=self.TEMPERATURE,
 #            max_tokens=self.MAX_TOKENS,
 #            call_type=call_type
 #        )
 #        return output
 #    except Exception as e:
 #        logging.error(f"Error in send_request: {e}")
 #        return f"Error: Unable to get response from LLM. Please check your AWS credentials and try again. Details: {str(e)}"
