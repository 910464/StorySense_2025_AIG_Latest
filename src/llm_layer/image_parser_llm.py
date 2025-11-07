import base64
import logging
from src.llm_layer.model_manual_test_llm import LLM

class ImageParserLLM:
    """
    A class to parse images using a Large Language Model.
    """

    def __init__(self, metrics_manager=None):
        """
        Initializes the ImageParserLLM.
        :param metrics_manager: Optional metrics manager instance.
        """
        self.llm = LLM(metrics_manager=metrics_manager)
        self.logger = logging.getLogger(__name__)

    def _encode_image_to_base64(self, image_path):
        """
        Encodes an image file to a base64 string.
        :param image_path: Path to the image file.
        :return: Base64 encoded string of the image.
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            self.logger.error(f"Error encoding image {image_path} to base64: {e}")
            raise

    def parse_image(self, image_path):
        """
        Parses an image using the LLM.
        :param image_path: Path to the image file.
        :return: The parsed text from the image.
        """
        try:
            # The prompt is now a simple instruction to the LLM.
            # The image data will be handled separately by the multimodal call.
            template_prompt = """
            You are an expert at analyzing images.
            Analyze the following image and provide a detailed description of its content.
            If there is text, extract it verbatim.
            If it is a diagram or wireframe, describe its components and flows.
            """
            
            # The input variables and dictionary are no longer needed here,
            # as the image data is not part of the template.
            
            # The `send_request_multimodal` method is responsible for
            # encoding the image and constructing the correct API payload.
            response = self.llm.send_request_multimodal(
                template_prompt=template_prompt,
                input_variables=[],
                input_variables_dict={},
                call_type="image_analysis",
                image_path=image_path
            )
            
            return response

        except Exception as e:
            self.logger.error(f"Error parsing image {image_path} with LLM: {e}")
            return f"Error: Unable to get response from LLM for image {image_path}. Details: {str(e)}"

    def parse_image_batch(self, image_paths):
        """
        Parses a batch of images using the LLM.
        :param image_paths: A list of paths to image files.
        :return: A list of parsed texts from the images.
        """
        # This is a placeholder for batch processing.
        # For simplicity, we will call parse_image for each image.
        # A more optimized version would make a single API call with multiple images if the API supports it.
        results = []
        for image_path in image_paths:
            results.append(self.parse_image(image_path))
        return results
