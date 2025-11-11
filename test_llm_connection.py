
"""
Test script to check if the AWS Bedrock connection and LLM response is working
"""

import os
import sys
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add src directory to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_llm_connection():
    """Test basic LLM connection and JSON response"""
    
    try:
        from src.metrics.metrics_manager import MetricsManager
        from src.llm_layer.model_manual_test_llm import LLM
        
        print("Testing LLM Connection...")
        print("=" * 50)
        
        # Create metrics manager and LLM instance
        metrics_manager = MetricsManager()
        llm = LLM('AWS', metrics_manager=metrics_manager)
        
        # Simple test prompt that should return JSON
        test_prompt = """
        Please respond with valid JSON only. No other text.
        
        Analyze this simple user story and respond with JSON:
        "As a user, I want to login so that I can access my account."
        
        Respond with this exact JSON structure:
        {
            "user_centered_score": 8,
            "user_centered_justification": "Good user focus",
            "user_centered_recommendations": ["Add more specific user type"],
            "test_status": "success"
        }
        """
        
        print("Sending test prompt to LLM...")
        response = llm.send_request_multimodal(test_prompt, image_path="", call_type="test")
        
        print(f"Response received (length: {len(response)})")
        print("Raw response:")
        print("-" * 30)
        print(response)
        print("-" * 30)
        
        # Try to parse as JSON
        try:
            json_data = json.loads(response.strip())
            print("✅ Successfully parsed as JSON!")
            print(f"Keys: {list(json_data.keys())}")
            
            if 'test_status' in json_data:
                print("✅ Test status found in response")
            else:
                print("⚠️  Test status not found - might be a different response")
                
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse as JSON: {e}")
            
            # Try our extraction method
            try:
                from src.prompt_layer.storysense_analyzer import StorySenseAnalyzer
                analyzer = StorySenseAnalyzer('AWS')
                extracted = analyzer._extract_and_parse_json(response)
                
                if isinstance(extracted, dict) and 'user_centered_score' in extracted:
                    print("✅ Extraction method worked!")
                else:
                    print("❌ Extraction method also failed")
                    print(f"Extracted type: {type(extracted)}")
                    
            except Exception as extraction_error:
                print(f"❌ Extraction method error: {extraction_error}")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

def test_environment_variables():
    """Test if all required environment variables are set"""
    
    print("\nTesting Environment Variables...")
    print("=" * 50)
    
    required_vars = [
        'AWS_ACCESS_KEY_ID',
        'AWS_SECRET_ACCESS_KEY', 
        'AWS_REGION',
        'LLM_FAMILY',
        'LLM_MODEL_ID',
        'LLM_TEMPERATURE',
        'LLM_MAX_TOKENS'
    ]
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Hide sensitive values
            if 'KEY' in var or 'SECRET' in var:
                display_value = f"{value[:10]}...{value[-5:]}" if len(value) > 15 else "***"
            else:
                display_value = value
            print(f"✅ {var}: {display_value}")
        else:
            print(f"❌ {var}: Not set")

if __name__ == "__main__":
    test_environment_variables()
    test_llm_connection()