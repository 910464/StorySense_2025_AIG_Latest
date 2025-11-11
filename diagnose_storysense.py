#!/usr/bin/env python3
"""
Comprehensive diagnostic script for StorySense LLM issues
"""

import os
import sys
import json
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def check_environment():
    """Check all environment variables and configurations"""
    print("🔍 Environment Check")
    print("=" * 50)
    
    # Required environment variables
    env_vars = {
        'AWS_ACCESS_KEY_ID': 'AWS Access Key',
        'AWS_SECRET_ACCESS_KEY': 'AWS Secret Key', 
        'AWS_SESSION_TOKEN': 'AWS Session Token (Optional)',
        'AWS_REGION': 'AWS Region',
        'LLM_FAMILY': 'LLM Family',
        'LLM_MODEL_ID': 'Model ID',
        'LLM_TEMPERATURE': 'Temperature',
        'LLM_MAX_TOKENS': 'Max Tokens'
    }
    
    all_good = True
    for var, description in env_vars.items():
        value = os.getenv(var)
        if value:
            if 'KEY' in var or 'SECRET' in var or 'TOKEN' in var:
                display_value = f"{value[:8]}..." if len(value) > 8 else "***"
            else:
                display_value = value
            print(f"✅ {description}: {display_value}")
        else:
            if var != 'AWS_SESSION_TOKEN':  # Session token is optional
                all_good = False
                print(f"❌ {description}: NOT SET")
            else:
                print(f"⚠️  {description}: Not set (optional)")
    
    return all_good

def test_aws_bedrock_basic():
    """Test basic AWS Bedrock connectivity"""
    print("\n🔗 AWS Bedrock Connectivity Test")
    print("=" * 50)
    
    try:
        import boto3
        
        # Test AWS credentials
        region = os.getenv('AWS_REGION', 'us-east-1')
        session = boto3.Session(region_name=region)
        
        # Try to create bedrock client
        bedrock = session.client('bedrock')
        bedrock_runtime = session.client('bedrock-runtime')
        
        print("✅ AWS Session created successfully")
        print("✅ Bedrock clients created successfully")
        
        # Try to list models (this tests permissions)
        try:
            models = bedrock.list_foundation_models()
            print(f"✅ Found {len(models.get('modelSummaries', []))} available models")
            
            # Check if our model is available
            model_id = os.getenv('LLM_MODEL_ID', 'anthropic.claude-3-sonnet-20240229-v1:0')
            available_models = [m['modelId'] for m in models.get('modelSummaries', [])]
            
            if model_id in available_models:
                print(f"✅ Target model {model_id} is available")
            else:
                print(f"❌ Target model {model_id} is NOT available")
                print("Available models:")
                for model in available_models[:5]:  # Show first 5
                    print(f"   - {model}")
                if len(available_models) > 5:
                    print(f"   ... and {len(available_models) - 5} more")
                return False
                
        except Exception as e:
            print(f"⚠️  Could not list models (permission issue?): {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ AWS Bedrock connectivity failed: {e}")
        return False

def test_llm_simple_request():
    """Test a simple LLM request"""
    print("\n🤖 Simple LLM Request Test")
    print("=" * 50)
    
    try:
        from src.metrics.metrics_manager import MetricsManager
        from src.llm_layer.model_manual_test_llm import LLM
        
        metrics_manager = MetricsManager()
        llm = LLM('AWS', metrics_manager=metrics_manager)
        
        # Very simple prompt
        simple_prompt = 'Respond with exactly this JSON: {"test": "success", "number": 42}'
        
        print("Sending simple test prompt...")
        response = llm.send_request_multimodal(simple_prompt, image_path="", call_type="diagnostic")
        
        print(f"Response received ({len(response)} chars):")
        print("-" * 30)
        print(response)
        print("-" * 30)
        
        # Check if response looks like JSON
        try:
            parsed = json.loads(response.strip())
            print("✅ Response is valid JSON!")
            print(f"   Keys: {list(parsed.keys())}")
            return True
        except json.JSONDecodeError:
            print("❌ Response is not valid JSON")
            
            # Check if it contains JSON
            if '{' in response and '}' in response:
                print("⚠️  Response contains braces, might have extra text")
            else:
                print("❌ Response doesn't even contain JSON braces")
            return False
            
    except Exception as e:
        print(f"❌ LLM test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_story_analysis():
    """Test the story analysis specifically"""
    print("\n📖 Story Analysis Test")
    print("=" * 50)
    
    try:
        from src.prompt_layer.storysense_analyzer import StorySenseAnalyzer
        from src.metrics.metrics_manager import MetricsManager
        
        metrics_manager = MetricsManager()
        analyzer = StorySenseAnalyzer('AWS', metrics_manager=metrics_manager)
        
        # Simple test user story
        user_story_data = {
            "us_id": "TEST-001",
            "userstory": "As a user, I want to login so that I can access my account.",
            "context": "",
            "context_file_types": {},
            "context_quality": "none",
            "context_count": 0
        }
        
        print("Testing story analysis...")
        result = analyzer.analyze_user_story(user_story_data)
        
        print(f"Analysis completed. Result type: {type(result)}")
        
        if isinstance(result, dict):
            print("✅ Got dictionary result")
            
            # Check for key fields
            key_fields = ['user_centered_score', 'overall_score', 'recommendation']
            for field in key_fields:
                if field in result:
                    print(f"✅ Found {field}: {result[field]}")
                else:
                    print(f"❌ Missing {field}")
            
            # Check if it's a fallback analysis
            if result.get('user_centered_justification', '').startswith('Could not analyze'):
                print("❌ Analysis failed - got fallback response")
                print(f"   Reason: {result.get('user_centered_justification', 'Unknown')}")
                return False
            else:
                print("✅ Analysis succeeded!")
                return True
        else:
            print(f"❌ Unexpected result type: {type(result)}")
            return False
            
    except Exception as e:
        print(f"❌ Story analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all diagnostic tests"""
    print(f"🔧 StorySense Diagnostic Tool")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    tests = [
        ("Environment Check", check_environment),
        ("AWS Bedrock Connectivity", test_aws_bedrock_basic),
        ("Simple LLM Request", test_llm_simple_request),
        ("Story Analysis", test_story_analysis)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 50)
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed < len(results):
        print("\n🔧 Troubleshooting Tips:")
        print("1. Make sure all AWS environment variables are set correctly")
        print("2. Check AWS permissions for Bedrock access")
        print("3. Verify the model ID is correct and available in your region")
        print("4. Check network connectivity to AWS")
        print("5. Try running the main StorySense app with debug logging enabled")

if __name__ == "__main__":
    main()