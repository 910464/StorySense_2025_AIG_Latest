#!/usr/bin/env python3
"""
Test script to validate JSON extraction from various LLM response formats
"""

import json
import sys
import os

# Add src directory to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.prompt_layer.storysense_analyzer import StorySenseAnalyzer

def test_json_extraction():
    """Test the JSON extraction with various response formats"""
    
    # Create analyzer instance (we just need the method)
    analyzer = StorySenseAnalyzer('AWS')
    
    # Test cases with different response formats
    test_cases = [
        # Case 1: Clean JSON response
        {
            "name": "Clean JSON",
            "response": '{"user_centered_score": 8, "user_centered_justification": "Good focus on user needs", "user_centered_recommendations": ["Be more specific"]}'
        },
        
        # Case 2: JSON with markdown formatting
        {
            "name": "Markdown JSON",
            "response": '''Here's the analysis:

```json
{
    "user_centered_score": 7,
    "user_centered_justification": "Decent user focus",
    "user_centered_recommendations": ["Add more context"]
}
```

Let me know if you need any clarification!'''
        },
        
        # Case 3: JSON with explanatory text
        {
            "name": "JSON with text",
            "response": '''Based on my analysis, here is the structured response:

{
    "user_centered_score": 6,
    "user_centered_justification": "Some user focus present",
    "user_centered_recommendations": ["Improve user persona"]
}

This analysis considers all the key factors.'''
        },
        
        # Case 4: Multiple JSON objects (should pick the right one)
        {
            "name": "Multiple JSON",
            "response": '''Here's some metadata: {"meta": "data"}

And here's the actual analysis:
{
    "user_centered_score": 9,
    "user_centered_justification": "Excellent user focus",
    "user_centered_recommendations": ["Maintain current approach"]
}'''
        },
        
        # Case 5: Malformed response that should trigger fallback
        {
            "name": "Malformed",
            "response": '''This is not JSON at all, just plain text response without any structured data.'''
        }
    ]
    
    print("Testing JSON Extraction Methods")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print("-" * 30)
        
        try:
            result = analyzer._extract_and_parse_json(test_case['response'])
            
            if isinstance(result, dict):
                print("✅ Successfully extracted JSON")
                print(f"   Keys found: {list(result.keys())[:5]}...")  # Show first 5 keys
                
                # Check if it looks like our expected structure
                if 'user_centered_score' in result:
                    print("✅ Contains expected user story analysis structure")
                else:
                    print("⚠️  Doesn't contain expected structure")
            else:
                print("❌ Result is not a dictionary")
                
        except Exception as e:
            print(f"❌ Exception occurred: {str(e)}")
        
    print("\n" + "=" * 50)
    print("Test completed!")

if __name__ == "__main__":
    test_json_extraction()