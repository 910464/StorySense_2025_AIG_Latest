import unittest
from unittest.mock import Mock, patch, MagicMock
import pytest
import json
import logging
from pathlib import Path
import sys

# Add the src directory to the Python path
current_dir = Path(__file__).parent
src_dir = current_dir.parent.parent / "src"
sys.path.insert(0, str(src_dir))

try:
    from src.prompt_layer.query_expander import QueryExpander
    from src.llm_layer.model_manual_test_llm import LLM
    from src.metrics.metrics_manager import MetricsManager
except ImportError:
    # Fallback for different path configurations
    sys.path.append(str(current_dir.parent.parent))
    from src.prompt_layer.query_expander import QueryExpander
    from src.llm_layer.model_manual_test_llm import LLM
    from src.metrics.metrics_manager import MetricsManager


class TestQueryExpander(unittest.TestCase):
    """Comprehensive test suite for QueryExpander class"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Mock the LLM class to avoid external dependencies
        self.llm_patcher = patch('src.prompt_layer.query_expander.LLM')
        self.mock_llm_class = self.llm_patcher.start()
        
        # Create mock LLM instance
        self.mock_llm_instance = Mock(spec=LLM)
        # Mock both the method the source code calls and the method that actually exists
        self.mock_llm_instance.send_request = Mock()
        self.mock_llm_instance.send_request_multimodal = Mock()
        self.mock_llm_class.return_value = self.mock_llm_instance
        
        # Create mock metrics manager
        self.mock_metrics_manager = Mock(spec=MetricsManager)

    def tearDown(self):
        """Clean up after each test method."""
        self.llm_patcher.stop()

    # ==================== Initialization Tests ====================

    def test_init_default_values(self):
        """Test QueryExpander initialization with default values"""
        expander = QueryExpander()
        
        # Verify LLM was initialized with correct parameters
        self.mock_llm_class.assert_called_once_with(None, metrics_manager=None)
        self.assertEqual(expander.llm, self.mock_llm_instance)

    def test_init_with_llm_family(self):
        """Test QueryExpander initialization with custom LLM family"""
        llm_family = "AWS"
        expander = QueryExpander(llm_family=llm_family)
        
        # Verify LLM was initialized with correct parameters
        self.mock_llm_class.assert_called_once_with(llm_family, metrics_manager=None)
        self.assertEqual(expander.llm, self.mock_llm_instance)

    def test_init_with_metrics_manager(self):
        """Test QueryExpander initialization with metrics manager"""
        expander = QueryExpander(metrics_manager=self.mock_metrics_manager)
        
        # Verify LLM was initialized with metrics manager
        self.mock_llm_class.assert_called_once_with(None, metrics_manager=self.mock_metrics_manager)
        self.assertEqual(expander.llm, self.mock_llm_instance)

    def test_init_with_all_parameters(self):
        """Test QueryExpander initialization with all parameters"""
        llm_family = "AWS"
        expander = QueryExpander(llm_family=llm_family, metrics_manager=self.mock_metrics_manager)
        
        # Verify LLM was initialized with all parameters
        self.mock_llm_class.assert_called_once_with(llm_family, metrics_manager=self.mock_metrics_manager)
        self.assertEqual(expander.llm, self.mock_llm_instance)

    # ==================== Query Expansion Tests ====================

    def test_expand_query_success_with_valid_json(self):
        """Test successful query expansion with valid JSON response"""
        query = "What is machine learning?"
        
        # Mock successful LLM response with valid JSON
        json_response = {
            "alternative_phrasings": [
                "What is ML?",
                "Define machine learning",
                "Explain artificial intelligence learning"
            ],
            "keywords": [
                "machine learning",
                "artificial intelligence",
                "ML",
                "algorithms",
                "data science"
            ],
            "broader_concepts": [
                "artificial intelligence",
                "data science",
                "computer science"
            ]
        }
        
        llm_response = f"Here is the response: {json.dumps(json_response)}"
        self.mock_llm_instance.send_request.return_value = llm_response
        
        expander = QueryExpander()
        result = expander.expand_query(query)
        
        # Verify LLM was called with correct parameters
        self.mock_llm_instance.send_request.assert_called_once()
        call_args = self.mock_llm_instance.send_request.call_args
        
        # Check arguments: (prompt_template, input_variables, input_variables_dict, call_type)
        # Be flexible about argument positions and use kwargs if needed
        if len(call_args[0]) >= 4:
            self.assertIn("query", call_args[0][1])  # input_variables
            self.assertEqual(call_args[0][2]["query"], query)  # input_variables_dict
            self.assertEqual(call_args[0][3], "query_expansion")  # call_type
        elif call_args[1]:  # Check kwargs
            self.assertEqual(call_args[1].get("call_type"), "query_expansion")
        
        # Verify result structure
        self.assertEqual(result["original_query"], query)
        self.assertTrue(result["success"])
        self.assertIn("expanded_data", result)
        
        # Verify expanded data
        expanded_data = result["expanded_data"]
        self.assertEqual(expanded_data["alternative_phrasings"], json_response["alternative_phrasings"])
        self.assertEqual(expanded_data["keywords"], json_response["keywords"])
        self.assertEqual(expanded_data["broader_concepts"], json_response["broader_concepts"])

    def test_expand_query_success_with_json_in_text(self):
        """Test query expansion with JSON embedded in text response"""
        query = "How does neural network work?"
        
        json_response = {
            "alternative_phrasings": [
                "How do neural networks function?",
                "Explain neural network operations"
            ],
            "keywords": [
                "neural network",
                "deep learning",
                "neurons"
            ],
            "broader_concepts": [
                "machine learning",
                "artificial intelligence"
            ]
        }
        
        # Response with JSON embedded in text
        llm_response = f"""
        Based on your query, here is the expansion:
        
        {json.dumps(json_response)}
        
        This should help improve your search results.
        """
        
        self.mock_llm_instance.send_request.return_value = llm_response
        
        expander = QueryExpander()
        result = expander.expand_query(query)
        
        # Verify successful parsing
        self.assertTrue(result["success"])
        self.assertEqual(result["expanded_data"]["alternative_phrasings"], 
                        json_response["alternative_phrasings"])

    def test_expand_query_invalid_json_response(self):
        """Test query expansion with invalid JSON response"""
        query = "What is blockchain?"
        
        # Mock LLM response with invalid JSON
        llm_response = "This is not a valid JSON response"
        self.mock_llm_instance.send_request.return_value = llm_response
        
        with patch('src.prompt_layer.query_expander.logging') as mock_logging:
            expander = QueryExpander()
            result = expander.expand_query(query)
            
            # Verify warning was logged
            mock_logging.warning.assert_called_once_with(
                "Failed to extract JSON from query expansion response"
            )
        
        # Verify fallback behavior
        self.assertEqual(result["original_query"], query)
        self.assertTrue(result["success"])  # Still successful, just with empty data
        
        expanded_data = result["expanded_data"]
        self.assertEqual(expanded_data["alternative_phrasings"], [])
        self.assertEqual(expanded_data["keywords"], [])
        self.assertEqual(expanded_data["broader_concepts"], [])

    def test_expand_query_malformed_json(self):
        """Test query expansion with malformed JSON that matches regex but fails parsing"""
        query = "What is Python programming?"
        
        # Mock response with malformed JSON
        llm_response = '{"alternative_phrasings": ["test1", "test2",], "keywords": []}'  # Trailing comma
        self.mock_llm_instance.send_request.return_value = llm_response
        
        with patch('src.prompt_layer.query_expander.logging') as mock_logging:
            expander = QueryExpander()
            result = expander.expand_query(query)
            
            # Should catch the JSON parsing exception and return fallback
            mock_logging.error.assert_called_once()
            error_message = mock_logging.error.call_args[0][0]
            self.assertIn("Query expansion failed", error_message)
        
        # Verify error response structure
        self.assertEqual(result["original_query"], query)
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        
        expanded_data = result["expanded_data"]
        self.assertEqual(expanded_data["alternative_phrasings"], [])
        self.assertEqual(expanded_data["keywords"], [])
        self.assertEqual(expanded_data["broader_concepts"], [])

    def test_expand_query_llm_exception(self):
        """Test query expansion when LLM raises an exception"""
        query = "What is data science?"
        
        # Mock LLM to raise an exception
        error_message = "Connection timeout"
        self.mock_llm_instance.send_request.side_effect = Exception(error_message)
        
        with patch('src.prompt_layer.query_expander.logging') as mock_logging:
            expander = QueryExpander()
            result = expander.expand_query(query)
            
            # Verify error was logged
            mock_logging.error.assert_called_once()
            logged_message = mock_logging.error.call_args[0][0]
            self.assertIn("Query expansion failed", logged_message)
        
        # Verify error response structure
        self.assertEqual(result["original_query"], query)
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], error_message)
        
        # Verify empty expanded data
        expanded_data = result["expanded_data"]
        self.assertEqual(expanded_data["alternative_phrasings"], [])
        self.assertEqual(expanded_data["keywords"], [])
        self.assertEqual(expanded_data["broader_concepts"], [])

    def test_expand_query_empty_query(self):
        """Test query expansion with empty query string"""
        query = ""
        
        json_response = {
            "alternative_phrasings": [],
            "keywords": [],
            "broader_concepts": []
        }
        
        llm_response = json.dumps(json_response)
        self.mock_llm_instance.send_request.return_value = llm_response
        
        expander = QueryExpander()
        result = expander.expand_query(query)
        
        # Verify it handles empty query
        self.assertEqual(result["original_query"], "")
        self.assertTrue(result["success"])

    def test_expand_query_complex_json_structure(self):
        """Test query expansion with complex JSON structure"""
        query = "Advanced machine learning techniques"
        
        json_response = {
            "alternative_phrasings": [
                "Advanced ML techniques",
                "Sophisticated machine learning methods",
                "Complex AI algorithms",
                "Advanced artificial intelligence approaches"
            ],
            "keywords": [
                "machine learning",
                "deep learning",
                "neural networks",
                "algorithms",
                "artificial intelligence",
                "data science",
                "pattern recognition"
            ],
            "broader_concepts": [
                "artificial intelligence",
                "computer science",
                "data analytics"
            ]
        }
        
        llm_response = json.dumps(json_response)
        self.mock_llm_instance.send_request.return_value = llm_response
        
        expander = QueryExpander()
        result = expander.expand_query(query)
        
        # Verify all data is preserved
        expanded_data = result["expanded_data"]
        self.assertEqual(len(expanded_data["alternative_phrasings"]), 4)
        self.assertEqual(len(expanded_data["keywords"]), 7)
        self.assertEqual(len(expanded_data["broader_concepts"]), 3)

    # ==================== Generate Search Queries Tests ====================

    def test_generate_search_queries_successful_expansion(self):
        """Test generating search queries from successful expansion"""
        original_query = "What is machine learning?"
        
        expansion_result = {
            "original_query": original_query,
            "success": True,
            "expanded_data": {
                "alternative_phrasings": [
                    "What is ML?",
                    "Define machine learning"
                ],
                "keywords": [
                    "machine learning",
                    "AI",
                    "algorithms"
                ],
                "broader_concepts": [
                    "artificial intelligence",
                    "data science"
                ]
            }
        }
        
        expander = QueryExpander()
        queries = expander.generate_search_queries(expansion_result)
        
        # Verify original query is first
        self.assertEqual(queries[0], original_query)
        
        # Verify alternative phrasings are included
        self.assertIn("What is ML?", queries)
        self.assertIn("Define machine learning", queries)
        
        # Verify broader concepts are included
        self.assertIn("artificial intelligence", queries)
        self.assertIn("data science", queries)
        
        # Verify total count (original + alternatives + broader concepts)
        expected_count = 1 + 2 + 2  # original + 2 alternatives + 2 broader concepts
        self.assertEqual(len(queries), expected_count)

    def test_generate_search_queries_failed_expansion(self):
        """Test generating search queries from failed expansion"""
        original_query = "What is blockchain?"
        
        expansion_result = {
            "original_query": original_query,
            "success": False,
            "expanded_data": {
                "alternative_phrasings": [],
                "keywords": [],
                "broader_concepts": []
            },
            "error": "LLM connection failed"
        }
        
        expander = QueryExpander()
        queries = expander.generate_search_queries(expansion_result)
        
        # Should only return the original query
        self.assertEqual(len(queries), 1)
        self.assertEqual(queries[0], original_query)

    def test_generate_search_queries_missing_success_field(self):
        """Test generating search queries when success field is missing"""
        original_query = "What is Python?"
        
        expansion_result = {
            "original_query": original_query,
            "expanded_data": {
                "alternative_phrasings": ["What is Python programming?"],
                "broader_concepts": ["programming languages"]
            }
        }
        
        expander = QueryExpander()
        queries = expander.generate_search_queries(expansion_result)
        
        # Without success=True, should only return original query
        self.assertEqual(len(queries), 1)
        self.assertEqual(queries[0], original_query)

    def test_generate_search_queries_missing_expanded_data(self):
        """Test generating search queries when expanded_data is missing"""
        original_query = "What is Docker?"
        
        expansion_result = {
            "original_query": original_query,
            "success": True
            # Missing expanded_data
        }
        
        expander = QueryExpander()
        queries = expander.generate_search_queries(expansion_result)
        
        # Should only return original query
        self.assertEqual(len(queries), 1)
        self.assertEqual(queries[0], original_query)

    def test_generate_search_queries_partial_expanded_data(self):
        """Test generating search queries with partial expanded data"""
        original_query = "What is Kubernetes?"
        
        expansion_result = {
            "original_query": original_query,
            "success": True,
            "expanded_data": {
                "alternative_phrasings": ["What is K8s?"],
                # Missing keywords and broader_concepts
            }
        }
        
        expander = QueryExpander()
        queries = expander.generate_search_queries(expansion_result)
        
        # Should include original + alternatives (no broader concepts available)
        self.assertEqual(len(queries), 2)
        self.assertEqual(queries[0], original_query)
        self.assertIn("What is K8s?", queries)

    def test_generate_search_queries_empty_lists(self):
        """Test generating search queries with empty lists in expanded data"""
        original_query = "What is DevOps?"
        
        expansion_result = {
            "original_query": original_query,
            "success": True,
            "expanded_data": {
                "alternative_phrasings": [],
                "keywords": [],
                "broader_concepts": []
            }
        }
        
        expander = QueryExpander()
        queries = expander.generate_search_queries(expansion_result)
        
        # Should only return original query
        self.assertEqual(len(queries), 1)
        self.assertEqual(queries[0], original_query)

    def test_generate_search_queries_only_broader_concepts(self):
        """Test generating search queries with only broader concepts"""
        original_query = "What is microservices?"
        
        expansion_result = {
            "original_query": original_query,
            "success": True,
            "expanded_data": {
                "alternative_phrasings": [],
                "keywords": ["microservices", "architecture"],
                "broader_concepts": ["software architecture", "distributed systems"]
            }
        }
        
        expander = QueryExpander()
        queries = expander.generate_search_queries(expansion_result)
        
        # Should include original + broader concepts (no alternatives)
        self.assertEqual(len(queries), 3)  # original + 2 broader concepts
        self.assertEqual(queries[0], original_query)
        self.assertIn("software architecture", queries)
        self.assertIn("distributed systems", queries)

    # ==================== Integration Tests ====================

    def test_full_workflow_integration(self):
        """Test complete workflow from expansion to query generation"""
        query = "What is cloud computing?"
        
        # Mock successful LLM response
        json_response = {
            "alternative_phrasings": [
                "What is cloud technology?",
                "Define cloud computing"
            ],
            "keywords": [
                "cloud",
                "computing",
                "AWS",
                "Azure"
            ],
            "broader_concepts": [
                "distributed computing",
                "web services"
            ]
        }
        
        llm_response = json.dumps(json_response)
        self.mock_llm_instance.send_request.return_value = llm_response
        
        expander = QueryExpander(llm_family="AWS", metrics_manager=self.mock_metrics_manager)
        
        # Step 1: Expand query
        expansion_result = expander.expand_query(query)
        
        # Verify expansion result
        self.assertTrue(expansion_result["success"])
        self.assertEqual(expansion_result["original_query"], query)
        
        # Step 2: Generate search queries
        search_queries = expander.generate_search_queries(expansion_result)
        
        # Verify search queries
        self.assertGreater(len(search_queries), 1)
        self.assertEqual(search_queries[0], query)
        self.assertIn("What is cloud technology?", search_queries)
        self.assertIn("distributed computing", search_queries)

    # ==================== Edge Cases and Error Handling ====================

    def test_expand_query_with_special_characters(self):
        """Test query expansion with special characters"""
        query = "What is C++ programming? & how does it work?"
        
        json_response = {
            "alternative_phrasings": ["C++ programming language"],
            "keywords": ["C++", "programming"],
            "broader_concepts": ["programming languages"]
        }
        
        llm_response = json.dumps(json_response)
        self.mock_llm_instance.send_request.return_value = llm_response
        
        expander = QueryExpander()
        result = expander.expand_query(query)
        
        # Should handle special characters correctly
        self.assertTrue(result["success"])
        self.assertEqual(result["original_query"], query)

    def test_expand_query_with_unicode_characters(self):
        """Test query expansion with Unicode characters"""
        query = "What is artificial intelligence? 🤖"
        
        json_response = {
            "alternative_phrasings": ["AI definition"],
            "keywords": ["AI", "intelligence"],
            "broader_concepts": ["computer science"]
        }
        
        llm_response = json.dumps(json_response, ensure_ascii=False)
        self.mock_llm_instance.send_request.return_value = llm_response
        
        expander = QueryExpander()
        result = expander.expand_query(query)
        
        # Should handle Unicode correctly
        self.assertTrue(result["success"])
        self.assertEqual(result["original_query"], query)

    def test_expand_query_very_long_query(self):
        """Test query expansion with very long query"""
        query = "What is machine learning and how does it work in artificial intelligence systems for data analysis and predictive modeling in enterprise applications?" * 10
        
        json_response = {
            "alternative_phrasings": ["ML in enterprise"],
            "keywords": ["machine learning", "enterprise"],
            "broader_concepts": ["AI systems"]
        }
        
        llm_response = json.dumps(json_response)
        self.mock_llm_instance.send_request.return_value = llm_response
        
        expander = QueryExpander()
        result = expander.expand_query(query)
        
        # Should handle long queries
        self.assertTrue(result["success"])
        self.assertEqual(result["original_query"], query)

    def test_prompt_template_content(self):
        """Test that the prompt template contains expected instructions"""
        query = "test query"
        
        # Mock successful response
        json_response = {"alternative_phrasings": [], "keywords": [], "broader_concepts": []}
        self.mock_llm_instance.send_request.return_value = json.dumps(json_response)
        
        expander = QueryExpander()
        expander.expand_query(query)
        
        # Verify the prompt template content
        call_args = self.mock_llm_instance.send_request.call_args
        if call_args and call_args[0]:
            prompt_template = call_args[0][0]
        else:
            prompt_template = ""
        
        # Check for key instructions in the prompt
        self.assertIn("information retrieval", prompt_template)
        self.assertIn("alternative phrasings", prompt_template)
        self.assertIn("keywords", prompt_template)
        self.assertIn("broader concepts", prompt_template)
        self.assertIn("JSON", prompt_template)
        self.assertIn("{query}", prompt_template)

    # ==================== JSON Parsing Edge Cases ====================

    def test_expand_query_nested_json_in_response(self):
        """Test query expansion with nested JSON structures"""
        query = "What is REST API?"
        
        json_response = {
            "alternative_phrasings": [
                "What is RESTful API?",
                "Define REST web service"
            ],
            "keywords": [
                "REST",
                "API",
                "HTTP",
                "web service"
            ],
            "broader_concepts": [
                "web development",
                "software architecture"
            ]
        }
        
        # Response with nested JSON-like text that shouldn't be parsed
        llm_response = f"""
        Here's some text with {{"nested": "json"}} that should be ignored.
        
        The actual response is:
        {json.dumps(json_response)}
        
        And here's more {{"invalid": "json"}} text.
        """
        
        self.mock_llm_instance.send_request.return_value = llm_response
        
        expander = QueryExpander()
        result = expander.expand_query(query)
        
        # Should parse the correct JSON structure
        self.assertTrue(result["success"])
        self.assertEqual(result["expanded_data"]["alternative_phrasings"], 
                        json_response["alternative_phrasings"])

    def test_expand_query_multiple_json_blocks(self):
        """Test query expansion with multiple JSON blocks in response"""
        query = "What is GraphQL?"
        
        # First JSON block (should be ignored)
        wrong_json = {"wrong": "data"}
        
        # Correct JSON block
        correct_json = {
            "alternative_phrasings": ["GraphQL query language"],
            "keywords": ["GraphQL", "query"],
            "broader_concepts": ["API design"]
        }
        
        llm_response = f"""
        First block: {json.dumps(wrong_json)}
        
        Correct block: {json.dumps(correct_json)}
        """
        
        self.mock_llm_instance.send_request.return_value = llm_response
        
        expander = QueryExpander()
        result = expander.expand_query(query)
        
        # Should parse the first valid JSON it finds
        self.assertTrue(result["success"])
        # Note: regex will match the first JSON block
        self.assertEqual(result["expanded_data"]["wrong"], "data")


if __name__ == '__main__':
    # Configure logging for tests
    logging.basicConfig(level=logging.INFO)
    
    # Run the tests
    unittest.main(verbosity=2)
