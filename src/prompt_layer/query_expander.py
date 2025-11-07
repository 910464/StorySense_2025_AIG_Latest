import logging
from typing import List, Dict, Any, Optional
from src.llm_layer.model_manual_test_llm  import LLM


class QueryExpander:
    """Expands queries to improve retrieval recall"""

    def __init__(self, llm_family=None, metrics_manager=None):
        """
        Initialize query expander

        Args:
            llm_family: LLM family to use for expansion
            metrics_manager: Metrics manager for tracking
        """
        self.llm = LLM(llm_family, metrics_manager=metrics_manager)

    def expand_query(self, query: str) -> Dict[str, Any]:
        """
        Expand a query to improve retrieval recall

        Args:
            query: Original query text

        Returns:
            Dict with original and expanded queries
        """
        # Define prompt for query expansion
        prompt_template = """
        You are an expert at information retrieval. Your task is to expand the following query to improve search results.

        Original Query: {query}

        Please provide:
        1. 3-5 alternative phrasings of the query
        2. 5-7 important keywords or key phrases from the query
        3. 2-3 broader concepts related to the query

        Format your response as JSON with the following structure:
        {{
            "alternative_phrasings": ["phrasing1", "phrasing2", ...],
            "keywords": ["keyword1", "keyword2", ...],
            "broader_concepts": ["concept1", "concept2", ...]
        }}

        IMPORTANT: Return ONLY the JSON with no additional text.
        """

        # Send request to LLM
        input_variables = ["query"]
        input_variables_dict = {"query": query}

        try:
            response = self.llm.send_request(
                prompt_template,
                input_variables,
                input_variables_dict,
                call_type="query_expansion"
            )

            # Parse JSON response
            import json
            import re

            # Extract JSON from response
            json_match = re.search(r'({.*})', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                expansion_data = json.loads(json_str)
            else:
                # Fallback if JSON extraction fails
                expansion_data = {
                    "alternative_phrasings": [],
                    "keywords": [],
                    "broader_concepts": []
                }
                logging.warning("Failed to extract JSON from query expansion response")

            return {
                "original_query": query,
                "expanded_data": expansion_data,
                "success": True
            }

        except Exception as e:
            logging.error(f"Query expansion failed: {str(e)}")
            return {
                "original_query": query,
                "expanded_data": {
                    "alternative_phrasings": [],
                    "keywords": [],
                    "broader_concepts": []
                },
                "success": False,
                "error": str(e)
            }

    def generate_search_queries(self, expansion_result: Dict[str, Any]) -> List[str]:
        """
        Generate search queries from expansion result

        Args:
            expansion_result: Result from expand_query

        Returns:
            List of search queries to use
        """
        queries = [expansion_result["original_query"]]

        if expansion_result.get("success", False):
            expanded_data = expansion_result.get("expanded_data", {})

            # Add alternative phrasings
            queries.extend(expanded_data.get("alternative_phrasings", []))

            # Add broader concepts
            queries.extend(expanded_data.get("broader_concepts", []))

        return queries