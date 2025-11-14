

import pytest
import sys
from unittest.mock import Mock, patch, call
import importlib.util

# Mock all problematic modules before any imports
mock_modules = {
    'torch': Mock(),
    'transformers': Mock(),
    'langchain': Mock(),
    'langchain.prompts': Mock(),
    'langchain_core': Mock(),
    'langchain_core.prompts': Mock(),
}

for module_name, mock_module in mock_modules.items():
    sys.modules[module_name] = mock_module

# Create mock QueryExpander module
mock_query_expander = Mock()
class MockQueryExpander:
    def __init__(self, metrics_manager=None):
        self.metrics_manager = metrics_manager
        
    def expand_query(self, query):
        return {"original_query": query, "expanded_data": {}}
        
    def generate_search_queries(self, expansion_result):
        original = expansion_result.get("original_query", "test")
        return [original, f"expanded_{original}"]

mock_query_expander.QueryExpander = MockQueryExpander
sys.modules['src.prompt_layer.query_expander'] = mock_query_expander

# Now we can safely import the module
from src.context_handler.context_storage_handler.pgvector_searcher import PGVectorSearcher


class TestPGVectorSearcher95Coverage:
    """Test suite designed to achieve 95%+ coverage by mocking all dependencies completely"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_orchestrator = Mock()
        self.mock_orchestrator.metrics_manager = Mock()
        self.mock_orchestrator.collection_name = "test_collection"
        self.mock_orchestrator.config_file_path = "../Config/Config.properties"
        self.mock_orchestrator.reconnect_if_needed = Mock()
        self.mock_orchestrator.threshold = 0.7
        self.mock_orchestrator.db = Mock()
        self.mock_orchestrator.db.connect_with_retry = Mock()
        self.mock_orchestrator.embeddings = Mock()
        self.mock_orchestrator.similarity_metric = "cosine"
        self.mock_orchestrator.metrics_reporter = Mock()
        self.mock_orchestrator.retrieval_dir = "test_retrieval"

    def test_initialization_success(self):
        """Test successful initialization - covers lines 15-19"""
        with patch('src.context_handler.context_storage_handler.pgvector_searcher.PGVectorRetriever'):
            searcher = PGVectorSearcher(self.mock_orchestrator)
            assert searcher.orch == self.mock_orchestrator
            assert searcher.metrics_manager == self.mock_orchestrator.metrics_manager
            assert searcher.collection_name == "test_collection"

    def test_initialization_failure(self):
        """Test initialization with None orchestrator"""
        with pytest.raises(AttributeError):
            PGVectorSearcher(None)

    def test_complete_search_execution(self):
        """Test complete search execution covering all major paths"""
        
        with patch('src.context_handler.context_storage_handler.pgvector_searcher.PGVectorRetriever') as mock_retriever_class:
            # Set up main retriever mock
            mock_retriever = Mock()
            mock_retriever.retrieval_context.return_value = (
                "Main context content",
                {0.95: "MainDoc1", 0.85: "MainDoc2"},
                {0.95: {"source": "main1.pdf"}, 0.85: {"source": "main2.pdf"}},
                0.7
            )
            mock_retriever_class.return_value = mock_retriever
            
            # Set up stories components
            mock_stories_orch = Mock()
            mock_stories_retriever = Mock()
            mock_stories_retriever.retrieval_context.return_value = (
                "Stories content",
                {0.9: "StoryDoc1", 0.8: "StoryDoc2"},
                {0.9: {"source": "story1.txt"}, 0.8: {"source": "story2.txt"}},
                0.7
            )
            
            searcher = PGVectorSearcher(self.mock_orchestrator)
            
            with patch('src.context_handler.context_storage_handler.pgvector_searcher.PGVectorOrchestrator', return_value=mock_stories_orch):
                with patch('src.context_handler.context_storage_handler.pgvector_searcher.PGVectorRetriever', return_value=mock_stories_retriever):
                    with patch('logging.info') as mock_logging:
                        
                        result = searcher.search_all_collections("test query", k=5)
                        
                        # Verify result structure
                        assert isinstance(result, tuple)
                        assert len(result) == 4
                        
                        combined_context, combined_docs, combined_metadata, threshold = result
                        
                        # Verify content
                        assert isinstance(combined_context, str)
                        assert "CONTEXT FROM QUERY" in combined_context
                        assert "STORIES FROM QUERY" in combined_context
                        assert "Main context content" in combined_context
                        assert "Stories content" in combined_context
                        
                        # Verify documents
                        assert len(combined_docs) == 4
                        assert len(combined_metadata) == 4
                        
                        # Verify sorting by score (documents are combined and sorted)
                        scores = list(combined_docs.keys())
                        assert len(scores) == 4
                        assert max(scores) == 0.95
                        assert min(scores) == 0.8
                        
                        # Verify method calls
                        self.mock_orchestrator.reconnect_if_needed.assert_called_once()
                        assert mock_logging.call_count > 0

    def test_empty_results_handling(self):
        """Test handling of empty results from both sources"""
        
        with patch('src.context_handler.context_storage_handler.pgvector_searcher.PGVectorRetriever') as mock_retriever_class:
            mock_retriever = Mock()
            mock_retriever.retrieval_context.return_value = ("", {}, {}, 0.7)
            mock_retriever_class.return_value = mock_retriever
            
            mock_stories_orch = Mock()
            mock_stories_retriever = Mock()
            mock_stories_retriever.retrieval_context.return_value = ("", {}, {}, 0.7)
            
            searcher = PGVectorSearcher(self.mock_orchestrator)
            
            with patch('src.context_handler.context_storage_handler.pgvector_searcher.PGVectorOrchestrator', return_value=mock_stories_orch):
                with patch('src.context_handler.context_storage_handler.pgvector_searcher.PGVectorRetriever', return_value=mock_stories_retriever):
                    
                    result = searcher.search_all_collections("test query", k=5)
                    combined_context, combined_docs, combined_metadata, threshold = result
                    
                    assert combined_context == ""
                    assert len(combined_docs) == 0
                    assert len(combined_metadata) == 0
                    assert threshold == 0.7

    def test_k_limiting_functionality(self):
        """Test k limiting when there are more results than requested"""
        
        with patch('src.context_handler.context_storage_handler.pgvector_searcher.PGVectorRetriever') as mock_retriever_class:
            # Create many documents
            many_docs = {0.95: "Doc1", 0.9: "Doc2", 0.85: "Doc3", 0.8: "Doc4", 0.75: "Doc5", 0.7: "Doc6"}
            many_metadata = {score: {"source": f"doc{i}.pdf"} for i, score in enumerate(many_docs.keys())}
            
            mock_retriever = Mock()
            mock_retriever.retrieval_context.return_value = ("Context", many_docs, many_metadata, 0.7)
            mock_retriever_class.return_value = mock_retriever
            
            mock_stories_orch = Mock()
            mock_stories_retriever = Mock()
            mock_stories_retriever.retrieval_context.return_value = ("", {}, {}, 0.7)
            
            searcher = PGVectorSearcher(self.mock_orchestrator)
            
            with patch('src.context_handler.context_storage_handler.pgvector_searcher.PGVectorOrchestrator', return_value=mock_stories_orch):
                with patch('src.context_handler.context_storage_handler.pgvector_searcher.PGVectorRetriever', return_value=mock_stories_retriever):
                    
                    result = searcher.search_all_collections("test query", k=3)
                    combined_context, combined_docs, combined_metadata, threshold = result
                    
                    # Should be limited to k=3
                    assert len(combined_docs) == 3
                    assert len(combined_metadata) == 3
                    
                    # Should be highest scoring docs
                    scores = list(combined_docs.keys())
                    assert scores == [0.95, 0.9, 0.85]

    def test_exception_handling_and_retry(self):
        """Test exception handling and retry logic"""
        
        with patch('src.context_handler.context_storage_handler.pgvector_searcher.PGVectorRetriever') as mock_retriever_class:
            # Make retrieval fail first, then succeed
            call_count = [0]
            def failing_retrieval(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    raise Exception("Connection error")
                return ("Context after retry", {0.9: "Doc1"}, {0.9: {"source": "doc1.pdf"}}, 0.7)
            
            mock_retriever = Mock()
            mock_retriever.retrieval_context = failing_retrieval
            mock_retriever_class.return_value = mock_retriever
            
            mock_stories_orch = Mock()
            mock_stories_retriever = Mock()
            mock_stories_retriever.retrieval_context = failing_retrieval
            
            searcher = PGVectorSearcher(self.mock_orchestrator)
            
            with patch('src.context_handler.context_storage_handler.pgvector_searcher.PGVectorOrchestrator', return_value=mock_stories_orch):
                with patch('src.context_handler.context_storage_handler.pgvector_searcher.PGVectorRetriever', return_value=mock_stories_retriever):
                    with patch('time.sleep') as mock_sleep:
                        with patch('logging.warning') as mock_warning:
                            
                            result = searcher.search_all_collections("test query", k=5)
                            
                            # Should succeed after retry
                            combined_context, combined_docs, combined_metadata, threshold = result
                            assert "Context after retry" in combined_context
                            assert len(combined_docs) > 0
                            
                            # Should have attempted retry
                            assert mock_sleep.call_count >= 1
                            assert mock_warning.call_count >= 1
                            assert self.mock_orchestrator.db.connect_with_retry.call_count >= 1

    def test_max_retries_exceeded(self):
        """Test behavior when max retries are exceeded"""
        
        with patch('src.context_handler.context_storage_handler.pgvector_searcher.PGVectorRetriever') as mock_retriever_class:
            # Always fail
            mock_retriever = Mock()
            mock_retriever.retrieval_context.side_effect = Exception("Persistent error")
            mock_retriever_class.return_value = mock_retriever
            
            mock_stories_orch = Mock()
            mock_stories_retriever = Mock()
            mock_stories_retriever.retrieval_context.side_effect = Exception("Persistent error")
            
            searcher = PGVectorSearcher(self.mock_orchestrator)
            
            with patch('src.context_handler.context_storage_handler.pgvector_searcher.PGVectorOrchestrator', return_value=mock_stories_orch):
                with patch('src.context_handler.context_storage_handler.pgvector_searcher.PGVectorRetriever', return_value=mock_stories_retriever):
                    with patch('time.sleep'):
                        with patch('logging.error') as mock_error:
                            
                            result = searcher.search_all_collections("test query", k=5)
                            
                            # Should return empty results
                            combined_context, combined_docs, combined_metadata, threshold = result
                            assert combined_context == ""
                            assert len(combined_docs) == 0
                            assert len(combined_metadata) == 0
                            assert threshold == self.mock_orchestrator.threshold
                            
                            # Should have logged error and recorded metrics
                            assert mock_error.call_count >= 1
                            self.mock_orchestrator.metrics_manager.record_error.assert_called()

    def test_metrics_error_handling(self):
        """Test exception handling in metrics recording"""
        
        with patch('src.context_handler.context_storage_handler.pgvector_searcher.PGVectorRetriever') as mock_retriever_class:
            # Make metrics fail
            self.mock_orchestrator.metrics_manager.record_error.side_effect = Exception("Metrics error")
            
            mock_retriever = Mock()
            mock_retriever.retrieval_context.side_effect = Exception("Search error")
            mock_retriever_class.return_value = mock_retriever
            
            mock_stories_orch = Mock()
            mock_stories_retriever = Mock()
            mock_stories_retriever.retrieval_context.side_effect = Exception("Search error")
            
            searcher = PGVectorSearcher(self.mock_orchestrator)
            
            with patch('src.context_handler.context_storage_handler.pgvector_searcher.PGVectorOrchestrator', return_value=mock_stories_orch):
                with patch('src.context_handler.context_storage_handler.pgvector_searcher.PGVectorRetriever', return_value=mock_stories_retriever):
                    with patch('time.sleep'):
                        
                        result = searcher.search_all_collections("test query", k=5)
                        
                        # Should still return empty results despite metrics error
                        combined_context, combined_docs, combined_metadata, threshold = result
                        assert combined_context == ""
                        assert len(combined_docs) == 0
                        assert len(combined_metadata) == 0

    def test_mixed_expansion_scenarios(self):
        """Test various query expansion scenarios"""
        
        # Test with custom expander that returns multiple queries
        class MultiQueryExpander:
            def __init__(self, metrics_manager=None):
                self.metrics_manager = metrics_manager
                
            def expand_query(self, query):
                return {"original_query": query, "expanded_data": {}}
                
            def generate_search_queries(self, expansion_result):
                return ["query1", "query2", "query3"]
        
        with patch('src.context_handler.context_storage_handler.pgvector_searcher.PGVectorRetriever') as mock_retriever_class:
            def query_specific_results(query, k_val):
                if "query1" in query:
                    return ("Context1", {0.9: "Doc1"}, {0.9: {"source": "doc1.pdf"}}, 0.7)
                elif "query2" in query:
                    return ("Context2", {0.8: "Doc2"}, {0.8: {"source": "doc2.pdf"}}, 0.7)
                else:
                    return ("Context3", {0.75: "Doc3"}, {0.75: {"source": "doc3.pdf"}}, 0.7)
            
            mock_retriever = Mock()
            mock_retriever.retrieval_context = query_specific_results
            mock_retriever_class.return_value = mock_retriever
            
            mock_stories_orch = Mock()
            mock_stories_retriever = Mock()
            mock_stories_retriever.retrieval_context = query_specific_results
            
            searcher = PGVectorSearcher(self.mock_orchestrator)
            
            with patch('src.context_handler.context_storage_handler.pgvector_searcher.PGVectorOrchestrator', return_value=mock_stories_orch):
                with patch('src.context_handler.context_storage_handler.pgvector_searcher.PGVectorRetriever', return_value=mock_stories_retriever):
                    # Test with default mock expander
                    with patch('logging.info') as mock_info:
                        
                        result = searcher.search_all_collections("test query", k=10)
                        combined_context, combined_docs, combined_metadata, threshold = result
                        
                        # Should have results from queries
                        assert len(combined_docs) > 0
                        assert "CONTEXT FROM QUERY" in combined_context or "STORIES FROM QUERY" in combined_context
                        
                        # Should have logged expansion info
                        log_calls = [call.args[0] for call in mock_info.call_args_list]
                        expansion_logs = [msg for msg in log_calls if "Query expansion" in msg]
                        assert len(expansion_logs) > 0

    def test_k_division_calculation(self):
        """Test k division calculation for multiple queries"""
        
        with patch('src.context_handler.context_storage_handler.pgvector_searcher.PGVectorRetriever') as mock_retriever_class:
            # Track k values passed to retrieval_context
            k_values = []
            def track_k_values(query, k_val):
                k_values.append(k_val)
                return (f"Context for {query}", {0.9: f"Doc_{query}"}, {0.9: {"source": f"{query}.pdf"}}, 0.7)
            
            mock_retriever = Mock()
            mock_retriever.retrieval_context = track_k_values
            mock_retriever_class.return_value = mock_retriever
            
            mock_stories_orch = Mock()
            mock_stories_retriever = Mock() 
            mock_stories_retriever.retrieval_context = track_k_values
            
            searcher = PGVectorSearcher(self.mock_orchestrator)
            
            with patch('src.context_handler.context_storage_handler.pgvector_searcher.PGVectorOrchestrator', return_value=mock_stories_orch):
                with patch('src.context_handler.context_storage_handler.pgvector_searcher.PGVectorRetriever', return_value=mock_stories_retriever):
                    
                    result = searcher.search_all_collections("test query", k=6)
                    
                    # With 2 default queries, k should be divided: 6 // 2 + 1 = 4
                    assert all(k_val == 4 for k_val in k_values)

    def test_context_formatting_logic(self):
        """Test context formatting with different scenarios"""
        
        with patch('src.context_handler.context_storage_handler.pgvector_searcher.PGVectorRetriever') as mock_retriever_class:
            # Test only main context
            mock_retriever = Mock()
            mock_retriever.retrieval_context.return_value = (
                "Main only content",
                {0.9: "MainDoc"},
                {0.9: {"source": "main.pdf"}},
                0.7
            )
            mock_retriever_class.return_value = mock_retriever
            
            mock_stories_orch = Mock()
            mock_stories_retriever = Mock()
            mock_stories_retriever.retrieval_context.return_value = ("", {}, {}, 0.7)
            
            searcher = PGVectorSearcher(self.mock_orchestrator)
            
            with patch('src.context_handler.context_storage_handler.pgvector_searcher.PGVectorOrchestrator', return_value=mock_stories_orch):
                with patch('src.context_handler.context_storage_handler.pgvector_searcher.PGVectorRetriever', return_value=mock_stories_retriever):
                    
                    result = searcher.search_all_collections("test query", k=5)
                    combined_context, combined_docs, combined_metadata, threshold = result
                    
                    assert "CONTEXT FROM QUERY" in combined_context
                    assert "Main only content" in combined_context
                    assert "STORIES FROM QUERY" not in combined_context
                    assert len(combined_docs) == 1

    def test_stories_only_scenario(self):
        """Test scenario with only stories results"""
        
        with patch('src.context_handler.context_storage_handler.pgvector_searcher.PGVectorRetriever') as mock_retriever_class:
            # Main retriever empty
            mock_retriever = Mock()
            mock_retriever.retrieval_context.return_value = ("", {}, {}, 0.7)
            mock_retriever_class.return_value = mock_retriever
            
            mock_stories_orch = Mock()
            mock_stories_retriever = Mock()
            mock_stories_retriever.retrieval_context.return_value = (
                "Stories only content",
                {0.8: "StoryDoc"},
                {0.8: {"source": "story.txt"}},
                0.7
            )
            
            searcher = PGVectorSearcher(self.mock_orchestrator)
            
            with patch('src.context_handler.context_storage_handler.pgvector_searcher.PGVectorOrchestrator', return_value=mock_stories_orch):
                with patch('src.context_handler.context_storage_handler.pgvector_searcher.PGVectorRetriever', return_value=mock_stories_retriever):
                    
                    result = searcher.search_all_collections("test query", k=5)
                    combined_context, combined_docs, combined_metadata, threshold = result
                    
                    assert "CONTEXT FROM QUERY" not in combined_context
                    assert "STORIES FROM QUERY" in combined_context
                    assert "Stories only content" in combined_context
                    assert len(combined_docs) == 1
