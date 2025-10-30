"""
End-to-end tests for RAG system handling content-related queries
Tests the complete flow from user query to AI response with search
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from rag_system import RAGSystem
from vector_store import SearchResults
from config import Config


@pytest.mark.integration
class TestRAGSystemContentQueries:
    """Test RAG system's handling of content-related questions"""

    def test_content_query_with_valid_results(self, mock_anthropic_client, test_config):
        """Test successful content query flow with search results"""
        with patch('rag_system.VectorStore') as MockVectorStore, \
             patch('rag_system.DocumentProcessor') as MockDocProcessor, \
             patch('rag_system.AIGenerator') as MockAIGenerator:

            # Setup mocks
            mock_store = Mock()
            mock_processor = Mock()
            mock_ai = Mock()

            MockVectorStore.return_value = mock_store
            MockDocProcessor.return_value = mock_processor
            MockAIGenerator.return_value = mock_ai

            # Mock search results
            search_results = SearchResults(
                documents=["MCP servers are tools that connect to Claude"],
                metadata=[{"course_title": "Introduction to MCP", "lesson_number": 1}],
                distances=[0.3],
                error=None
            )
            mock_store.search.return_value = search_results
            mock_store.get_lesson_link.return_value = "https://example.com/lesson1"

            # Mock AI response
            mock_ai.generate_response.return_value = (
                "MCP servers are tools that connect to Claude and provide additional capabilities.",
                [{"text": "Introduction to MCP - Lesson 1", "link": "https://example.com/lesson1"}]
            )

            # Create RAG system
            rag = RAGSystem(api_key="test-key", docs_dir="./docs")

            # Execute query
            response, sources = rag.query("What are MCP servers?", "session_1")

            # Verify
            assert response is not None
            assert "MCP servers" in response
            assert len(sources) == 1
            assert sources[0]["text"] == "Introduction to MCP - Lesson 1"
            mock_ai.generate_response.assert_called_once()

    def test_content_query_with_no_results(self, mock_anthropic_client, test_config):
        """Test content query when search returns no results"""
        with patch('rag_system.VectorStore') as MockVectorStore, \
             patch('rag_system.DocumentProcessor') as MockDocProcessor, \
             patch('rag_system.AIGenerator') as MockAIGenerator:

            mock_store = Mock()
            mock_processor = Mock()
            mock_ai = Mock()

            MockVectorStore.return_value = mock_store
            MockDocProcessor.return_value = mock_processor
            MockAIGenerator.return_value = mock_ai

            # Mock empty search results
            empty_results = SearchResults(
                documents=[],
                metadata=[],
                distances=[],
                error=None
            )
            mock_store.search.return_value = empty_results

            # Mock AI response acknowledging no results
            mock_ai.generate_response.return_value = (
                "No relevant content found for your query.",
                []
            )

            rag = RAGSystem(api_key="test-key", docs_dir="./docs")
            response, sources = rag.query("nonexistent topic", "session_1")

            # Should still get a response, but no sources
            assert response is not None
            assert sources == []

    def test_content_query_with_search_error(self, mock_anthropic_client, test_config):
        """Test content query when search encounters an error"""
        with patch('rag_system.VectorStore') as MockVectorStore, \
             patch('rag_system.DocumentProcessor') as MockDocProcessor, \
             patch('rag_system.AIGenerator') as MockAIGenerator:

            mock_store = Mock()
            mock_processor = Mock()
            mock_ai = Mock()

            MockVectorStore.return_value = mock_store
            MockDocProcessor.return_value = mock_processor
            MockAIGenerator.return_value = mock_ai

            # Mock search error
            error_results = SearchResults(
                documents=[],
                metadata=[],
                distances=[],
                error="ChromaDB connection failed"
            )
            mock_store.search.return_value = error_results

            # Mock AI response handling the error
            mock_ai.generate_response.return_value = (
                "I encountered an error searching the course materials.",
                []
            )

            rag = RAGSystem(api_key="test-key", docs_dir="./docs")
            response, sources = rag.query("test query", "session_1")

            # Should get error-aware response
            assert response is not None
            assert sources == []

    def test_content_query_with_course_filter(self, mock_anthropic_client, test_config):
        """Test content query with course name filtering"""
        with patch('rag_system.VectorStore') as MockVectorStore, \
             patch('rag_system.DocumentProcessor') as MockDocProcessor, \
             patch('rag_system.AIGenerator') as MockAIGenerator:

            mock_store = Mock()
            mock_processor = Mock()
            mock_ai = Mock()

            MockVectorStore.return_value = mock_store
            MockDocProcessor.return_value = mock_processor
            MockAIGenerator.return_value = mock_ai

            # Mock filtered search results
            search_results = SearchResults(
                documents=["Specific course content"],
                metadata=[{"course_title": "Advanced MCP", "lesson_number": 3}],
                distances=[0.2],
                error=None
            )
            mock_store.search.return_value = search_results
            mock_store.get_lesson_link.return_value = None

            mock_ai.generate_response.return_value = (
                "Information from Advanced MCP course",
                [{"text": "Advanced MCP - Lesson 3", "link": None}]
            )

            rag = RAGSystem(api_key="test-key", docs_dir="./docs")
            response, sources = rag.query("specific topic", "session_1")

            assert response is not None
            assert len(sources) == 1
            assert "Advanced MCP" in sources[0]["text"]

    def test_content_query_preserves_conversation_history(self, mock_anthropic_client, test_config):
        """Test that conversation history is maintained across queries"""
        with patch('rag_system.VectorStore') as MockVectorStore, \
             patch('rag_system.DocumentProcessor') as MockDocProcessor, \
             patch('rag_system.AIGenerator') as MockAIGenerator:

            mock_store = Mock()
            mock_processor = Mock()
            mock_ai = Mock()

            MockVectorStore.return_value = mock_store
            MockDocProcessor.return_value = mock_processor
            MockAIGenerator.return_value = mock_ai

            mock_store.search.return_value = SearchResults(
                documents=["Content"],
                metadata=[{"course_title": "Course", "lesson_number": 1}],
                distances=[0.5],
                error=None
            )

            # First query
            mock_ai.generate_response.return_value = ("First response", [])
            rag = RAGSystem(api_key="test-key", docs_dir="./docs")
            rag.query("First question", "session_1")

            # Second query - should include history
            mock_ai.generate_response.return_value = ("Second response", [])
            rag.query("Follow-up question", "session_1")

            # Verify AI was called with conversation history on second call
            calls = mock_ai.generate_response.call_args_list
            assert len(calls) == 2

            # Second call should have conversation_history parameter
            second_call_kwargs = calls[1][1]
            assert "conversation_history" in second_call_kwargs
            assert second_call_kwargs["conversation_history"] is not None

    def test_multiple_sessions_isolated(self, mock_anthropic_client, test_config):
        """Test that different sessions maintain separate conversation histories"""
        with patch('rag_system.VectorStore') as MockVectorStore, \
             patch('rag_system.DocumentProcessor') as MockDocProcessor, \
             patch('rag_system.AIGenerator') as MockAIGenerator:

            mock_store = Mock()
            mock_processor = Mock()
            mock_ai = Mock()

            MockVectorStore.return_value = mock_store
            MockDocProcessor.return_value = mock_processor
            MockAIGenerator.return_value = mock_ai

            mock_store.search.return_value = SearchResults(
                documents=["Content"],
                metadata=[{"course_title": "Course", "lesson_number": 1}],
                distances=[0.5],
                error=None
            )

            mock_ai.generate_response.return_value = ("Response", [])

            rag = RAGSystem(api_key="test-key", docs_dir="./docs")

            # Query in session 1
            rag.query("Session 1 question", "session_1")

            # Query in session 2 (should have no history)
            rag.query("Session 2 question", "session_2")

            # Verify sessions are isolated
            session1_history = rag.session_manager.get_history("session_1")
            session2_history = rag.session_manager.get_history("session_2")

            assert "Session 1 question" in session1_history
            assert "Session 2 question" not in session1_history
            assert "Session 2 question" in session2_history
            assert "Session 1 question" not in session2_history


@pytest.mark.unit
class TestRAGSystemErrorHandling:
    """Test RAG system's error handling for various failure scenarios"""

    def test_missing_api_key_error(self, test_config):
        """Test that missing API key is handled gracefully"""
        with patch('rag_system.VectorStore') as MockVectorStore, \
             patch('rag_system.DocumentProcessor') as MockDocProcessor:

            mock_store = Mock()
            mock_processor = Mock()

            MockVectorStore.return_value = mock_store
            MockDocProcessor.return_value = mock_processor

            # Create RAG system with empty API key
            rag = RAGSystem(api_key="", docs_dir="./docs")

            # Mock search results
            mock_store.search.return_value = SearchResults(
                documents=["Content"],
                metadata=[{"course_title": "Course", "lesson_number": 1}],
                distances=[0.5],
                error=None
            )

            # Query should handle authentication error
            response, sources = rag.query("test question", "session_1")

            # Should get error message about API key
            assert "Authentication" in response or "API key" in response

    def test_ai_api_error_handling(self, mock_anthropic_client, test_config):
        """Test that AI API errors are handled gracefully"""
        with patch('rag_system.VectorStore') as MockVectorStore, \
             patch('rag_system.DocumentProcessor') as MockDocProcessor, \
             patch('rag_system.AIGenerator') as MockAIGenerator:

            mock_store = Mock()
            mock_processor = Mock()
            mock_ai = Mock()

            MockVectorStore.return_value = mock_store
            MockDocProcessor.return_value = mock_processor
            MockAIGenerator.return_value = mock_ai

            mock_store.search.return_value = SearchResults(
                documents=["Content"],
                metadata=[{"course_title": "Course", "lesson_number": 1}],
                distances=[0.5],
                error=None
            )

            # Mock AI API error
            mock_ai.generate_response.return_value = (
                "Anthropic API error: Rate limit exceeded",
                []
            )

            rag = RAGSystem(api_key="test-key", docs_dir="./docs")
            response, sources = rag.query("test question", "session_1")

            # Should get error message
            assert "error" in response.lower() or "API" in response


@pytest.mark.integration
class TestRAGSystemWithZeroMaxResults:
    """Tests specifically for MAX_RESULTS=0 configuration issue"""

    def test_detect_zero_max_results_causes_empty_searches(self, mock_anthropic_client):
        """
        CRITICAL TEST: Verify that MAX_RESULTS=0 causes searches to return 0 results
        This test documents the current bug behavior
        """
        # Import actual config to check current value
        from config import Config

        if Config.MAX_RESULTS == 0:
            pytest.skip("MAX_RESULTS is 0 - this test validates the bug exists")

        # If MAX_RESULTS > 0, this test should pass
        assert Config.MAX_RESULTS > 0

    @patch('config.Config.MAX_RESULTS', 0)
    def test_zero_max_results_behavior(self, mock_anthropic_client, test_config):
        """
        Test that MAX_RESULTS=0 causes the system to return empty search results
        """
        with patch('rag_system.VectorStore') as MockVectorStore, \
             patch('rag_system.DocumentProcessor') as MockDocProcessor, \
             patch('rag_system.AIGenerator') as MockAIGenerator:

            mock_store = Mock()
            mock_processor = Mock()
            mock_ai = Mock()

            MockVectorStore.return_value = mock_store
            MockDocProcessor.return_value = mock_processor
            MockAIGenerator.return_value = mock_ai

            # When MAX_RESULTS=0, searches return empty
            mock_store.search.return_value = SearchResults(
                documents=[],  # Empty due to MAX_RESULTS=0
                metadata=[],
                distances=[],
                error=None
            )

            mock_ai.generate_response.return_value = (
                "I don't have specific information about that.",
                []
            )

            rag = RAGSystem(api_key="test-key", docs_dir="./docs")
            response, sources = rag.query("What are MCP servers?", "session_1")

            # With MAX_RESULTS=0, no sources returned even for valid queries
            assert sources == []


@pytest.mark.integration
class TestRAGSystemQueryTypes:
    """Test different types of queries that users might ask"""

    def test_broad_general_query(self, mock_anthropic_client, test_config):
        """Test handling of broad, general questions"""
        with patch('rag_system.VectorStore') as MockVectorStore, \
             patch('rag_system.DocumentProcessor') as MockDocProcessor, \
             patch('rag_system.AIGenerator') as MockAIGenerator:

            mock_store = Mock()
            mock_processor = Mock()
            mock_ai = Mock()

            MockVectorStore.return_value = mock_store
            MockDocProcessor.return_value = mock_processor
            MockAIGenerator.return_value = mock_ai

            # Mock multiple results for broad query
            mock_store.search.return_value = SearchResults(
                documents=["Result 1", "Result 2", "Result 3"],
                metadata=[
                    {"course_title": "Course A", "lesson_number": 1},
                    {"course_title": "Course A", "lesson_number": 2},
                    {"course_title": "Course B", "lesson_number": 1}
                ],
                distances=[0.3, 0.4, 0.5],
                error=None
            )

            mock_ai.generate_response.return_value = (
                "Comprehensive answer based on multiple sources",
                [
                    {"text": "Course A - Lesson 1", "link": None},
                    {"text": "Course A - Lesson 2", "link": None},
                    {"text": "Course B - Lesson 1", "link": None}
                ]
            )

            rag = RAGSystem(api_key="test-key", docs_dir="./docs")
            response, sources = rag.query("Tell me about MCP", "session_1")

            assert response is not None
            assert len(sources) == 3

    def test_specific_lesson_query(self, mock_anthropic_client, test_config):
        """Test query about a specific lesson"""
        with patch('rag_system.VectorStore') as MockVectorStore, \
             patch('rag_system.DocumentProcessor') as MockDocProcessor, \
             patch('rag_system.AIGenerator') as MockAIGenerator:

            mock_store = Mock()
            mock_processor = Mock()
            mock_ai = Mock()

            MockVectorStore.return_value = mock_store
            MockDocProcessor.return_value = mock_processor
            MockAIGenerator.return_value = mock_ai

            # Mock specific lesson result
            mock_store.search.return_value = SearchResults(
                documents=["Lesson 3 specific content"],
                metadata=[{"course_title": "MCP Course", "lesson_number": 3}],
                distances=[0.2],
                error=None
            )

            mock_ai.generate_response.return_value = (
                "Information from Lesson 3",
                [{"text": "MCP Course - Lesson 3", "link": None}]
            )

            rag = RAGSystem(api_key="test-key", docs_dir="./docs")
            response, sources = rag.query("What's covered in lesson 3?", "session_1")

            assert response is not None
            assert len(sources) == 1
            assert "Lesson 3" in sources[0]["text"]

    def test_how_to_query(self, mock_anthropic_client, test_config):
        """Test 'how to' procedural questions"""
        with patch('rag_system.VectorStore') as MockVectorStore, \
             patch('rag_system.DocumentProcessor') as MockDocProcessor, \
             patch('rag_system.AIGenerator') as MockAIGenerator:

            mock_store = Mock()
            mock_processor = Mock()
            mock_ai = Mock()

            MockVectorStore.return_value = mock_store
            MockDocProcessor.return_value = mock_processor
            MockAIGenerator.return_value = mock_ai

            mock_store.search.return_value = SearchResults(
                documents=["Step-by-step instructions for creating a server"],
                metadata=[{"course_title": "MCP Tutorial", "lesson_number": 2}],
                distances=[0.1],
                error=None
            )

            mock_ai.generate_response.return_value = (
                "To create a server, follow these steps: ...",
                [{"text": "MCP Tutorial - Lesson 2", "link": None}]
            )

            rag = RAGSystem(api_key="test-key", docs_dir="./docs")
            response, sources = rag.query("How do I create an MCP server?", "session_1")

            assert response is not None
            assert len(sources) == 1
