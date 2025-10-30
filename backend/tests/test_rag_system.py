import os
import tempfile
from typing import Tuple
from unittest.mock import MagicMock, Mock, patch

import pytest
from config import Config
from models import Course, CourseChunk, Lesson
from rag_system import RAGSystem


class TestRAGSystem:
    """Integration tests for RAG System orchestration"""

    @pytest.mark.integration
    def test_initialization(self, test_config):
        """Test RAG system initialization with all components"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore"),
            patch("rag_system.AIGenerator"),
            patch("rag_system.SessionManager"),
            patch("rag_system.CourseSearchTool") as mock_search_tool,
        ):

            rag = RAGSystem(test_config)

            # Verify all components are initialized
            assert rag.config == test_config
            assert rag.document_processor is not None
            assert rag.vector_store is not None
            assert rag.ai_generator is not None
            assert rag.session_manager is not None
            assert rag.tool_manager is not None
            assert rag.search_tool is not None

            # Verify search tool is registered
            assert len(rag.tool_manager.tools) == 1

    @pytest.mark.integration
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.SessionManager")
    @patch("rag_system.CourseSearchTool")
    def test_add_course_document_success(
        self,
        mock_search_tool,
        mock_session,
        mock_ai,
        mock_vector,
        mock_doc_processor,
        test_config,
    ):
        """Test successfully adding a course document"""
        # Setup mocks
        mock_course = Course(title="Test Course", instructor="Test Instructor")
        mock_chunks = [
            CourseChunk(
                content="Test content",
                course_title="Test Course",
                lesson_number=1,
                chunk_index=0,
            )
        ]

        mock_doc_processor.return_value.process_course_document.return_value = (
            mock_course,
            mock_chunks,
        )
        mock_vector_instance = Mock()
        mock_vector.return_value = mock_vector_instance

        rag = RAGSystem(test_config)

        # Test adding document
        course, chunk_count = rag.add_course_document("/path/to/test.txt")

        # Verify process was called correctly
        mock_doc_processor.return_value.process_course_document.assert_called_once_with(
            "/path/to/test.txt"
        )
        mock_vector_instance.add_course_metadata.assert_called_once_with(mock_course)
        mock_vector_instance.add_course_content.assert_called_once_with(mock_chunks)

        assert course == mock_course
        assert chunk_count == 1

    @pytest.mark.integration
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.SessionManager")
    @patch("rag_system.CourseSearchTool")
    def test_add_course_document_error(
        self,
        mock_search_tool,
        mock_session,
        mock_ai,
        mock_vector,
        mock_doc_processor,
        test_config,
    ):
        """Test error handling when adding course document fails"""
        # Setup mock to raise exception
        mock_doc_processor.return_value.process_course_document.side_effect = Exception(
            "Processing failed"
        )

        rag = RAGSystem(test_config)

        # Test error handling
        course, chunk_count = rag.add_course_document("/path/to/bad.txt")

        assert course is None
        assert chunk_count == 0

    @pytest.mark.integration
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.SessionManager")
    @patch("rag_system.CourseSearchTool")
    @patch("os.path.exists")
    @patch("os.listdir")
    def test_add_course_folder_success(
        self,
        mock_listdir,
        mock_exists,
        mock_search_tool,
        mock_session,
        mock_ai,
        mock_vector,
        mock_doc_processor,
        test_config,
    ):
        """Test adding courses from a folder"""
        # Setup mocks
        mock_exists.return_value = True
        mock_listdir.return_value = ["course1.txt", "course2.pdf", "ignored.log"]

        mock_course1 = Course(title="Course 1", instructor="Instructor 1")
        mock_course2 = Course(title="Course 2", instructor="Instructor 2")
        mock_chunks1 = [
            CourseChunk(
                content="Content 1",
                course_title="Course 1",
                lesson_number=1,
                chunk_index=0,
            )
        ]
        mock_chunks2 = [
            CourseChunk(
                content="Content 2",
                course_title="Course 2",
                lesson_number=1,
                chunk_index=0,
            )
        ]

        mock_doc_processor.return_value.process_course_document.side_effect = [
            (mock_course1, mock_chunks1),
            (mock_course2, mock_chunks2),
        ]

        mock_vector_instance = Mock()
        mock_vector.return_value = mock_vector_instance
        mock_vector_instance.get_existing_course_titles.return_value = []

        rag = RAGSystem(test_config)

        # Test adding folder
        total_courses, total_chunks = rag.add_course_folder("/test/folder")

        assert total_courses == 2
        assert total_chunks == 2

        # Verify both courses were processed
        assert mock_doc_processor.return_value.process_course_document.call_count == 2

    @pytest.mark.integration
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.SessionManager")
    @patch("rag_system.CourseSearchTool")
    @patch("os.path.exists")
    def test_add_course_folder_nonexistent(
        self,
        mock_exists,
        mock_search_tool,
        mock_session,
        mock_ai,
        mock_vector,
        mock_doc_processor,
        test_config,
    ):
        """Test adding courses from non-existent folder"""
        mock_exists.return_value = False

        rag = RAGSystem(test_config)

        total_courses, total_chunks = rag.add_course_folder("/nonexistent/folder")

        assert total_courses == 0
        assert total_chunks == 0

    @pytest.mark.integration
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.SessionManager")
    @patch("rag_system.CourseSearchTool")
    def test_query_without_session(
        self,
        mock_search_tool,
        mock_session,
        mock_ai,
        mock_vector,
        mock_doc_processor,
        test_config,
    ):
        """Test query processing without session context"""
        # Setup mocks
        mock_ai_instance = Mock()
        mock_ai.return_value = mock_ai_instance
        mock_ai_instance.generate_response.return_value = "AI response to the query"

        mock_tool_manager = Mock()
        mock_tool_manager.get_tool_definitions.return_value = [{"name": "search_tool"}]
        mock_tool_manager.get_last_sources.return_value = []
        mock_tool_manager.reset_sources.return_value = None

        rag = RAGSystem(test_config)
        rag.tool_manager = mock_tool_manager

        # Test query
        response, sources = rag.query("What is Python?")

        # Verify AI generator was called correctly
        mock_ai_instance.generate_response.assert_called_once_with(
            query="Answer this question about course materials: What is Python?",
            conversation_history=None,
            tools=[{"name": "search_tool"}],
            tool_manager=mock_tool_manager,
        )

        assert response == "AI response to the query"
        assert sources == []

    @pytest.mark.integration
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.SessionManager")
    @patch("rag_system.CourseSearchTool")
    def test_query_with_session(
        self,
        mock_search_tool,
        mock_session,
        mock_ai,
        mock_vector,
        mock_doc_processor,
        test_config,
    ):
        """Test query processing with session context"""
        # Setup mocks
        mock_ai_instance = Mock()
        mock_ai.return_value = mock_ai_instance
        mock_ai_instance.generate_response.return_value = "Contextual AI response"

        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.get_conversation_history.return_value = (
            "Previous conversation"
        )

        mock_tool_manager = Mock()
        mock_tool_manager.get_tool_definitions.return_value = []
        mock_tool_manager.get_last_sources.return_value = [
            {"text": "Source 1", "link": None}
        ]

        rag = RAGSystem(test_config)
        rag.tool_manager = mock_tool_manager

        # Test query with session
        response, sources = rag.query("Follow up question", session_id="session_1")

        # Verify session manager was used
        mock_session_instance.get_conversation_history.assert_called_once_with(
            "session_1"
        )
        mock_session_instance.add_exchange.assert_called_once_with(
            "session_1", "Follow up question", "Contextual AI response"
        )

        assert response == "Contextual AI response"
        assert sources == [{"text": "Source 1", "link": None}]

    @pytest.mark.integration
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.SessionManager")
    @patch("rag_system.CourseSearchTool")
    def test_query_with_tool_usage(
        self,
        mock_search_tool,
        mock_session,
        mock_ai,
        mock_vector,
        mock_doc_processor,
        test_config,
    ):
        """Test query that triggers tool usage"""
        # Setup mocks for tool usage scenario
        mock_ai_instance = Mock()
        mock_ai.return_value = mock_ai_instance
        mock_ai_instance.generate_response.return_value = (
            "Based on search results: Python is a programming language"
        )

        mock_tool_manager = Mock()
        mock_tool_manager.get_tool_definitions.return_value = [
            {"name": "search_course_content"}
        ]
        mock_tool_manager.get_last_sources.return_value = [
            {"text": "Python Course - Lesson 1", "link": "http://example.com/lesson1"}
        ]

        rag = RAGSystem(test_config)
        rag.tool_manager = mock_tool_manager

        # Test query that should use tools
        response, sources = rag.query("What is Python programming?")

        # Verify tools were provided to AI
        mock_ai_instance.generate_response.assert_called_once()
        call_args = mock_ai_instance.generate_response.call_args
        assert call_args[1]["tools"] == [{"name": "search_course_content"}]
        assert call_args[1]["tool_manager"] == mock_tool_manager

        # Verify sources were retrieved and reset
        mock_tool_manager.get_last_sources.assert_called_once()
        mock_tool_manager.reset_sources.assert_called_once()

        assert "Based on search results" in response
        assert len(sources) == 1
        assert sources[0]["text"] == "Python Course - Lesson 1"

    @pytest.mark.integration
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.SessionManager")
    @patch("rag_system.CourseSearchTool")
    def test_get_course_analytics(
        self,
        mock_search_tool,
        mock_session,
        mock_ai,
        mock_vector,
        mock_doc_processor,
        test_config,
    ):
        """Test retrieving course analytics"""
        # Setup mock
        mock_vector_instance = Mock()
        mock_vector.return_value = mock_vector_instance
        mock_vector_instance.get_course_count.return_value = 3
        mock_vector_instance.get_existing_course_titles.return_value = [
            "Course 1",
            "Course 2",
            "Course 3",
        ]

        rag = RAGSystem(test_config)

        analytics = rag.get_course_analytics()

        assert analytics["total_courses"] == 3
        assert analytics["course_titles"] == ["Course 1", "Course 2", "Course 3"]

    @pytest.mark.integration
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.SessionManager")
    @patch("rag_system.CourseSearchTool")
    @patch("os.path.exists")
    @patch("os.listdir")
    def test_skip_existing_courses(
        self,
        mock_listdir,
        mock_exists,
        mock_search_tool,
        mock_session,
        mock_ai,
        mock_vector,
        mock_doc_processor,
        test_config,
    ):
        """Test that existing courses are skipped during folder processing"""
        # Setup mocks
        mock_exists.return_value = True
        mock_listdir.return_value = ["course1.txt", "course2.txt"]

        mock_course1 = Course(title="Existing Course", instructor="Instructor")
        mock_course2 = Course(title="New Course", instructor="Instructor")

        mock_doc_processor.return_value.process_course_document.side_effect = [
            (mock_course1, []),
            (
                mock_course2,
                [
                    CourseChunk(
                        content="New content",
                        course_title="New Course",
                        lesson_number=1,
                        chunk_index=0,
                    )
                ],
            ),
        ]

        mock_vector_instance = Mock()
        mock_vector.return_value = mock_vector_instance
        mock_vector_instance.get_existing_course_titles.return_value = [
            "Existing Course"
        ]

        rag = RAGSystem(test_config)

        total_courses, total_chunks = rag.add_course_folder(
            "/test/folder", clear_existing=False
        )

        # Only new course should be added
        assert total_courses == 1
        assert total_chunks == 1

        # Verify only new course was added to vector store
        mock_vector_instance.add_course_metadata.assert_called_once_with(mock_course2)

    @pytest.mark.integration
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.SessionManager")
    @patch("rag_system.CourseSearchTool")
    @patch("os.path.exists")
    def test_clear_existing_data(
        self,
        mock_exists,
        mock_search_tool,
        mock_session,
        mock_ai,
        mock_vector,
        mock_doc_processor,
        test_config,
    ):
        """Test clearing existing data when clear_existing=True"""
        mock_exists.return_value = True

        mock_vector_instance = Mock()
        mock_vector.return_value = mock_vector_instance

        rag = RAGSystem(test_config)

        rag.add_course_folder("/test/folder", clear_existing=True)

        # Verify clear_all_data was called
        mock_vector_instance.clear_all_data.assert_called_once()

    @pytest.mark.integration
    def test_real_rag_system_initialization(self, test_config):
        """Test RAG system with real components (no mocking)"""
        # This test uses real components but with test config
        rag = RAGSystem(test_config)

        # Verify components are properly initialized
        assert rag.config == test_config
        assert hasattr(rag, "document_processor")
        assert hasattr(rag, "vector_store")
        assert hasattr(rag, "ai_generator")
        assert hasattr(rag, "session_manager")
        assert hasattr(rag, "tool_manager")
        assert hasattr(rag, "search_tool")

        # Verify tool manager has search tool registered
        tool_definitions = rag.tool_manager.get_tool_definitions()
        assert len(tool_definitions) == 1
        assert tool_definitions[0]["name"] == "search_course_content"

    @pytest.mark.slow
    @pytest.mark.integration
    def test_full_workflow_with_test_data(
        self, test_config, sample_course, sample_course_chunks
    ):
        """Test complete workflow with test data (slow test)"""
        rag = RAGSystem(test_config)

        # Add test data
        rag.vector_store.add_course_metadata(sample_course)
        rag.vector_store.add_course_content(sample_course_chunks)

        # Test analytics
        analytics = rag.get_course_analytics()
        assert analytics["total_courses"] == 1
        assert sample_course.title in analytics["course_titles"]

        # Test that vector store has data
        results = rag.vector_store.search("introduction")
        assert not results.is_empty()

        # Test search tool directly
        search_result = rag.search_tool.execute("introduction")
        assert "Test Course" in search_result

        # Verify tool manager can execute search
        tool_result = rag.tool_manager.execute_tool(
            "search_course_content", query="introduction"
        )
        assert isinstance(tool_result, str)
        assert len(tool_result) > 0

    @pytest.mark.integration
    @patch("rag_system.AIGenerator")
    def test_query_error_handling(self, mock_ai, test_config):
        """Test error handling in query processing"""
        # Setup AI to raise exception
        mock_ai_instance = Mock()
        mock_ai.return_value = mock_ai_instance
        mock_ai_instance.generate_response.side_effect = Exception("AI API failed")

        rag = RAGSystem(test_config)

        # Query should handle AI errors gracefully
        with pytest.raises(Exception, match="AI API failed"):
            rag.query("What is Python?")

    @pytest.mark.integration
    def test_session_integration(self, test_config):
        """Test session manager integration with RAG system"""
        rag = RAGSystem(test_config)

        # Create a session via session manager
        session_id = rag.session_manager.create_session()

        # Add some conversation history
        rag.session_manager.add_exchange(session_id, "First question", "First answer")

        # Get history and verify format
        history = rag.session_manager.get_conversation_history(session_id)
        assert "User: First question" in history
        assert "Assistant: First answer" in history
