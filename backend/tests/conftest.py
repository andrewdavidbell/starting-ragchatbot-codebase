import os
import shutil
import sys
import tempfile
from typing import Any, Dict, List
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ai_generator import AIGenerator
from config import Config
from models import Course, CourseChunk, Lesson
from rag_system import RAGSystem
from search_tools import CourseSearchTool, ToolManager
from session_manager import SessionManager
from vector_store import SearchResults, VectorStore


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_config(temp_dir):
    """Create test configuration"""
    config = Config()
    config.CHROMA_PATH = os.path.join(temp_dir, "test_chroma_db")
    config.ANTHROPIC_API_KEY = "test_api_key"
    config.MAX_RESULTS = 3
    config.CHUNK_SIZE = 200
    config.CHUNK_OVERLAP = 50
    return config


@pytest.fixture
def sample_course():
    """Create a sample course for testing"""
    lessons = [
        Lesson(
            lesson_number=1,
            title="Introduction",
            lesson_link="http://example.com/lesson1",
        ),
        Lesson(
            lesson_number=2,
            title="Getting Started",
            lesson_link="http://example.com/lesson2",
        ),
        Lesson(lesson_number=3, title="Advanced Topics"),
    ]
    return Course(
        title="Test Course",
        course_link="http://example.com/course",
        instructor="Test Instructor",
        lessons=lessons,
    )


@pytest.fixture
def sample_course_chunks(sample_course):
    """Create sample course chunks for testing"""
    return [
        CourseChunk(
            content="This is lesson 1 content about introduction to the topic.",
            course_title=sample_course.title,
            lesson_number=1,
            chunk_index=0,
        ),
        CourseChunk(
            content="This covers getting started with basic concepts and setup procedures.",
            course_title=sample_course.title,
            lesson_number=2,
            chunk_index=1,
        ),
        CourseChunk(
            content="Advanced topics include complex configurations and troubleshooting.",
            course_title=sample_course.title,
            lesson_number=3,
            chunk_index=2,
        ),
    ]


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store"""
    mock_store = Mock(spec=VectorStore)

    # Configure default return values
    mock_store.search.return_value = SearchResults(
        documents=["Test document content"],
        metadata=[{"course_title": "Test Course", "lesson_number": 1}],
        distances=[0.1],
    )

    mock_store.get_existing_course_titles.return_value = ["Test Course"]
    mock_store.get_course_count.return_value = 1
    mock_store.get_lesson_link.return_value = "http://example.com/lesson1"

    return mock_store


@pytest.fixture
def real_vector_store(test_config):
    """Create a real vector store for integration tests"""
    store = VectorStore(
        chroma_path=test_config.CHROMA_PATH,
        embedding_model=test_config.EMBEDDING_MODEL,
        max_results=test_config.MAX_RESULTS,
    )
    yield store
    # Cleanup
    if os.path.exists(test_config.CHROMA_PATH):
        shutil.rmtree(test_config.CHROMA_PATH)


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client for testing"""
    with patch("anthropic.Anthropic") as mock_client_class:
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Configure mock response
        mock_response = Mock()
        mock_response.content = [Mock(text="Test AI response")]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response

        yield mock_client


@pytest.fixture
def mock_anthropic_with_tools():
    """Create a mock Anthropic client that simulates tool calling"""
    with patch("anthropic.Anthropic") as mock_client_class:
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Mock tool use response
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.input = {"query": "test query"}
        mock_tool_block.id = "tool_call_123"

        mock_initial_response = Mock()
        mock_initial_response.content = [mock_tool_block]
        mock_initial_response.stop_reason = "tool_use"

        # Mock final response after tool execution
        mock_final_response = Mock()
        mock_final_response.content = [
            Mock(text="Based on the search results, here is the answer.")
        ]
        mock_final_response.stop_reason = "end_turn"

        mock_client.messages.create.side_effect = [
            mock_initial_response,
            mock_final_response,
        ]

        yield mock_client


@pytest.fixture
def search_tool(mock_vector_store):
    """Create a CourseSearchTool with mock vector store"""
    return CourseSearchTool(mock_vector_store)


@pytest.fixture
def tool_manager(search_tool):
    """Create a ToolManager with registered search tool"""
    manager = ToolManager()
    manager.register_tool(search_tool)
    return manager


@pytest.fixture
def ai_generator(test_config):
    """Create an AIGenerator for testing"""
    return AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)


@pytest.fixture
def session_manager():
    """Create a SessionManager for testing"""
    return SessionManager(max_history=2)


@pytest.fixture
def rag_system(test_config):
    """Create a RAGSystem for integration testing"""
    with (
        patch("rag_system.DocumentProcessor"),
        patch("rag_system.VectorStore"),
        patch("rag_system.AIGenerator"),
        patch("rag_system.SessionManager"),
        patch("rag_system.CourseSearchTool"),
    ):
        return RAGSystem(test_config)


# Test data helpers
def create_search_results(
    documents: List[str],
    course_title: str = "Test Course",
    lesson_numbers: List[int] = None,
) -> SearchResults:
    """Helper to create SearchResults for testing"""
    if lesson_numbers is None:
        lesson_numbers = [1] * len(documents)

    metadata = [
        {"course_title": course_title, "lesson_number": lesson_num}
        for lesson_num in lesson_numbers
    ]
    distances = [0.1 + i * 0.1 for i in range(len(documents))]

    return SearchResults(documents=documents, metadata=metadata, distances=distances)


def create_empty_search_results(error_msg: str = None) -> SearchResults:
    """Helper to create empty SearchResults for testing"""
    return SearchResults(documents=[], metadata=[], distances=[], error=error_msg)
