import pytest
import tempfile
import shutil
import os
from unittest.mock import Mock, patch
from typing import List, Dict, Any

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import Course, Lesson, CourseChunk
from vector_store import VectorStore, SearchResults
from search_tools import CourseSearchTool, ToolManager
from ai_generator import AIGenerator
from rag_system import RAGSystem
from session_manager import SessionManager
from config import Config

# FastAPI imports for API testing
from fastapi.testclient import TestClient
from fastapi import FastAPI


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
        Lesson(lesson_number=1, title="Introduction", lesson_link="http://example.com/lesson1"),
        Lesson(lesson_number=2, title="Getting Started", lesson_link="http://example.com/lesson2"),
        Lesson(lesson_number=3, title="Advanced Topics")
    ]
    return Course(
        title="Test Course", 
        course_link="http://example.com/course",
        instructor="Test Instructor",
        lessons=lessons
    )


@pytest.fixture 
def sample_course_chunks(sample_course):
    """Create sample course chunks for testing"""
    return [
        CourseChunk(
            content="This is lesson 1 content about introduction to the topic.",
            course_title=sample_course.title,
            lesson_number=1,
            chunk_index=0
        ),
        CourseChunk(
            content="This covers getting started with basic concepts and setup procedures.",
            course_title=sample_course.title,
            lesson_number=2,
            chunk_index=1
        ),
        CourseChunk(
            content="Advanced topics include complex configurations and troubleshooting.",
            course_title=sample_course.title,
            lesson_number=3,
            chunk_index=2
        )
    ]


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store"""
    mock_store = Mock(spec=VectorStore)
    
    # Configure default return values
    mock_store.search.return_value = SearchResults(
        documents=["Test document content"],
        metadata=[{"course_title": "Test Course", "lesson_number": 1}],
        distances=[0.1]
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
        max_results=test_config.MAX_RESULTS
    )
    yield store
    # Cleanup
    if os.path.exists(test_config.CHROMA_PATH):
        shutil.rmtree(test_config.CHROMA_PATH)


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client for testing"""
    with patch('anthropic.Anthropic') as mock_client_class:
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
    with patch('anthropic.Anthropic') as mock_client_class:
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
        mock_final_response.content = [Mock(text="Based on the search results, here is the answer.")]
        mock_final_response.stop_reason = "end_turn"
        
        mock_client.messages.create.side_effect = [mock_initial_response, mock_final_response]
        
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
    with patch('rag_system.DocumentProcessor'), \
         patch('rag_system.VectorStore'), \
         patch('rag_system.AIGenerator'), \
         patch('rag_system.SessionManager'), \
         patch('rag_system.CourseSearchTool'):
        return RAGSystem(test_config)


# Test data helpers
def create_search_results(documents: List[str], course_title: str = "Test Course", lesson_numbers: List[int] = None) -> SearchResults:
    """Helper to create SearchResults for testing"""
    if lesson_numbers is None:
        lesson_numbers = [1] * len(documents)
    
    metadata = [{"course_title": course_title, "lesson_number": lesson_num} 
               for lesson_num in lesson_numbers]
    distances = [0.1 + i * 0.1 for i in range(len(documents))]
    
    return SearchResults(documents=documents, metadata=metadata, distances=distances)


def create_empty_search_results(error_msg: str = None) -> SearchResults:
    """Helper to create empty SearchResults for testing"""
    return SearchResults(documents=[], metadata=[], distances=[], error=error_msg)


# ==================== API Testing Fixtures ====================

@pytest.fixture
def mock_rag_for_api():
    """Create a comprehensive mock RAG system for API endpoint testing"""
    mock_system = Mock(spec=RAGSystem)

    # Session manager mock
    mock_session_manager = Mock()
    mock_session_manager.create_session.return_value = "test_session_1"
    mock_session_manager.get_session_history.return_value = []
    mock_session_manager.add_to_history.return_value = None
    mock_system.session_manager = mock_session_manager

    # Default query response
    mock_system.query.return_value = (
        "This is a test answer from the RAG system.",
        [
            {
                "course": "Test Course",
                "lesson": 1,
                "link": "http://example.com/lesson1"
            }
        ]
    )

    # Course analytics response
    mock_system.get_course_analytics.return_value = {
        "total_courses": 3,
        "course_titles": ["Course A", "Course B", "Course C"]
    }

    # Document loading response
    mock_system.add_course_folder.return_value = (3, 250)

    return mock_system


@pytest.fixture
def api_test_client(mock_rag_for_api):
    """
    Create a TestClient with a minimal FastAPI app for testing.
    This avoids the static file mounting issue in app.py.
    """
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import Optional, List, Any

    # Define models locally to avoid importing app.py
    class QueryRequest(BaseModel):
        """Request model for course queries"""
        query: str
        session_id: Optional[str] = None

    class QueryResponse(BaseModel):
        """Response model for course queries"""
        answer: str
        sources: List[Any]
        session_id: str

    class CourseStats(BaseModel):
        """Response model for course statistics"""
        total_courses: int
        course_titles: List[str]

    # Create a minimal test app
    test_app = FastAPI(title="RAG System Test API")

    # Add CORS
    test_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store RAG system in app state
    test_app.state.rag_system = mock_rag_for_api

    # Define endpoints
    @test_app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = test_app.state.rag_system.session_manager.create_session()

            answer, sources = test_app.state.rag_system.query(request.query, session_id)

            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @test_app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = test_app.state.rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @test_app.get("/health")
    async def health_check():
        return {"status": "healthy"}

    return TestClient(test_app)


@pytest.fixture
def sample_api_query_request():
    """Sample query request for API testing"""
    return {
        "query": "What is machine learning?",
        "session_id": "test_session_123"
    }


@pytest.fixture
def sample_api_query_response():
    """Sample query response for API testing"""
    return {
        "answer": "Machine learning is a subset of artificial intelligence.",
        "sources": [
            {
                "course": "ML Fundamentals",
                "lesson": 1,
                "link": "http://example.com/ml/lesson1"
            },
            {
                "course": "AI Basics",
                "lesson": 2,
                "link": "http://example.com/ai/lesson2"
            }
        ],
        "session_id": "test_session_123"
    }


@pytest.fixture
def mock_anthropic_response():
    """Create a mock response from Anthropic API"""
    mock_response = Mock()
    mock_response.content = [Mock(text="This is a test response from Claude.")]
    mock_response.stop_reason = "end_turn"
    mock_response.usage = Mock(input_tokens=100, output_tokens=50)
    return mock_response


@pytest.fixture
def mock_anthropic_tool_response():
    """Create a mock tool-calling response from Anthropic API"""
    # First response: tool use
    tool_block = Mock()
    tool_block.type = "tool_use"
    tool_block.name = "search_course_content"
    tool_block.input = {"query": "machine learning", "course_name": None, "lesson_number": None}
    tool_block.id = "toolu_123abc"

    tool_response = Mock()
    tool_response.content = [tool_block]
    tool_response.stop_reason = "tool_use"

    # Second response: final answer
    final_response = Mock()
    final_response.content = [Mock(text="Based on the search results, machine learning is...")]
    final_response.stop_reason = "end_turn"

    return (tool_response, final_response)


# ==================== Database and File System Helpers ====================

@pytest.fixture
def clean_test_environment(temp_dir):
    """
    Ensure clean test environment by clearing any cached data.
    Useful for integration tests.
    """
    # Clear any existing ChromaDB data
    chroma_path = os.path.join(temp_dir, "test_chroma_db")
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)

    yield temp_dir

    # Cleanup after test
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)


@pytest.fixture
def sample_course_document():
    """Sample course document content for testing document processing"""
    return """Course Title: Introduction to Machine Learning
Course Link: http://example.com/ml-course
Course Instructor: Dr. Jane Smith

Lesson 0: Course Overview
This course provides a comprehensive introduction to machine learning concepts and techniques.

Lesson 1: What is Machine Learning?
Lesson Link: http://example.com/ml-course/lesson1
Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed.

Lesson 2: Supervised Learning
Lesson Link: http://example.com/ml-course/lesson2
Supervised learning involves training models on labelled data to make predictions on new, unseen data.
"""


@pytest.fixture
def sample_course_file(temp_dir, sample_course_document):
    """Create a temporary course file for testing"""
    file_path = os.path.join(temp_dir, "test_course.txt")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(sample_course_document)
    return file_path


# ==================== Pytest Markers and Hooks ====================

def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "api: mark test as an API endpoint test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_api_key: mark test as requiring external API keys"
    )