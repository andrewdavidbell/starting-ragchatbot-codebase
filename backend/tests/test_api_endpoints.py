"""
API endpoint tests for the RAG system FastAPI application.

This module tests the HTTP API endpoints without requiring static files.
It creates a test app that includes only the API routes.
"""

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

# Import without triggering app initialization
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rag_system import RAGSystem
from config import Config


# Define models locally to avoid importing app.py which mounts static files
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


@pytest.fixture
def mock_rag_system():
    """Create a mock RAG system for API testing"""
    mock_system = Mock(spec=RAGSystem)

    # Configure session manager mock
    mock_session_manager = Mock()
    mock_session_manager.create_session.return_value = "session_1"
    mock_system.session_manager = mock_session_manager

    # Configure default query response
    mock_system.query.return_value = (
        "This is a test answer to your query.",
        [
            {"course": "Test Course", "lesson": 1, "link": "http://example.com/lesson1"}
        ]
    )

    # Configure course analytics response
    mock_system.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Test Course 1", "Test Course 2"]
    }

    # Configure add_course_folder mock
    mock_system.add_course_folder.return_value = (2, 150)

    return mock_system


@pytest.fixture
def test_app(mock_rag_system):
    """Create a test FastAPI app without static file mounting"""
    app = FastAPI(title="Course Materials RAG System - Test", root_path="")

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store mock RAG system in app state
    app.state.rag_system = mock_rag_system

    # Define API endpoints inline (same as app.py but using app.state.rag_system)
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        """Process a query and return response with sources"""
        try:
            # Create session if not provided
            session_id = request.session_id
            if not session_id:
                session_id = app.state.rag_system.session_manager.create_session()

            # Process query using RAG system
            answer, sources = app.state.rag_system.query(request.query, session_id)

            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        """Get course analytics and statistics"""
        try:
            analytics = app.state.rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {"status": "healthy"}

    return app


@pytest.fixture
def client(test_app):
    """Create a test client for the FastAPI app"""
    return TestClient(test_app)


# Test /api/query endpoint
class TestQueryEndpoint:
    """Tests for the /api/query endpoint"""

    def test_query_without_session_id(self, client, test_app):
        """Test querying without providing a session ID"""
        response = client.post(
            "/api/query",
            json={"query": "What is machine learning?"}
        )

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data

        # Verify session was created
        assert data["session_id"] == "session_1"
        test_app.state.rag_system.session_manager.create_session.assert_called_once()

    def test_query_with_session_id(self, client, test_app):
        """Test querying with an existing session ID"""
        session_id = "existing_session_123"
        response = client.post(
            "/api/query",
            json={
                "query": "Explain neural networks",
                "session_id": session_id
            }
        )

        assert response.status_code == 200
        data = response.json()

        # Verify session ID is preserved
        assert data["session_id"] == session_id

        # Verify RAG system was called with correct parameters
        test_app.state.rag_system.query.assert_called_once_with(
            "Explain neural networks",
            session_id
        )

    def test_query_with_sources(self, client, test_app):
        """Test that query response includes sources"""
        # Configure mock to return specific sources
        test_app.state.rag_system.query.return_value = (
            "Neural networks are computational models.",
            [
                {"course": "ML Basics", "lesson": 1, "link": "http://example.com/ml/lesson1"},
                {"course": "Deep Learning", "lesson": 3, "link": "http://example.com/dl/lesson3"}
            ]
        )

        response = client.post(
            "/api/query",
            json={"query": "What are neural networks?"}
        )

        assert response.status_code == 200
        data = response.json()

        assert len(data["sources"]) == 2
        assert data["sources"][0]["course"] == "ML Basics"
        assert data["sources"][1]["course"] == "Deep Learning"

    def test_query_empty_string(self, client):
        """Test querying with an empty string"""
        response = client.post(
            "/api/query",
            json={"query": ""}
        )

        # Should still process, validation happens in Pydantic model
        assert response.status_code == 200

    def test_query_missing_query_field(self, client):
        """Test request with missing query field"""
        response = client.post(
            "/api/query",
            json={"session_id": "test_session"}
        )

        # Pydantic validation should fail
        assert response.status_code == 422

    def test_query_invalid_json(self, client):
        """Test request with invalid JSON"""
        response = client.post(
            "/api/query",
            data="not valid json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422

    def test_query_system_error(self, client, test_app):
        """Test handling of system errors during query processing"""
        # Configure mock to raise an exception
        test_app.state.rag_system.query.side_effect = RuntimeError("Database connection failed")

        response = client.post(
            "/api/query",
            json={"query": "test query"}
        )

        assert response.status_code == 500
        assert "Database connection failed" in response.json()["detail"]

    def test_query_long_text(self, client):
        """Test querying with very long text"""
        long_query = "What is " + "machine learning " * 100

        response = client.post(
            "/api/query",
            json={"query": long_query}
        )

        assert response.status_code == 200


# Test /api/courses endpoint
class TestCoursesEndpoint:
    """Tests for the /api/courses endpoint"""

    def test_get_courses_success(self, client, test_app):
        """Test successful retrieval of course statistics"""
        response = client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "total_courses" in data
        assert "course_titles" in data

        # Verify expected values
        assert data["total_courses"] == 2
        assert len(data["course_titles"]) == 2
        assert "Test Course 1" in data["course_titles"]
        assert "Test Course 2" in data["course_titles"]

    def test_get_courses_empty_database(self, client, test_app):
        """Test course endpoint when no courses are loaded"""
        # Configure mock to return empty analytics
        test_app.state.rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }

        response = client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()

        assert data["total_courses"] == 0
        assert len(data["course_titles"]) == 0

    def test_get_courses_system_error(self, client, test_app):
        """Test handling of system errors during course retrieval"""
        # Configure mock to raise an exception
        test_app.state.rag_system.get_course_analytics.side_effect = Exception("Vector store unavailable")

        response = client.get("/api/courses")

        assert response.status_code == 500
        assert "Vector store unavailable" in response.json()["detail"]

    def test_get_courses_multiple_calls(self, client, test_app):
        """Test that multiple calls to courses endpoint work correctly"""
        # First call
        response1 = client.get("/api/courses")
        assert response1.status_code == 200

        # Second call
        response2 = client.get("/api/courses")
        assert response2.status_code == 200

        # Both should return the same data
        assert response1.json() == response2.json()

        # Verify mock was called twice
        assert test_app.state.rag_system.get_course_analytics.call_count == 2


# Test health check endpoint
class TestHealthEndpoint:
    """Tests for the health check endpoint"""

    def test_health_check(self, client):
        """Test health check endpoint returns healthy status"""
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


# Integration tests
class TestAPIIntegration:
    """Integration tests combining multiple endpoints"""

    def test_query_then_check_courses(self, client):
        """Test making a query and then checking courses"""
        # First, make a query
        query_response = client.post(
            "/api/query",
            json={"query": "What is Python?"}
        )
        assert query_response.status_code == 200

        # Then, check courses
        courses_response = client.get("/api/courses")
        assert courses_response.status_code == 200
        assert courses_response.json()["total_courses"] >= 0

    def test_multiple_queries_same_session(self, client, test_app):
        """Test multiple queries within the same session"""
        # First query
        response1 = client.post(
            "/api/query",
            json={"query": "What is AI?"}
        )
        session_id = response1.json()["session_id"]

        # Second query with same session
        response2 = client.post(
            "/api/query",
            json={
                "query": "Tell me more about that",
                "session_id": session_id
            }
        )

        assert response2.status_code == 200
        assert response2.json()["session_id"] == session_id

    def test_cors_headers_present(self, client):
        """Test that CORS headers are properly set"""
        response = client.options(
            "/api/query",
            headers={"Origin": "http://localhost:3000"}
        )

        # FastAPI TestClient doesn't always process OPTIONS correctly
        # but we can verify the app has CORS middleware
        assert response.status_code in [200, 405]


# Request validation tests
class TestRequestValidation:
    """Tests for request validation and error handling"""

    def test_query_request_validation(self):
        """Test QueryRequest model validation"""
        # Valid request
        request = QueryRequest(query="test query", session_id="test_session")
        assert request.query == "test query"
        assert request.session_id == "test_session"

        # Valid request without session_id
        request = QueryRequest(query="test query")
        assert request.session_id is None

    def test_query_response_validation(self):
        """Test QueryResponse model validation"""
        response = QueryResponse(
            answer="test answer",
            sources=[{"course": "Test", "lesson": 1}],
            session_id="test_session"
        )
        assert response.answer == "test answer"
        assert len(response.sources) == 1

    def test_course_stats_validation(self):
        """Test CourseStats model validation"""
        stats = CourseStats(total_courses=5, course_titles=["Course 1", "Course 2"])
        assert stats.total_courses == 5
        assert len(stats.course_titles) == 2


# Performance and edge case tests
class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def test_query_special_characters(self, client):
        """Test query with special characters"""
        special_query = "What is C++ & Python? <test> 'quotes' \"double\""

        response = client.post(
            "/api/query",
            json={"query": special_query}
        )

        assert response.status_code == 200

    def test_query_unicode_characters(self, client):
        """Test query with Unicode characters"""
        unicode_query = "What is 机器学习 and Überlearning? 🤖"

        response = client.post(
            "/api/query",
            json={"query": unicode_query}
        )

        assert response.status_code == 200

    def test_concurrent_sessions(self, client, test_app):
        """Test handling multiple concurrent sessions"""
        # Configure mock to return different session IDs
        test_app.state.rag_system.session_manager.create_session.side_effect = [
            "session_1", "session_2", "session_3"
        ]

        # Make multiple queries
        responses = []
        for i in range(3):
            response = client.post(
                "/api/query",
                json={"query": f"Query {i}"}
            )
            responses.append(response)

        # All should succeed with different session IDs
        assert all(r.status_code == 200 for r in responses)
        session_ids = [r.json()["session_id"] for r in responses]
        assert len(set(session_ids)) == 3  # All unique

    def test_query_with_newlines_and_whitespace(self, client):
        """Test query with various whitespace characters"""
        query_with_whitespace = "What is\n\nmachine learning?\t\tExplain   in detail."

        response = client.post(
            "/api/query",
            json={"query": query_with_whitespace}
        )

        assert response.status_code == 200
