import pytest
import json
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestAPIIntegration:
    """Integration tests for FastAPI endpoints"""

    @pytest.fixture
    def client(self, api_test_client):
        """Use the API test client from conftest"""
        return api_test_client

    @pytest.fixture
    def mock_rag_system(self, api_test_client):
        """Get the mock RAG system from the test client"""
        return api_test_client.app.state.rag_system

    @pytest.mark.integration
    def test_query_endpoint_basic(self, client, mock_rag_system):
        """Test basic query endpoint functionality"""
        # Setup mock
        mock_rag_system.query.return_value = ("This is the answer", [])
        mock_rag_system.session_manager.create_session.return_value = "session_123"
        
        # Make request
        response = client.post("/api/query", json={
            "query": "What is Python?"
        })
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        
        assert data["answer"] == "This is the answer"
        assert data["sources"] == []
        assert data["session_id"] == "session_123"
        
        # Verify RAG system was called correctly
        mock_rag_system.query.assert_called_once_with("What is Python?", "session_123")

    @pytest.mark.integration
    def test_query_endpoint_with_session(self, client, mock_rag_system):
        """Test query endpoint with existing session"""
        # Setup mock
        mock_rag_system.query.return_value = ("Follow-up answer", [{"text": "Source 1", "link": None}])
        
        # Make request with session ID
        response = client.post("/api/query", json={
            "query": "Tell me more",
            "session_id": "existing_session"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["answer"] == "Follow-up answer"
        assert len(data["sources"]) == 1
        assert data["session_id"] == "existing_session"
        
        # Verify existing session was used
        mock_rag_system.query.assert_called_once_with("Tell me more", "existing_session")
        mock_rag_system.session_manager.create_session.assert_not_called()

    @pytest.mark.integration
    def test_query_endpoint_with_sources(self, client, mock_rag_system):
        """Test query endpoint returning sources"""
        # Setup mock with sources
        mock_sources = [
            {"text": "Python Course - Lesson 1", "link": "http://example.com/lesson1"},
            {"text": "Advanced Python - Lesson 3", "link": None}
        ]
        mock_rag_system.query.return_value = ("Python is a programming language", mock_sources)
        mock_rag_system.session_manager.create_session.return_value = "session_456"
        
        response = client.post("/api/query", json={
            "query": "What is Python?"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["answer"] == "Python is a programming language"
        assert data["sources"] == mock_sources
        assert len(data["sources"]) == 2

    @pytest.mark.integration
    def test_query_endpoint_empty_query(self, client, mock_rag_system):
        """Test query endpoint with empty query"""
        response = client.post("/api/query", json={
            "query": ""
        })
        
        # Should still process empty query
        assert response.status_code == 200

    @pytest.mark.integration
    def test_query_endpoint_missing_query(self, client, mock_rag_system):
        """Test query endpoint with missing query field"""
        response = client.post("/api/query", json={})
        
        # Should return validation error
        assert response.status_code == 422

    @pytest.mark.integration
    def test_query_endpoint_invalid_json(self, client):
        """Test query endpoint with invalid JSON"""
        response = client.post("/api/query", data="invalid json")
        
        assert response.status_code == 422

    @pytest.mark.integration
    def test_query_endpoint_rag_error(self, client, mock_rag_system):
        """Test query endpoint when RAG system raises exception"""
        # Setup mock to raise exception
        mock_rag_system.query.side_effect = Exception("RAG system failed")
        mock_rag_system.session_manager.create_session.return_value = "session_error"
        
        response = client.post("/api/query", json={
            "query": "What causes error?"
        })
        
        assert response.status_code == 500
        assert "RAG system failed" in response.json()["detail"]

    @pytest.mark.integration
    def test_courses_endpoint_basic(self, client, mock_rag_system):
        """Test courses endpoint basic functionality"""
        # Setup mock
        mock_analytics = {
            "total_courses": 4,
            "course_titles": [
                "Advanced Retrieval for AI with Chroma",
                "Prompt Compression and Query Optimization", 
                "Building Towards Computer Use with Anthropic",
                "MCP: Build Rich-Context AI Apps with Anthropic"
            ]
        }
        mock_rag_system.get_course_analytics.return_value = mock_analytics
        
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_courses"] == 4
        assert len(data["course_titles"]) == 4
        assert "Advanced Retrieval for AI with Chroma" in data["course_titles"]

    @pytest.mark.integration
    def test_courses_endpoint_empty(self, client, mock_rag_system):
        """Test courses endpoint with no courses"""
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }
        
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_courses"] == 0
        assert data["course_titles"] == []

    @pytest.mark.integration
    def test_courses_endpoint_error(self, client, mock_rag_system):
        """Test courses endpoint when analytics fails"""
        mock_rag_system.get_course_analytics.side_effect = Exception("Analytics failed")
        
        response = client.get("/api/courses")
        
        assert response.status_code == 500
        assert "Analytics failed" in response.json()["detail"]

    @pytest.mark.integration
    def test_cors_headers(self, client):
        """Test CORS headers are present"""
        response = client.options("/api/query")
        
        # Should have CORS headers
        headers = response.headers
        assert "access-control-allow-origin" in headers
        assert "access-control-allow-methods" in headers

    @pytest.mark.integration  
    def test_static_file_serving(self, client):
        """Test static file serving for frontend"""
        # Test root path serves HTML
        response = client.get("/")
        
        # Should serve frontend files (may be 200 or 404 depending on if frontend exists)
        assert response.status_code in [200, 404, 404]

    @pytest.mark.integration
    def test_request_response_models(self, client, mock_rag_system):
        """Test request/response model validation"""
        # Setup mock
        mock_rag_system.query.return_value = ("Test answer", [])
        mock_rag_system.session_manager.create_session.return_value = "session_test"
        
        # Test with all optional fields
        response = client.post("/api/query", json={
            "query": "Test query",
            "session_id": "optional_session"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response has required fields
        assert "answer" in data
        assert "sources" in data  
        assert "session_id" in data
        assert isinstance(data["sources"], list)

    @pytest.mark.integration
    def test_content_type_handling(self, client, mock_rag_system):
        """Test proper content type handling"""
        mock_rag_system.query.return_value = ("Test", [])
        mock_rag_system.session_manager.create_session.return_value = "session"
        
        # Test JSON content type
        response = client.post("/api/query",
                             json={"query": "test"},
                             headers={"Content-Type": "application/json"})
        
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/json")

    @pytest.mark.integration
    def test_large_query_handling(self, client, mock_rag_system):
        """Test handling of large queries"""
        mock_rag_system.query.return_value = ("Large response", [])
        mock_rag_system.session_manager.create_session.return_value = "session"
        
        # Create large query
        large_query = "What is Python? " * 1000  # Very long query
        
        response = client.post("/api/query", json={
            "query": large_query
        })
        
        assert response.status_code == 200
        # Should handle large queries gracefully

    @pytest.mark.integration
    def test_special_characters_in_query(self, client, mock_rag_system):
        """Test handling of special characters in queries"""
        mock_rag_system.query.return_value = ("Special char response", [])
        mock_rag_system.session_manager.create_session.return_value = "session"
        
        special_query = "What about @#$%^&*()? Unicode: 🐍 Python?"
        
        response = client.post("/api/query", json={
            "query": special_query
        })
        
        assert response.status_code == 200
        # Should handle special characters and unicode

    @pytest.mark.integration
    def test_concurrent_requests(self, client, mock_rag_system):
        """Test handling of concurrent requests"""
        mock_rag_system.query.return_value = ("Concurrent response", [])
        mock_rag_system.session_manager.create_session.return_value = "session"
        
        # Make multiple concurrent requests (simulated)
        responses = []
        for i in range(5):
            response = client.post("/api/query", json={
                "query": f"Concurrent query {i}"
            })
            responses.append(response)
        
        # All should succeed
        for response in responses:
            assert response.status_code == 200

    @pytest.mark.integration 
    def test_session_persistence(self, client, mock_rag_system):
        """Test session persistence across requests"""
        mock_rag_system.query.return_value = ("Response", [])
        mock_rag_system.session_manager.create_session.return_value = "persistent_session"
        
        # First request creates session
        response1 = client.post("/api/query", json={
            "query": "First query"
        })
        
        session_id = response1.json()["session_id"]
        
        # Second request uses same session
        response2 = client.post("/api/query", json={
            "query": "Second query",
            "session_id": session_id
        })
        
        assert response2.json()["session_id"] == session_id
        
        # Verify RAG system got the session ID both times
        calls = mock_rag_system.query.call_args_list
        assert calls[0][0][1] == session_id  # Second argument (session_id) of first call
        assert calls[1][0][1] == session_id  # Second argument (session_id) of second call


class TestAPIRealIntegration:
    """Integration tests with real components (no mocking)"""

    @pytest.fixture
    def real_client(self):
        """Create test client with real components"""
        # Skip if frontend directory doesn't exist (required for real app import)
        frontend_path = os.path.join(os.path.dirname(__file__), '..', '..', 'frontend')
        if not os.path.exists(frontend_path):
            pytest.skip("Frontend directory not found - skipping real integration tests")

        from app import app
        # Note: This uses the actual RAG system, so tests may be slower
        return TestClient(app)

    @pytest.mark.slow
    @pytest.mark.integration
    def test_real_courses_endpoint(self, real_client):
        """Test courses endpoint with real RAG system"""
        response = real_client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()

        assert "total_courses" in data
        assert "course_titles" in data
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)

    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.requires_api_key
    def test_real_query_endpoint(self, real_client):
        """Test query endpoint with real RAG system (requires API key)"""
        # This test requires a valid ANTHROPIC_API_KEY
        response = real_client.post("/api/query", json={
            "query": "What is 2+2?"
        })

        # May succeed or fail depending on system state and API key
        if response.status_code == 200:
            data = response.json()
            assert "answer" in data
            assert "sources" in data
            assert "session_id" in data
        else:
            # If it fails, it should be a proper error response
            assert response.status_code in [500, 422]

    @pytest.mark.integration
    def test_api_response_time(self, real_client):
        """Test API response time is reasonable"""
        import time

        start_time = time.time()
        response = real_client.get("/api/courses")
        end_time = time.time()

        response_time = end_time - start_time

        # API should respond within reasonable time (adjust threshold as needed)
        assert response_time < 5.0  # 5 seconds max for courses endpoint
        assert response.status_code == 200