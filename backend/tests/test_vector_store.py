import pytest
import os
import shutil
from unittest.mock import Mock, patch

from vector_store import VectorStore, SearchResults
from models import Course, Lesson, CourseChunk


class TestSearchResults:
    """Unit tests for SearchResults dataclass"""

    @pytest.mark.unit
    def test_from_chroma_with_results(self):
        """Test creating SearchResults from ChromaDB query results"""
        chroma_results = {
            'documents': [['Document 1', 'Document 2']],
            'metadatas': [[{'course_title': 'Course 1'}, {'course_title': 'Course 2'}]],
            'distances': [[0.1, 0.2]]
        }
        
        results = SearchResults.from_chroma(chroma_results)
        
        assert results.documents == ['Document 1', 'Document 2']
        assert results.metadata == [{'course_title': 'Course 1'}, {'course_title': 'Course 2'}]
        assert results.distances == [0.1, 0.2]
        assert results.error is None

    @pytest.mark.unit
    def test_from_chroma_empty_results(self):
        """Test creating SearchResults from empty ChromaDB results"""
        chroma_results = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
        
        results = SearchResults.from_chroma(chroma_results)
        
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error is None

    @pytest.mark.unit
    def test_empty_results_with_error(self):
        """Test creating empty results with error message"""
        error_msg = "Database connection failed"
        results = SearchResults.empty(error_msg)
        
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error == error_msg

    @pytest.mark.unit
    def test_is_empty(self):
        """Test is_empty method"""
        # Empty results
        empty_results = SearchResults([], [], [])
        assert empty_results.is_empty() is True
        
        # Non-empty results
        results = SearchResults(['doc'], [{'meta': 'data'}], [0.1])
        assert results.is_empty() is False


class TestVectorStore:
    """Unit tests for VectorStore class"""

    @pytest.mark.unit
    def test_initialization(self, test_config):
        """Test VectorStore initialization"""
        store = VectorStore(test_config.CHROMA_PATH, test_config.EMBEDDING_MODEL, test_config.MAX_RESULTS)
        
        assert store.max_results == test_config.MAX_RESULTS
        assert store.client is not None
        assert store.embedding_function is not None
        assert store.course_catalog is not None
        assert store.course_content is not None

    @pytest.mark.integration
    def test_add_course_metadata(self, real_vector_store, sample_course):
        """Test adding course metadata to catalog"""
        real_vector_store.add_course_metadata(sample_course)
        
        # Verify course was added
        existing_titles = real_vector_store.get_existing_course_titles()
        assert sample_course.title in existing_titles
        
        # Verify course count
        assert real_vector_store.get_course_count() == 1

    @pytest.mark.integration
    def test_add_course_content(self, real_vector_store, sample_course_chunks):
        """Test adding course content chunks"""
        real_vector_store.add_course_content(sample_course_chunks)
        
        # Verify we can search for the content
        results = real_vector_store.search("introduction")
        assert not results.is_empty()
        assert any("introduction" in doc.lower() for doc in results.documents)

    @pytest.mark.integration
    def test_search_basic(self, real_vector_store, sample_course, sample_course_chunks):
        """Test basic search functionality"""
        # Add test data
        real_vector_store.add_course_metadata(sample_course)
        real_vector_store.add_course_content(sample_course_chunks)
        
        # Search for content
        results = real_vector_store.search("introduction")
        
        assert not results.is_empty()
        assert len(results.documents) > 0
        assert len(results.metadata) > 0
        assert len(results.distances) > 0
        assert results.error is None

    @pytest.mark.integration
    def test_search_with_course_filter(self, real_vector_store, sample_course, sample_course_chunks):
        """Test search with course name filtering"""
        # Add test data
        real_vector_store.add_course_metadata(sample_course)
        real_vector_store.add_course_content(sample_course_chunks)
        
        # Search with course filter
        results = real_vector_store.search("content", course_name="Test Course")
        
        assert not results.is_empty()
        # All results should be from the specified course
        for meta in results.metadata:
            assert meta['course_title'] == "Test Course"

    @pytest.mark.integration
    def test_search_with_lesson_filter(self, real_vector_store, sample_course, sample_course_chunks):
        """Test search with lesson number filtering"""
        # Add test data
        real_vector_store.add_course_metadata(sample_course)
        real_vector_store.add_course_content(sample_course_chunks)
        
        # Search with lesson filter
        results = real_vector_store.search("content", lesson_number=2)
        
        assert not results.is_empty()
        # All results should be from lesson 2
        for meta in results.metadata:
            assert meta['lesson_number'] == 2

    @pytest.mark.integration
    def test_search_with_both_filters(self, real_vector_store, sample_course, sample_course_chunks):
        """Test search with both course and lesson filters"""
        # Add test data
        real_vector_store.add_course_metadata(sample_course)
        real_vector_store.add_course_content(sample_course_chunks)
        
        # Search with both filters
        results = real_vector_store.search("getting started", 
                                         course_name="Test Course", 
                                         lesson_number=2)
        
        assert not results.is_empty()
        # Results should match both filters
        for meta in results.metadata:
            assert meta['course_title'] == "Test Course"
            assert meta['lesson_number'] == 2

    @pytest.mark.integration
    def test_search_nonexistent_course(self, real_vector_store, sample_course, sample_course_chunks):
        """Test search for non-existent course"""
        # Add test data
        real_vector_store.add_course_metadata(sample_course)
        real_vector_store.add_course_content(sample_course_chunks)
        
        # Search for non-existent course
        results = real_vector_store.search("content", course_name="Nonexistent Course")
        
        assert results.error is not None
        assert "No course found matching" in results.error

    @pytest.mark.integration
    def test_course_name_resolution(self, real_vector_store, sample_course):
        """Test course name resolution via semantic search"""
        # Add course metadata
        real_vector_store.add_course_metadata(sample_course)
        
        # Test exact match
        resolved = real_vector_store._resolve_course_name("Test Course")
        assert resolved == "Test Course"
        
        # Test partial match
        resolved = real_vector_store._resolve_course_name("Test")
        assert resolved == "Test Course"

    @pytest.mark.unit
    def test_build_filter_no_filters(self, real_vector_store):
        """Test filter building with no filters"""
        filter_dict = real_vector_store._build_filter(None, None)
        assert filter_dict is None

    @pytest.mark.unit
    def test_build_filter_course_only(self, real_vector_store):
        """Test filter building with course filter only"""
        filter_dict = real_vector_store._build_filter("Test Course", None)
        assert filter_dict == {"course_title": "Test Course"}

    @pytest.mark.unit
    def test_build_filter_lesson_only(self, real_vector_store):
        """Test filter building with lesson filter only"""
        filter_dict = real_vector_store._build_filter(None, 2)
        assert filter_dict == {"lesson_number": 2}

    @pytest.mark.unit
    def test_build_filter_both(self, real_vector_store):
        """Test filter building with both filters"""
        filter_dict = real_vector_store._build_filter("Test Course", 2)
        expected = {"$and": [
            {"course_title": "Test Course"},
            {"lesson_number": 2}
        ]}
        assert filter_dict == expected

    @pytest.mark.integration
    def test_get_lesson_link(self, real_vector_store, sample_course):
        """Test retrieving lesson links"""
        # Add course with lesson links
        real_vector_store.add_course_metadata(sample_course)
        
        # Get lesson link
        link = real_vector_store.get_lesson_link("Test Course", 1)
        assert link == "http://example.com/lesson1"
        
        # Test lesson without link
        link = real_vector_store.get_lesson_link("Test Course", 3)
        assert link is None  # Lesson 3 has no link in sample data
        
        # Test nonexistent course
        link = real_vector_store.get_lesson_link("Nonexistent", 1)
        assert link is None

    @pytest.mark.integration
    def test_get_course_link(self, real_vector_store, sample_course):
        """Test retrieving course links"""
        # Add course
        real_vector_store.add_course_metadata(sample_course)
        
        # Get course link
        link = real_vector_store.get_course_link("Test Course")
        assert link == "http://example.com/course"
        
        # Test nonexistent course
        link = real_vector_store.get_course_link("Nonexistent")
        assert link is None

    @pytest.mark.integration
    def test_get_all_courses_metadata(self, real_vector_store, sample_course):
        """Test retrieving all courses metadata"""
        # Add course
        real_vector_store.add_course_metadata(sample_course)
        
        metadata = real_vector_store.get_all_courses_metadata()
        
        assert len(metadata) == 1
        course_meta = metadata[0]
        assert course_meta['title'] == "Test Course"
        assert course_meta['instructor'] == "Test Instructor"
        assert 'lessons' in course_meta
        assert len(course_meta['lessons']) == 3

    @pytest.mark.integration
    def test_clear_all_data(self, real_vector_store, sample_course, sample_course_chunks):
        """Test clearing all data from collections"""
        # Add data
        real_vector_store.add_course_metadata(sample_course)
        real_vector_store.add_course_content(sample_course_chunks)
        
        # Verify data exists
        assert real_vector_store.get_course_count() > 0
        
        # Clear data
        real_vector_store.clear_all_data()
        
        # Verify data is cleared
        assert real_vector_store.get_course_count() == 0
        results = real_vector_store.search("content")
        assert results.is_empty()

    @pytest.mark.integration
    def test_search_limit_parameter(self, real_vector_store, sample_course, sample_course_chunks):
        """Test search with custom limit parameter"""
        # Add test data
        real_vector_store.add_course_metadata(sample_course)
        real_vector_store.add_course_content(sample_course_chunks)
        
        # Search with custom limit
        results = real_vector_store.search("content", limit=2)
        
        assert len(results.documents) <= 2
        assert len(results.metadata) <= 2
        assert len(results.distances) <= 2

    @pytest.mark.integration
    def test_empty_course_chunks(self, real_vector_store):
        """Test adding empty course chunks list"""
        # Should handle empty list gracefully
        real_vector_store.add_course_content([])
        
        # No error should occur
        assert real_vector_store.get_course_count() == 0

    @pytest.mark.integration
    def test_duplicate_course_handling(self, real_vector_store, sample_course):
        """Test handling duplicate course additions"""
        # Add course twice
        real_vector_store.add_course_metadata(sample_course)
        real_vector_store.add_course_metadata(sample_course)
        
        # Should still only have one course (ChromaDB handles duplicates by ID)
        count = real_vector_store.get_course_count()
        assert count == 1

    @pytest.mark.unit
    @patch('vector_store.chromadb.PersistentClient')
    def test_chroma_connection_error(self, mock_client, test_config):
        """Test handling of ChromaDB connection errors"""
        mock_client.side_effect = Exception("ChromaDB connection failed")
        
        with pytest.raises(Exception, match="ChromaDB connection failed"):
            VectorStore(test_config.CHROMA_PATH, test_config.EMBEDDING_MODEL, test_config.MAX_RESULTS)

    @pytest.mark.integration
    def test_search_with_special_characters(self, real_vector_store, sample_course):
        """Test search with special characters and edge cases"""
        # Add course
        real_vector_store.add_course_metadata(sample_course)
        
        # Add content with special characters
        special_chunks = [
            CourseChunk(
                content="Special characters: @#$%^&*()!",
                course_title=sample_course.title,
                lesson_number=1,
                chunk_index=0
            )
        ]
        real_vector_store.add_course_content(special_chunks)
        
        # Search should handle special characters
        results = real_vector_store.search("special characters")
        assert not results.is_empty()

    @pytest.mark.integration
    def test_search_error_handling(self, real_vector_store):
        """Test search error handling with malformed queries"""
        # This test depends on how ChromaDB handles errors
        # Search in empty database should return empty results, not error
        results = real_vector_store.search("")
        # Empty query might return empty results or all results depending on ChromaDB behavior
        assert results.error is None or results.is_empty()

    @pytest.mark.integration
    def test_large_content_chunks(self, real_vector_store, sample_course):
        """Test handling of large content chunks"""
        # Add course
        real_vector_store.add_course_metadata(sample_course)
        
        # Create large content chunk
        large_content = "Large content chunk. " * 1000  # Very large chunk
        large_chunks = [
            CourseChunk(
                content=large_content,
                course_title=sample_course.title,
                lesson_number=1,
                chunk_index=0
            )
        ]
        
        # Should handle large chunks
        real_vector_store.add_course_content(large_chunks)
        
        results = real_vector_store.search("large content")
        assert not results.is_empty()