"""
Enhanced tests for CourseSearchTool.execute() method
Tests edge cases, error scenarios, and MAX_RESULTS configuration issues
"""

import pytest
from unittest.mock import Mock, patch
from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


@pytest.mark.unit
class TestCourseSearchToolExecute:
    """Enhanced tests for CourseSearchTool.execute() method"""

    def test_execute_with_zero_max_results(self, mock_vector_store):
        """Test that MAX_RESULTS=0 causes empty results - THIS SHOULD REVEAL THE BUG"""
        tool = CourseSearchTool(mock_vector_store)

        # Configure mock to return empty results (simulating MAX_RESULTS=0)
        mock_vector_store.search.return_value = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )

        result = tool.execute(query="test query")

        # Should return "no relevant content found" message
        assert "No relevant content found" in result
        assert tool.last_sources == []

    def test_execute_with_valid_results(self, mock_vector_store, create_search_results):
        """Test execute returns formatted results correctly"""
        tool = CourseSearchTool(mock_vector_store)

        # Create valid search results
        search_results = create_search_results(
            documents=["Lesson content about MCP servers"],
            metadata=[{
                "course_title": "Introduction to MCP",
                "lesson_number": 1,
                "chunk_index": 0
            }]
        )
        mock_vector_store.search.return_value = search_results

        result = tool.execute(query="MCP servers")

        # Verify formatted output
        assert "[Introduction to MCP - Lesson 1]" in result
        assert "Lesson content about MCP servers" in result
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["text"] == "Introduction to MCP - Lesson 1"

    def test_execute_with_error_from_vector_store(self, mock_vector_store, create_empty_search_results):
        """Test that errors from VectorStore are properly propagated"""
        tool = CourseSearchTool(mock_vector_store)

        # Create error result
        error_result = create_empty_search_results(error_msg="ChromaDB connection failed")
        mock_vector_store.search.return_value = error_result

        result = tool.execute(query="test query")

        # Should return the error message
        assert result == "ChromaDB connection failed"
        assert tool.last_sources == []

    def test_execute_with_course_name_filter(self, mock_vector_store, create_search_results):
        """Test execute passes course_name filter to vector store"""
        tool = CourseSearchTool(mock_vector_store)

        search_results = create_search_results(
            documents=["MCP content"],
            metadata=[{"course_title": "Introduction to MCP", "lesson_number": 1}]
        )
        mock_vector_store.search.return_value = search_results

        result = tool.execute(query="servers", course_name="Introduction to MCP")

        # Verify vector store was called with correct parameters
        mock_vector_store.search.assert_called_once_with(
            query="servers",
            course_name="Introduction to MCP",
            lesson_number=None
        )
        assert "MCP content" in result

    def test_execute_with_lesson_number_filter(self, mock_vector_store, create_search_results):
        """Test execute passes lesson_number filter to vector store"""
        tool = CourseSearchTool(mock_vector_store)

        search_results = create_search_results(
            documents=["Lesson 2 content"],
            metadata=[{"course_title": "MCP Course", "lesson_number": 2}]
        )
        mock_vector_store.search.return_value = search_results

        result = tool.execute(query="servers", lesson_number=2)

        # Verify vector store was called with lesson filter
        mock_vector_store.search.assert_called_once_with(
            query="servers",
            course_name=None,
            lesson_number=2
        )
        assert "Lesson 2 content" in result

    def test_execute_with_both_filters(self, mock_vector_store, create_search_results):
        """Test execute with both course_name and lesson_number filters"""
        tool = CourseSearchTool(mock_vector_store)

        search_results = create_search_results(
            documents=["Specific lesson content"],
            metadata=[{"course_title": "MCP Course", "lesson_number": 3}]
        )
        mock_vector_store.search.return_value = search_results

        result = tool.execute(
            query="authentication",
            course_name="MCP Course",
            lesson_number=3
        )

        # Verify both filters were passed
        mock_vector_store.search.assert_called_once_with(
            query="authentication",
            course_name="MCP Course",
            lesson_number=3
        )
        assert "Specific lesson content" in result

    def test_execute_no_results_with_course_filter(self, mock_vector_store, create_empty_search_results):
        """Test meaningful message when no results found with course filter"""
        tool = CourseSearchTool(mock_vector_store)

        mock_vector_store.search.return_value = create_empty_search_results()

        result = tool.execute(query="nonexistent", course_name="Some Course")

        # Should include course name in message
        assert "No relevant content found in course 'Some Course'" in result

    def test_execute_no_results_with_lesson_filter(self, mock_vector_store, create_empty_search_results):
        """Test meaningful message when no results found with lesson filter"""
        tool = CourseSearchTool(mock_vector_store)

        mock_vector_store.search.return_value = create_empty_search_results()

        result = tool.execute(query="test", lesson_number=99)

        # Should include lesson number in message
        assert "No relevant content found in lesson 99" in result

    def test_execute_multiple_results_source_tracking(self, mock_vector_store, create_search_results):
        """Test that multiple results are tracked correctly in sources"""
        tool = CourseSearchTool(mock_vector_store)

        search_results = create_search_results(
            documents=["Content 1", "Content 2", "Content 3"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course A", "lesson_number": 2},
                {"course_title": "Course B", "lesson_number": 1}
            ]
        )
        mock_vector_store.search.return_value = search_results

        result = tool.execute(query="test")

        # Should have 3 sources
        assert len(tool.last_sources) == 3
        assert tool.last_sources[0]["text"] == "Course A - Lesson 1"
        assert tool.last_sources[1]["text"] == "Course A - Lesson 2"
        assert tool.last_sources[2]["text"] == "Course B - Lesson 1"

    def test_execute_with_lesson_links(self, mock_vector_store, create_search_results):
        """Test that lesson links are properly included in sources"""
        tool = CourseSearchTool(mock_vector_store)

        search_results = create_search_results(
            documents=["Content with link"],
            metadata=[{"course_title": "MCP Course", "lesson_number": 1}]
        )
        mock_vector_store.search.return_value = search_results

        # Mock get_lesson_link to return a link
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"

        result = tool.execute(query="test")

        # Verify lesson link is in sources
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["link"] == "https://example.com/lesson1"
        mock_vector_store.get_lesson_link.assert_called_once_with("MCP Course", 1)

    def test_execute_without_lesson_links(self, mock_vector_store, create_search_results):
        """Test handling when lesson links are not available"""
        tool = CourseSearchTool(mock_vector_store)

        search_results = create_search_results(
            documents=["Content without link"],
            metadata=[{"course_title": "MCP Course", "lesson_number": 1}]
        )
        mock_vector_store.search.return_value = search_results

        # Mock get_lesson_link to return None
        mock_vector_store.get_lesson_link.return_value = None

        result = tool.execute(query="test")

        # Source should have None for link
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["link"] is None

    def test_execute_with_missing_metadata(self, mock_vector_store):
        """Test execute handles missing or malformed metadata gracefully"""
        tool = CourseSearchTool(mock_vector_store)

        # Create results with incomplete metadata
        search_results = SearchResults(
            documents=["Some content"],
            metadata=[{}],  # Empty metadata
            distances=[0.5],
            error=None
        )
        mock_vector_store.search.return_value = search_results

        result = tool.execute(query="test")

        # Should handle gracefully with 'unknown' course
        assert "[unknown]" in result
        assert "Some content" in result


@pytest.mark.unit
class TestCourseSearchToolFormatResults:
    """Tests for the internal _format_results method"""

    def test_format_results_with_no_lesson_number(self, mock_vector_store, create_search_results):
        """Test formatting when lesson_number is None"""
        tool = CourseSearchTool(mock_vector_store)

        search_results = create_search_results(
            documents=["Course overview"],
            metadata=[{"course_title": "MCP Course", "lesson_number": None}]
        )
        mock_vector_store.search.return_value = search_results

        result = tool.execute(query="test")

        # Should not include lesson number in header
        assert "[MCP Course]" in result
        assert "Lesson" not in result.split('\n')[0]  # First line is header


@pytest.mark.integration
class TestCourseSearchToolWithRealConfig:
    """Integration tests that check actual configuration issues"""

    def test_detect_zero_max_results_configuration(self):
        """
        CRITICAL TEST: Detect when MAX_RESULTS is set to 0
        This test should FAIL if MAX_RESULTS=0 in config.py
        """
        from config import Config

        # This assertion should fail with current config
        assert Config.MAX_RESULTS > 0, (
            f"MAX_RESULTS is set to {Config.MAX_RESULTS}. "
            "This will cause all searches to return 0 results! "
            "Change to at least 5 in backend/config.py"
        )

    def test_max_results_reasonable_value(self):
        """Test that MAX_RESULTS is set to a reasonable value"""
        from config import Config

        assert Config.MAX_RESULTS >= 3, (
            f"MAX_RESULTS={Config.MAX_RESULTS} is too low. "
            "Recommended: 5-10 for good search quality"
        )
        assert Config.MAX_RESULTS <= 20, (
            f"MAX_RESULTS={Config.MAX_RESULTS} is very high. "
            "This may cause token limit issues. Recommended: 5-10"
        )


@pytest.mark.unit
class TestToolManagerSourceTracking:
    """Tests for ToolManager's source tracking functionality"""

    def test_get_last_sources_from_search_tool(self, mock_vector_store, create_search_results):
        """Test that ToolManager can retrieve sources from CourseSearchTool"""
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(search_tool)

        # Setup search results
        search_results = create_search_results(
            documents=["Test content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}]
        )
        mock_vector_store.search.return_value = search_results

        # Execute tool
        manager.execute_tool("search_course_content", query="test")

        # Get sources
        sources = manager.get_last_sources()

        assert len(sources) == 1
        assert sources[0]["text"] == "Test Course - Lesson 1"

    def test_reset_sources_clears_all_tools(self, mock_vector_store, create_search_results):
        """Test that reset_sources clears sources from all tools"""
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(search_tool)

        # Execute search to populate sources
        search_results = create_search_results(
            documents=["Test"],
            metadata=[{"course_title": "Course", "lesson_number": 1}]
        )
        mock_vector_store.search.return_value = search_results
        manager.execute_tool("search_course_content", query="test")

        # Verify sources exist
        assert len(manager.get_last_sources()) > 0

        # Reset
        manager.reset_sources()

        # Sources should be empty
        assert manager.get_last_sources() == []
        assert search_tool.last_sources == []
