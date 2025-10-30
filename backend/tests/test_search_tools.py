import pytest
from unittest.mock import Mock, patch
from typing import List

from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager, Tool
from vector_store import SearchResults
from .conftest import create_search_results, create_empty_search_results


class TestCourseSearchTool:
    """Unit tests for CourseSearchTool class"""

    @pytest.mark.unit
    def test_get_tool_definition(self, search_tool):
        """Test that tool definition is correctly formatted"""
        definition = search_tool.get_tool_definition()
        
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        
        schema = definition["input_schema"]
        assert schema["type"] == "object"
        assert "query" in schema["properties"]
        assert schema["required"] == ["query"]
        
        # Check optional parameters are present
        assert "course_name" in schema["properties"] 
        assert "lesson_number" in schema["properties"]

    @pytest.mark.unit
    def test_execute_basic_search(self, mock_vector_store):
        """Test basic search execution without filters"""
        # Setup
        mock_vector_store.search.return_value = create_search_results(
            documents=["This is test content about Python"],
            course_title="Python Basics",
            lesson_numbers=[1]
        )
        
        search_tool = CourseSearchTool(mock_vector_store)
        
        # Execute
        result = search_tool.execute("Python basics")
        
        # Verify
        mock_vector_store.search.assert_called_once_with(
            query="Python basics",
            course_name=None,
            lesson_number=None
        )
        
        assert "[Python Basics - Lesson 1]" in result
        assert "This is test content about Python" in result
        assert len(search_tool.last_sources) == 1
        assert search_tool.last_sources[0]["text"] == "Python Basics - Lesson 1"

    @pytest.mark.unit
    def test_execute_with_course_filter(self, mock_vector_store):
        """Test search with course name filter"""
        mock_vector_store.search.return_value = create_search_results(
            documents=["Advanced Python concepts"],
            course_title="Python Advanced",
            lesson_numbers=[3]
        )
        
        search_tool = CourseSearchTool(mock_vector_store)
        
        result = search_tool.execute("concepts", course_name="Python Advanced")
        
        mock_vector_store.search.assert_called_once_with(
            query="concepts",
            course_name="Python Advanced",
            lesson_number=None
        )
        
        assert "[Python Advanced - Lesson 3]" in result
        assert "Advanced Python concepts" in result

    @pytest.mark.unit
    def test_execute_with_lesson_filter(self, mock_vector_store):
        """Test search with lesson number filter"""
        mock_vector_store.search.return_value = create_search_results(
            documents=["Lesson 2 specific content"],
            course_title="Test Course",
            lesson_numbers=[2]
        )
        
        search_tool = CourseSearchTool(mock_vector_store)
        
        result = search_tool.execute("specific content", lesson_number=2)
        
        mock_vector_store.search.assert_called_once_with(
            query="specific content",
            course_name=None,
            lesson_number=2
        )
        
        assert "[Test Course - Lesson 2]" in result

    @pytest.mark.unit
    def test_execute_with_both_filters(self, mock_vector_store):
        """Test search with both course and lesson filters"""
        mock_vector_store.search.return_value = create_search_results(
            documents=["Specific lesson content"],
            course_title="Specific Course",
            lesson_numbers=[4]
        )
        
        search_tool = CourseSearchTool(mock_vector_store)
        
        result = search_tool.execute("content", course_name="Specific Course", lesson_number=4)
        
        mock_vector_store.search.assert_called_once_with(
            query="content",
            course_name="Specific Course", 
            lesson_number=4
        )

    @pytest.mark.unit
    def test_execute_empty_results(self, mock_vector_store):
        """Test handling of empty search results"""
        mock_vector_store.search.return_value = create_empty_search_results()
        
        search_tool = CourseSearchTool(mock_vector_store)
        
        result = search_tool.execute("nonexistent content")
        
        assert "No relevant content found" in result
        assert len(search_tool.last_sources) == 0

    @pytest.mark.unit
    def test_execute_empty_results_with_filters(self, mock_vector_store):
        """Test empty results message includes filter information"""
        mock_vector_store.search.return_value = create_empty_search_results()
        
        search_tool = CourseSearchTool(mock_vector_store)
        
        result = search_tool.execute("content", course_name="Missing Course", lesson_number=5)
        
        assert "No relevant content found in course 'Missing Course' in lesson 5" in result

    @pytest.mark.unit
    def test_execute_search_error(self, mock_vector_store):
        """Test handling of search errors"""
        mock_vector_store.search.return_value = create_empty_search_results("Database connection failed")
        
        search_tool = CourseSearchTool(mock_vector_store)
        
        result = search_tool.execute("any query")
        
        assert result == "Database connection failed"
        assert len(search_tool.last_sources) == 0

    @pytest.mark.unit
    def test_source_tracking_with_links(self, mock_vector_store):
        """Test that sources are tracked with lesson links when available"""
        mock_vector_store.search.return_value = create_search_results(
            documents=["Content with link"],
            course_title="Linked Course",
            lesson_numbers=[1]
        )
        mock_vector_store.get_lesson_link.return_value = "http://example.com/lesson1"
        
        search_tool = CourseSearchTool(mock_vector_store)
        
        result = search_tool.execute("content")
        
        assert len(search_tool.last_sources) == 1
        source = search_tool.last_sources[0]
        assert source["text"] == "Linked Course - Lesson 1"
        assert source["link"] == "http://example.com/lesson1"
        
        # Verify get_lesson_link was called
        mock_vector_store.get_lesson_link.assert_called_once_with("Linked Course", 1)

    @pytest.mark.unit
    def test_source_tracking_without_links(self, mock_vector_store):
        """Test source tracking when no lesson links are available"""
        mock_vector_store.search.return_value = create_search_results(
            documents=["Content without link"],
            course_title="Unlinked Course",
            lesson_numbers=[2]
        )
        mock_vector_store.get_lesson_link.return_value = None
        
        search_tool = CourseSearchTool(mock_vector_store)
        
        result = search_tool.execute("content")
        
        assert len(search_tool.last_sources) == 1
        source = search_tool.last_sources[0]
        assert source["text"] == "Unlinked Course - Lesson 2"
        assert source["link"] is None

    @pytest.mark.unit
    def test_multiple_results_formatting(self, mock_vector_store):
        """Test formatting of multiple search results"""
        mock_vector_store.search.return_value = create_search_results(
            documents=[
                "First result content",
                "Second result content", 
                "Third result content"
            ],
            course_title="Multi Course",
            lesson_numbers=[1, 2, 3]
        )
        
        search_tool = CourseSearchTool(mock_vector_store)
        
        result = search_tool.execute("content")
        
        # Check all results are included
        assert "[Multi Course - Lesson 1]" in result
        assert "[Multi Course - Lesson 2]" in result 
        assert "[Multi Course - Lesson 3]" in result
        assert "First result content" in result
        assert "Second result content" in result
        assert "Third result content" in result
        
        # Check sources tracking
        assert len(search_tool.last_sources) == 3

    @pytest.mark.unit
    def test_missing_metadata_handling(self, mock_vector_store):
        """Test handling of missing or malformed metadata"""
        # Create results with missing metadata
        results = SearchResults(
            documents=["Content with missing metadata"],
            metadata=[{}],  # Empty metadata
            distances=[0.1]
        )
        mock_vector_store.search.return_value = results
        
        search_tool = CourseSearchTool(mock_vector_store)
        
        result = search_tool.execute("content")
        
        # Should handle missing metadata gracefully
        assert "[unknown]" in result
        assert "Content with missing metadata" in result


class TestCourseOutlineTool:
    """Unit tests for CourseOutlineTool class"""

    @pytest.mark.unit
    def test_get_tool_definition(self, mock_vector_store):
        """Test that tool definition is correctly formatted"""
        outline_tool = CourseOutlineTool(mock_vector_store)
        definition = outline_tool.get_tool_definition()

        assert definition["name"] == "get_course_outline"
        assert "description" in definition
        assert "input_schema" in definition

        schema = definition["input_schema"]
        assert schema["type"] == "object"
        assert "course_name" in schema["properties"]
        assert schema["required"] == ["course_name"]

    @pytest.mark.unit
    def test_execute_successful_outline_retrieval(self, mock_vector_store):
        """Test successful retrieval of course outline"""
        # Setup mock responses
        mock_vector_store._resolve_course_name.return_value = "Introduction to MCP Servers"
        mock_vector_store.get_all_courses_metadata.return_value = [
            {
                "title": "Introduction to MCP Servers",
                "course_link": "http://example.com/mcp-course",
                "instructor": "John Doe",
                "lessons": [
                    {"lesson_number": 0, "lesson_title": "Introduction", "lesson_link": "http://example.com/lesson0"},
                    {"lesson_number": 1, "lesson_title": "Getting Started", "lesson_link": "http://example.com/lesson1"},
                    {"lesson_number": 2, "lesson_title": "Advanced Features", "lesson_link": None}
                ]
            }
        ]

        outline_tool = CourseOutlineTool(mock_vector_store)
        result = outline_tool.execute("MCP")

        # Verify the result format
        assert "Course: Introduction to MCP Servers" in result
        assert "Course Link: http://example.com/mcp-course" in result
        assert "Instructor: John Doe" in result
        assert "Lessons:" in result
        assert "0. Introduction - http://example.com/lesson0" in result
        assert "1. Getting Started - http://example.com/lesson1" in result
        assert "2. Advanced Features" in result

        # Verify fuzzy matching was used
        mock_vector_store._resolve_course_name.assert_called_once_with("MCP")

    @pytest.mark.unit
    def test_execute_with_fuzzy_course_name(self, mock_vector_store):
        """Test that fuzzy course name matching works correctly"""
        mock_vector_store._resolve_course_name.return_value = "Python Programming Basics"
        mock_vector_store.get_all_courses_metadata.return_value = [
            {
                "title": "Python Programming Basics",
                "course_link": "http://example.com/python",
                "instructor": "Jane Smith",
                "lessons": [
                    {"lesson_number": 1, "lesson_title": "Variables", "lesson_link": None}
                ]
            }
        ]

        outline_tool = CourseOutlineTool(mock_vector_store)
        result = outline_tool.execute("Python")

        assert "Course: Python Programming Basics" in result
        mock_vector_store._resolve_course_name.assert_called_once_with("Python")

    @pytest.mark.unit
    def test_execute_course_not_found(self, mock_vector_store):
        """Test handling when course name cannot be resolved"""
        mock_vector_store._resolve_course_name.return_value = None

        outline_tool = CourseOutlineTool(mock_vector_store)
        result = outline_tool.execute("Nonexistent Course")

        assert "Course 'Nonexistent Course' not found" in result
        assert "Please check the course name" in result

    @pytest.mark.unit
    def test_execute_course_not_in_metadata(self, mock_vector_store):
        """Test handling when resolved course is not in metadata"""
        mock_vector_store._resolve_course_name.return_value = "Test Course"
        mock_vector_store.get_all_courses_metadata.return_value = [
            {
                "title": "Different Course",
                "course_link": "http://example.com/different",
                "instructor": "Someone",
                "lessons": []
            }
        ]

        outline_tool = CourseOutlineTool(mock_vector_store)
        result = outline_tool.execute("Test")

        assert "Course 'Test' not found in the catalogue" in result

    @pytest.mark.unit
    def test_execute_no_lessons(self, mock_vector_store):
        """Test handling of course with no lessons"""
        mock_vector_store._resolve_course_name.return_value = "Empty Course"
        mock_vector_store.get_all_courses_metadata.return_value = [
            {
                "title": "Empty Course",
                "course_link": "http://example.com/empty",
                "instructor": "Test Instructor",
                "lessons": []
            }
        ]

        outline_tool = CourseOutlineTool(mock_vector_store)
        result = outline_tool.execute("Empty Course")

        assert "Course: Empty Course" in result
        assert "No lessons found for this course" in result

    @pytest.mark.unit
    def test_execute_missing_optional_fields(self, mock_vector_store):
        """Test handling when optional fields are missing"""
        mock_vector_store._resolve_course_name.return_value = "Minimal Course"
        mock_vector_store.get_all_courses_metadata.return_value = [
            {
                "title": "Minimal Course",
                "course_link": None,  # No course link
                "instructor": None,   # No instructor
                "lessons": [
                    {"lesson_number": 1, "lesson_title": "Lesson One", "lesson_link": None}
                ]
            }
        ]

        outline_tool = CourseOutlineTool(mock_vector_store)
        result = outline_tool.execute("Minimal")

        assert "Course: Minimal Course" in result
        # Should show course link as not available when None
        assert "Course Link: Not available" in result
        # Should not include instructor line if None
        assert "Instructor:" not in result
        # Should still include lessons
        assert "1. Lesson One" in result
        # No actual HTTP links should appear
        assert result.count("http://") == 0

    @pytest.mark.unit
    def test_format_with_all_fields(self, mock_vector_store):
        """Test output format includes all available fields"""
        mock_vector_store._resolve_course_name.return_value = "Complete Course"
        mock_vector_store.get_all_courses_metadata.return_value = [
            {
                "title": "Complete Course",
                "course_link": "http://example.com/complete",
                "instructor": "Dr. Complete",
                "lessons": [
                    {"lesson_number": 1, "lesson_title": "First Lesson", "lesson_link": "http://example.com/l1"},
                    {"lesson_number": 2, "lesson_title": "Second Lesson", "lesson_link": "http://example.com/l2"}
                ]
            }
        ]

        outline_tool = CourseOutlineTool(mock_vector_store)
        result = outline_tool.execute("Complete")

        lines = result.split("\n")
        assert "Course: Complete Course" in lines[0]
        assert "Course Link: http://example.com/complete" in lines[1]
        assert "Instructor: Dr. Complete" in lines[2]
        # Check blank line exists
        assert lines[3] == ""
        assert "Lessons:" in lines[4]
        assert "1. First Lesson - http://example.com/l1" in lines[5]
        assert "2. Second Lesson - http://example.com/l2" in lines[6]


class TestToolManager:
    """Unit tests for ToolManager class"""

    @pytest.mark.unit
    def test_tool_registration(self):
        """Test registering tools with ToolManager"""
        manager = ToolManager()
        mock_tool = Mock(spec=Tool)
        mock_tool.get_tool_definition.return_value = {"name": "test_tool", "description": "Test"}
        
        manager.register_tool(mock_tool)
        
        assert "test_tool" in manager.tools
        assert manager.tools["test_tool"] == mock_tool

    @pytest.mark.unit
    def test_tool_registration_missing_name(self):
        """Test that tools without names raise appropriate errors"""
        manager = ToolManager()
        mock_tool = Mock(spec=Tool)
        mock_tool.get_tool_definition.return_value = {"description": "Test"}  # Missing name
        
        with pytest.raises(ValueError, match="Tool must have a 'name'"):
            manager.register_tool(mock_tool)

    @pytest.mark.unit  
    def test_get_tool_definitions(self, tool_manager):
        """Test getting all tool definitions"""
        definitions = tool_manager.get_tool_definitions()
        
        assert isinstance(definitions, list)
        assert len(definitions) == 1  # Only search tool registered in fixture
        assert definitions[0]["name"] == "search_course_content"

    @pytest.mark.unit
    def test_execute_tool_success(self, tool_manager):
        """Test successful tool execution"""
        # The search tool in fixture is already mocked
        result = tool_manager.execute_tool("search_course_content", query="test query")
        
        # Should return formatted search results from mock
        assert isinstance(result, str)
        assert "Test document content" in result

    @pytest.mark.unit
    def test_execute_nonexistent_tool(self, tool_manager):
        """Test execution of non-existent tool"""
        result = tool_manager.execute_tool("nonexistent_tool", query="test")
        
        assert result == "Tool 'nonexistent_tool' not found"

    @pytest.mark.unit
    def test_get_last_sources(self, tool_manager, search_tool):
        """Test retrieving sources from last search"""
        # Execute a search to populate sources
        search_tool.execute("test query")
        
        sources = tool_manager.get_last_sources()
        
        assert isinstance(sources, list)
        assert len(sources) > 0
        assert sources[0]["text"] == "Test Course - Lesson 1"

    @pytest.mark.unit
    def test_get_last_sources_empty(self):
        """Test getting sources when no searches have been performed"""
        manager = ToolManager()
        mock_tool = Mock(spec=Tool)
        mock_tool.get_tool_definition.return_value = {"name": "test_tool"}
        manager.register_tool(mock_tool)
        
        sources = manager.get_last_sources()
        
        assert sources == []

    @pytest.mark.unit
    def test_reset_sources(self, tool_manager, search_tool):
        """Test resetting sources after search"""
        # Execute search and verify sources exist
        search_tool.execute("test query")
        assert len(tool_manager.get_last_sources()) > 0
        
        # Reset and verify sources are cleared
        tool_manager.reset_sources()
        assert tool_manager.get_last_sources() == []