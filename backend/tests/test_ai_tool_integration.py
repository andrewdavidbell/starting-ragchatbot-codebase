"""
Tests for AI Generator's integration with CourseSearchTool
Verifies that ai_generator.py correctly calls and uses the CourseSearchTool
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
from ai_generator import AIGenerator, ConversationState
from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


@pytest.mark.unit
class TestAIGeneratorToolCalling:
    """Test that AIGenerator correctly invokes CourseSearchTool"""

    def test_ai_generator_calls_search_tool_with_correct_params(
        self, mock_anthropic_client, mock_vector_store
    ):
        """Test that AI generator passes correct parameters to search tool"""
        # Setup
        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
        generator.client = mock_anthropic_client

        # Setup tool manager with search tool
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        tool_manager.register_tool(search_tool)

        # Mock AI response with tool use
        mock_response = MagicMock()
        mock_response.stop_reason = "tool_use"
        mock_response.content = [
            MagicMock(
                type="tool_use",
                id="tool_123",
                name="search_course_content",
                input={"query": "MCP servers", "course_name": "Introduction to MCP"},
            )
        ]
        mock_anthropic_client.messages.create.return_value = mock_response

        # Mock vector store to return results
        search_results = SearchResults(
            documents=["MCP server content"],
            metadata=[{"course_title": "Introduction to MCP", "lesson_number": 1}],
            distances=[0.5],
            error=None,
        )
        mock_vector_store.search.return_value = search_results

        # Second call returns text response
        text_response = MagicMock()
        text_response.stop_reason = "end_turn"
        text_response.content = [
            MagicMock(type="text", text="Here's info about MCP servers")
        ]
        mock_anthropic_client.messages.create.side_effect = [
            mock_response,
            text_response,
        ]

        # Execute
        tools = tool_manager.get_tool_definitions()
        response, sources = generator.generate_response(
            query="Tell me about MCP servers", tools=tools, tool_manager=tool_manager
        )

        # Verify search was called with correct params
        mock_vector_store.search.assert_called_once_with(
            query="MCP servers", course_name="Introduction to MCP", lesson_number=None
        )

    def test_ai_generator_accumulates_sources_across_rounds(
        self, mock_anthropic_client, mock_vector_store
    ):
        """Test that sources are accumulated across multiple tool calling rounds"""
        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
        generator.client = mock_anthropic_client

        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        tool_manager.register_tool(search_tool)

        # Mock first tool call
        first_tool_response = MagicMock()
        first_tool_response.stop_reason = "tool_use"
        first_tool_response.content = [
            MagicMock(
                type="tool_use",
                id="tool_1",
                name="search_course_content",
                input={"query": "servers"},
            )
        ]

        # Mock second tool call
        second_tool_response = MagicMock()
        second_tool_response.stop_reason = "tool_use"
        second_tool_response.content = [
            MagicMock(
                type="tool_use",
                id="tool_2",
                name="search_course_content",
                input={"query": "authentication"},
            )
        ]

        # Final text response
        final_response = MagicMock()
        final_response.stop_reason = "end_turn"
        final_response.content = [MagicMock(type="text", text="Combined response")]

        mock_anthropic_client.messages.create.side_effect = [
            first_tool_response,
            second_tool_response,
            final_response,
        ]

        # Mock search results for both calls
        def search_side_effect(query, course_name=None, lesson_number=None):
            if query == "servers":
                return SearchResults(
                    documents=["Server info"],
                    metadata=[{"course_title": "Course A", "lesson_number": 1}],
                    distances=[0.5],
                    error=None,
                )
            else:  # authentication
                return SearchResults(
                    documents=["Auth info"],
                    metadata=[{"course_title": "Course B", "lesson_number": 2}],
                    distances=[0.4],
                    error=None,
                )

        mock_vector_store.search.side_effect = search_side_effect

        # Execute
        tools = tool_manager.get_tool_definitions()
        response, sources = generator.generate_response(
            query="Test query", tools=tools, tool_manager=tool_manager
        )

        # Should have sources from both rounds
        assert len(sources) == 2
        assert sources[0]["text"] == "Course A - Lesson 1"
        assert sources[1]["text"] == "Course B - Lesson 2"

    def test_ai_generator_handles_tool_error_response(
        self, mock_anthropic_client, mock_vector_store
    ):
        """Test that AI generator handles errors returned by tools"""
        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
        generator.client = mock_anthropic_client

        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        tool_manager.register_tool(search_tool)

        # Mock AI requesting tool use
        tool_use_response = MagicMock()
        tool_use_response.stop_reason = "tool_use"
        tool_use_response.content = [
            MagicMock(
                type="tool_use",
                id="tool_1",
                name="search_course_content",
                input={"query": "test"},
            )
        ]

        # Final response after seeing error
        final_response = MagicMock()
        final_response.stop_reason = "end_turn"
        final_response.content = [
            MagicMock(type="text", text="I couldn't find that information")
        ]

        mock_anthropic_client.messages.create.side_effect = [
            tool_use_response,
            final_response,
        ]

        # Mock vector store returning error
        error_result = SearchResults(
            documents=[], metadata=[], distances=[], error="ChromaDB connection failed"
        )
        mock_vector_store.search.return_value = error_result

        # Execute
        tools = tool_manager.get_tool_definitions()
        response, sources = generator.generate_response(
            query="Test query", tools=tools, tool_manager=tool_manager
        )

        # Should still get a response (AI handles the error)
        assert response is not None
        assert sources == []  # No sources since there was an error

    def test_ai_generator_respects_max_rounds_limit(
        self, mock_anthropic_client, mock_vector_store
    ):
        """Test that AI generator stops after MAX_ROUNDS tool calls"""
        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
        generator.client = mock_anthropic_client

        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        tool_manager.register_tool(search_tool)

        # Mock tool use response
        tool_use_response = MagicMock()
        tool_use_response.stop_reason = "tool_use"
        tool_use_response.content = [
            MagicMock(
                type="tool_use",
                id=f"tool_{i}",
                name="search_course_content",
                input={"query": f"query_{i}"},
            )
        ]

        # Final synthesis response
        final_response = MagicMock()
        final_response.stop_reason = "end_turn"
        final_response.content = [
            MagicMock(type="text", text="Final synthesized response")
        ]

        # Return tool use for first 2 calls, then final response
        mock_anthropic_client.messages.create.side_effect = [
            tool_use_response,  # Round 1
            tool_use_response,  # Round 2 (max)
            final_response,  # Final synthesis
        ]

        # Mock search results
        mock_vector_store.search.return_value = SearchResults(
            documents=["Content"],
            metadata=[{"course_title": "Course", "lesson_number": 1}],
            distances=[0.5],
            error=None,
        )

        # Execute
        tools = tool_manager.get_tool_definitions()
        response, sources = generator.generate_response(
            query="Test query", tools=tools, tool_manager=tool_manager
        )

        # Should have made exactly 3 API calls (2 tool rounds + 1 final synthesis)
        assert mock_anthropic_client.messages.create.call_count == 3
        assert response == "Final synthesized response"

    def test_ai_generator_without_tools(self, mock_anthropic_client):
        """Test that AI generator works without tools (direct response)"""
        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
        generator.client = mock_anthropic_client

        # Mock direct text response (no tools)
        text_response = MagicMock()
        text_response.stop_reason = "end_turn"
        text_response.content = [
            MagicMock(type="text", text="Direct answer without searching")
        ]

        mock_anthropic_client.messages.create.return_value = text_response

        # Execute without tools
        response, sources = generator.generate_response(
            query="What is Python?", tools=None, tool_manager=None
        )

        # Should get response without tool calling
        assert response == "Direct answer without searching"
        assert sources == []
        assert mock_anthropic_client.messages.create.call_count == 1


@pytest.mark.unit
class TestConversationState:
    """Test the ConversationState dataclass used in multi-round conversations"""

    def test_conversation_state_initialization(self):
        """Test ConversationState initializes correctly"""
        state = ConversationState(
            initial_query="test query",
            system_content="system prompt",
            tools=[{"name": "test_tool"}],
            tool_manager=Mock(),
            max_rounds=2,
            messages=[{"role": "user", "content": "test"}],
        )

        assert state.current_round == 0
        assert state.accumulated_sources == []
        assert len(state.messages) == 1

    def test_add_assistant_message(self):
        """Test adding assistant messages to conversation state"""
        state = ConversationState(
            initial_query="test",
            system_content="system",
            tools=None,
            tool_manager=None,
            max_rounds=2,
            messages=[],
        )

        state.add_assistant_message("assistant response")

        assert len(state.messages) == 1
        assert state.messages[0]["role"] == "assistant"
        assert state.messages[0]["content"] == "assistant response"

    def test_add_tool_results(self):
        """Test adding tool results to conversation state"""
        state = ConversationState(
            initial_query="test",
            system_content="system",
            tools=None,
            tool_manager=None,
            max_rounds=2,
            messages=[],
        )

        tool_results = [
            {"type": "tool_result", "tool_use_id": "123", "content": "result"}
        ]
        state.add_tool_results(tool_results)

        assert len(state.messages) == 1
        assert state.messages[0]["role"] == "user"
        assert state.messages[0]["content"] == tool_results

    def test_increment_round(self):
        """Test incrementing round counter"""
        state = ConversationState(
            initial_query="test",
            system_content="system",
            tools=None,
            tool_manager=None,
            max_rounds=2,
            messages=[],
        )

        assert state.current_round == 0
        state.increment_round()
        assert state.current_round == 1
        state.increment_round()
        assert state.current_round == 2

    def test_can_use_tools(self):
        """Test can_use_tools checks all requirements"""
        tool_manager = Mock()
        tools = [{"name": "test"}]

        # All conditions met
        state1 = ConversationState(
            initial_query="test",
            system_content="system",
            tools=tools,
            tool_manager=tool_manager,
            max_rounds=2,
            messages=[],
        )
        assert state1.can_use_tools() is True

        # No tools
        state2 = ConversationState(
            initial_query="test",
            system_content="system",
            tools=None,
            tool_manager=tool_manager,
            max_rounds=2,
            messages=[],
        )
        assert state2.can_use_tools() is False

        # No tool manager
        state3 = ConversationState(
            initial_query="test",
            system_content="system",
            tools=tools,
            tool_manager=None,
            max_rounds=2,
            messages=[],
        )
        assert state3.can_use_tools() is False

        # At max rounds
        state4 = ConversationState(
            initial_query="test",
            system_content="system",
            tools=tools,
            tool_manager=tool_manager,
            max_rounds=2,
            messages=[],
            current_round=2,
        )
        assert state4.can_use_tools() is False


@pytest.mark.integration
class TestAIGeneratorToolIntegration:
    """Integration tests with real tool definitions"""

    def test_tool_definitions_format(self):
        """Test that CourseSearchTool provides correct tool definition format"""
        mock_store = Mock()
        tool = CourseSearchTool(mock_store)
        tool_def = tool.get_tool_definition()

        # Verify structure matches Anthropic's requirements
        assert "name" in tool_def
        assert tool_def["name"] == "search_course_content"
        assert "description" in tool_def
        assert "input_schema" in tool_def
        assert tool_def["input_schema"]["type"] == "object"
        assert "properties" in tool_def["input_schema"]
        assert "query" in tool_def["input_schema"]["properties"]
        assert "required" in tool_def["input_schema"]
        assert "query" in tool_def["input_schema"]["required"]

    def test_tool_manager_provides_correct_format_to_ai(self, mock_vector_store):
        """Test that ToolManager provides tools in correct format for AI"""
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(search_tool)

        tool_defs = manager.get_tool_definitions()

        # Should return list of tool definitions
        assert isinstance(tool_defs, list)
        assert len(tool_defs) == 1
        assert tool_defs[0]["name"] == "search_course_content"
