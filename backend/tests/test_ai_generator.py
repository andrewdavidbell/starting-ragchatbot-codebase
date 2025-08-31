import pytest
from unittest.mock import Mock, patch, call
import anthropic

from ai_generator import AIGenerator
from search_tools import ToolManager


class TestAIGenerator:
    """Unit tests for AIGenerator class"""

    @pytest.mark.unit
    def test_initialization(self, test_config):
        """Test AIGenerator initialization"""
        ai_gen = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
        
        assert ai_gen.model == test_config.ANTHROPIC_MODEL
        assert ai_gen.base_params["model"] == test_config.ANTHROPIC_MODEL
        assert ai_gen.base_params["temperature"] == 0
        assert ai_gen.base_params["max_tokens"] == 800

    @pytest.mark.unit 
    def test_generate_response_without_tools(self, mock_anthropic_client, test_config):
        """Test basic response generation without tool calling"""
        ai_gen = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
        
        response = ai_gen.generate_response("What is Python?")
        
        # Verify Anthropic API was called correctly
        mock_anthropic_client.messages.create.assert_called_once()
        call_args = mock_anthropic_client.messages.create.call_args[1]
        
        assert call_args["model"] == test_config.ANTHROPIC_MODEL
        assert call_args["temperature"] == 0
        assert call_args["max_tokens"] == 800
        assert call_args["messages"] == [{"role": "user", "content": "What is Python?"}]
        assert "tools" not in call_args
        
        assert response == "Test AI response"

    @pytest.mark.unit
    def test_generate_response_with_conversation_history(self, mock_anthropic_client, test_config):
        """Test response generation with conversation history"""
        ai_gen = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
        
        history = "User: Previous question\nAssistant: Previous answer"
        response = ai_gen.generate_response("Follow up question", conversation_history=history)
        
        call_args = mock_anthropic_client.messages.create.call_args[1]
        
        # Check system prompt includes history
        assert history in call_args["system"]
        assert ai_gen.SYSTEM_PROMPT in call_args["system"]

    @pytest.mark.unit
    def test_generate_response_with_tools_no_tool_use(self, mock_anthropic_client, test_config, tool_manager):
        """Test response generation with tools available but not used"""
        ai_gen = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
        
        tools = tool_manager.get_tool_definitions()
        response = ai_gen.generate_response("What is 2+2?", tools=tools, tool_manager=tool_manager)
        
        call_args = mock_anthropic_client.messages.create.call_args[1]
        
        # Verify tools were provided
        assert "tools" in call_args
        assert call_args["tools"] == tools
        assert call_args["tool_choice"] == {"type": "auto"}
        
        assert response == "Test AI response"

    @pytest.mark.unit
    def test_generate_response_with_tool_use(self, mock_anthropic_with_tools, test_config, tool_manager):
        """Test response generation when AI uses tools"""
        ai_gen = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
        
        tools = tool_manager.get_tool_definitions()
        response = ai_gen.generate_response("Search for Python content", tools=tools, tool_manager=tool_manager)
        
        # Verify two API calls were made (initial + follow-up after tool execution)
        assert mock_anthropic_with_tools.messages.create.call_count == 2
        
        # Verify final response
        assert response == "Based on the search results, here is the answer."

    @pytest.mark.unit
    def test_handle_tool_execution(self, mock_anthropic_with_tools, test_config, tool_manager):
        """Test the tool execution handling flow"""
        ai_gen = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
        
        # Create mock tool response
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.input = {"query": "Python basics"}
        mock_tool_block.id = "tool_call_456"
        
        mock_initial_response = Mock()
        mock_initial_response.content = [mock_tool_block]
        mock_initial_response.stop_reason = "tool_use"
        
        base_params = {
            "messages": [{"role": "user", "content": "Find Python content"}],
            "system": ai_gen.SYSTEM_PROMPT,
            "model": test_config.ANTHROPIC_MODEL,
            "temperature": 0,
            "max_tokens": 800
        }
        
        result = ai_gen._handle_tool_execution(mock_initial_response, base_params, tool_manager)
        
        # Verify tool manager was called correctly
        # Note: The search tool in fixture is mocked and will return predefined results
        
        # Verify second API call was made with tool results
        assert mock_anthropic_with_tools.messages.create.call_count == 1
        call_args = mock_anthropic_with_tools.messages.create.call_args[1]
        
        # Check that messages include the assistant response and tool results
        messages = call_args["messages"]
        assert len(messages) == 3  # Original user message + assistant tool use + tool results
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        
        # Check tool results structure
        tool_results_content = messages[2]["content"]
        assert isinstance(tool_results_content, list)
        assert tool_results_content[0]["type"] == "tool_result"
        assert tool_results_content[0]["tool_use_id"] == "tool_call_456"

    @pytest.mark.unit
    def test_tool_execution_multiple_tools(self, test_config):
        """Test handling multiple tool calls in one response"""
        with patch('anthropic.Anthropic') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            
            # Create mock with multiple tool calls
            mock_tool_block_1 = Mock()
            mock_tool_block_1.type = "tool_use"
            mock_tool_block_1.name = "search_course_content"
            mock_tool_block_1.input = {"query": "Python"}
            mock_tool_block_1.id = "tool_1"
            
            mock_tool_block_2 = Mock()
            mock_tool_block_2.type = "tool_use"  
            mock_tool_block_2.name = "search_course_content"
            mock_tool_block_2.input = {"query": "JavaScript"}
            mock_tool_block_2.id = "tool_2"
            
            mock_initial_response = Mock()
            mock_initial_response.content = [mock_tool_block_1, mock_tool_block_2]
            mock_initial_response.stop_reason = "tool_use"
            
            mock_final_response = Mock()
            mock_final_response.content = [Mock(text="Multiple tool results processed")]
            
            mock_client.messages.create.return_value = mock_final_response
            
            ai_gen = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
            
            # Create mock tool manager
            mock_tool_manager = Mock()
            mock_tool_manager.execute_tool.side_effect = ["Result 1", "Result 2"]
            
            base_params = {
                "messages": [{"role": "user", "content": "Search for content"}],
                "system": ai_gen.SYSTEM_PROMPT,
                "model": test_config.ANTHROPIC_MODEL,
                "temperature": 0,
                "max_tokens": 800
            }
            
            result = ai_gen._handle_tool_execution(mock_initial_response, base_params, mock_tool_manager)
            
            # Verify both tools were executed
            assert mock_tool_manager.execute_tool.call_count == 2
            mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="Python")
            mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="JavaScript")
            
            # Verify final API call included both tool results
            call_args = mock_client.messages.create.call_args[1]
            tool_results = call_args["messages"][2]["content"]
            assert len(tool_results) == 2
            assert tool_results[0]["tool_use_id"] == "tool_1"
            assert tool_results[1]["tool_use_id"] == "tool_2"

    @pytest.mark.unit
    @patch('anthropic.Anthropic')
    def test_anthropic_api_error_handling(self, mock_client_class, test_config):
        """Test handling of Anthropic API errors"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.messages.create.side_effect = anthropic.APIError("API Error")
        
        ai_gen = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
        
        with pytest.raises(anthropic.APIError):
            ai_gen.generate_response("Test query")

    @pytest.mark.unit
    def test_system_prompt_structure(self, test_config):
        """Test that the system prompt contains expected instructions"""
        ai_gen = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
        
        prompt = ai_gen.SYSTEM_PROMPT
        
        # Check key instructions are present
        assert "search tool" in prompt.lower()
        assert "one search per query maximum" in prompt.lower()
        assert "course-specific questions" in prompt.lower()
        assert "brief, concise and focused" in prompt.lower()

    @pytest.mark.unit
    def test_base_params_configuration(self, test_config):
        """Test that base parameters are correctly configured"""
        ai_gen = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
        
        params = ai_gen.base_params
        
        assert params["model"] == test_config.ANTHROPIC_MODEL
        assert params["temperature"] == 0  # Deterministic responses
        assert params["max_tokens"] == 800
        assert len(params) == 3  # Only these core params should be preset

    @pytest.mark.unit
    def test_tool_choice_configuration(self, mock_anthropic_client, test_config, tool_manager):
        """Test that tool_choice is set to auto when tools are provided"""
        ai_gen = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
        
        tools = tool_manager.get_tool_definitions()
        ai_gen.generate_response("Test query", tools=tools, tool_manager=tool_manager)
        
        call_args = mock_anthropic_client.messages.create.call_args[1]
        
        assert call_args["tool_choice"] == {"type": "auto"}

    @pytest.mark.unit 
    def test_no_tool_manager_with_tools(self, mock_anthropic_with_tools, test_config, tool_manager):
        """Test behavior when tools are provided but no tool_manager"""
        ai_gen = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
        
        tools = tool_manager.get_tool_definitions()
        # Call with tools but without tool_manager - should not attempt tool execution
        response = ai_gen.generate_response("Search query", tools=tools, tool_manager=None)
        
        # Should only make initial call, no tool execution attempt
        assert mock_anthropic_with_tools.messages.create.call_count == 1

    @pytest.mark.requires_api
    def test_real_api_integration(self, test_config):
        """Integration test with real Anthropic API (requires valid API key)"""
        if not test_config.ANTHROPIC_API_KEY or test_config.ANTHROPIC_API_KEY == "test_api_key":
            pytest.skip("No valid API key for integration test")
        
        ai_gen = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
        
        response = ai_gen.generate_response("What is 2+2?")
        
        assert isinstance(response, str)
        assert len(response) > 0
        assert "4" in response  # Should contain the answer