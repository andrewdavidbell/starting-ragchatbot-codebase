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
        
        response, sources = ai_gen.generate_response("What is Python?")
        
        # Verify Anthropic API was called correctly
        mock_anthropic_client.messages.create.assert_called_once()
        call_args = mock_anthropic_client.messages.create.call_args[1]
        
        assert call_args["model"] == test_config.ANTHROPIC_MODEL
        assert call_args["temperature"] == 0
        assert call_args["max_tokens"] == 800
        assert call_args["messages"] == [{"role": "user", "content": "What is Python?"}]
        assert "tools" not in call_args
        
        assert response == "Test AI response"
        assert sources == []

    @pytest.mark.unit
    def test_generate_response_with_conversation_history(self, mock_anthropic_client, test_config):
        """Test response generation with conversation history"""
        ai_gen = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
        
        history = "User: Previous question\nAssistant: Previous answer"
        response, sources = ai_gen.generate_response("Follow up question", conversation_history=history)
        
        call_args = mock_anthropic_client.messages.create.call_args[1]
        
        # Check system prompt includes history
        assert history in call_args["system"]
        assert ai_gen.SYSTEM_PROMPT in call_args["system"]

    @pytest.mark.unit
    def test_generate_response_with_tools_no_tool_use(self, mock_anthropic_client, test_config, tool_manager):
        """Test response generation with tools available but not used"""
        ai_gen = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
        
        tools = tool_manager.get_tool_definitions()
        response, sources = ai_gen.generate_response("What is 2+2?", tools=tools, tool_manager=tool_manager)
        
        call_args = mock_anthropic_client.messages.create.call_args[1]
        
        # Verify tools were provided
        assert "tools" in call_args
        assert call_args["tools"] == tools
        assert call_args["tool_choice"] == {"type": "auto"}
        
        assert response == "Test AI response"
        assert sources == []

    @pytest.mark.unit
    def test_generate_response_with_tool_use(self, mock_anthropic_with_tools, test_config, tool_manager):
        """Test response generation when AI uses tools"""
        ai_gen = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
        
        tools = tool_manager.get_tool_definitions()
        response, sources = ai_gen.generate_response("Search for Python content", tools=tools, tool_manager=tool_manager)
        
        # Verify two API calls were made (initial + follow-up after tool execution)
        assert mock_anthropic_with_tools.messages.create.call_count == 2
        
        # Verify final response
        assert response == "Based on the search results, here is the answer."
        assert isinstance(sources, list)

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
        
        response, sources = ai_gen.generate_response("Test query")
        assert "Anthropic API error" in response
        assert sources == []

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
        response, sources = ai_gen.generate_response("Search query", tools=tools, tool_manager=None)
        
        # Should only make initial call, no tool execution attempt
        assert mock_anthropic_with_tools.messages.create.call_count == 1
        assert isinstance(sources, list)

    @pytest.mark.unit
    def test_sequential_two_round_tool_execution(self, test_config, tool_manager):
        """Test Claude makes two sequential tool calls across rounds"""
        with patch('anthropic.Anthropic') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            # Mock round 0: Claude requests first tool
            mock_tool_block_1 = Mock()
            mock_tool_block_1.type = "tool_use"
            mock_tool_block_1.name = "get_course_outline"
            mock_tool_block_1.input = {"course_name": "Python Basics"}
            mock_tool_block_1.id = "tool_1"

            mock_response_1 = Mock()
            mock_response_1.content = [mock_tool_block_1]
            mock_response_1.stop_reason = "tool_use"

            # Mock round 1: Claude requests second tool
            mock_tool_block_2 = Mock()
            mock_tool_block_2.type = "tool_use"
            mock_tool_block_2.name = "search_course_content"
            mock_tool_block_2.input = {"query": "variables", "course_name": "Python Basics"}
            mock_tool_block_2.id = "tool_2"

            mock_response_2 = Mock()
            mock_response_2.content = [mock_tool_block_2]
            mock_response_2.stop_reason = "tool_use"

            # Mock final synthesis without tools
            mock_text_block = Mock()
            mock_text_block.text = "Python Basics covers variables in lesson 2."

            mock_response_3 = Mock()
            mock_response_3.content = [mock_text_block]
            mock_response_3.stop_reason = "end_turn"

            mock_client.messages.create.side_effect = [
                mock_response_1,
                mock_response_2,
                mock_response_3
            ]

            ai_gen = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
            tools = tool_manager.get_tool_definitions()

            response, sources = ai_gen.generate_response(
                "What course covers variables?",
                tools=tools,
                tool_manager=tool_manager
            )

            # Verify 3 API calls were made
            assert mock_client.messages.create.call_count == 3

            # Verify first two calls include tools parameter
            call_1_args = mock_client.messages.create.call_args_list[0][1]
            call_2_args = mock_client.messages.create.call_args_list[1][1]
            assert "tools" in call_1_args
            assert "tools" in call_2_args

            # Verify third call excludes tools (synthesis)
            call_3_args = mock_client.messages.create.call_args_list[2][1]
            assert "tools" not in call_3_args

            # Verify final response
            assert response == "Python Basics covers variables in lesson 2."
            assert isinstance(sources, list)

    @pytest.mark.unit
    def test_early_termination_single_round(self, test_config, tool_manager):
        """Test Claude responds after single tool use without requesting more tools"""
        with patch('anthropic.Anthropic') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            # Mock round 0: Claude requests tool
            mock_tool_block = Mock()
            mock_tool_block.type = "tool_use"
            mock_tool_block.name = "search_course_content"
            mock_tool_block.input = {"query": "Python"}
            mock_tool_block.id = "tool_1"

            mock_response_1 = Mock()
            mock_response_1.content = [mock_tool_block]
            mock_response_1.stop_reason = "tool_use"

            # Mock round 1: Claude responds with text (no more tools)
            mock_text_block = Mock()
            mock_text_block.text = "Python is a programming language covered in course X."

            mock_response_2 = Mock()
            mock_response_2.content = [mock_text_block]
            mock_response_2.stop_reason = "end_turn"

            mock_client.messages.create.side_effect = [
                mock_response_1,
                mock_response_2
            ]

            ai_gen = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
            tools = tool_manager.get_tool_definitions()

            response, sources = ai_gen.generate_response(
                "What is Python?",
                tools=tools,
                tool_manager=tool_manager
            )

            # Verify only 2 API calls (early termination)
            assert mock_client.messages.create.call_count == 2

            # Verify second call still offered tools (Claude chose not to use)
            call_2_args = mock_client.messages.create.call_args_list[1][1]
            assert "tools" in call_2_args

            # Verify response
            assert response == "Python is a programming language covered in course X."

    @pytest.mark.unit
    def test_max_rounds_final_synthesis(self, test_config, tool_manager):
        """Test final synthesis call when max rounds reached"""
        with patch('anthropic.Anthropic') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            # Mock round 0: First tool use
            mock_tool_1 = Mock()
            mock_tool_1.type = "tool_use"
            mock_tool_1.name = "get_course_outline"
            mock_tool_1.input = {"course_name": "Python"}
            mock_tool_1.id = "tool_1"

            mock_response_1 = Mock()
            mock_response_1.content = [mock_tool_1]
            mock_response_1.stop_reason = "tool_use"

            # Mock round 1: Second tool use
            mock_tool_2 = Mock()
            mock_tool_2.type = "tool_use"
            mock_tool_2.name = "search_course_content"
            mock_tool_2.input = {"query": "lesson 4"}
            mock_tool_2.id = "tool_2"

            mock_response_2 = Mock()
            mock_response_2.content = [mock_tool_2]
            mock_response_2.stop_reason = "tool_use"

            # Mock final synthesis (max rounds reached)
            mock_text = Mock()
            mock_text.text = "Synthesised answer from both tool results."

            mock_response_3 = Mock()
            mock_response_3.content = [mock_text]
            mock_response_3.stop_reason = "end_turn"

            mock_client.messages.create.side_effect = [
                mock_response_1,
                mock_response_2,
                mock_response_3
            ]

            ai_gen = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
            tools = tool_manager.get_tool_definitions()

            response, sources = ai_gen.generate_response(
                "Complex multi-part query",
                tools=tools,
                tool_manager=tool_manager
            )

            # Verify 3 API calls made
            assert mock_client.messages.create.call_count == 3

            # Verify final call has NO tools parameter (forced synthesis)
            final_call_args = mock_client.messages.create.call_args_list[2][1]
            assert "tools" not in final_call_args

            # Verify message array includes all tool results
            final_messages = final_call_args["messages"]
            assert len(final_messages) >= 5  # user, assistant+tool1, user+result1, assistant+tool2, user+result2

            # Verify response
            assert response == "Synthesised answer from both tool results."
            assert isinstance(sources, list)

    @pytest.mark.unit
    def test_tool_failure_in_second_round(self, test_config, tool_manager):
        """Test graceful handling when tool execution fails in round 2"""
        with patch('anthropic.Anthropic') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            # Mock round 0: Successful tool use
            mock_tool_1 = Mock()
            mock_tool_1.type = "tool_use"
            mock_tool_1.name = "search_course_content"
            mock_tool_1.input = {"query": "Python"}
            mock_tool_1.id = "tool_1"

            mock_response_1 = Mock()
            mock_response_1.content = [mock_tool_1]
            mock_response_1.stop_reason = "tool_use"

            # Mock round 1: Tool use that will fail
            mock_tool_2 = Mock()
            mock_tool_2.type = "tool_use"
            mock_tool_2.name = "search_course_content"
            mock_tool_2.input = {"query": "JavaScript"}
            mock_tool_2.id = "tool_2"

            mock_response_2 = Mock()
            mock_response_2.content = [mock_tool_2]
            mock_response_2.stop_reason = "tool_use"

            mock_client.messages.create.side_effect = [
                mock_response_1,
                mock_response_2
            ]

            # Configure tool manager to fail on second execution
            tool_manager.execute_tool = Mock(side_effect=[
                "Result from first tool",
                Exception("Tool execution failed")
            ])

            ai_gen = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
            tools = tool_manager.get_tool_definitions()

            response, sources = ai_gen.generate_response(
                "Test query",
                tools=tools,
                tool_manager=tool_manager
            )

            # Verify error message returned
            assert "error" in response.lower() or "encountered" in response.lower()

            # Verify sources is still a list (empty or with round 1 sources)
            assert isinstance(sources, list)

    @pytest.mark.unit
    def test_source_accumulation_across_rounds(self, test_config):
        """Test sources from multiple rounds are accumulated correctly"""
        with patch('anthropic.Anthropic') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            # Create mock tool manager with source tracking
            mock_tool_manager = Mock()

            # Mock tool execution returns and source tracking
            mock_tool_manager.execute_tool.side_effect = [
                "Result 1",
                "Result 2"
            ]

            # Mock sources from each round
            mock_tool_manager.get_last_sources.side_effect = [
                [{"course": "Python Basics", "lesson": 1}],
                [{"course": "Advanced Python", "lesson": 3}]
            ]

            # Mock API responses
            mock_tool_1 = Mock()
            mock_tool_1.type = "tool_use"
            mock_tool_1.name = "search_course_content"
            mock_tool_1.input = {"query": "basics"}
            mock_tool_1.id = "tool_1"

            mock_response_1 = Mock()
            mock_response_1.content = [mock_tool_1]
            mock_response_1.stop_reason = "tool_use"

            mock_tool_2 = Mock()
            mock_tool_2.type = "tool_use"
            mock_tool_2.name = "search_course_content"
            mock_tool_2.input = {"query": "advanced"}
            mock_tool_2.id = "tool_2"

            mock_response_2 = Mock()
            mock_response_2.content = [mock_tool_2]
            mock_response_2.stop_reason = "tool_use"

            mock_text = Mock()
            mock_text.text = "Final answer"

            mock_response_3 = Mock()
            mock_response_3.content = [mock_text]
            mock_response_3.stop_reason = "end_turn"

            mock_client.messages.create.side_effect = [
                mock_response_1,
                mock_response_2,
                mock_response_3
            ]

            ai_gen = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)

            response, sources = ai_gen.generate_response(
                "Query requiring multiple searches",
                tools=[{"name": "search_course_content"}],
                tool_manager=mock_tool_manager
            )

            # Verify sources from both rounds are present
            assert len(sources) == 2
            assert sources[0] == {"course": "Python Basics", "lesson": 1}
            assert sources[1] == {"course": "Advanced Python", "lesson": 3}

            # Verify reset_sources was called after each round
            assert mock_tool_manager.reset_sources.call_count == 2

    @pytest.mark.unit
    def test_conversation_state_transitions(self, test_config, tool_manager):
        """Test ConversationState methods work correctly"""
        from ai_generator import ConversationState

        # Create conversation state
        state = ConversationState(
            initial_query="test query",
            system_content="system prompt",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
            max_rounds=2,
            messages=[{"role": "user", "content": "test query"}]
        )

        # Test initial state
        assert state.current_round == 0
        assert len(state.messages) == 1
        assert len(state.accumulated_sources) == 0
        assert state.can_use_tools() is True

        # Test add_assistant_message
        state.add_assistant_message([{"type": "tool_use", "name": "test"}])
        assert len(state.messages) == 2
        assert state.messages[1]["role"] == "assistant"

        # Test add_tool_results
        tool_results = [{"type": "tool_result", "content": "result"}]
        state.add_tool_results(tool_results)
        assert len(state.messages) == 3
        assert state.messages[2]["role"] == "user"
        assert state.messages[2]["content"] == tool_results

        # Test increment_round
        state.increment_round()
        assert state.current_round == 1
        assert state.can_use_tools() is True

        # Test can_use_tools at max rounds
        state.increment_round()
        assert state.current_round == 2
        assert state.can_use_tools() is False

        # Test accumulated_sources
        state.accumulated_sources.extend([{"source": "test"}])
        assert len(state.accumulated_sources) == 1

    @pytest.mark.unit
    def test_message_array_structure_multi_round(self, test_config, tool_manager):
        """Test message array maintains correct structure across rounds"""
        with patch('anthropic.Anthropic') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            # Mock two tool uses
            mock_tool_1 = Mock()
            mock_tool_1.type = "tool_use"
            mock_tool_1.name = "search_course_content"
            mock_tool_1.input = {"query": "test1"}
            mock_tool_1.id = "tool_1"

            mock_response_1 = Mock()
            mock_response_1.content = [mock_tool_1]
            mock_response_1.stop_reason = "tool_use"

            mock_tool_2 = Mock()
            mock_tool_2.type = "tool_use"
            mock_tool_2.name = "search_course_content"
            mock_tool_2.input = {"query": "test2"}
            mock_tool_2.id = "tool_2"

            mock_response_2 = Mock()
            mock_response_2.content = [mock_tool_2]
            mock_response_2.stop_reason = "tool_use"

            mock_text = Mock()
            mock_text.text = "Final"

            mock_response_3 = Mock()
            mock_response_3.content = [mock_text]

            mock_client.messages.create.side_effect = [
                mock_response_1,
                mock_response_2,
                mock_response_3
            ]

            ai_gen = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
            tools = tool_manager.get_tool_definitions()

            response, sources = ai_gen.generate_response(
                "Test query",
                tools=tools,
                tool_manager=tool_manager
            )

            # Check final message structure
            final_call_messages = mock_client.messages.create.call_args_list[2][1]["messages"]

            # Should have: user query, assistant tool_use_1, user tool_result_1,
            #              assistant tool_use_2, user tool_result_2
            assert len(final_call_messages) == 5
            assert final_call_messages[0]["role"] == "user"
            assert final_call_messages[1]["role"] == "assistant"
            assert final_call_messages[2]["role"] == "user"
            assert final_call_messages[3]["role"] == "assistant"
            assert final_call_messages[4]["role"] == "user"

            # Verify tool results structure
            assert isinstance(final_call_messages[2]["content"], list)
            assert final_call_messages[2]["content"][0]["type"] == "tool_result"
            assert isinstance(final_call_messages[4]["content"], list)
            assert final_call_messages[4]["content"][0]["type"] == "tool_result"

    @pytest.mark.unit
    def test_system_prompt_multi_round_instructions(self, test_config):
        """Test system prompt contains multi-round guidance"""
        ai_gen = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)

        prompt = ai_gen.SYSTEM_PROMPT

        # Check for multi-round instructions
        assert "UP TO 2 ROUNDS" in prompt or "2 ROUNDS" in prompt
        assert "Round 1" in prompt or "round 1" in prompt.lower()
        assert "Round 2" in prompt or "round 2" in prompt.lower()
        assert "Synthesize ALL search results" in prompt or "synthesize" in prompt.lower()

        # Check for multi-round strategy guidance
        assert "multi-round" in prompt.lower() or "multiple" in prompt.lower()

    @pytest.mark.requires_api
    def test_real_api_integration(self, test_config):
        """Integration test with real Anthropic API (requires valid API key)"""
        if not test_config.ANTHROPIC_API_KEY or test_config.ANTHROPIC_API_KEY == "test_api_key":
            pytest.skip("No valid API key for integration test")

        ai_gen = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)

        response, sources = ai_gen.generate_response("What is 2+2?")

        assert isinstance(response, str)
        assert len(response) > 0
        assert "4" in response  # Should contain the answer
        assert isinstance(sources, list)