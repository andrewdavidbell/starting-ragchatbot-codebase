from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import anthropic


@dataclass
class ConversationState:
    """Manages conversation state across multiple tool calling rounds."""

    initial_query: str
    system_content: str
    tools: Optional[List]
    tool_manager: Any
    max_rounds: int

    # Dynamic state
    messages: List[Dict[str, Any]]
    current_round: int = 0
    accumulated_sources: List = None

    def __post_init__(self):
        if self.accumulated_sources is None:
            self.accumulated_sources = []

    def add_assistant_message(self, content):
        """Add assistant message to conversation."""
        self.messages.append({"role": "assistant", "content": content})

    def add_tool_results(self, tool_results: List[Dict]):
        """Add tool execution results to conversation."""
        if tool_results:
            self.messages.append({"role": "user", "content": tool_results})

    def increment_round(self):
        """Move to the next round."""
        self.current_round += 1

    def can_use_tools(self) -> bool:
        """Check if tools can still be used."""
        return (
            self.tools is not None
            and self.tool_manager is not None
            and self.current_round < self.max_rounds
        )


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Multi-Round Search Protocol:
- You can make UP TO 2 ROUNDS of tool calls per user query
- Use Round 1 for broad searches or initial exploration  
- Use Round 2 for follow-up searches, clarification, or additional details
- **Maximum 2 rounds total** - plan your search strategy accordingly
- Synthesize ALL search results from both rounds into your final response

Tool Selection Guidelines:
- **get_course_outline**: Use for structural/outline queries about courses
  - When users ask "what lessons are in...", "show course structure", "course outline"
  - Returns: Course title, course link, instructor, and complete lesson list (numbers, titles, links)
  - Example queries: "What's in the MCP course?", "Show lessons for Introduction to X"

- **search_course_content**: Use for questions about specific course content or detailed materials
  - When users ask about concepts, explanations, how-to information within courses
  - Returns: Relevant content chunks from course materials with context
  - Example queries: "How do I create a server?", "Explain authentication in lesson 3"

- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course-specific questions**: Choose the appropriate tool first, then answer based on results
- If first search yields limited results, consider a follow-up search with different parameters
- If search yields no results, state this clearly without offering alternatives

Multi-Round Strategy Examples:
- Round 1: Search broad course topic → Round 2: Search specific lesson details
- Round 1: Search by course name → Round 2: Search by different course if first was wrong
- Round 1: Search general concept → Round 2: Search for examples or implementations
- Single Round: Use when initial search provides sufficient information

Response Protocol:
- **No meta-commentary**: Provide direct answers only
- Do not mention "based on search results" or explain your search process
- Do not mention round numbers or search strategies  
- Synthesize information from multiple searches seamlessly

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> Tuple[str, List]:
        """
        Generate AI response with support for up to 2 rounds of sequential tool calling.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Tuple of (response text, sources list)
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Initialize conversation state
        state = ConversationState(
            initial_query=query,
            system_content=system_content,
            tools=tools,
            tool_manager=tool_manager,
            max_rounds=2,
            messages=[{"role": "user", "content": query}],
        )

        try:
            # Execute multi-round conversation
            response_text = self._execute_multi_round_conversation(state)
            return response_text, state.accumulated_sources
        except anthropic.AuthenticationError as e:
            return (
                f"Authentication failed: Invalid API key. Please check your ANTHROPIC_API_KEY in .env file. Error: {str(e)}",
                [],
            )
        except anthropic.APIError as e:
            return f"Anthropic API error: {str(e)}", []
        except Exception as e:
            return f"Unexpected error connecting to AI service: {str(e)}", []

    def _execute_multi_round_conversation(self, state: ConversationState) -> str:
        """Execute the multi-round conversation with tool calling."""

        while state.current_round < state.max_rounds:
            # Make API call with current conversation state
            response = self._make_api_call(state)

            # Handle different response scenarios
            if response.stop_reason == "tool_use" and state.can_use_tools():
                # Execute tools and prepare for next round
                tool_execution_success = self._execute_tools_and_update_state(
                    response, state
                )

                if not tool_execution_success:
                    # Tool execution failed - return error response
                    return "I encountered an error while searching for information. Please try rephrasing your query."

                state.increment_round()

                # Continue to next round unless this was the final round
                if state.current_round >= state.max_rounds:
                    # Final round - make one more call without tools to synthesize
                    return self._make_final_synthesis_call(state)

            else:
                # No tool use requested or we're done - return response
                return response.content[0].text

        # Fallback - shouldn't reach here
        return "Unable to generate response after maximum rounds."

    def _make_api_call(self, state: ConversationState):
        """Make API call with current conversation state."""
        api_params = {
            **self.base_params,
            "messages": state.messages.copy(),
            "system": state.system_content,
        }

        # Add tools if available and not at max rounds
        if state.can_use_tools():
            api_params["tools"] = state.tools
            api_params["tool_choice"] = {"type": "auto"}

        return self.client.messages.create(**api_params)

    def _execute_tools_and_update_state(
        self, response, state: ConversationState
    ) -> bool:
        """Execute tools and update conversation state for next round."""

        try:
            # Add AI's tool use response to messages
            state.add_assistant_message(response.content)

            # Execute all tool calls and collect results
            tool_results = []
            current_round_sources = []

            for content_block in response.content:
                if content_block.type == "tool_use":
                    # Execute the tool
                    tool_result = state.tool_manager.execute_tool(
                        content_block.name, **content_block.input
                    )

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": tool_result,
                        }
                    )

                    # Collect sources from this tool execution
                    round_sources = state.tool_manager.get_last_sources()
                    current_round_sources.extend(round_sources)

            # Add tool results to conversation state
            state.add_tool_results(tool_results)

            # Accumulate sources across rounds
            state.accumulated_sources.extend(current_round_sources)

            # Reset tool manager sources for next round
            state.tool_manager.reset_sources()

            return True

        except Exception as e:
            print(f"Tool execution error in round {state.current_round + 1}: {e}")
            return False

    def _make_final_synthesis_call(self, state: ConversationState) -> str:
        """Make final API call without tools to synthesize response."""

        try:
            # Get API params without tools for final synthesis
            api_params = {
                **self.base_params,
                "messages": state.messages.copy(),
                "system": state.system_content,
            }

            # Make the call
            final_response = self.client.messages.create(**api_params)
            return final_response.content[0].text

        except Exception as e:
            return f"Error in final synthesis: {str(e)}"
