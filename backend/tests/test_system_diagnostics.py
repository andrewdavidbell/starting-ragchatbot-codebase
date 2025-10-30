import os
import sys
from unittest.mock import patch

import anthropic
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ai_generator import AIGenerator
from config import config
from rag_system import RAGSystem
from search_tools import CourseSearchTool, ToolManager
from session_manager import SessionManager
from vector_store import VectorStore


class TestSystemDiagnostics:
    """Diagnostic tests to identify specific system failures"""

    @pytest.mark.integration
    def test_chroma_database_exists_and_accessible(self):
        """Test that ChromaDB exists and is accessible"""
        chroma_path = config.CHROMA_PATH

        # Check if database directory exists
        if os.path.exists(chroma_path):
            print(f"✓ ChromaDB path exists: {chroma_path}")
        else:
            pytest.fail(f"✗ ChromaDB path does not exist: {chroma_path}")

        # Try to initialize VectorStore
        try:
            store = VectorStore(chroma_path, config.EMBEDDING_MODEL, config.MAX_RESULTS)
            print("✓ VectorStore initialization successful")
        except Exception as e:
            pytest.fail(f"✗ VectorStore initialization failed: {e}")

    @pytest.mark.integration
    def test_vector_database_has_course_data(self):
        """Test that the vector database contains expected course data"""
        store = VectorStore(
            config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS
        )

        # Check course count
        course_count = store.get_course_count()
        print(f"Course count in database: {course_count}")

        if course_count == 0:
            pytest.fail("✗ No courses found in database")
        else:
            print(f"✓ Found {course_count} courses in database")

        # Check course titles
        course_titles = store.get_existing_course_titles()
        print(f"Course titles: {course_titles}")

        if not course_titles:
            pytest.fail("✗ No course titles found")
        else:
            print(f"✓ Found course titles: {course_titles}")

    @pytest.mark.integration
    def test_vector_search_functionality(self):
        """Test that vector search returns results"""
        store = VectorStore(
            config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS
        )

        # Try basic search
        test_queries = ["introduction", "Python", "getting started", "MCP", "Anthropic"]

        search_results = {}
        for query in test_queries:
            try:
                results = store.search(query)
                search_results[query] = {
                    "success": not results.is_empty(),
                    "doc_count": len(results.documents),
                    "error": results.error,
                }
                if not results.is_empty():
                    print(
                        f"✓ Search for '{query}' returned {len(results.documents)} results"
                    )
                else:
                    print(
                        f"✗ Search for '{query}' returned no results. Error: {results.error}"
                    )
            except Exception as e:
                search_results[query] = {"success": False, "error": str(e)}
                print(f"✗ Search for '{query}' failed with exception: {e}")

        # Check if any searches succeeded
        successful_searches = [q for q, r in search_results.items() if r["success"]]
        if not successful_searches:
            pytest.fail(f"✗ All search queries failed: {search_results}")
        else:
            print(f"✓ {len(successful_searches)} search queries succeeded")

    @pytest.mark.integration
    def test_course_name_resolution(self):
        """Test that course name resolution works"""
        store = VectorStore(
            config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS
        )

        # Get existing course titles
        existing_titles = store.get_existing_course_titles()

        if not existing_titles:
            pytest.skip("No courses in database to test resolution")

        # Test exact match
        first_course = existing_titles[0]
        resolved = store._resolve_course_name(first_course)

        if resolved != first_course:
            pytest.fail(
                f"✗ Course name resolution failed. Expected: {first_course}, Got: {resolved}"
            )
        else:
            print(f"✓ Course name resolution works for exact match: {first_course}")

        # Test partial match (if course name has multiple words)
        if len(first_course.split()) > 1:
            partial_name = first_course.split()[0]  # First word
            resolved_partial = store._resolve_course_name(partial_name)

            if resolved_partial:
                print(
                    f"✓ Course name resolution works for partial match: '{partial_name}' -> '{resolved_partial}'"
                )
            else:
                print(
                    f"✗ Course name resolution failed for partial match: '{partial_name}'"
                )

    @pytest.mark.integration
    def test_search_tool_functionality(self):
        """Test that CourseSearchTool works correctly"""
        store = VectorStore(
            config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS
        )
        search_tool = CourseSearchTool(store)

        # Test tool definition
        definition = search_tool.get_tool_definition()
        assert definition["name"] == "search_course_content"
        print("✓ CourseSearchTool definition is correct")

        # Test basic search
        try:
            result = search_tool.execute("introduction")
            if "No relevant content found" in result:
                pytest.fail("✗ CourseSearchTool returned no results for 'introduction'")
            else:
                print(
                    f"✓ CourseSearchTool search successful: {len(result)} characters returned"
                )
                print(f"Sample result: {result[:200]}...")
        except Exception as e:
            pytest.fail(f"✗ CourseSearchTool execution failed: {e}")

    @pytest.mark.integration
    def test_tool_manager_functionality(self):
        """Test that ToolManager works correctly"""
        store = VectorStore(
            config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS
        )
        search_tool = CourseSearchTool(store)
        tool_manager = ToolManager()
        tool_manager.register_tool(search_tool)

        # Test tool registration
        definitions = tool_manager.get_tool_definitions()
        assert len(definitions) == 1
        assert definitions[0]["name"] == "search_course_content"
        print("✓ ToolManager registration works")

        # Test tool execution
        try:
            result = tool_manager.execute_tool(
                "search_course_content", query="introduction"
            )
            if "Tool" in result and "not found" in result:
                pytest.fail(f"✗ ToolManager failed to find registered tool: {result}")
            else:
                print("✓ ToolManager execution works")
        except Exception as e:
            pytest.fail(f"✗ ToolManager execution failed: {e}")

    @pytest.mark.requires_api
    def test_anthropic_api_connectivity(self):
        """Test that Anthropic API is accessible with current config"""
        if not config.ANTHROPIC_API_KEY or config.ANTHROPIC_API_KEY == "test_api_key":
            pytest.skip("No valid API key configured")

        ai_generator = AIGenerator(config.ANTHROPIC_API_KEY, config.ANTHROPIC_MODEL)

        try:
            # Test simple query without tools
            response = ai_generator.generate_response("What is 2+2?")

            if not response or len(response.strip()) == 0:
                pytest.fail("✗ Anthropic API returned empty response")
            else:
                print(
                    f"✓ Anthropic API connection successful. Response: {response[:100]}..."
                )

        except anthropic.APIError as e:
            pytest.fail(f"✗ Anthropic API error: {e}")
        except Exception as e:
            pytest.fail(f"✗ Unexpected error with Anthropic API: {e}")

    @pytest.mark.requires_api
    def test_anthropic_tool_calling(self):
        """Test that Anthropic API can handle tool calling"""
        if not config.ANTHROPIC_API_KEY or config.ANTHROPIC_API_KEY == "test_api_key":
            pytest.skip("No valid API key configured")

        # Setup components
        store = VectorStore(
            config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS
        )
        search_tool = CourseSearchTool(store)
        tool_manager = ToolManager()
        tool_manager.register_tool(search_tool)
        ai_generator = AIGenerator(config.ANTHROPIC_API_KEY, config.ANTHROPIC_MODEL)

        try:
            # Test with tools available
            tools = tool_manager.get_tool_definitions()
            response = ai_generator.generate_response(
                "Search for information about introduction to courses",
                tools=tools,
                tool_manager=tool_manager,
            )

            if not response:
                pytest.fail("✗ Anthropic API with tools returned empty response")
            else:
                print(
                    f"✓ Anthropic API tool calling works. Response: {response[:150]}..."
                )

                # Check if sources were populated (indicates tool was used)
                sources = tool_manager.get_last_sources()
                if sources:
                    print(f"✓ Tool was actually used - found {len(sources)} sources")
                else:
                    print("? Tool may not have been used (no sources found)")

        except Exception as e:
            pytest.fail(f"✗ Anthropic tool calling failed: {e}")

    @pytest.mark.integration
    def test_session_manager_functionality(self):
        """Test that SessionManager works correctly"""
        session_manager = SessionManager(config.MAX_HISTORY)

        # Test session creation
        session_id = session_manager.create_session()
        assert session_id.startswith("session_")
        print(f"✓ Session creation works: {session_id}")

        # Test adding exchanges
        session_manager.add_exchange(session_id, "Test question", "Test answer")

        # Test history retrieval
        history = session_manager.get_conversation_history(session_id)
        assert "Test question" in history
        assert "Test answer" in history
        print("✓ Session history management works")

    @pytest.mark.integration
    def test_full_rag_system_initialization(self):
        """Test that RAGSystem initializes correctly"""
        try:
            rag = RAGSystem(config)
            print("✓ RAGSystem initialization successful")

            # Test analytics
            analytics = rag.get_course_analytics()
            print(f"✓ Course analytics: {analytics}")

        except Exception as e:
            pytest.fail(f"✗ RAGSystem initialization failed: {e}")

    @pytest.mark.slow
    @pytest.mark.requires_api
    def test_end_to_end_query_flow(self):
        """Test complete end-to-end query processing"""
        if not config.ANTHROPIC_API_KEY or config.ANTHROPIC_API_KEY == "test_api_key":
            pytest.skip("No valid API key configured")

        # Initialize RAG system
        rag = RAGSystem(config)

        # Test queries that should and shouldn't use search
        test_cases = [
            {
                "query": "What is 2+2?",
                "should_use_search": False,
                "description": "General knowledge query",
            },
            {
                "query": "Tell me about the introduction to MCP course",
                "should_use_search": True,
                "description": "Course-specific query",
            },
            {
                "query": "What are the main topics covered in the courses?",
                "should_use_search": True,
                "description": "Course content query",
            },
        ]

        for test_case in test_cases:
            print(f"\nTesting: {test_case['description']}")
            print(f"Query: {test_case['query']}")

            try:
                response, sources = rag.query(test_case["query"])

                print(f"Response length: {len(response)} characters")
                print(f"Response preview: {response[:150]}...")
                print(f"Sources found: {len(sources)}")

                if test_case["should_use_search"] and not sources:
                    print(f"⚠ Expected search tool usage but no sources found")
                elif not test_case["should_use_search"] and sources:
                    print(f"⚠ Unexpected search tool usage (found sources)")
                else:
                    print("✓ Tool usage as expected")

                # Check for "failed query" response
                if "failed query" in response.lower():
                    pytest.fail(
                        f"✗ Got 'failed query' response for: {test_case['query']}"
                    )

            except Exception as e:
                pytest.fail(
                    f"✗ End-to-end query failed for '{test_case['query']}': {e}"
                )

    @pytest.mark.integration
    def test_diagnose_failed_query_issue(self):
        """Specific diagnostic test to identify 'failed query' issue"""
        print("\n=== DIAGNOSING 'FAILED QUERY' ISSUE ===")

        # Test each component in isolation
        print("\n1. Testing VectorStore...")
        store = VectorStore(
            config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS
        )
        search_result = store.search("introduction")
        print(f"   Vector search success: {not search_result.is_empty()}")
        if search_result.error:
            print(f"   Vector search error: {search_result.error}")

        print("\n2. Testing SearchTool...")
        search_tool = CourseSearchTool(store)
        tool_result = search_tool.execute("introduction")
        print(f"   Search tool result length: {len(tool_result)}")
        print(f"   Search tool result preview: {tool_result[:100]}...")

        print("\n3. Testing ToolManager...")
        tool_manager = ToolManager()
        tool_manager.register_tool(search_tool)
        manager_result = tool_manager.execute_tool(
            "search_course_content", query="introduction"
        )
        print(f"   Tool manager result length: {len(manager_result)}")

        if config.ANTHROPIC_API_KEY and config.ANTHROPIC_API_KEY != "test_api_key":
            print("\n4. Testing AIGenerator...")
            ai_generator = AIGenerator(config.ANTHROPIC_API_KEY, config.ANTHROPIC_MODEL)

            # Test without tools first
            simple_response = ai_generator.generate_response("What is 2+2?")
            print(f"   Simple AI response: {simple_response[:100]}...")

            # Test with tools
            tools = tool_manager.get_tool_definitions()
            tool_response = ai_generator.generate_response(
                "Tell me about course introductions",
                tools=tools,
                tool_manager=tool_manager,
            )
            print(f"   AI response with tools: {tool_response[:100]}...")

            # Check for failure indicators
            if "failed query" in tool_response.lower():
                print("   ✗ FOUND 'failed query' in AI response!")
            else:
                print("   ✓ No 'failed query' in AI response")
        else:
            print("\n4. Skipping AIGenerator test (no API key)")

    @pytest.mark.integration
    def test_document_integrity(self):
        """Test that course documents are properly loaded"""
        docs_path = "../docs"

        if not os.path.exists(docs_path):
            pytest.skip(f"Documents directory not found: {docs_path}")

        # Check document files
        doc_files = [
            f
            for f in os.listdir(docs_path)
            if f.lower().endswith((".txt", ".pdf", ".docx"))
        ]

        print(f"Found {len(doc_files)} document files: {doc_files}")

        if not doc_files:
            pytest.fail("No document files found in docs directory")

        # Test that VectorStore has content from these documents
        store = VectorStore(
            config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS
        )
        course_count = store.get_course_count()

        if course_count == 0:
            pytest.fail(
                "No courses loaded in VectorStore despite having document files"
            )
        else:
            print(
                f"✓ {course_count} courses loaded from {len(doc_files)} document files"
            )

    @pytest.mark.integration
    def test_environment_configuration(self):
        """Test that environment is properly configured"""
        print("\n=== ENVIRONMENT CONFIGURATION ===")

        print(
            f"ANTHROPIC_API_KEY configured: {'Yes' if config.ANTHROPIC_API_KEY and config.ANTHROPIC_API_KEY != 'test_api_key' else 'No'}"
        )
        print(f"ANTHROPIC_MODEL: {config.ANTHROPIC_MODEL}")
        print(f"EMBEDDING_MODEL: {config.EMBEDDING_MODEL}")
        print(f"CHROMA_PATH: {config.CHROMA_PATH}")
        print(f"MAX_RESULTS: {config.MAX_RESULTS}")
        print(f"CHUNK_SIZE: {config.CHUNK_SIZE}")
        print(f"CHUNK_OVERLAP: {config.CHUNK_OVERLAP}")

        # Test ChromaDB path
        if os.path.exists(config.CHROMA_PATH):
            print(f"✓ ChromaDB directory exists")

            # Check contents
            contents = os.listdir(config.CHROMA_PATH)
            print(f"ChromaDB contents: {contents}")
        else:
            print(f"✗ ChromaDB directory does not exist: {config.CHROMA_PATH}")

        # Validate configuration values
        assert config.MAX_RESULTS > 0, "MAX_RESULTS must be positive"
        assert config.CHUNK_SIZE > 0, "CHUNK_SIZE must be positive"
        assert config.CHUNK_OVERLAP >= 0, "CHUNK_OVERLAP must be non-negative"
        print("✓ Configuration values are valid")
