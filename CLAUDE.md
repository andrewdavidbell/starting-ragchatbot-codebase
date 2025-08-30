# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Course Materials RAG (Retrieval-Augmented Generation) system that enables semantic search and AI-powered Q&A over educational content. It's a full-stack application with a Python FastAPI backend and vanilla JavaScript frontend.

## Development Commands

### Running the Application
```bash
# Quick start (recommended)
chmod +x run.sh && ./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000

# Install dependencies
uv sync
```

### Environment Setup
```bash
# Copy and configure environment file
cp .env.example .env
# Add ANTHROPIC_API_KEY to .env file
```

### Access Points
- Web Interface: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Architecture Overview

### Core System Flow
The application follows a tool-based RAG architecture where Claude AI uses search tools to retrieve relevant course content:

1. **Query Processing**: User input → FastAPI endpoint → RAG system orchestration
2. **AI Decision Making**: Claude determines if/how to search course content using tools
3. **Semantic Search**: Vector similarity search across course chunks with metadata filtering
4. **Response Generation**: AI synthesizes search results into natural language answers

### Key Components

**RAG System (`backend/rag_system.py`)**
- Central orchestrator that coordinates all components
- Manages document ingestion, query processing, and response generation
- Initializes and connects: DocumentProcessor, VectorStore, AIGenerator, SessionManager, ToolManager

**Document Processing Pipeline**
- `DocumentProcessor`: Parses structured course documents, extracts metadata, creates overlapping chunks
- `VectorStore`: Manages ChromaDB collections (course_catalog + course_content), handles semantic search
- Document format: `Course Title:`, `Course Link:`, `Course Instructor:`, then `Lesson N:` markers

**AI Integration (`ai_generator.py`)**
- Anthropic Claude API integration with tool calling capabilities
- System prompt optimized for educational content and concise responses
- Handles tool execution flow: initial call → tool use → follow-up call with results

**Search Tools (`search_tools.py`)**
- `CourseSearchTool`: Implements semantic course name resolution and content filtering
- `ToolManager`: Registers and executes available tools, tracks source citations
- Tool calling protocol: Claude requests search → tool executes → formatted results returned

**Vector Storage Strategy**
- **Two-collection design**: `course_catalog` (metadata) + `course_content` (chunks)
- **Smart course resolution**: Fuzzy matching on course names using vector similarity
- **Contextual chunking**: Sentence-based with overlap, lesson context preserved in chunk content

### Session Management
- In-memory conversation tracking with configurable history limits
- Session-based context preservation across multiple queries
- Automatic session creation for new conversations

### Configuration (`backend/config.py`)
Key settings controlled via environment variables and dataclass:
- `CHUNK_SIZE: 800` - Text chunk size for vector storage
- `CHUNK_OVERLAP: 100` - Character overlap between chunks  
- `MAX_RESULTS: 5` - Vector search result limit
- `EMBEDDING_MODEL: "all-MiniLM-L6-v2"` - Sentence transformer model
- `ANTHROPIC_MODEL: "claude-sonnet-4-20250514"` - Claude model version

### Document Structure Requirements
Course documents in `/docs` must follow this format:
```
Course Title: [title]
Course Link: [url] 
Course Instructor: [instructor]

Lesson 0: Introduction
[lesson content]

Lesson 1: Getting Started
Lesson Link: [optional lesson url]
[lesson content]
```

### Data Models (`backend/models.py`)
- `Course`: Title, instructor, lessons, course link
- `Lesson`: Number, title, optional lesson link
- `CourseChunk`: Content, course title, lesson number, chunk index

## Important Implementation Notes

### Vector Store Behavior
- **Duplicate Prevention**: Existing courses are not reprocessed on startup
- **Course Name Resolution**: Uses semantic search on course titles to handle partial/fuzzy matches
- **Metadata Filtering**: Supports filtering by exact course title and/or lesson number
- **Context Preservation**: First chunk of each lesson gets "Lesson N content:" prefix

### AI Tool Usage Pattern
- Claude decides autonomously whether to search based on query content
- **One search per query maximum** enforced by system prompt
- Search results are formatted with `[Course - Lesson N]` headers
- Sources are tracked separately for UI display

### Session Architecture
- Session IDs generated as `session_N` incrementally
- History truncated to `MAX_HISTORY * 2` messages (user + assistant pairs)
- Sessions persist in memory only (not database backed)

### Frontend Integration
- Static file serving from `/frontend` directory with development no-cache headers
- WebSocket-free design using standard HTTP requests
- Markdown rendering of AI responses using marked.js library
- Collapsible source citations in UI

## Key File Relationships

- `app.py` → `rag_system.py` → orchestrates all components
- `rag_system.py` → `ai_generator.py` → handles Claude API calls with tools
- `ai_generator.py` → `search_tools.py` → executes semantic search when needed  
- `search_tools.py` → `vector_store.py` → performs ChromaDB queries
- `vector_store.py` → `document_processor.py` → processes and chunks documents
- `session_manager.py` → standalone conversation history tracking

The system is designed for educational content Q&A with intelligent course material retrieval and maintains conversation context while providing source attribution for all answers.
- Always use uv to run the server, do not use pip directly