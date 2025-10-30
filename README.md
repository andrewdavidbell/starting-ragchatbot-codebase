# Course Materials RAG System

A Retrieval-Augmented Generation (RAG) system designed to answer questions about course materials using semantic search and AI-powered responses.

## Overview

This application is a full-stack web application that enables users to query course materials and receive intelligent, context-aware responses. It uses ChromaDB for vector storage, Anthropic's Claude for AI generation, and provides a web interface for interaction.


## Prerequisites

- Python 3.13 or higher
- uv (Python package manager)
- An Anthropic API key (for Claude AI)
- **For Windows**: Use Git Bash to run the application commands - [Download Git for Windows](https://git-scm.com/downloads/win)

## Installation

1. **Install uv** (if not already installed)
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install Python dependencies**
   ```bash
   uv sync
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the root directory:
   ```bash
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ```

## Running the Application

### Quick Start

Use the provided shell script:
```bash
chmod +x run.sh
./run.sh
```

### Manual Start

```bash
cd backend
uv run uvicorn app:app --reload --port 8000
```

The application will be available at:
- Web Interface: `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`

## Development Workflow

### Code Quality Tools

This project uses several code quality tools to maintain consistent formatting and catch potential issues:

- **black**: Automatic code formatting
- **isort**: Import statement sorting
- **flake8**: Linting and style checking
- **mypy**: Static type checking
- **pytest**: Testing framework with coverage reporting

### Development Scripts

The following scripts are available to help maintain code quality:

#### Format Code
Automatically format code with black and sort imports with isort:
```bash
./format.sh
```

#### Quality Checks
Run all quality checks (formatting, linting, type checking):
```bash
./quality-check.sh
```

#### Run Tests
Execute the test suite with coverage reporting:
```bash
./test.sh
```

#### Complete Check
Run formatting, quality checks, and tests in sequence:
```bash
./check-all.sh
```

### Recommended Development Workflow

1. Make your code changes
2. Run `./format.sh` to automatically format your code
3. Run `./quality-check.sh` to verify code quality
4. Run `./test.sh` to ensure tests pass
5. Commit your changes

Or simply run `./check-all.sh` to execute all steps at once before committing.

### Configuration

Code quality tool configurations are located in:
- `pyproject.toml`: black, isort, mypy, and pytest settings
- `.flake8`: flake8 configuration

