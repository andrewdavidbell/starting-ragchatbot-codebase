#!/bin/bash
# Run tests with coverage

set -e

echo "🧪 Running tests with coverage..."
cd backend && uv run pytest

echo "✅ Tests complete! Check htmlcov/index.html for detailed coverage report."
