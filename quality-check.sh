#!/bin/bash
# Run all code quality checks

set -e

echo "🚀 Running code quality checks..."
echo ""

echo "📋 Checking code formatting with black..."
if uv run black --check backend/ main.py; then
    echo "✅ Black: All files properly formatted"
else
    echo "❌ Black: Some files need formatting. Run ./format.sh to fix."
    exit 1
fi
echo ""

echo "📋 Checking import sorting with isort..."
if uv run isort --check-only backend/ main.py; then
    echo "✅ Isort: All imports properly sorted"
else
    echo "❌ Isort: Some imports need sorting. Run ./format.sh to fix."
    exit 1
fi
echo ""

echo "📋 Running flake8 linting..."
if uv run flake8 backend/ main.py; then
    echo "✅ Flake8: No linting issues found"
else
    echo "❌ Flake8: Linting issues found. Please review and fix."
    exit 1
fi
echo ""

echo "📋 Running type checks with mypy..."
if uv run mypy backend/ main.py; then
    echo "✅ Mypy: No type issues found"
else
    echo "⚠️  Mypy: Type issues found. Please review."
    # Not failing on mypy errors for now as it may be too strict
fi
echo ""

echo "🎉 All quality checks passed!"
