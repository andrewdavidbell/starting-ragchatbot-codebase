#!/bin/bash
# Run all checks: format, quality checks, and tests

set -e

echo "🔄 Running complete codebase checks..."
echo ""

echo "Step 1: Formatting code..."
./format.sh
echo ""

echo "Step 2: Running quality checks..."
./quality-check.sh
echo ""

echo "Step 3: Running tests..."
./test.sh
echo ""

echo "🎉 All checks passed! Code is ready to commit."
