#!/usr/bin/env bash
# Run all tests in the tests directory

set -e  # Exit on error

echo "=========================================="
echo "Running BDH Test Suite"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Track results
PASSED=0
FAILED=0

# Run each test file
for test_file in tests/test_*.py; do
    if [ -f "$test_file" ]; then
        echo "Running $(basename $test_file)..."
        if python "$test_file"; then
            PASSED=$((PASSED + 1))
            echo -e "${GREEN}✓ $(basename $test_file) passed${NC}"
        else
            FAILED=$((FAILED + 1))
            echo -e "${RED}✗ $(basename $test_file) failed${NC}"
        fi
        echo ""
    fi
done

# Summary
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo -e "Passed: ${GREEN}${PASSED}${NC}"
if [ $FAILED -gt 0 ]; then
    echo -e "Failed: ${RED}${FAILED}${NC}"
    exit 1
else
    echo -e "Failed: ${FAILED}"
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
fi
