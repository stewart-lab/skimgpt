#!/bin/bash

# Build and publish skimgpt package to PyPI

set -e  # Exit on any error

echo "Building skimgpt package..."

# Clean previous builds
rm -rf build/ dist/ *.egg-info/

# Install build dependencies
pip install --upgrade build twine

# Build the package
python -m build

echo "Package built successfully!"
echo "Contents of dist/:"
ls -la dist/

echo ""
echo "To publish to PyPI:"
echo "1. Test upload to TestPyPI first:"
echo "   python -m twine upload --repository testpypi dist/*"
echo ""
echo "2. If test upload works, upload to PyPI:"
echo "   python -m twine upload dist/*"
echo ""
echo "3. Install from PyPI:"
echo "   pip install skimgpt"

# Optionally, you can uncomment the following lines to auto-publish
# echo "Publishing to TestPyPI..."
# python -m twine upload --repository testpypi dist/*

# echo "Publishing to PyPI..."
# python -m twine upload dist/* 