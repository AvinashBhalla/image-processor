# build.sh

#!/bin/bash

echo "Building Image Processor..."

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Build package
python setup.py sdist bdist_wheel

echo "Build complete!"