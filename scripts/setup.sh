#!/bin/bash

# RAG Agent Setup Script
# This script sets up the RAG Agent environment and installs all dependencies

set -e

echo "RAG Agent Setup Script"
echo "======================"

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check if Python version is 3.8 or higher
if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "Error: Python 3.8 or higher is required"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p data logs temp chroma_db examples/documents

# Check if Ollama is installed
echo "Checking Ollama installation..."
if command -v ollama &> /dev/null; then
    echo "✓ Ollama is installed"
    
    # Check if Ollama is running
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✓ Ollama server is running"
    else
        echo "⚠ Ollama server is not running. Please start it with: ollama serve"
    fi
else
    echo "⚠ Ollama is not installed"
    echo "Please install Ollama from: https://ollama.ai"
    echo "Then run: ollama pull llama3"
fi

# Install the package in development mode
echo "Installing RAG Agent package..."
pip install -e .

echo ""
echo "Setup complete!"
echo "=============="
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Start Ollama if not running: ollama serve"
echo "3. Pull a model: ollama pull llama3"
echo "4. Initialize RAG Agent: python main.py init"
echo "5. Ingest documents: python main.py ingest examples/documents"
echo "6. Start querying: python main.py interactive"
echo ""
echo "For help: python main.py --help"