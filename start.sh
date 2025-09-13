#!/bin/bash

# RAG Agent Startup Script
# Quick startup script to launch the RAG Agent

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Starting RAG Agent for Custom Data Analysis${NC}"
echo "=========================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Please run install.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if Ollama is running
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama service..."
    ollama serve > /dev/null 2>&1 &
    sleep 3
fi

# Display menu
echo
echo "Choose an interface:"
echo "1. Command Line Interface (CLI)"
echo "2. Web Interface (Streamlit)"
echo "3. Run Tests"
echo "4. Add Documents"
echo

read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo -e "${GREEN}Starting CLI...${NC}"
        python interfaces/cli.py
        ;;
    2)
        echo -e "${GREEN}Starting Web Interface...${NC}"
        echo "Open your browser to: http://localhost:8501"
        streamlit run interfaces/web_app.py
        ;;
    3)
        echo -e "${GREEN}Running tests...${NC}"
        python test_installation.py
        ;;
    4)
        read -p "Enter path to documents: " doc_path
        echo -e "${GREEN}Adding documents...${NC}"
        python interfaces/cli.py --add-docs "$doc_path"
        ;;
    *)
        echo "Invalid choice. Starting CLI..."
        python interfaces/cli.py
        ;;
esac
