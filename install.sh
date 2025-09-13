#!/bin/bash

# RAG Agent Installation Script for Unix/Linux/macOS
# This script automates the installation and setup of the RAG Agent

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check system requirements
check_requirements() {
    print_status "Checking system requirements..."
    
    # Check Python version
    if command_exists python3; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        print_status "Found Python $PYTHON_VERSION"
        
        # Check if Python version is >= 3.8
        if python3 -c 'import sys; exit(0 if sys.version_info >= (3,8) else 1)'; then
            print_success "Python version is compatible"
        else
            print_error "Python 3.8+ is required. Found Python $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3 is not installed"
        exit 1
    fi
    
    # Check available memory
    if command_exists free; then
        MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
        print_status "Available memory: ${MEMORY_GB}GB"
        
        if [ "$MEMORY_GB" -lt 8 ]; then
            print_warning "Recommended memory is 16GB+. You have ${MEMORY_GB}GB"
            read -p "Continue anyway? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        fi
    fi
    
    # Check available disk space
    AVAILABLE_SPACE=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    print_status "Available disk space: ${AVAILABLE_SPACE}GB"
    
    if [ "$AVAILABLE_SPACE" -lt 20 ]; then
        print_warning "Recommended disk space is 50GB+. You have ${AVAILABLE_SPACE}GB"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Function to install Ollama
install_ollama() {
    print_status "Installing Ollama..."
    
    if command_exists ollama; then
        print_success "Ollama is already installed"
        ollama --version
    else
        print_status "Downloading and installing Ollama..."
        curl -fsSL https://ollama.ai/install.sh | sh
        
        if command_exists ollama; then
            print_success "Ollama installed successfully"
        else
            print_error "Failed to install Ollama"
            exit 1
        fi
    fi
    
    # Start Ollama service
    print_status "Starting Ollama service..."
    if command_exists systemctl; then
        sudo systemctl start ollama || true
        sudo systemctl enable ollama || true
    else
        # For macOS or systems without systemctl
        ollama serve > /dev/null 2>&1 &
        sleep 2
    fi
}

# Function to pull language model
pull_model() {
    print_status "Pulling language model..."
    
    echo "Available models:"
    echo "1. llama3 (Recommended for general use - ~4GB)"
    echo "2. mistral (Smaller model for limited resources - ~2GB)"
    echo "3. codellama (For code analysis - ~4GB)"
    echo "4. llama3:70b (Best quality, requires 32GB+ RAM - ~40GB)"
    
    read -p "Select model (1-4) [1]: " MODEL_CHOICE
    MODEL_CHOICE=${MODEL_CHOICE:-1}
    
    case $MODEL_CHOICE in
        1)
            MODEL_NAME="llama3"
            ;;
        2)
            MODEL_NAME="mistral"
            ;;
        3)
            MODEL_NAME="codellama"
            ;;
        4)
            MODEL_NAME="llama3:70b"
            print_warning "This model requires 32GB+ RAM"
            read -p "Continue? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                MODEL_NAME="llama3"
            fi
            ;;
        *)
            print_warning "Invalid choice, using llama3"
            MODEL_NAME="llama3"
            ;;
    esac
    
    print_status "Pulling model: $MODEL_NAME"
    ollama pull $MODEL_NAME
    
    if [ $? -eq 0 ]; then
        print_success "Model $MODEL_NAME pulled successfully"
    else
        print_error "Failed to pull model $MODEL_NAME"
        exit 1
    fi
}

# Function to setup Python environment
setup_python_env() {
    print_status "Setting up Python environment..."
    
    # Check if virtual environment exists
    if [ -d "venv" ]; then
        print_status "Virtual environment already exists"
    else
        print_status "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    print_status "Activating virtual environment..."
    source venv/bin/activate
    
    # Upgrade pip
    print_status "Upgrading pip..."
    pip install --upgrade pip
    
    # Install requirements
    print_status "Installing Python dependencies..."
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    else
        print_error "requirements.txt not found"
        exit 1
    fi
    
    print_success "Python environment setup complete"
}

# Function to setup configuration
setup_config() {
    print_status "Setting up configuration..."
    
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            cp .env.example .env
            print_status "Created .env file from template"
        else
            # Create basic .env file
            cat > .env << EOF
# RAG System Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=$MODEL_NAME

# Embedding Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

# ChromaDB Configuration
CHROMA_DB_PATH=./data/vectordb
COLLECTION_NAME=knowledge_base

# Document Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_CHUNK_SIZE=1500

# Retrieval Configuration
RETRIEVAL_K=5
SIMILARITY_THRESHOLD=0.7
RERANK_TOP_K=3

# Response Generation
TEMPERATURE=0.7
MAX_TOKENS=2048
CONTEXT_WINDOW=4096

# Performance Settings
BATCH_SIZE=100
MAX_WORKERS=4

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/rag_agent.log

# Security
ENCRYPT_SENSITIVE_DATA=true
EOF
            print_status "Created basic .env file"
        fi
    else
        print_status ".env file already exists"
    fi
    
    # Create necessary directories
    mkdir -p data/documents
    mkdir -p data/vectordb
    mkdir -p logs
    
    print_success "Configuration setup complete"
}

# Function to test installation
test_installation() {
    print_status "Testing installation..."
    
    # Test Ollama connection
    print_status "Testing Ollama connection..."
    if ollama list > /dev/null 2>&1; then
        print_success "Ollama is running and accessible"
    else
        print_error "Ollama connection failed"
        return 1
    fi
    
    # Test Python imports
    print_status "Testing Python dependencies..."
    python3 -c "
import sys
sys.path.append('src')
try:
    from src.core.rag_agent import rag_agent
    print('✓ RAG agent import successful')
    
    # Test health check
    health = rag_agent.health_check()
    if health['status'] in ['healthy', 'degraded']:
        print('✓ RAG agent health check passed')
    else:
        print('✗ RAG agent health check failed')
        sys.exit(1)
        
except Exception as e:
    print(f'✗ Import failed: {e}')
    sys.exit(1)
"
    
    if [ $? -eq 0 ]; then
        print_success "Python dependencies test passed"
    else
        print_error "Python dependencies test failed"
        return 1
    fi
    
    print_success "Installation test completed successfully"
}

# Function to display usage information
show_usage() {
    print_success "Installation completed successfully!"
    echo
    echo "Quick Start Guide:"
    echo "=================="
    echo
    echo "1. Activate the virtual environment:"
    echo "   source venv/bin/activate"
    echo
    echo "2. Start the CLI interface:"
    echo "   python interfaces/cli.py"
    echo
    echo "3. Start the web interface:"
    echo "   streamlit run interfaces/web_app.py"
    echo
    echo "4. Add documents to your knowledge base:"
    echo "   python interfaces/cli.py --add-docs /path/to/your/documents"
    echo
    echo "5. Ask questions:"
    echo "   python interfaces/cli.py --query \"What are the main findings?\""
    echo
    echo "Configuration:"
    echo "============="
    echo "- Edit .env file to customize settings"
    echo "- Log files are stored in ./logs/"
    echo "- Vector database is stored in ./data/vectordb/"
    echo
    echo "For more information, see README.md"
}

# Main installation process
main() {
    echo "================================================"
    echo "  RAG Agent for Custom Data Analysis Installer"
    echo "================================================"
    echo
    
    check_requirements
    install_ollama
    pull_model
    setup_python_env
    setup_config
    test_installation
    show_usage
}

# Check if running as root (not recommended)
if [ "$EUID" -eq 0 ]; then
    print_warning "Running as root is not recommended"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Run main installation
main
