@echo off
REM RAG Agent Installation Script for Windows
REM This script automates the installation and setup of the RAG Agent

setlocal enabledelayedexpansion

echo ================================================
echo   RAG Agent for Custom Data Analysis Installer
echo ================================================
echo.

REM Function to check if command exists
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

REM Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [INFO] Found Python %PYTHON_VERSION%

REM Check if Python version is >= 3.8
python -c "import sys; exit(0 if sys.version_info >= (3,8) else 1)" 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python 3.8+ is required. Found Python %PYTHON_VERSION%
    pause
    exit /b 1
)
echo [SUCCESS] Python version is compatible

REM Check for Ollama
where ollama >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Ollama not found. Please install Ollama first.
    echo.
    echo 1. Download Ollama from: https://ollama.ai/download
    echo 2. Run the installer
    echo 3. Restart this script
    echo.
    pause
    exit /b 1
) else (
    echo [SUCCESS] Ollama is installed
)

REM Start Ollama if not running
ollama list >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Starting Ollama service...
    start /B ollama serve
    timeout /t 5 >nul
)

REM Check available models and pull if needed
echo [INFO] Checking available models...
ollama list | findstr llama3 >nul
if %errorlevel% neq 0 (
    echo [INFO] Pulling recommended model (llama3)...
    echo This may take several minutes depending on your internet connection.
    ollama pull llama3
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to pull model
        pause
        exit /b 1
    )
    echo [SUCCESS] Model llama3 pulled successfully
) else (
    echo [SUCCESS] Model llama3 is already available
)

REM Setup Python virtual environment
echo [INFO] Setting up Python environment...

if exist "venv" (
    echo [INFO] Virtual environment already exists
) else (
    echo [INFO] Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo [INFO] Installing Python dependencies...
if exist "requirements.txt" (
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install requirements
        pause
        exit /b 1
    )
) else (
    echo [ERROR] requirements.txt not found
    pause
    exit /b 1
)

echo [SUCCESS] Python environment setup complete

REM Setup configuration
echo [INFO] Setting up configuration...

if not exist ".env" (
    if exist ".env.example" (
        copy .env.example .env
        echo [INFO] Created .env file from template
    ) else (
        REM Create basic .env file
        (
        echo # RAG System Configuration
        echo OLLAMA_BASE_URL=http://localhost:11434
        echo OLLAMA_MODEL=llama3
        echo.
        echo # Embedding Configuration
        echo EMBEDDING_MODEL=all-MiniLM-L6-v2
        echo EMBEDDING_DIMENSION=384
        echo.
        echo # ChromaDB Configuration
        echo CHROMA_DB_PATH=./data/vectordb
        echo COLLECTION_NAME=knowledge_base
        echo.
        echo # Document Processing
        echo CHUNK_SIZE=1000
        echo CHUNK_OVERLAP=200
        echo MAX_CHUNK_SIZE=1500
        echo.
        echo # Retrieval Configuration
        echo RETRIEVAL_K=5
        echo SIMILARITY_THRESHOLD=0.7
        echo RERANK_TOP_K=3
        echo.
        echo # Response Generation
        echo TEMPERATURE=0.7
        echo MAX_TOKENS=2048
        echo CONTEXT_WINDOW=4096
        echo.
        echo # Performance Settings
        echo BATCH_SIZE=100
        echo MAX_WORKERS=4
        echo.
        echo # Logging
        echo LOG_LEVEL=INFO
        echo LOG_FILE=./logs/rag_agent.log
        echo.
        echo # Security
        echo ENCRYPT_SENSITIVE_DATA=true
        ) > .env
        echo [INFO] Created basic .env file
    )
) else (
    echo [INFO] .env file already exists
)

REM Create necessary directories
if not exist "data\documents" mkdir data\documents
if not exist "data\vectordb" mkdir data\vectordb
if not exist "logs" mkdir logs

echo [SUCCESS] Configuration setup complete

REM Test installation
echo [INFO] Testing installation...

REM Test Ollama connection
ollama list >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Ollama connection failed
    pause
    exit /b 1
) else (
    echo [SUCCESS] Ollama is running and accessible
)

REM Test Python imports
echo [INFO] Testing Python dependencies...
python -c "import sys; sys.path.append('src'); from src.core.rag_agent import rag_agent; health = rag_agent.health_check(); print('✓ RAG agent test passed' if health['status'] in ['healthy', 'degraded'] else '✗ RAG agent test failed'); exit(0 if health['status'] in ['healthy', 'degraded'] else 1)" 2>nul

if %errorlevel% neq 0 (
    echo [ERROR] Python dependencies test failed
    echo Please check the installation and try again
    pause
    exit /b 1
) else (
    echo [SUCCESS] Python dependencies test passed
)

echo [SUCCESS] Installation test completed successfully

REM Display usage information
echo.
echo ================================================
echo   Installation completed successfully!
echo ================================================
echo.
echo Quick Start Guide:
echo ==================
echo.
echo 1. Activate the virtual environment:
echo    venv\Scripts\activate.bat
echo.
echo 2. Start the CLI interface:
echo    python interfaces\cli.py
echo.
echo 3. Start the web interface:
echo    streamlit run interfaces\web_app.py
echo.
echo 4. Add documents to your knowledge base:
echo    python interfaces\cli.py --add-docs "C:\path\to\your\documents"
echo.
echo 5. Ask questions:
echo    python interfaces\cli.py --query "What are the main findings?"
echo.
echo Configuration:
echo =============
echo - Edit .env file to customize settings
echo - Log files are stored in .\logs\
echo - Vector database is stored in .\data\vectordb\
echo.
echo For more information, see README.md
echo.
pause
