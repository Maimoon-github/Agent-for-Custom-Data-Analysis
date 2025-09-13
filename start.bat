@echo off
REM RAG Agent Startup Script for Windows
REM Quick startup script to launch the RAG Agent

echo Starting RAG Agent for Custom Data Analysis
echo ==========================================

REM Check if virtual environment exists
if not exist "venv" (
    echo Virtual environment not found. Please run install.bat first.
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if Ollama is running (basic check)
ollama list >nul 2>&1
if %errorlevel% neq 0 (
    echo Starting Ollama service...
    start /B ollama serve
    timeout /t 5 >nul
)

REM Display menu
echo.
echo Choose an interface:
echo 1. Command Line Interface ^(CLI^)
echo 2. Web Interface ^(Streamlit^)
echo 3. Run Tests
echo 4. Add Documents
echo.

set /p choice="Enter choice (1-4): "

if "%choice%"=="1" (
    echo Starting CLI...
    python interfaces\cli.py
) else if "%choice%"=="2" (
    echo Starting Web Interface...
    echo Open your browser to: http://localhost:8501
    streamlit run interfaces\web_app.py
) else if "%choice%"=="3" (
    echo Running tests...
    python test_installation.py
    pause
) else if "%choice%"=="4" (
    set /p doc_path="Enter path to documents: "
    echo Adding documents...
    python interfaces\cli.py --add-docs "!doc_path!"
) else (
    echo Invalid choice. Starting CLI...
    python interfaces\cli.py
)
