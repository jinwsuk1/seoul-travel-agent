@echo off
chcp 65001 > nul
echo ============================================
echo  Seoul Travel Agent - Execution Script
echo ============================================
echo.

:: 1. Terminate existing Ollama
echo [1/4] Terminating existing Ollama...
taskkill /F /IM ollama.exe /T 2>nul
timeout /t 3 /nobreak > nul
echo      Done!

:: 2. Migrate Ollama model path (Move)
echo [2/4] Migrating Ollama model path...
set SRC_DIR=%USERPROFILE%\.ollama\models
set DEST_DIR=C:\Users\Public\ollama_models

if exist "%SRC_DIR%" (
    if not exist "%DEST_DIR%" (
        echo      Moving model folder to public path...
        move "%SRC_DIR%" "%DEST_DIR%" > nul
        echo      Migration completed!
    ) else (
        echo      Target folder already exists.
    )
) else (
    echo      Source models folder not found or already migrated.
)

:: 3. Restart Ollama serve
echo [3/4] Starting Ollama server...
set OLLAMA_MODELS=C:\Users\Public\ollama_models
start /B "" ollama serve
timeout /t 5 /nobreak > nul
echo      Ollama server started!

:: 4. Run Agent
echo [4/4] Starting Seoul Travel Agent...
echo ============================================
echo.
set OLLAMA_MODELS=C:\Users\Public\ollama_models
set PYTHONIOENCODING=utf-8
python travel_agent.py

pause

