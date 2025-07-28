@echo off
REM Enhanced Document Intelligence System - Docker Runner (Windows)
REM Quick start script for Docker deployment on Windows

echo === Enhanced Document Intelligence System - Docker Runner ===

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo Error: Docker is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)

REM Build the Docker image
echo Building Docker image...
docker build -t document-intelligence:latest .
if errorlevel 1 (
    echo Error: Failed to build Docker image
    pause
    exit /b 1
)

REM Create necessary directories if they don't exist
if not exist "input" mkdir input
if not exist "output" mkdir output
if not exist "logs" mkdir logs

REM Check if input directory has PDF files
dir /b input\*.pdf >nul 2>&1
if errorlevel 1 (
    echo Warning: No PDF files found in input\ directory
    echo Please add PDF files to the input\ directory before running
)

REM Run the container
echo Starting container...
docker run --rm ^
    -v "%cd%\input:/app/input:ro" ^
    -v "%cd%\output:/app/output" ^
    -v "%cd%\logs:/app/logs" ^
    --memory=2g ^
    --cpus=2.0 ^
    document-intelligence:latest

echo === Analysis completed. Check output\ directory for results ===
pause