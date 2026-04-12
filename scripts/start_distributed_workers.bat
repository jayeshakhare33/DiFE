@echo off
REM Start Distributed Feature Extraction Workers
REM This script starts 4 workers and 1 aggregator

echo ============================================================
echo Starting Distributed Feature Extraction System
echo ============================================================
echo.

echo Step 1: Triggering extraction requests...
python scripts/run_distributed_feature_extraction.py --workers 4

echo.
echo Step 2: Starting feature extraction workers...
echo Starting Worker-0...
start "Worker-0" cmd /k "python storage/kafka_consumer.py --worker-id 0"

timeout /t 2 /nobreak >nul

echo Starting Worker-1...
start "Worker-1" cmd /k "python storage/kafka_consumer.py --worker-id 1"

timeout /t 2 /nobreak >nul

echo Starting Worker-2...
start "Worker-2" cmd /k "python storage/kafka_consumer.py --worker-id 2"

timeout /t 2 /nobreak >nul

echo Starting Worker-3...
start "Worker-3" cmd /k "python storage/kafka_consumer.py --worker-id 3"

timeout /t 2 /nobreak >nul

echo.
echo Step 3: Starting feature aggregator...
start "Aggregator" cmd /k "python storage/feature_aggregator.py --interval 5.0"

echo.
echo ============================================================
echo All workers and aggregator started!
echo ============================================================
echo.
echo Workers are running in separate windows.
echo Close windows to stop workers.
echo.
pause

