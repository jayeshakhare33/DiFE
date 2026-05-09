@echo off
echo ========================================
echo Fraud Detection System - Quick Start
echo ========================================
echo.

echo Step 1: Starting Docker services...
docker-compose up -d

echo.
echo Waiting for services to start...
timeout /t 15 /nobreak

echo.
echo Step 2: Testing Neo4j connection...
python scripts\test_neo4j.py
if %ERRORLEVEL% NEQ 0 (
    echo Warning: Neo4j test failed. Wait a bit longer and try again.
)

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Initialize Neo4j: Open http://localhost:7474 and run scripts/init_neo4j.cypher
echo 2. Extract features: python scripts/extract_all_features.py
echo.
echo Services:
echo - Neo4j Browser: http://localhost:7474
echo.
pause

