@echo off
echo ========================================
echo Starting Fraud Detection Services
echo ========================================
echo.

echo Step 1: Starting Docker services...
docker-compose up -d

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Failed to start services!
    echo.
    echo Troubleshooting:
    echo 1. Check if Docker Desktop is running
    echo 2. Check if ports are already in use
    echo 3. Try: docker-compose down
    echo 4. Then: docker-compose up -d
    pause
    exit /b 1
)

echo.
echo Services starting... Please wait 15-30 seconds
echo.

timeout /t 5 /nobreak

echo Step 2: Checking service status...
docker-compose ps

echo.
echo Step 3: Waiting for services to be ready...
timeout /t 10 /nobreak

echo.
echo ========================================
echo Service Status
echo ========================================
docker-compose ps

echo.
echo ========================================
echo Service URLs
echo ========================================
echo PostgreSQL:    localhost:5432
echo Neo4j Browser: http://localhost:7474
echo Neo4j Bolt:    bolt://localhost:7687
echo Kafka:         localhost:9092
echo Kafka UI:      http://localhost:8080
echo Redis:         localhost:6379
echo.

echo ========================================
echo Next Steps
echo ========================================
echo 1. Wait 15-30 seconds for all services to start
echo 2. Test connections: python scripts\test_postgres.py
echo 3. Initialize Neo4j: Open http://localhost:7474
echo 4. Create Kafka topics: scripts\create_kafka_topics.bat
echo 5. Extract features: python scripts\extract_all_features.py
echo.

echo To view logs: docker-compose logs -f
echo To stop services: docker-compose down
echo.

pause

