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
echo Step 2: Creating Kafka topics...
call scripts\create_kafka_topics.bat

echo.
echo Step 3: Testing connections...
echo.
echo Testing PostgreSQL...
python scripts\test_postgres.py
if %ERRORLEVEL% NEQ 0 (
    echo Warning: PostgreSQL test failed. Check if PostgreSQL is running.
)

echo.
echo Testing Neo4j...
python scripts\test_neo4j.py
if %ERRORLEVEL% NEQ 0 (
    echo Warning: Neo4j test failed. Wait a bit longer and try again.
)

echo.
echo Testing Kafka...
python scripts\test_kafka.py
if %ERRORLEVEL% NEQ 0 (
    echo Warning: Kafka test failed. Wait a bit longer and try again.
)

echo.
echo Testing Redis...
python scripts\test_redis.py
if %ERRORLEVEL% NEQ 0 (
    echo Warning: Redis test failed. Check if Redis is running.
)

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Initialize Neo4j: Open http://localhost:7474 and run scripts/init_neo4j.cypher
echo 2. Extract features: python scripts/extract_all_features.py
echo 3. Verify features: python scripts/verify_features_in_redis.py
echo.
echo Services:
echo - PostgreSQL: localhost:5432
echo - Neo4j Browser: http://localhost:7474
echo - Kafka UI: http://localhost:8080
echo - Redis: localhost:6379
echo.
pause

